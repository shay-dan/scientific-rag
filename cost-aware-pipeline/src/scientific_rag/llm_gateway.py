"""Provider-agnostic chat-model gateway with selective automatic fallback."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import os
import re
import inspect
import json
import time
from typing import Any, Callable, Iterable
from uuid import uuid4

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from model_policy import ModelPolicy
from .provider_throttle import ProviderThrottled, admit as shared_admit, cool_down as shared_cool_down, release as shared_release
from .provider_throttle import estimate_input_tokens
from .costing import calculate_cost, rate_for
from .provider_capabilities import (
    REASONING_EFFORT_SUPPORT, ROUTE_CAPABILITIES, UnsupportedReasoningEffort, validate_reasoning_effort,
)
from .provider_failures import FAILURE_CLASSES, TransmissionRefused, describe_provider_failure
from . import routing_policy


logger = logging.getLogger("ResearchAgent.LLMGateway")


class InvalidStructuredResponse(ValueError):
    """A provider responded, but its output did not satisfy the requested schema."""


def _validate_role_output(role: str, result: Any, payload: Any) -> None:
    """Reject provider control/safety metadata masquerading as a final answer."""
    if role != "expert" or not isinstance(result, str):
        return
    context = payload.get("context") if isinstance(payload, dict) else None
    # Only enforce the research-answer contract for the synthesis path, whose
    # evidence blocks carry stable source labels. Other expert calls (including
    # provider health checks) intentionally return short uncited strings.
    if not isinstance(context, str) or not re.search(r"---\s*\[S\d+:", context):
        return
    answer = result.strip()
    if re.fullmatch(r"(?:user\s+safety|safety)\s*:\s*(?:safe|unsafe)", answer, flags=re.IGNORECASE):
        raise InvalidStructuredResponse("Provider returned safety metadata instead of a research answer")
    if len(answer) < 80:
        raise InvalidStructuredResponse("Provider returned an implausibly short research answer")
    if not re.search(r"\[S\d+\]", answer):
        raise InvalidStructuredResponse("Research answer did not cite any supplied source label")

class OutputTokenLimitError(ValueError):
    """Base class for refusals of an explicitly requested output-token limit (R6D)."""


class InvalidOutputTokenLimit(OutputTokenLimitError):
    """The requested limit is not a strictly positive integer."""


class UnsupportedOutputTokenLimit(OutputTokenLimitError):
    """A configured provider cannot enforce a verified output-token limit."""


class OutputTokenLimitNotEnforced(OutputTokenLimitError):
    """The outbound provider request did not carry the requested limit, or could not be verified."""


# R6D: providers whose outbound request is verified to carry an output-token limit, with the
# request-body parameter names accepted as that limit. Both use the OpenAI-compatible chat
# completions API through ``ChatOpenAI`` (``max_tokens`` constructor argument). Gemini is not
# supported for bounded requests: its outbound request is not verified by this gateway.
OUTPUT_TOKEN_LIMIT_PARAMETERS: dict[str, tuple[str, ...]] = {
    "groq": ("max_completion_tokens", "max_tokens"),
    "openrouter": ("max_completion_tokens", "max_tokens"),
}


def validate_max_output_tokens(value: Any) -> int | None:
    """Return the limit, or None when unspecified. Only a strictly positive int is accepted."""
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise InvalidOutputTokenLimit("max_output_tokens must be a strictly positive integer")
    return value


def output_token_limit_parameters(provider: str) -> tuple[str, ...]:
    """Outbound parameter names that enforce the limit for a provider; refuses unsupported ones."""
    names = OUTPUT_TOKEN_LIMIT_PARAMETERS.get(provider)
    if not names:
        raise UnsupportedOutputTokenLimit(
            f"provider {provider!r} has no verified output-token limit; bounded requests are refused")
    return names


def output_token_limit_support() -> dict[str, tuple[str, ...] | None]:
    return {provider: OUTPUT_TOKEN_LIMIT_PARAMETERS.get(provider) for provider in DEFAULT_PROVIDER_ORDER}


class BoundedRequestLimitExceeded(OutputTokenLimitNotEnforced):
    """A bounded provider attempt tried to send more than one outbound request."""


class ReasoningEffortNotEnforced(OutputTokenLimitNotEnforced):
    """The outbound bounded request did not carry the requested reasoning effort; it was not sent."""


class _OutputLimitGuard:
    """Checks every outbound request of a bounded model, before it is sent.

    One guard belongs to one bounded model, which the gateway builds for one provider attempt.
    It admits at most one outbound request: a provider attempt is recorded and budgeted as one
    call, so a second request (a chain invoking the model again, or a hand-rolled retry, also
    after a failed first request) is refused before it is transmitted. ``check`` has no await,
    so concurrent requests cannot both be admitted. The first refusal is kept in ``refusal``
    so the gateway fails the attempt even if the chain swallowed the error.
    """

    def __init__(self, provider: str, limit: int, reasoning_effort: str | None = None,
                 required_fields: dict[str, Any] | None = None) -> None:
        self.provider, self.limit, self.verified_requests = provider, limit, 0
        self.reasoning_effort = reasoning_effort
        # Stage 20G: further request-body fields the outbound request must carry exactly
        # (e.g. OpenRouter provider preferences that disable hidden upstream fallback).
        self.required_fields = dict(required_fields or {})
        self.refusal: OutputTokenLimitError | None = None
        self.response_status: int | None = None   # HTTP status of the one response, once received
        # Provider-reported usage read from the one successful HTTP response (numbers only). It
        # survives a local parsing/validation failure, which can prevent LangChain callbacks.
        self.response_usage: tuple[int, int] | None = None
        self.response_reasoning_tokens: int | None = None
        self.response_finish_reason: str | None = None

    def check(self, content: bytes) -> None:
        try:
            self._check(content)
        except OutputTokenLimitError as error:
            if self.refusal is None:
                self.refusal = error
            raise

    def _check(self, content: bytes) -> None:
        if self.verified_requests >= 1:
            raise BoundedRequestLimitExceeded(
                f"a bounded {self.provider} attempt may send one outbound request; a further request was not sent")
        try:
            body = json.loads(content)
        except (ValueError, TypeError):
            body = None
        sent = ({name: body[name] for name in OUTPUT_TOKEN_LIMIT_PARAMETERS[self.provider]
                 if body.get(name) is not None} if isinstance(body, dict) else {})
        if not sent or any(type(value) is not int or value != self.limit for value in sent.values()):
            raise OutputTokenLimitNotEnforced(
                f"outbound {self.provider} request does not carry the requested output-token limit; not sent")
        if body.get("n") not in (None, 1):   # several completions would multiply the bounded output
            raise OutputTokenLimitNotEnforced(
                f"outbound {self.provider} request asks for more than one completion; not sent")
        if self.reasoning_effort is not None and body.get("reasoning_effort") != self.reasoning_effort:
            raise ReasoningEffortNotEnforced(
                f"outbound {self.provider} request does not carry the requested reasoning effort; not sent")
        if any(body.get(name) != value for name, value in self.required_fields.items()):
            raise OutputTokenLimitNotEnforced(
                f"outbound {self.provider} request does not carry the required routing fields; not sent")
        self.verified_requests += 1

    def sync_hook(self, request: httpx.Request) -> None:
        self.check(request.read())

    async def async_hook(self, request: httpx.Request) -> None:
        self.check(await request.aread())

    def _record_response(self, status: int, content: bytes) -> None:
        self.response_status = status
        if status != 200:
            return
        try:
            body = json.loads(content)
        except (ValueError, TypeError):
            return
        if not isinstance(body, dict):
            return
        usage = body.get("usage")
        input_tokens, output_tokens = _usage_pair(usage)
        if input_tokens is not None and output_tokens is not None:
            self.response_usage = (input_tokens, output_tokens)
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if type(reasoning) is int and reasoning >= 0:
            self.response_reasoning_tokens = reasoning
        choices = body.get("choices")
        finish = choices[0].get("finish_reason") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if isinstance(finish, str) and re.fullmatch(r"[a-z_]{1,32}", finish):
            self.response_finish_reason = finish

    def sync_response_hook(self, response: httpx.Response) -> None:
        self._record_response(response.status_code, response.read())

    async def async_response_hook(self, response: httpx.Response) -> None:
        self._record_response(response.status_code, await response.aread())


class _WireRecorder:
    """Stage 20G: transmission facts for an unbounded routed attempt (no request checks).

    The request hook runs immediately before the transport sends; the response hook runs
    once a status line has been received. Together they distinguish "never sent", "sent with
    a definitive response" and "sent with an unknown outcome".
    """

    def __init__(self) -> None:
        self.verified_requests = 0
        self.response_status: int | None = None
        self.refusal = None

    def _single_request(self) -> None:
        if self.verified_requests:
            self.refusal = BoundedRequestLimitExceeded(
                "a routed provider attempt may transmit at most one physical request")
            raise self.refusal
        self.verified_requests += 1

    def sync_hook(self, _request: httpx.Request) -> None:
        self._single_request()

    async def async_hook(self, _request: httpx.Request) -> None:
        self._single_request()

    def sync_response_hook(self, response: httpx.Response) -> None:
        self.response_status = response.status_code

    async def async_response_hook(self, response: httpx.Response) -> None:
        self.response_status = response.status_code


def _outbound_transport() -> httpx.BaseTransport | None:
    """Transport for bounded-request clients. None selects the normal network transport."""
    return None


def _output_limit_guard(model: Any) -> _OutputLimitGuard:
    guard = getattr(getattr(model, "http_async_client", None), "output_limit_guard", None)
    if not isinstance(guard, _OutputLimitGuard):
        raise OutputTokenLimitNotEnforced("the chat model has no outbound output-token limit guard")
    return guard


def _output_limit_cause(error: BaseException) -> OutputTokenLimitError | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, OutputTokenLimitError):
            return current
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return None


async def _close_bounded_clients(model: Any) -> None:
    for name in ("http_async_client", "http_client"):
        client = getattr(model, name, None)
        try:
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
            elif isinstance(client, httpx.Client):
                client.close()
        except Exception as error:   # closing is best effort
            logger.warning("llm_gateway_bounded_client_close_failed error_type=%s", type(error).__name__)


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROVIDER_ORDER = ("groq", "openrouter", "gemini")
RETRYABLE_STATUS_CODES = {408, 409, 429, *range(500, 600)}
RETRYABLE_MARKERS = (
    "connection error",
    "connection reset",
    "deadline exceeded",
    "model not found",
    "model unavailable",
    "rate limit",
    "resource exhausted",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
    "invalid json output",
    "output_parsing_failure",
)
RATE_LIMIT_HEADER_NAMES = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    # Separate input/output token limits, when a provider or account reports them. Captured
    # only in the opt-in rate-limit capture mode used by the R7D evaluation runner.
    "x-ratelimit-limit-input-tokens",
    "x-ratelimit-remaining-input-tokens",
    "x-ratelimit-reset-input-tokens",
    "x-ratelimit-limit-output-tokens",
    "x-ratelimit-remaining-output-tokens",
    "x-ratelimit-reset-output-tokens",
)


def _rate_limit_headers(value: Any) -> dict[str, str]:
    if not value or not hasattr(value, "items"):
        return {}
    lowered = {str(key).lower(): str(item) for key, item in value.items()}
    return {name: lowered[name] for name in RATE_LIMIT_HEADER_NAMES if name in lowered}


def _error_rate_limit_headers(error: BaseException) -> dict[str, str]:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        headers = getattr(getattr(current, "response", None), "headers", None) or getattr(current, "headers", None)
        captured = _rate_limit_headers(headers)
        if captured:
            return captured
        current = current.__cause__ or current.__context__
    return {}


@dataclass(frozen=True)
class ProviderAttempt:
    call_id: str
    attempt_index: int
    provider: str
    model: str
    role: str
    latency_ms: float
    success: bool
    retryable: bool
    status_code: int | None = None
    error_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    recovery: bool = False
    context_chars: int | None = None
    skipped: bool = False
    admission_wait_ms: float = 0.0
    rate_limit_headers: dict[str, str] | None = None
    # R7D-C: sanitized structured diagnostics (no text, prompts or credentials). Serialized only
    # in the opt-in capture mode, so production run traces keep their existing shape.
    diagnostics: dict[str, Any] | None = None


_trace: ContextVar[list[ProviderAttempt] | None] = ContextVar("llm_gateway_trace", default=None)
_provider_cooldowns: ContextVar[dict[str, float] | None] = ContextVar("llm_gateway_provider_cooldowns", default=None)
_provider_failures: ContextVar[dict[str, int] | None] = ContextVar("llm_gateway_provider_failures", default=None)
_shared_redis: ContextVar[Any | None] = ContextVar("llm_gateway_shared_redis", default=None)
_capture_rate_limits: ContextVar[bool] = ContextVar("llm_gateway_capture_rate_limits", default=False)
# Stage 20G (consolidated routing only): an optional accounting hook consulted before every
# physical attempt, and an optional observer receiving sanitized routing decisions.
_routing_budget: ContextVar[Any | None] = ContextVar("llm_gateway_routing_budget", default=None)
_routing_observer: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "llm_gateway_routing_observer", default=None)
# Injectable clock for deterministic tests of bounded waits.
_monotonic = time.monotonic
_sleep = asyncio.sleep


class RoutingStopped(RuntimeError):
    """The consolidated routing policy found no admissible transmission (nothing was sent)."""

    def __init__(self, role: str, reason: str) -> None:
        super().__init__(f"no admissible route for role {role}: {reason}")
        self.role, self.reason = role, reason
        self.routing_reason = reason


ROUTING_MODES = ("legacy", "consolidated")


def routing_mode() -> str:
    value = os.getenv("LLM_ROUTING_POLICY", "legacy").strip().lower() or "legacy"
    if value not in ROUTING_MODES:
        raise ValueError(f"LLM_ROUTING_POLICY must be one of {', '.join(ROUTING_MODES)}")
    return value


class routing_controls:
    """Install an accounting ``budget`` (``can_reserve(provider, model) -> bool``) and/or an
    ``observer`` for consolidated routing in the current context."""

    def __init__(self, *, budget: Any | None = None, observer: Callable[[dict[str, Any]], None] | None = None):
        self.budget, self.observer = budget, observer
        self._tokens: tuple[Token, Token] | None = None

    def __enter__(self):
        self._tokens = (_routing_budget.set(self.budget), _routing_observer.set(self.observer))
        return self

    def __exit__(self, *_exc):
        budget_token, observer_token = self._tokens
        _routing_observer.reset(observer_token)
        _routing_budget.reset(budget_token)
        return False


class _TokenUsageHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.has_usage = False
        self.rate_limit_headers: dict[str, str] = {}
        self.reasoning_tokens: int | None = None
        self.finish_reason: str | None = None

    def on_llm_end(self, response, **kwargs) -> None:
        usage = getattr(response, "llm_output", None) or {}
        reasoning, finish = _reasoning_and_finish(response)
        if reasoning is not None:
            self.reasoning_tokens = reasoning
        if finish is not None:
            self.finish_reason = finish
        self.rate_limit_headers.update(_rate_limit_headers(usage.get("headers") or usage.get("response_headers")))
        messages = [getattr(generation, "message", None)
                    for group in getattr(response, "generations", []) or [] for generation in group]
        for message in messages:
            metadata = getattr(message, "response_metadata", None) or {}
            self.rate_limit_headers.update(
                _rate_limit_headers(metadata.get("headers") or metadata.get("response_headers")))
        token_usage = usage.get("token_usage") or usage.get("usage") or {}
        input_tokens, output_tokens = _usage_pair(token_usage)
        if input_tokens is not None and output_tokens is not None:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.has_usage = True
            return
        for message in messages:
            input_tokens, output_tokens = _usage_counts(message)
            if input_tokens is not None and output_tokens is not None:
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.has_usage = True


def _reasoning_and_finish(response: Any) -> tuple[int | None, str | None]:
    """Provider-reported reasoning tokens and finish reason, when present; never estimated."""
    reasoning: int | None = None
    finish: str | None = None
    usage_sources: list[Any] = []
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        usage_sources.append(llm_output.get("token_usage") or llm_output.get("usage"))
    for group in getattr(response, "generations", []) or []:
        for generation in group:
            info = getattr(generation, "generation_info", None) or {}
            message = getattr(generation, "message", None)
            metadata = getattr(message, "response_metadata", None) or {}
            for candidate in (info.get("finish_reason"), metadata.get("finish_reason")):
                if isinstance(candidate, str) and re.fullmatch(r"[a-z_]{1,32}", candidate):
                    finish = candidate
            usage_sources.extend([metadata.get("token_usage"), getattr(message, "usage_metadata", None)])
    for usage in usage_sources:
        if not isinstance(usage, dict):
            continue
        details = usage.get("completion_tokens_details") or usage.get("output_token_details") or {}
        value = details.get("reasoning_tokens", details.get("reasoning")) if isinstance(details, dict) else None
        if type(value) is int and value >= 0:
            reasoning = value
    return reasoning, finish


def _usage_pair(usage: Any) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if input_tokens is None or output_tokens is None:
        return None, None
    return int(input_tokens), int(output_tokens)


def begin_trace(
    redis: Any | None = None, *, capture_rate_limits: bool = False,
) -> tuple[Token, Token, Token, Token, Token]:
    """Start an isolated trace for one research run."""
    return (_trace.set([]), _provider_cooldowns.set({}), _provider_failures.set({}),
            _shared_redis.set(redis), _capture_rate_limits.set(bool(capture_rate_limits)))


def end_trace(token: tuple[Token, Token, Token, Token, Token]) -> list[dict[str, Any]]:
    """Return serializable attempt metadata and restore the prior context."""
    attempts = list(_trace.get() or [])
    capture_rate_limits = _capture_rate_limits.get()
    _trace.reset(token[0])
    _provider_cooldowns.reset(token[1])
    _provider_failures.reset(token[2])
    _shared_redis.reset(token[3])
    _capture_rate_limits.reset(token[4])
    serialized = [asdict(attempt) for attempt in attempts]
    if not capture_rate_limits:
        for attempt in serialized:
            attempt.pop("rate_limit_headers", None)
            attempt.pop("diagnostics", None)
    return serialized


def summarize_trace(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    token_usage_available = False
    usage_attempts = 0
    estimated_cost_usd = 0.0
    cost_available = True
    for attempt in attempts:
        provider = str(attempt["provider"])
        model = str(attempt["model"])
        bucket = by_provider.setdefault(provider, {"attempts": 0, "successes": 0, "failures": 0, "skips": 0, "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0})
        if attempt.get("skipped"):
            bucket["skips"] += 1
            continue
        bucket["attempts"] += 1
        bucket["successes" if attempt["success"] else "failures"] += 1
        bucket["latency_ms"] = round(bucket["latency_ms"] + float(attempt["latency_ms"]), 1)
        model_bucket = by_model.setdefault(f"{provider}:{model}", {"provider": provider, "model": model, "attempts": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cost_available": True})
        model_bucket["attempts"] += 1
        input_tokens = attempt.get("input_tokens")
        output_tokens = attempt.get("output_tokens")
        if input_tokens is not None and output_tokens is not None:
            token_usage_available = True
            usage_attempts += 1
            total_input_tokens += int(input_tokens)
            total_output_tokens += int(output_tokens)
            bucket["input_tokens"] += int(input_tokens)
            bucket["output_tokens"] += int(output_tokens)
            model_bucket["input_tokens"] += int(input_tokens)
            model_bucket["output_tokens"] += int(output_tokens)
            call_cost, rate_source = calculate_cost(provider, model, int(input_tokens), int(output_tokens))
            if call_cost is None:
                cost_available = False
                model_bucket["cost_available"] = False
            else:
                estimated_cost_usd += call_cost
                model_bucket["estimated_cost_usd"] += call_cost
                model_bucket["rate_source"] = rate_source
        else:
            cost_available = False
            model_bucket["cost_available"] = False
    failures = sum(1 for attempt in attempts if not attempt["success"] and not attempt.get("skipped"))
    recovery_attempts = [attempt for attempt in attempts if attempt.get("recovery")]
    skipped_attempts = [attempt for attempt in attempts if attempt.get("skipped")]
    throttle_skips = [attempt for attempt in skipped_attempts if attempt.get("error_type") == "ProviderThrottled"]
    policy_skips = [attempt for attempt in skipped_attempts if attempt.get("error_type") == "ExpertContextLimit"]
    call_groups: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        call_groups.setdefault(str(attempt["call_id"]), []).append(attempt)
    provider_failovers = 0
    throttle_reroutes = 0
    for call_attempts in call_groups.values():
        called = [attempt for attempt in call_attempts if not attempt.get("skipped")]
        failed_providers = {str(attempt["provider"]) for attempt in called if not attempt.get("success")}
        successful_providers = {str(attempt["provider"]) for attempt in called if attempt.get("success")}
        if failed_providers and successful_providers and failed_providers != successful_providers:
            provider_failovers += 1
        if any(attempt.get("skipped") and attempt.get("error_type") == "ProviderThrottled" for attempt in call_attempts) and called:
            throttle_reroutes += 1
    for bucket in by_model.values():
        bucket["estimated_cost_usd"] = round(bucket["estimated_cost_usd"], 8) if bucket["cost_available"] else None
    for bucket in by_provider.values():
        bucket["average_latency_ms"] = round(bucket["latency_ms"] / bucket["attempts"], 1) if bucket["attempts"] else 0.0
    return {
        "attempts": attempts,
        "successful_calls": sum(1 for attempt in attempts if attempt["success"]),
        "failed_attempts": failures,
        "throttle_skips": len(throttle_skips),
        "policy_skips": len(policy_skips),
        "provider_calls": len(attempts) - len(skipped_attempts),
        "admission_wait_ms": round(sum(float(attempt.get("admission_wait_ms") or 0) for attempt in attempts), 1),
        "recovery_attempts": len(recovery_attempts),
        "recovery_latency_ms": round(sum(float(attempt["latency_ms"]) for attempt in recovery_attempts), 1),
        # Backwards-compatible name, now limited to real failed-provider ->
        # successful-provider transitions. Admission skips are reported
        # separately and no longer inflate the fallback number.
        "fallback_count": provider_failovers,
        "provider_failover_count": provider_failovers,
        "throttle_reroute_count": throttle_reroutes,
        "by_provider": by_provider,
        "by_model": by_model,
        "input_tokens": total_input_tokens if token_usage_available else None,
        "output_tokens": total_output_tokens if token_usage_available else None,
        "usage_attempts": usage_attempts,
        "total_attempts": len(attempts) - len(skipped_attempts),
        "token_usage_complete": bool(len(attempts) - len(skipped_attempts)) and usage_attempts == len(attempts) - len(skipped_attempts),
        "estimated_cost_usd": round(estimated_cost_usd, 8) if cost_available and token_usage_available else None,
    }


def _usage_counts(result: Any) -> tuple[int | None, int | None]:
    response_metadata = getattr(result, "response_metadata", None) or {}
    candidates = [getattr(result, "usage_metadata", None), response_metadata.get("token_usage")]
    if isinstance(result, dict):
        raw = result.get("raw")
        if raw is not None:
            raw_metadata = getattr(raw, "response_metadata", None) or {}
            candidates.extend([getattr(raw, "usage_metadata", None), raw_metadata.get("token_usage")])
    for usage in candidates:
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_tokens is not None and output_tokens is not None:
            return int(input_tokens), int(output_tokens)
    return None, None


async def _ainvoke_with_usage(chain: Any, payload: Any, usage_handler: _TokenUsageHandler) -> Any:
    try:
        parameters = inspect.signature(chain.ainvoke).parameters.values()
        accepts_config = any(parameter.name == "config" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    except (TypeError, ValueError):
        accepts_config = False
    if accepts_config:
        return await chain.ainvoke(payload, config={"callbacks": [usage_handler]})
    return await chain.ainvoke(payload)


def _attempt_diagnostics(usage_handler: "_TokenUsageHandler", limit: int | None, reasoning_effort: str | None,
                         usage_complete: bool) -> dict[str, Any]:
    return {"reasoning_effort": reasoning_effort, "max_output_tokens": limit, "usage_complete": usage_complete,
            "reasoning_tokens": usage_handler.reasoning_tokens, "finish_reason": usage_handler.finish_reason}


def _merge_wire_details(usage_handler: "_TokenUsageHandler", guard: "_OutputLimitGuard | None") -> None:
    if guard is None:
        return
    if usage_handler.reasoning_tokens is None:
        usage_handler.reasoning_tokens = guard.response_reasoning_tokens
    if usage_handler.finish_reason is None:
        usage_handler.finish_reason = guard.response_finish_reason


def _record(attempt: ProviderAttempt) -> None:
    attempts = _trace.get()
    if attempts is not None:
        attempts.append(attempt)


def provider_order(role: str | None = None) -> tuple[str, ...]:
    role_override = os.getenv(f"LLM_PROVIDER_ORDER_{role.upper()}") if role else None
    configured = role_override or os.getenv("LLM_PROVIDER_ORDER", ",".join(DEFAULT_PROVIDER_ORDER))
    providers = tuple(item.strip().lower() for item in configured.split(",") if item.strip())
    unknown = set(providers) - set(DEFAULT_PROVIDER_ORDER)
    if unknown:
        raise ValueError(f"Unknown LLM provider(s): {', '.join(sorted(unknown))}")
    if not providers:
        raise ValueError("LLM_PROVIDER_ORDER must include at least one provider")
    return providers


def _key_for(provider: str) -> str:
    return {
        "groq": os.getenv("GROQ_API_KEY", ""),
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
        "gemini": os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""),
    }[provider]


def model_for(provider: str, role: str) -> str:
    role_key = role.upper()
    if provider == "groq":
        role_default = "openai/gpt-oss-20b" if role == "query_repair" else os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return os.getenv(f"GROQ_MODEL_{role_key}", role_default)
    if provider == "openrouter":
        role_default = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-35b-a3b")
        return os.getenv(f"OPENROUTER_MODEL_{role_key}", role_default)
    return ModelPolicy.from_environment().route_for(role).primary


def primary_route_label(role: str) -> str:
    cooldowns = _provider_cooldowns.get() or {}
    for provider in provider_order(role):
        if _key_for(provider) and cooldowns.get(provider, 0) <= time.monotonic():
            return f"{provider}:{model_for(provider, role)}"
    return "unconfigured"


def openrouter_provider_preferences(model: str) -> dict[str, Any]:
    """Stage 20G routed OpenRouter requests: no hidden upstream fallback, every parameter
    honoured, and a per-token price ceiling so a modeled reservation is a real upper bound.

    The ceiling defaults to the dated rate in the pricing table; operators may set
    ``LLM_OPENROUTER_MAX_PRICE_PROMPT``/``LLM_OPENROUTER_MAX_PRICE_COMPLETION`` (USD per
    million tokens). Without any rate the route is refused.
    """
    rate = rate_for("openrouter", model)
    prompt = os.getenv("LLM_OPENROUTER_MAX_PRICE_PROMPT")
    completion = os.getenv("LLM_OPENROUTER_MAX_PRICE_COMPLETION")
    prompt_value = float(prompt) if prompt else (rate.input_usd_per_million if rate else None)
    completion_value = float(completion) if completion else (rate.output_usd_per_million if rate else None)
    if prompt_value is None or completion_value is None or min(prompt_value, completion_value) < 0:
        raise ValueError(f"routed OpenRouter model {model!r} has no price ceiling")
    return {"require_parameters": True, "allow_fallbacks": False,
            "max_price": {"prompt": prompt_value, "completion": completion_value}}


def create_chat_model(provider: str, role: str, temperature: float, max_output_tokens: int | None = None,
                      reasoning_effort: str | None = None, *, model: str | None = None, routing: bool = False):
    """Build the chat model for a provider and role.

    ``max_output_tokens`` (R6D) is optional. When omitted the model is built exactly as
    before. When given, the provider must support a verified limit: the limit is set as a
    request parameter, and a guard on the model's own HTTP clients refuses any outbound
    request that does not carry it.

    ``reasoning_effort`` (R7D-C) is optional and only valid with a bounded request on a
    supported route; the same guard refuses an outbound request that does not carry it.

    ``model``/``routing`` (Stage 20G) are used by the consolidated routing path: the route's
    model is explicit, every OpenAI-compatible attempt records its wire transmission facts,
    and routed OpenRouter requests carry (and the guard verifies) fixed provider preferences.
    """
    limit = validate_max_output_tokens(max_output_tokens)
    if limit is not None:
        output_token_limit_parameters(provider)
    model = model or model_for(provider, role)
    effort = validate_reasoning_effort(provider, model, reasoning_effort, routing=routing)
    if effort is not None and limit is None:
        raise UnsupportedReasoningEffort("reasoning effort requires a verified bounded request (max_output_tokens)")
    timeout = provider_timeout_seconds(provider)
    api_key = _key_for(provider)
    if not api_key:
        raise ValueError(f"{provider.upper()} API key is not configured")

    if provider in {"groq", "openrouter"}:
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": GROQ_BASE_URL if provider == "groq" else OPENROUTER_BASE_URL,
            "timeout": timeout,
            "max_retries": 0,
            "temperature": temperature,
        }
        if provider == "openrouter":
            kwargs["default_headers"] = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/shay-dan/scientific-rag"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Scientific RAG"),
            }
        if _capture_rate_limits.get():
            kwargs["include_response_headers"] = True
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        required: dict[str, Any] = {}
        if routing and provider == "openrouter":
            required = {"provider": openrouter_provider_preferences(model)}
            kwargs["extra_body"] = dict(required)
        if limit is None and routing:
            recorder = _WireRecorder()
            kwargs["http_client"] = httpx.Client(
                timeout=timeout, follow_redirects=False, transport=_outbound_transport(),
                event_hooks={"request": [recorder.sync_hook], "response": [recorder.sync_response_hook]})
            kwargs["http_async_client"] = httpx.AsyncClient(
                timeout=timeout, follow_redirects=False, transport=_outbound_transport(),
                event_hooks={"request": [recorder.async_hook], "response": [recorder.async_response_hook]})
            kwargs["http_async_client"].output_limit_guard = recorder
        if limit is not None:
            guard = _OutputLimitGuard(provider, limit, effort, required)
            kwargs["max_tokens"] = limit
            # Redirects are not followed: a redirected POST would be a second, uncounted request.
            kwargs["http_client"] = httpx.Client(
                timeout=timeout, follow_redirects=False, transport=_outbound_transport(),
                event_hooks={"request": [guard.sync_hook], "response": [guard.sync_response_hook]})
            kwargs["http_async_client"] = httpx.AsyncClient(
                timeout=timeout, follow_redirects=False, transport=_outbound_transport(),
                event_hooks={"request": [guard.async_hook], "response": [guard.async_response_hook]})
            kwargs["http_async_client"].output_limit_guard = guard
        return ChatOpenAI(**kwargs)

    kwargs = {"model": model, "google_api_key": api_key, "timeout": timeout, "max_retries": 0}
    kwargs.update(ModelPolicy.from_environment().generation_parameters_for_model(model, temperature))
    return ChatGoogleGenerativeAI(**kwargs)


def provider_timeout_seconds(provider: str) -> float:
    """A per-attempt deadline; the legacy global value remains an upper bound."""
    defaults = {"groq": 20, "openrouter": 12, "gemini": 45}
    ceiling = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90"))
    selected = float(os.getenv(f"LLM_{provider.upper()}_TIMEOUT_SECONDS", str(defaults[provider])))
    if ceiling <= 0 or selected <= 0:
        raise ValueError("LLM provider timeouts must be positive")
    return min(selected, ceiling)


def _status_code(error: BaseException) -> int | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        value = getattr(current, "status_code", None)
        if isinstance(value, int):
            return value
        response = getattr(current, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
        current = current.__cause__ or current.__context__
    return None


def is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, InvalidStructuredResponse):
        return True
    # Some LangChain integrations raise OutputParserException directly instead
    # of returning include_raw parsing metadata. Treat malformed model output as
    # provider-specific so the next configured provider can try the same schema.
    if error.__class__.__name__ == "OutputParserException":
        return True
    status = _status_code(error)
    if status is not None:
        if status in RETRYABLE_STATUS_CODES:
            return True
        if status != 404:
            return False
        # A provider/router may use 404 for temporary model availability, but a
        # generic endpoint 404 is a configuration error and must remain visible.
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "model not found",
                "model_not_found",
                "model unavailable",
                "model does not exist",
                "does not exist or you do not have access",
            )
        )
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    message = str(error).lower()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def _configured_providers(role: str | None = None) -> Iterable[str]:
    for provider in provider_order(role):
        if _key_for(provider):
            yield provider
        else:
            logger.info("Skipping unconfigured LLM provider %s", provider)


def _retry_after_seconds(error: BaseException) -> float | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        headers = getattr(getattr(current, "response", None), "headers", None) or getattr(current, "headers", None)
        value = headers.get("retry-after") if headers else None
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        current = current.__cause__ or current.__context__
    return None


def create_primary_chat_model(role: str, temperature: float):
    """Build the first configured provider model for constructor compatibility."""
    provider = next(iter(_configured_providers(role)), None)
    if provider is None:
        raise ValueError("No LLM provider API key is configured")
    return create_chat_model(provider, role, temperature)


async def ainvoke(role: str, temperature: float, chain_builder: Callable[[Any], Any], payload: Any, *,
                  max_output_tokens: int | None = None, reasoning_effort: str | None = None):
    """Invoke a chain and fail over only for transient/provider-availability errors.

    ``max_output_tokens`` (R6D, optional) bounds every provider attempt of this call. It is
    refused unless every configured provider for the role can enforce it, so a bounded call
    can never fall back to an unbounded request. Without it, behavior is unchanged.

    ``reasoning_effort`` (R7D-C, optional) requires ``max_output_tokens`` and a supported
    route for every configured provider; it is never silently dropped or applied elsewhere.
    """
    limit = validate_max_output_tokens(max_output_tokens)
    if routing_mode() == "consolidated":
        return await _ainvoke_routed(role, temperature, chain_builder, payload, limit=limit,
                                     reasoning_effort=reasoning_effort)
    providers = list(_configured_providers(role))
    if not providers:
        raise ValueError("No LLM provider API key is configured")
    if limit is not None:
        for configured in providers:
            output_token_limit_parameters(configured)
    if reasoning_effort is not None:
        if limit is None:
            raise UnsupportedReasoningEffort("reasoning effort requires a verified bounded request (max_output_tokens)")
        for configured in providers:
            validate_reasoning_effort(configured, model_for(configured, role), reasoning_effort)

    last_error: BaseException | None = None
    call_id = str(uuid4())
    cooldowns = _provider_cooldowns.get()
    failures = _provider_failures.get()
    shared_redis = _shared_redis.get()
    call_budget = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90"))
    recovery_min_remaining = float(os.getenv("LLM_RECOVERY_MIN_REMAINING_SECONDS", "10"))
    cooldown_seconds = float(os.getenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60"))
    if min(call_budget, recovery_min_remaining, cooldown_seconds) <= 0:
        raise ValueError("LLM gateway time budgets must be positive")
    deadline = time.monotonic() + call_budget
    skipped: list[str] = []
    attempted: set[str] = set()
    pending = list(providers)
    recovery_phase = False
    context_chars = len(payload["context"]) if role == "expert" and isinstance(payload, dict) and isinstance(payload.get("context"), str) else None
    openrouter_expert_max_context = int(os.getenv("LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS", "6000"))
    if openrouter_expert_max_context < 0:
        raise ValueError("LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS cannot be negative")
    while pending and len(attempted) < 3:
        provider = pending.pop(0)
        if not recovery_phase and cooldowns is not None and cooldowns.get(provider, 0) > time.monotonic() and pending:
            skipped.append(provider)
            logger.info("llm_gateway_skip_cooldown provider=%s role=%s", provider, role)
            continue
        if recovery_phase and cooldowns is not None and cooldowns.get(provider, 0) > time.monotonic():
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        model = model_for(provider, role)
        if (provider == "openrouter" and role == "expert" and context_chars is not None
                and openrouter_expert_max_context > 0 and context_chars > openrouter_expert_max_context and pending):
            skipped.append(provider)
            _record(ProviderAttempt(call_id, len(attempted), provider, model, role, 0.0, False, True,
                                    error_type="ExpertContextLimit", context_chars=context_chars, skipped=True))
            logger.info("llm_gateway_skip_large_expert_context provider=%s context_chars=%s limit=%s",
                        provider, context_chars, openrouter_expert_max_context)
            continue
        admission = None
        if shared_redis is not None:
            try:
                admission = await shared_admit(shared_redis, provider, role, payload, remaining)
            except ProviderThrottled as error:
                last_error = error
                skipped.append(provider)
                _record(ProviderAttempt(call_id, len(attempted), provider, model, role, 0.0, False, True,
                                        error_type=type(error).__name__, recovery=recovery_phase, skipped=True))
                logger.info("llm_gateway_shared_throttle provider=%s role=%s reason=%s retry_after_ms=%s",
                            provider, role, error.reason, error.retry_after_ms)
                continue
        attempt_index = len(attempted)
        attempted.add(provider)
        started = time.perf_counter()
        usage_handler = _TokenUsageHandler()
        actual_tokens: int | None = None
        bounded_model = None
        guard = None
        chain = None
        try:
            if limit is None:
                chain = chain_builder(create_chat_model(provider, role, temperature))
            else:
                bounded_model = create_chat_model(provider, role, temperature, max_output_tokens=limit,
                                                  reasoning_effort=reasoning_effort)
                guard = _output_limit_guard(bounded_model)
                chain = chain_builder(bounded_model)
            result = await asyncio.wait_for(
                _ainvoke_with_usage(chain, payload, usage_handler),
                timeout=min(provider_timeout_seconds(provider), max(0.001, deadline - time.monotonic())),
            )
            if limit is not None:
                if guard.refusal is not None:   # refused request(s), even if the chain swallowed the error
                    raise guard.refusal
                if guard.verified_requests != 1:
                    raise OutputTokenLimitNotEnforced("the response did not come from one verified bounded request")
            _validate_role_output(role, result, payload)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            input_tokens, output_tokens = _usage_counts(result)
            if input_tokens is None and usage_handler.has_usage:
                input_tokens, output_tokens = usage_handler.input_tokens, usage_handler.output_tokens
            if input_tokens is None and guard is not None and guard.response_usage is not None:
                input_tokens, output_tokens = guard.response_usage
            _merge_wire_details(usage_handler, guard)
            if input_tokens is not None and output_tokens is not None:
                actual_tokens = input_tokens + output_tokens
            _record(ProviderAttempt(call_id, attempt_index, provider, model, role, latency_ms, True, False,
                                    input_tokens=input_tokens, output_tokens=output_tokens,
                                    recovery=recovery_phase, context_chars=context_chars,
                                    admission_wait_ms=admission.wait_ms if admission else 0,
                                    rate_limit_headers=usage_handler.rate_limit_headers or None,
                                    diagnostics=_attempt_diagnostics(
                                        usage_handler, limit, reasoning_effort,
                                        input_tokens is not None and output_tokens is not None)))
            if failures is not None:
                failures[provider] = 0
            if cooldowns is not None:
                cooldowns.pop(provider, None)
            logger.info("llm_gateway_success provider=%s model=%s role=%s latency_ms=%s", provider, model, role, latency_ms)
            return result
        except Exception as error:
            last_error = error
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            limit_error = None
            if limit is not None:   # a refusal is never retried or routed to another provider
                limit_error = _output_limit_cause(error) or (guard.refusal if guard is not None else None)
            retryable = is_retryable_error(error) and limit_error is None
            status = _status_code(error)
            if status == 429 and cooldowns is not None:
                delay = _retry_after_seconds(error)
                delay = delay if delay is not None else cooldown_seconds
                cooldowns[provider] = time.monotonic() + delay
                if shared_redis is not None:
                    try:
                        await shared_cool_down(shared_redis, provider, delay)
                    except Exception as redis_error:
                        logger.warning("llm_gateway_shared_cooldown_failed provider=%s error_type=%s", provider, type(redis_error).__name__)
            elif retryable and failures is not None and cooldowns is not None:
                failures[provider] = failures.get(provider, 0) + 1
                if failures[provider] >= 2:
                    cooldowns[provider] = time.monotonic() + cooldown_seconds
                    logger.info("llm_gateway_circuit_open provider=%s role=%s failures=%s", provider, role, failures[provider])
            input_tokens = usage_handler.input_tokens if usage_handler.has_usage else None
            output_tokens = usage_handler.output_tokens if usage_handler.has_usage else None
            if input_tokens is None and guard is not None and guard.response_usage is not None:
                input_tokens, output_tokens = guard.response_usage   # e.g. HTTP 200, then local parsing failed
            _merge_wire_details(usage_handler, guard)
            failure = describe_provider_failure(error, getattr(chain, "response_schema", None))
            _record(ProviderAttempt(call_id, attempt_index, provider, model, role, latency_ms, False,
                                    retryable, status, type(error).__name__, input_tokens, output_tokens,
                                    recovery_phase, context_chars, False,
                                    admission.wait_ms if admission else 0,
                                    _error_rate_limit_headers(error) or None,
                                    {**_attempt_diagnostics(usage_handler, limit, reasoning_effort,
                                                            input_tokens is not None and output_tokens is not None),
                                     **failure}))
            logger.warning(
                "llm_gateway_failure provider=%s model=%s role=%s latency_ms=%s status=%s retryable=%s "
                "error_type=%s failure_class=%s",
                provider, model, role, latency_ms, status, retryable, type(error).__name__, failure["failure_class"],
            )
            if limit_error is not None and limit_error is not error:
                raise limit_error from error
            if not retryable:
                raise
            if pending:
                logger.info("llm_gateway_fallback from=%s to=%s role=%s", provider, pending[0], role)
        finally:
            if bounded_model is not None:
                await _close_bounded_clients(bounded_model)
            if admission is not None:
                try:
                    await shared_release(shared_redis, admission, actual_tokens)
                except Exception as redis_error:
                    logger.warning("llm_gateway_shared_release_failed provider=%s error_type=%s", provider, type(redis_error).__name__)
        if not pending and not recovery_phase and last_error is not None and len(attempted) < 3:
            if deadline - time.monotonic() >= recovery_min_remaining:
                pending = [item for item in skipped if cooldowns is None or cooldowns.get(item, 0) <= time.monotonic()]
                recovery_phase = bool(pending)
                if recovery_phase:
                    logger.info("llm_gateway_recovery role=%s candidates=%s", role, ",".join(pending))
    if last_error is not None:
        raise last_error
    raise TimeoutError("LLM gateway call budget exhausted")


def routing_candidates(role: str) -> list[routing_policy.Route]:
    """Approved, declared and configured routes for ``role``, in preference order.

    ``LLM_ROUTING_ROUTES_<ROLE>`` (or ``LLM_ROUTING_ROUTES``) lists ``provider:model``
    labels. Nothing outside that allowlist is ever attempted; an approved route without
    declared capabilities is a configuration error, and one without a key is skipped.
    """
    raw = os.getenv(f"LLM_ROUTING_ROUTES_{role.upper()}") or os.getenv("LLM_ROUTING_ROUTES", "")
    labels = routing_policy.parse_route_allowlist(raw)
    if not labels:
        raise ValueError("consolidated routing requires LLM_ROUTING_ROUTES (provider:model, comma separated)")
    declared: dict[str, routing_policy.Route] = {}
    for (provider, model), facts in ROUTE_CAPABILITIES.items():
        declared[f"{provider}:{model}"] = routing_policy.Route(
            provider=provider, model=model, quota_group=facts["quota_group"],
            context_tokens=facts["context_tokens"], max_output_tokens=facts["max_output_tokens"],
            structured_output=facts["structured_output"], bounded_output=provider in OUTPUT_TOKEN_LIMIT_PARAMETERS,
            reasoning_efforts=tuple(facts.get("reasoning_efforts") or REASONING_EFFORT_SUPPORT.get((provider, model), ())),
            priced=rate_for(provider, model) is not None)
    routes = routing_policy.ordered_candidates(labels, declared)
    for route in routes:
        if route.provider not in DEFAULT_PROVIDER_ORDER:
            raise ValueError(f"unknown routed provider {route.provider}")
    return [route for route in routes if _key_for(route.provider)]


def _observe(event: dict[str, Any]) -> None:
    observer = _routing_observer.get()
    if observer is not None:
        observer(dict(event))


def _transmission_state(error: BaseException, wire: Any | None) -> str:
    """How far a failed attempt got, from wire facts first, then the error itself."""
    chain = _error_chain(error)
    if any(isinstance(item, TransmissionRefused) for item in chain):
        return routing_policy.NOT_TRANSMITTED
    if _output_limit_cause(error) is not None and (wire is None or wire.verified_requests == 0):
        return routing_policy.NOT_TRANSMITTED
    transient = any(isinstance(item, (TimeoutError, ConnectionError, asyncio.CancelledError, httpx.TransportError))
                    or type(item).__name__ in ("APITimeoutError", "APIConnectionError") for item in chain)
    if wire is not None:
        if wire.verified_requests == 0:
            return routing_policy.NOT_TRANSMITTED
        if transient or wire.response_status is None:
            return routing_policy.UNCERTAIN
        return (routing_policy.COMPLETED_INVALID if 200 <= wire.response_status < 300
                else routing_policy.DEFINITIVE_RESPONSE)
    status = _status_code(error)
    if transient:
        return routing_policy.UNCERTAIN
    if status is not None:
        return routing_policy.COMPLETED_INVALID if 200 <= status < 300 else routing_policy.DEFINITIVE_RESPONSE
    if isinstance(error, InvalidStructuredResponse) or error.__class__.__name__ == "OutputParserException":
        return routing_policy.COMPLETED_INVALID
    return routing_policy.UNCERTAIN


async def _ainvoke_routed(role: str, temperature: float, chain_builder: Callable[[Any], Any], payload: Any, *,
                          limit: int | None, reasoning_effort: str | None):
    """Stage 20G consolidated routing: one WAIT / FALLBACK / SEND / STOP policy per call.

    * Admission is taken lazily, preferred route first; a request that can never fit a
      route's whole window is not waited for. Waits are bounded and use the injectable clock.
    * Every physical attempt is checked against the optional accounting hook first; the hook's
      transport-level reservation (if any) happens before bytes are sent.
    * A failed attempt is classified by how far it got. Only a definitive rejection that the
      policy allows (rate limit, outage, route capacity) may move to another approved route;
      the failed route is never re-sent. Uncertain or completed-but-invalid attempts stop.
    * SDK retries are disabled (``max_retries=0``); routed OpenRouter requests forbid hidden
      upstream fallbacks.
    """
    routes = routing_candidates(role)
    if not routes:
        raise ValueError("No approved LLM route has an API key configured")
    if reasoning_effort is not None and limit is None:
        raise UnsupportedReasoningEffort("reasoning effort requires a verified bounded request (max_output_tokens)")
    structured = bool(getattr(chain_builder, "structured_output", False))
    request = routing_policy.RouteRequest(role, estimate_input_tokens(role, payload), limit, reasoning_effort,
                                          structured)
    config = routing_policy.config_from_environment(dict(os.environ))
    call_budget = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90"))
    cooldown_seconds = float(os.getenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60"))
    if min(call_budget, cooldown_seconds) <= 0:
        raise ValueError("LLM gateway time budgets must be positive")
    shared_redis = _shared_redis.get()
    budget = _routing_budget.get()
    call_id = str(uuid4())
    deadline = _monotonic() + call_budget
    state = routing_policy.CallState()
    admissions: dict[str, Any] = {}
    last_error: BaseException | None = None
    context_chars = len(payload["context"]) if role == "expert" and isinstance(payload, dict) and isinstance(payload.get("context"), str) else None

    def can_reserve(route: routing_policy.Route) -> bool:
        return budget is None or bool(budget.can_reserve(route.provider, route.model))

    async def probe(route: routing_policy.Route) -> routing_policy.Capacity:
        if shared_redis is None:
            return routing_policy.Capacity(True)
        try:
            admission = await shared_admit(shared_redis, route.provider, role, payload,
                                           max(0.001, deadline - _monotonic()), max_output_tokens=limit)
        except ProviderThrottled as error:
            reason = (routing_policy.ADMISSION_UNAVAILABLE if error.reason.startswith("Redis unavailable")
                      else error.reason)
            _record(ProviderAttempt(call_id, state.attempts, route.provider, route.model, role, 0.0, False, True,
                                    error_type=type(error).__name__, context_chars=context_chars, skipped=True,
                                    diagnostics={"routing_reason": reason, "retry_after_ms": error.retry_after_ms}))
            logger.info("llm_gateway_shared_throttle provider=%s role=%s reason=%s retry_after_ms=%s",
                        route.provider, role, reason, error.retry_after_ms)
            return routing_policy.Capacity(False, reason, error.retry_after_ms / 1000)
        if admission is not None:
            admissions[route.label] = admission
        return routing_policy.Capacity(True)

    async def release_unused() -> None:
        for admission in list(admissions.values()):
            try:
                await shared_release(shared_redis, admission, None)
            except Exception as redis_error:
                logger.warning("llm_gateway_shared_release_failed provider=%s error_type=%s",
                               admission.provider, type(redis_error).__name__)
        admissions.clear()

    try:
        while True:
            decision = await routing_policy.decide_admission(
                [route for route in routes], request, state, remaining_seconds=deadline - _monotonic(),
                probe=probe, can_reserve=can_reserve, config=config)
            _observe({"event": "routing_decision", "call_id": call_id, "role": role, "action": decision.action,
                      "reason": decision.reason, "route": decision.route.label if decision.route else None,
                      "wait_seconds": round(decision.wait_seconds, 3), "attempts": state.attempts})
            logger.info("llm_gateway_routing role=%s action=%s reason=%s route=%s wait_s=%.3f", role,
                        decision.action, decision.reason, decision.route.label if decision.route else None,
                        decision.wait_seconds)
            if decision.action == routing_policy.WAIT:
                await _sleep(decision.wait_seconds)
                state.waited_seconds += decision.wait_seconds
                continue
            if decision.action == routing_policy.STOP:
                if last_error is not None:
                    last_error.routing_reason = decision.reason
                    raise last_error
                raise RoutingStopped(role, decision.reason)
            route = decision.route
            admission = admissions.pop(route.label, None)
            state.attempts += 1
            state.excluded_routes.add(route.label)
            outcome = await _routed_attempt(route, role, temperature, chain_builder, payload, limit, reasoning_effort,
                                            call_id, state.attempts - 1, admission, shared_redis, deadline,
                                            context_chars, decision.action == routing_policy.FALLBACK)
            if outcome["ok"]:
                return outcome["result"]
            error = outcome["error"]
            last_error = error
            transmission = outcome["transmission"]
            failure = outcome["failure"]
            disposition = routing_policy.failure_disposition(
                transmission, failure.get("failure_class"), http_status=failure.get("http_status"),
                provider_error_code=failure.get("provider_error_code"), config=config)
            _observe({"event": "attempt_failed", "call_id": call_id, "role": role, "route": route.label,
                      "transmission": transmission, "failure_class": failure.get("failure_class"),
                      "http_status": failure.get("http_status"), "may_fallback": disposition.may_fallback,
                      "reason": disposition.reason})
            if disposition.exhaust_quota_group:
                state.exhausted_quota_groups.add(route.quota_group)
            if failure.get("http_status") == 429:
                delay = _retry_after_seconds(error)
                delay = delay if delay is not None else cooldown_seconds
                cooldowns = _provider_cooldowns.get()
                if cooldowns is not None:
                    cooldowns[route.provider] = _monotonic() + delay
                if shared_redis is not None:
                    try:
                        await shared_cool_down(shared_redis, route.provider, delay)
                    except Exception as redis_error:
                        logger.warning("llm_gateway_shared_cooldown_failed provider=%s error_type=%s",
                                       route.provider, type(redis_error).__name__)
            limit_error = _output_limit_cause(error)
            if any(isinstance(item, TransmissionRefused) for item in _error_chain(error)):
                raise error
            if limit_error is not None and limit_error is not error:
                limit_error.routing_reason = disposition.reason
                raise limit_error from error
            if not disposition.may_fallback:
                error.routing_reason = disposition.reason
                raise error
    finally:
        await release_unused()


def _error_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


async def _routed_attempt(route: routing_policy.Route, role: str, temperature: float, chain_builder, payload,
                          limit: int | None, reasoning_effort: str | None, call_id: str, attempt_index: int,
                          admission: Any, shared_redis: Any, deadline: float, context_chars: int | None,
                          recovery: bool) -> dict[str, Any]:
    started = time.perf_counter()
    usage_handler = _TokenUsageHandler()
    actual_tokens: int | None = None
    model_object = None
    wire = None
    chain = None
    try:
        model_object = create_chat_model(route.provider, role, temperature, max_output_tokens=limit,
                                         reasoning_effort=reasoning_effort, model=route.model, routing=True)
        wire = getattr(getattr(model_object, "http_async_client", None), "output_limit_guard", None)
        chain = chain_builder(model_object)
        result = await asyncio.wait_for(
            _ainvoke_with_usage(chain, payload, usage_handler),
            timeout=min(provider_timeout_seconds(route.provider), max(0.001, deadline - _monotonic())),
        )
        if wire is None or wire.refusal is not None or wire.verified_requests != 1:
            if wire is not None and wire.refusal is not None:
                raise wire.refusal
            raise OutputTokenLimitNotEnforced("the routed attempt did not make exactly one verified request")
        _validate_role_output(role, result, payload)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        input_tokens, output_tokens = _usage_counts(result)
        if input_tokens is None and usage_handler.has_usage:
            input_tokens, output_tokens = usage_handler.input_tokens, usage_handler.output_tokens
        guard = wire if isinstance(wire, _OutputLimitGuard) else None
        if input_tokens is None and guard is not None and guard.response_usage is not None:
            input_tokens, output_tokens = guard.response_usage
        _merge_wire_details(usage_handler, guard)
        if input_tokens is not None and output_tokens is not None:
            actual_tokens = input_tokens + output_tokens
        _record(ProviderAttempt(call_id, attempt_index, route.provider, route.model, role, latency_ms, True, False,
                                input_tokens=input_tokens, output_tokens=output_tokens, recovery=recovery,
                                context_chars=context_chars, admission_wait_ms=admission.wait_ms if admission else 0,
                                rate_limit_headers=usage_handler.rate_limit_headers or None,
                                diagnostics={**_attempt_diagnostics(usage_handler, limit, reasoning_effort,
                                                                    input_tokens is not None and output_tokens is not None),
                                             "transmission": "completed"}))
        _observe({"event": "attempt_succeeded", "call_id": call_id, "role": role, "route": route.label,
                  "fallback": recovery, "input_tokens": input_tokens, "output_tokens": output_tokens})
        logger.info("llm_gateway_success provider=%s model=%s role=%s latency_ms=%s", route.provider, route.model,
                    role, latency_ms)
        return {"ok": True, "result": result}
    except Exception as error:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        transmission = _transmission_state(error, wire)
        guard = wire if isinstance(wire, _OutputLimitGuard) else None
        input_tokens = usage_handler.input_tokens if usage_handler.has_usage else None
        output_tokens = usage_handler.output_tokens if usage_handler.has_usage else None
        if input_tokens is None and guard is not None and guard.response_usage is not None:
            input_tokens, output_tokens = guard.response_usage
        _merge_wire_details(usage_handler, guard)
        failure = describe_provider_failure(error, getattr(chain, "response_schema", None))
        if failure.get("http_status") is None and wire is not None and wire.response_status not in (None, 200):
            failure["http_status"] = wire.response_status
        _record(ProviderAttempt(call_id, attempt_index, route.provider, route.model, role, latency_ms, False,
                                False, _status_code(error), type(error).__name__, input_tokens, output_tokens,
                                recovery, context_chars, False, admission.wait_ms if admission else 0,
                                _error_rate_limit_headers(error) or None,
                                {**_attempt_diagnostics(usage_handler, limit, reasoning_effort,
                                                        input_tokens is not None and output_tokens is not None),
                                 **failure, "transmission": transmission}))
        logger.warning("llm_gateway_failure provider=%s model=%s role=%s latency_ms=%s status=%s transmission=%s "
                       "error_type=%s failure_class=%s", route.provider, route.model, role, latency_ms,
                       _status_code(error), transmission, type(error).__name__, failure["failure_class"])
        return {"ok": False, "error": error, "transmission": transmission, "failure": failure}
    except BaseException:
        # Cancellation mid-attempt: the request may have been sent. Never replayed here.
        _observe({"event": "attempt_cancelled", "call_id": call_id, "role": role, "route": route.label,
                  "transmission": routing_policy.UNCERTAIN if wire is None or wire.verified_requests else
                  routing_policy.NOT_TRANSMITTED})
        raise
    finally:
        if model_object is not None:
            await _close_bounded_clients(model_object)
        if admission is not None:
            try:
                await shared_release(shared_redis, admission, actual_tokens)
            except Exception as redis_error:
                logger.warning("llm_gateway_shared_release_failed provider=%s error_type=%s", route.provider,
                               type(redis_error).__name__)


async def ainvoke_structured(role: str, temperature: float, prompt: str, response_schema: type, *,
                             max_output_tokens: int | None = None, reasoning_effort: str | None = None):
    class ValidatedStructuredChain:
        def __init__(self, chain):
            self.chain = chain
            self.response_schema = response_schema   # for sanitized failure diagnostics only

        async def ainvoke(self, payload, config=None):
            try:
                parameters = inspect.signature(self.chain.ainvoke).parameters.values()
                accepts_config = any(parameter.name == "config" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
            except (TypeError, ValueError):
                accepts_config = False
            try:
                raw = await self.chain.ainvoke(payload, config=config) if accepts_config else await self.chain.ainvoke(payload)
            except ValidationError as error:
                raise InvalidStructuredResponse("Provider returned output incompatible with the requested schema") from error
            if isinstance(raw, dict) and "parsed" in raw:
                if raw.get("parsing_error") is not None:
                    raise InvalidStructuredResponse(f"Provider returned invalid structured output: {raw['parsing_error']}") from raw["parsing_error"]
                raw = raw["parsed"]
            if raw is None:
                raise InvalidStructuredResponse("Provider returned no structured response")
            try:
                if hasattr(response_schema, "model_validate"):
                    return response_schema.model_validate(raw)
                if isinstance(raw, response_schema):
                    return raw
                if isinstance(raw, dict) and callable(response_schema):
                    return response_schema(**raw)
                return raw
            except Exception as error:
                raise InvalidStructuredResponse(f"Provider response failed schema validation: {error}") from error

    def structured_model(model):
        try:
            return ValidatedStructuredChain(model.with_structured_output(response_schema, include_raw=True))
        except TypeError as error:
            if "include_raw" not in str(error):
                raise
            return ValidatedStructuredChain(model.with_structured_output(response_schema))

    structured_model.structured_output = True   # Stage 20G: routing needs structured-output support

    if reasoning_effort is not None:
        return await ainvoke(role, temperature, structured_model, prompt, max_output_tokens=max_output_tokens,
                             reasoning_effort=reasoning_effort)
    if max_output_tokens is not None:
        return await ainvoke(role, temperature, structured_model, prompt, max_output_tokens=max_output_tokens)
    return await ainvoke(
        role,
        temperature,
        structured_model,
        prompt,
    )
