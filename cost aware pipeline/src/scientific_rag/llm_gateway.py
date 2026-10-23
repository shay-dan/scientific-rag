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
import time
from typing import Any, Callable, Iterable
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from model_policy import ModelPolicy
from .provider_throttle import ProviderThrottled, admit as shared_admit, cool_down as shared_cool_down, release as shared_release


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


_trace: ContextVar[list[ProviderAttempt] | None] = ContextVar("llm_gateway_trace", default=None)
_provider_cooldowns: ContextVar[dict[str, float] | None] = ContextVar("llm_gateway_provider_cooldowns", default=None)
_provider_failures: ContextVar[dict[str, int] | None] = ContextVar("llm_gateway_provider_failures", default=None)
_shared_redis: ContextVar[Any | None] = ContextVar("llm_gateway_shared_redis", default=None)


class _TokenUsageHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.has_usage = False

    def on_llm_end(self, response, **kwargs) -> None:
        usage = getattr(response, "llm_output", None) or {}
        token_usage = usage.get("token_usage") or usage.get("usage") or {}
        input_tokens, output_tokens = _usage_pair(token_usage)
        if input_tokens is not None and output_tokens is not None:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.has_usage = True
            return
        for group in getattr(response, "generations", []) or []:
            for generation in group:
                input_tokens, output_tokens = _usage_counts(getattr(generation, "message", None))
                if input_tokens is not None and output_tokens is not None:
                    self.input_tokens += input_tokens
                    self.output_tokens += output_tokens
                    self.has_usage = True


def _usage_pair(usage: Any) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if input_tokens is None or output_tokens is None:
        return None, None
    return int(input_tokens), int(output_tokens)


def begin_trace(redis: Any | None = None) -> tuple[Token, Token, Token, Token]:
    """Start an isolated trace for one research run."""
    return _trace.set([]), _provider_cooldowns.set({}), _provider_failures.set({}), _shared_redis.set(redis)


def end_trace(token: tuple[Token, Token, Token, Token]) -> list[dict[str, Any]]:
    """Return serializable attempt metadata and restore the prior context."""
    attempts = list(_trace.get() or [])
    _trace.reset(token[0])
    _provider_cooldowns.reset(token[1])
    _provider_failures.reset(token[2])
    _shared_redis.reset(token[3])
    return [asdict(attempt) for attempt in attempts]


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
            model_rate_key = re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
            input_rate = os.getenv(f"LLM_{provider.upper()}_{model_rate_key}_INPUT_USD_PER_MILLION") or os.getenv(f"LLM_{provider.upper()}_INPUT_USD_PER_MILLION")
            output_rate = os.getenv(f"LLM_{provider.upper()}_{model_rate_key}_OUTPUT_USD_PER_MILLION") or os.getenv(f"LLM_{provider.upper()}_OUTPUT_USD_PER_MILLION")
            if input_rate is None or output_rate is None:
                cost_available = False
                model_bucket["cost_available"] = False
            else:
                call_cost = int(input_tokens) * float(input_rate) / 1_000_000
                call_cost += int(output_tokens) * float(output_rate) / 1_000_000
                estimated_cost_usd += call_cost
                model_bucket["estimated_cost_usd"] += call_cost
        else:
            cost_available = False
            model_bucket["cost_available"] = False
    failures = sum(1 for attempt in attempts if not attempt["success"] and not attempt.get("skipped"))
    recovery_attempts = [attempt for attempt in attempts if attempt.get("recovery")]
    skipped_attempts = [attempt for attempt in attempts if attempt.get("skipped")]
    throttle_skips = [attempt for attempt in skipped_attempts if attempt.get("error_type") == "ProviderThrottled"]
    policy_skips = [attempt for attempt in skipped_attempts if attempt.get("error_type") == "ExpertContextLimit"]
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
        "fallback_count": len({attempt["call_id"] for attempt in attempts if int(attempt["attempt_index"]) > 0 or attempt.get("skipped")}),
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


def _record(attempt: ProviderAttempt) -> None:
    attempts = _trace.get()
    if attempts is not None:
        attempts.append(attempt)


def provider_order() -> tuple[str, ...]:
    configured = os.getenv("LLM_PROVIDER_ORDER", ",".join(DEFAULT_PROVIDER_ORDER))
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
        return os.getenv(f"GROQ_MODEL_{role_key}", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    if provider == "openrouter":
        return os.getenv(f"OPENROUTER_MODEL_{role_key}", os.getenv("OPENROUTER_MODEL", "openrouter/free"))
    return ModelPolicy.from_environment().route_for(role).primary


def primary_route_label(role: str) -> str:
    cooldowns = _provider_cooldowns.get() or {}
    for provider in provider_order():
        if _key_for(provider) and cooldowns.get(provider, 0) <= time.monotonic():
            return f"{provider}:{model_for(provider, role)}"
    return "unconfigured"


def create_chat_model(provider: str, role: str, temperature: float):
    model = model_for(provider, role)
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


def _configured_providers() -> Iterable[str]:
    for provider in provider_order():
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
    provider = next(iter(_configured_providers()), None)
    if provider is None:
        raise ValueError("No LLM provider API key is configured")
    return create_chat_model(provider, role, temperature)


async def ainvoke(role: str, temperature: float, chain_builder: Callable[[Any], Any], payload: Any):
    """Invoke a chain and fail over only for transient/provider-availability errors."""
    providers = list(_configured_providers())
    if not providers:
        raise ValueError("No LLM provider API key is configured")

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
        try:
            chain = chain_builder(create_chat_model(provider, role, temperature))
            result = await asyncio.wait_for(
                _ainvoke_with_usage(chain, payload, usage_handler),
                timeout=min(provider_timeout_seconds(provider), max(0.001, deadline - time.monotonic())),
            )
            _validate_role_output(role, result, payload)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            input_tokens, output_tokens = _usage_counts(result)
            if input_tokens is None and usage_handler.has_usage:
                input_tokens, output_tokens = usage_handler.input_tokens, usage_handler.output_tokens
            if input_tokens is not None and output_tokens is not None:
                actual_tokens = input_tokens + output_tokens
            _record(ProviderAttempt(call_id, attempt_index, provider, model, role, latency_ms, True, False, input_tokens=input_tokens, output_tokens=output_tokens, recovery=recovery_phase, context_chars=context_chars, admission_wait_ms=admission.wait_ms if admission else 0))
            if failures is not None:
                failures[provider] = 0
            if cooldowns is not None:
                cooldowns.pop(provider, None)
            logger.info("llm_gateway_success provider=%s model=%s role=%s latency_ms=%s", provider, model, role, latency_ms)
            return result
        except Exception as error:
            last_error = error
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            retryable = is_retryable_error(error)
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
            _record(ProviderAttempt(call_id, attempt_index, provider, model, role, latency_ms, False, retryable, status, type(error).__name__, input_tokens, output_tokens, recovery_phase, context_chars, False, admission.wait_ms if admission else 0))
            logger.warning(
                "llm_gateway_failure provider=%s model=%s role=%s latency_ms=%s status=%s retryable=%s error_type=%s",
                provider, model, role, latency_ms, status, retryable, type(error).__name__,
            )
            if not retryable:
                raise
            if pending:
                logger.info("llm_gateway_fallback from=%s to=%s role=%s", provider, pending[0], role)
        finally:
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


async def ainvoke_structured(role: str, temperature: float, prompt: str, response_schema: type):
    class ValidatedStructuredChain:
        def __init__(self, chain):
            self.chain = chain

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

    return await ainvoke(
        role,
        temperature,
        structured_model,
        prompt,
    )
