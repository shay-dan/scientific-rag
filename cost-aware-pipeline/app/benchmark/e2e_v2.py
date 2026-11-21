"""Stage 20G evaluation-only E2E campaign with the consolidated routing policy.

Successor protocol to the terminal Stage 20F v1 campaign (``e2e_budget``, unchanged). The
production pipeline runs with ``LLM_ROUTING_POLICY=consolidated``; chat requests may use the
approved Groq route or, if approved, the OpenRouter route for the same model. Every physical
request (chat or embedding) is reserved in this campaign *before* its transport sends it, and
fixed reservations are never refunded.

Stops are scoped:
* case-level (that question fails, the campaign continues): per-case request ceiling,
  question deadline, a gateway failure the production pipeline does not recover from;
* campaign-level (nothing further is sent): campaign ceilings or deadline, journal failure,
  a request outside the approved contract, unknown tickets or usage beyond the allowance.

No provider clients, credentials or network defaults: callers supply transports.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import threading
import time
from typing import Callable

import httpx

from scientific_rag.provider_failures import TransmissionRefused

from .e2e_budget import CASES, EMBED_MODEL, GROQ_MODEL, PrivateJournal

CAMPAIGN_ID = "stage20-e2e-v2"
PROTOCOL_SCHEMA = "scientific-rag-e2e-protocol/v2"
AUTHORIZATION_SCHEMA = "scientific-rag-e2e-authorization/v2"
EXPECTED_MANIFEST_SHA256 = "c0a96c97fc840c854575cd27d8fa2680c176670b420780ad59b426b1157a394a"
OPENROUTER_MODEL = "openai/gpt-oss-120b"
NANO_USD = 1_000_000_000
# Integer nanodollars per token (USD per million tokens x 1000).
RATES = {"groq": (150, 600), "openrouter": (150, 600), "embedding": (200, 0)}
RATE_SOURCES = {"groq": "builtin-2026-10-24 (Groq on-demand gpt-oss-120b)",
                "openrouter": "price ceiling enforced per request by provider.max_price",
                "embedding": "builtin-2026-10-24 (gemini-embedding-2)"}
INPUT_LIMIT = {"chat": 32768, "embedding": 8192}
OUTPUT_LIMIT = 2000
FRAMING_ALLOWANCE = 512
PER_CASE_LIMITS = {"chat": 20, "embedding": 25}
TOTAL_LIMITS = {"chat": 100, "embedding": 125}
QUESTION_SECONDS = 600
CAMPAIGN_SECONDS = 3000
OPENROUTER_PREFERENCES = {"require_parameters": True, "allow_fallbacks": False,
                          "max_price": {"prompt": 0.15, "completion": 0.6}}
CHAT_ROUTES = {
    "groq": {"host": "api.groq.com", "path": "/openai/v1/chat/completions", "model": GROQ_MODEL},
    "openrouter": {"host": "openrouter.ai", "path": "/api/v1/chat/completions", "model": OPENROUTER_MODEL},
}
EMBEDDING_HOST = "generativelanguage.googleapis.com"
ROLES = ("orchestrator", "planner", "gatekeeper", "expert", "qa_generator", "query_repair")
# Routing policy values fixed by the protocol (see scientific_rag.routing_policy).
ROUTING = {"max_wait_seconds": 65, "prefer_primary_within_seconds": 5, "max_physical_attempts": 2,
           "min_attempt_seconds": 10, "fallback_on_invalid_output": False, "fallback_after_uncertain": False,
           "call_budget_seconds": 150, "groq_timeout_seconds": 30, "openrouter_timeout_seconds": 45,
           "groq_tpm_budget": 6000, "groq_rpm_budget": 20, "openrouter_rpm_budget": 10}

CASE_STOPS = frozenset({"per_case_request_ceiling", "question_deadline", "case_failed"})


def reservation(provider: str) -> int:
    rate_in, rate_out = RATES[provider]
    if provider == "embedding":
        return INPUT_LIMIT["embedding"] * rate_in
    return INPUT_LIMIT["chat"] * rate_in + OUTPUT_LIMIT * rate_out


def maximum_reservation_nanousd(fallback: bool = True) -> int:
    chat = max(reservation(p) for p in (("groq", "openrouter") if fallback else ("groq",)))
    return TOTAL_LIMITS["chat"] * chat + TOTAL_LIMITS["embedding"] * reservation("embedding")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def chat_routes(fallback: bool) -> list[str]:
    routes = [f"groq:{GROQ_MODEL}"]
    if fallback:
        routes.append(f"openrouter:{OPENROUTER_MODEL}")
    return routes


def protocol(*, fallback: bool = True) -> dict:
    limit = maximum_reservation_nanousd(fallback)
    return {
        "schema": PROTOCOL_SCHEMA, "campaign_id": CAMPAIGN_ID, "predecessor": {
            "campaign": "stage20-e2e-v1", "fingerprint":
                "422aabe98dd6ed242ea7c2b363d72a80b666404528073583e8a1cd3c082e308e",
            "status": "terminal (gateway_failure: local Groq TPM admission refusal before E001 completed)",
            "resumed": False, "budget_transferred": False},
        "cases": list(CASES), "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "mode": "fast", "cache_mode": "refresh_sources", "generation_prompt_version": "v16",
        "quote_recovery_mode": "unicode", "claim_recovery_mode": "degrade",
        "semantic_verification_mode": "enforce", "claim_audit_mode": "audit", "synthetic_evaluation": False,
        "routing_policy": "consolidated", "chat_routes": chat_routes(fallback),
        "openrouter_provider_preferences": OPENROUTER_PREFERENCES if fallback else None,
        "reasoning_effort": "low", "temperature": 0, "max_output_tokens": OUTPUT_LIMIT,
        "embedding_model": EMBED_MODEL, "embedding_dimensions": 768, "embedding_fallback": False,
        "routing": dict(ROUTING),
        "per_case_requests": dict(PER_CASE_LIMITS), "campaign_requests": dict(TOTAL_LIMITS),
        "input_allowance": dict(INPUT_LIMIT), "framing_allowance": FRAMING_ALLOWANCE,
        "rates_nanousd_per_token": {key: list(value) for key, value in RATES.items()
                                    if fallback or key != "openrouter"},
        "rate_sources": {key: value for key, value in RATE_SOURCES.items() if fallback or key != "openrouter"},
        "reservation_nanousd": {key: reservation(key) for key in RATES if fallback or key != "openrouter"},
        "campaign_limit_nanousd": limit, "maximum_modeled_reservation_nanousd": limit,
        "question_seconds": QUESTION_SECONDS, "campaign_seconds": CAMPAIGN_SECONDS,
        "transmission_rules": {"retransmission_same_route": False, "sdk_retries": 0,
                               "hidden_upstream_fallback": False, "resume": False,
                               "fallback_after": ["provider_rate_limit", "provider_unavailable",
                                                  "route_capacity_rejected", "local_capacity_or_window"]},
    }


def fingerprint(protocol_value: dict, *, manifest_sha256: str, code_sha: str, image_id: str) -> str:
    """Bind the owner authorization to the exact Docker image, not a mutable tag."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("authorization requires an exact Docker image ID")
    return canonical_sha256({"schema": AUTHORIZATION_SCHEMA, "campaign_id": protocol_value["campaign_id"],
                             "protocol_sha256": canonical_sha256(protocol_value),
                             "manifest_sha256": manifest_sha256, "code_sha": code_sha,
                             "image_id": image_id})


class ExecutionStopped(TransmissionRefused):
    """A request was refused before transmission; ``reason`` is a bounded code."""

    def __init__(self, reason: str, *, scope: str) -> None:
        super().__init__(reason)
        self.reason, self.scope = reason, scope


@dataclass(frozen=True)
class Ticket:
    index: int
    provider: str
    pool: str
    case: str
    reserved_nanousd: int


class Campaign:
    """Single isolated process; one lock covers sync embedding threads and async chat."""

    def __init__(self, append: Callable[[dict], None], *, fallback: bool = True, clock=time.monotonic):
        self._append, self._clock, self.fallback = append, clock, fallback
        self._lock = threading.RLock()
        self._started = clock()
        self._case_started = None
        self.case: str | None = None
        self.case_failure: str | None = None
        self.completed: list[dict] = []
        self.per_case = dict.fromkeys(TOTAL_LIMITS, 0)
        self.counts = dict.fromkeys(TOTAL_LIMITS, 0)
        self.by_provider = {"groq": 0, "openrouter": 0, "embedding": 0}
        self.reserved = 0
        self.stop_reason: str | None = None
        self.limit = maximum_reservation_nanousd(fallback)
        self._inflight: dict[int, Ticket] = {}
        self._index = 0

    # --- journal and stops ---------------------------------------------------------------
    def _write(self, event: dict) -> None:
        try:
            self._append(event)
        except Exception as exc:
            self.stop_reason = self.stop_reason or "journal_failure"
            raise ExecutionStopped("journal_failure", scope="campaign") from exc

    def stop(self, reason: str) -> None:
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                self._write({"event": "stop", "scope": "campaign", "case": self.case, "reason": reason})

    def fail_case(self, reason: str) -> None:
        with self._lock:
            if self.case is not None and self.case_failure is None:
                self.case_failure = reason
                self._write({"event": "stop", "scope": "case", "case": self.case, "reason": reason})

    def _refuse(self, reason: str):
        if reason in CASE_STOPS:
            self.fail_case(reason)
            raise ExecutionStopped(reason, scope="case")
        self.stop(reason)
        raise ExecutionStopped(self.stop_reason, scope="campaign")

    def ensure_running(self) -> None:
        with self._lock:
            if self.stop_reason is not None:
                raise ExecutionStopped(self.stop_reason, scope="campaign")
            now = self._clock()
            if now - self._started >= CAMPAIGN_SECONDS:
                self._refuse("campaign_deadline")
            if self.case_failure is not None:
                raise ExecutionStopped(self.case_failure, scope="case")
            if self._case_started is not None and now - self._case_started >= QUESTION_SECONDS:
                self._refuse("question_deadline")

    # --- cases -----------------------------------------------------------------------------
    def begin_case(self, case: str) -> None:
        with self._lock:
            if self.stop_reason is not None:
                raise ExecutionStopped(self.stop_reason, scope="campaign")
            if self._clock() - self._started >= CAMPAIGN_SECONDS:
                self._refuse("campaign_deadline")
            if (self.case is not None or self._inflight or len(self.completed) >= len(CASES)
                    or case != CASES[len(self.completed)]):
                self._refuse("case_order_or_pending_work")
            self._write({"event": "begin_case", "case": case})
            self.case, self.case_failure = case, None
            self._case_started = self._clock()
            self.per_case = dict.fromkeys(TOTAL_LIMITS, 0)

    def end_case(self, status: str, *, reason: str | None = None) -> dict:
        with self._lock:
            if self.case is None:
                self._refuse("case_not_settled")
            if self._inflight:
                # An unsettled ticket at case end is a defect; it stays reserved and is reported.
                self._write({"event": "unsettled_at_case_end", "case": self.case, "tickets": sorted(self._inflight)})
                # Do NOT clear tickets: they still represent possibly transmitting work.
                # A terminal stop prevents all further questions.
                self.stop("unsettled_at_case_end")
            if self._clock() - self._started >= CAMPAIGN_SECONDS:
                self.stop("campaign_deadline")
            if self._case_started is not None and self._clock() - self._case_started >= QUESTION_SECONDS:
                self.fail_case("question_deadline")
            terminal = self.stop_reason or self.case_failure
            if terminal:
                status, reason = "stopped", terminal
            outcome = {"case": self.case, "status": status, "reason": reason,
                       "requests": dict(self.per_case)}
            self._write({"event": "end_case", **outcome})
            self.completed.append(outcome)
            self.case, self.case_failure, self._case_started = None, None, None
            return outcome

    # --- routing budget hook (consulted by the gateway before each physical attempt) ------
    def can_reserve(self, provider: str, model: str) -> bool:
        with self._lock:
            if self.stop_reason is not None or self.case is None or self.case_failure is not None:
                return False
            route = CHAT_ROUTES.get(provider)
            if route is None or route["model"] != model or (provider == "openrouter" and not self.fallback):
                return False
            return (self.per_case["chat"] < PER_CASE_LIMITS["chat"] and self.counts["chat"] < TOTAL_LIMITS["chat"]
                    and self.reserved + reservation(provider) <= self.limit)

    def record_routing(self, event: dict) -> None:
        allowed = {"event", "call_id", "role", "action", "reason", "route", "wait_seconds", "attempts",
                   "transmission", "failure_class", "http_status", "may_fallback", "fallback",
                   "input_tokens", "output_tokens"}
        with self._lock:
            self._write({"event": "routing", "case": self.case,
                         **{key: value for key, value in event.items() if key in allowed and key != "event"},
                         "kind": event.get("event")})

    # --- transport-side admission ---------------------------------------------------------
    def _validate_request(self, request: httpx.Request, provider: str) -> int:
        if request.method != "POST" or request.url.scheme != "https" or request.url.port not in (None, 443):
            self._refuse("unexpected_method_or_destination")
        pool = "embedding" if provider == "embedding" else "chat"
        host = EMBEDDING_HOST if provider == "embedding" else CHAT_ROUTES.get(provider, {}).get("host")
        if host is None or request.url.host != host or (provider == "openrouter" and not self.fallback):
            self._refuse("unexpected_destination")
        body_bytes = request.content
        bound = len(body_bytes) + FRAMING_ALLOWANCE
        if bound > INPUT_LIMIT[pool]:
            # A wire request too large for the modeled allowance is a contract fault for chat
            # (the routing policy should have refused it) and for embeddings alike.
            self._refuse("input_allowance")
        try:
            body = json.loads(body_bytes)
        except (ValueError, TypeError):
            self._refuse("invalid_request_body")
        if not isinstance(body, dict):
            self._refuse("invalid_request_body")
        if pool == "chat":
            route = CHAT_ROUTES[provider]
            caps = [body[k] for k in ("max_tokens", "max_completion_tokens") if body.get(k) is not None]
            if (request.url.path != route["path"] or body.get("model") != route["model"]
                    or body.get("reasoning_effort") != "low"
                    or not caps or any(type(x) is not int or x != OUTPUT_LIMIT for x in caps)
                    or body.get("temperature") != 0 or body.get("n") not in (None, 1)
                    or body.get("stream") not in (None, False)
                    or (provider == "openrouter" and body.get("provider") != OPENROUTER_PREFERENCES)
                    or (provider == "groq" and "provider" in body)):
                self._refuse("chat_contract")
        else:
            prefix = f"/v1beta/models/{EMBED_MODEL}:"
            if request.url.path not in (prefix + "embedContent", prefix + "batchEmbedContents"):
                self._refuse("embedding_contract")
            items = body.get("requests") if request.url.path.endswith(":batchEmbedContents") else [body]
            if not isinstance(items, list) or not items:
                self._refuse("embedding_contract")
            for item in items:
                if (not isinstance(item, dict)
                        or item.get("model") not in (None, f"models/{EMBED_MODEL}", EMBED_MODEL)
                        or item.get("outputDimensionality") != 768
                        or item.get("taskType") not in ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")):
                    self._refuse("embedding_contract")
                content = item.get("content")
                parts = content.get("parts") if isinstance(content, dict) else None
                if not isinstance(parts, list) or not parts or any(
                        not isinstance(part, dict) or set(part) != {"text"}
                        or not isinstance(part["text"], str) or not part["text"] for part in parts):
                    self._refuse("embedding_contract")
        return bound

    def admit(self, request: httpx.Request, provider: str) -> Ticket:
        with self._lock:
            self.ensure_running()
            if self.case is None:
                self._refuse("no_active_case")
            bound = self._validate_request(request, provider)
            pool = "embedding" if provider == "embedding" else "chat"
            charge = reservation(provider)
            if self.per_case[pool] >= PER_CASE_LIMITS[pool]:
                self._refuse("per_case_request_ceiling")
            if self.counts[pool] >= TOTAL_LIMITS[pool] or self.reserved + charge > self.limit:
                self._refuse("campaign_ceiling")
            self._index += 1
            ticket = Ticket(self._index, provider, pool, self.case, charge)
            self._write({"event": "reserved_before_transport", "case": ticket.case, "index": ticket.index,
                         "provider": provider, "input_allowance_used": bound, "reserved_nanousd": charge})
            self.counts[pool] += 1
            self.per_case[pool] += 1
            self.by_provider[provider] += 1
            self.reserved += charge
            self._inflight[ticket.index] = ticket
            return ticket

    def returned(self, ticket: Ticket, status: int, usage: dict | None = None) -> None:
        with self._lock:
            if self._inflight.pop(ticket.index, None) != ticket:
                self._refuse("unknown_ticket")
            event = {"event": "transport_returned", "index": ticket.index, "provider": ticket.provider,
                     "status": status}
            usage_invalid = False
            if ticket.pool == "chat" and status == 200 and isinstance(usage, dict):
                inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if type(inp) is int and type(out) is int and min(inp, out) >= 0:
                    event.update({"input_tokens": inp, "output_tokens": out})
                    usage_invalid = inp > INPUT_LIMIT["chat"] or out > OUTPUT_LIMIT
            self._write(event)
            # A definitive non-200 response is settled here and decided by the routing policy;
            # usage beyond the modeled allowance means the accounting bound failed: stop.
            if usage_invalid:
                self.stop("provider_usage_exceeded_allowance")

    def uncertain(self, ticket: Ticket) -> None:
        with self._lock:
            self._inflight.pop(ticket.index, None)
            # Already fully reserved; never replayed (the gateway stops that call).
            self._write({"event": "transport_uncertain", "index": ticket.index, "provider": ticket.provider})

    def summary(self) -> dict:
        with self._lock:
            return {"campaign_id": CAMPAIGN_ID, "requests_reserved": dict(self.counts),
                    "requests_by_provider": dict(self.by_provider), "reserved_nanousd": self.reserved,
                    "limit_nanousd": self.limit, "pending": len(self._inflight), "stop_reason": self.stop_reason,
                    "cases": list(self.completed)}


class GuardedTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Reserve in the campaign immediately before the supplied transport sends."""

    def __init__(self, campaign: Campaign, provider: str, *, sync: httpx.BaseTransport | None = None,
                 async_: httpx.AsyncBaseTransport | None = None):
        self.campaign, self.provider, self.sync, self.async_ = campaign, provider, sync, async_

    @staticmethod
    def _usage(response):
        if response.status_code != 200:
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        return body.get("usage") if isinstance(body, dict) else None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.sync is None:
            self.campaign.stop("missing_sync_transport")
            raise ExecutionStopped("missing_sync_transport", scope="campaign")
        request.read()
        ticket = self.campaign.admit(request, self.provider)
        try:
            response = self.sync.handle_request(request)
            response.read()
        except BaseException:
            self.campaign.uncertain(ticket)
            raise
        self.campaign.returned(ticket, response.status_code, self._usage(response))
        return response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.async_ is None:
            self.campaign.stop("missing_async_transport")
            raise ExecutionStopped("missing_async_transport", scope="campaign")
        await request.aread()
        ticket = self.campaign.admit(request, self.provider)
        try:
            response = await self.async_.handle_async_request(request)
            await response.aread()
        except BaseException:
            self.campaign.uncertain(ticket)
            raise
        self.campaign.returned(ticket, response.status_code, self._usage(response))
        return response

    def close(self):
        if self.sync is not None:
            self.sync.close()

    async def aclose(self):
        if self.async_ is not None:
            await self.async_.aclose()


class ChatDispatchTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """One transport for every routed chat client: dispatch by destination host to the
    provider's guarded transport; any other destination is refused before sending."""

    def __init__(self, campaign: Campaign, routes: dict[str, GuardedTransport]):
        self.campaign = campaign
        self.by_host = {CHAT_ROUTES[provider]["host"]: transport for provider, transport in routes.items()}

    def _select(self, request: httpx.Request) -> GuardedTransport:
        transport = self.by_host.get(request.url.host)
        if transport is None:
            self.campaign.stop("unexpected_destination")
            raise ExecutionStopped("unexpected_destination", scope="campaign")
        return transport

    def handle_request(self, request):
        return self._select(request).handle_request(request)

    async def handle_async_request(self, request):
        return await self._select(request).handle_async_request(request)


__all__ = [
    "AUTHORIZATION_SCHEMA", "CAMPAIGN_ID", "CASES", "CHAT_ROUTES", "Campaign", "ChatDispatchTransport",
    "ExecutionStopped", "GuardedTransport", "OPENROUTER_PREFERENCES", "PrivateJournal", "Ticket",
    "canonical_sha256", "fingerprint", "maximum_reservation_nanousd", "protocol", "reservation",
]
