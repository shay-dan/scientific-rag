"""Stage 20 evaluation-only, fail-closed physical request accounting.

No provider clients, credentials, network defaults or production configuration.
The caller must supply a transport. Fixed reservations are never refunded; this
is a conservative modeled accounting limit, not a provider billing guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

import httpx

CASES = tuple(f"E{i:03d}" for i in range(1, 6))
GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "gemini-embedding-2"
# Integer nanodollars; deliberately reserve the maximum per admitted physical request.
# Groq: 32768 * 150 + 2000 * 600 = 6115200.
# Embedding: 8192 * 200 = 1638400.
NANO_USD = 1_000_000_000
CAMPAIGN_LIMIT = 850_000_000
PER_CASE_LIMITS = {"groq": 20, "embedding": 25}
TOTAL_LIMITS = {"groq": 100, "embedding": 125}
RESERVATION = {"groq": 6_115_200, "embedding": 1_638_400}
INPUT_LIMIT = {"groq": 32768, "embedding": 8192}
OUTPUT_LIMIT = 2000
FRAMING_ALLOWANCE = 512
QUESTION_SECONDS = 600
CAMPAIGN_SECONDS = 3000


def protocol() -> dict:
    return {
        "schema": "scientific-rag-e2e-protocol/v1", "cases": list(CASES),
        "mode": "fast", "cache_mode": "refresh_sources",
        "generation_prompt_version": "v16", "quote_recovery_mode": "unicode",
        "claim_recovery_mode": "degrade", "semantic_verification_mode": "enforce",
        "claim_audit_mode": "audit", "synthetic_evaluation": False,
        "groq_model": GROQ_MODEL, "reasoning_effort": "low", "temperature": 0,
        "embedding_model": EMBED_MODEL, "embedding_dimensions": 768,
        "per_case_requests": dict(PER_CASE_LIMITS), "campaign_requests": dict(TOTAL_LIMITS),
        "input_allowance": dict(INPUT_LIMIT), "max_output_tokens": OUTPUT_LIMIT,
        "framing_allowance": FRAMING_ALLOWANCE,
        "reservation_nanousd": dict(RESERVATION), "campaign_limit_nanousd": CAMPAIGN_LIMIT,
        "maximum_modeled_reservation_nanousd": sum(
            TOTAL_LIMITS[k] * RESERVATION[k] for k in TOTAL_LIMITS
        ),
        "question_seconds": QUESTION_SECONDS, "campaign_seconds": CAMPAIGN_SECONDS,
        "fallback": False, "retransmission": False, "resume": False,
    }


class ExecutionStopped(RuntimeError):
    """Safety stop; reason is a bounded code, never a raw provider error."""


class PrivateJournal:
    """Exclusive, durable append-only journal. No resume or reuse of this path."""

    def __init__(self, path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._file = os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8")
        # Commit the directory entry too; a crash must not make this path reusable.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def append(self, event: dict) -> None:
        self._file.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


@dataclass(frozen=True)
class Ticket:
    index: int
    provider: str
    case: str


class Campaign:
    """Single isolated process; lock covers sync embedding threads and async chat."""

    def __init__(self, append: Callable[[dict], None], *, clock=time.monotonic):
        self._append = append
        self._clock = clock
        self._lock = threading.RLock()
        self._started = clock()
        self._case_started = None
        self.case = None
        self.completed = []
        self.per_case = dict.fromkeys(TOTAL_LIMITS, 0)
        self.counts = dict.fromkeys(TOTAL_LIMITS, 0)
        self.reserved = 0
        self.stop_reason = None
        self._inflight: dict[int, Ticket] = {}

    def _write(self, event: dict) -> None:
        try:
            self._append(event)
        except Exception as exc:
            self.stop_reason = "journal_failure"
            raise ExecutionStopped(self.stop_reason) from exc

    def stop(self, reason: str) -> None:
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                self._write({"event": "stop", "case": self.case, "reason": reason})

    def _refuse(self, reason: str):
        self.stop(reason)
        raise ExecutionStopped(self.stop_reason)

    def ensure_running(self) -> None:
        with self._lock:
            if self.stop_reason is not None:
                raise ExecutionStopped(self.stop_reason)
            now = self._clock()
            if now - self._started >= CAMPAIGN_SECONDS:
                self._refuse("campaign_deadline")
            if self._case_started is not None and now - self._case_started >= QUESTION_SECONDS:
                self._refuse("question_deadline")

    def begin_case(self, case: str) -> None:
        with self._lock:
            self.ensure_running()
            if (self.case is not None or self._inflight or len(self.completed) >= len(CASES)
                    or case != CASES[len(self.completed)]):
                self._refuse("case_order_or_pending_work")
            self._write({"event": "begin_case", "case": case})
            self.case = case
            self._case_started = self._clock()
            self.per_case = dict.fromkeys(TOTAL_LIMITS, 0)

    def end_case(self) -> None:
        with self._lock:
            self.ensure_running()
            if self.case is None or self._inflight:
                self._refuse("case_not_settled")
            self._write({"event": "end_case", "case": self.case})
            self.completed.append(self.case)
            self.case = None
            self._case_started = None

    def _validate_request(self, request: httpx.Request, provider: str) -> int:
        if provider not in TOTAL_LIMITS or request.method != "POST":
            self._refuse("unexpected_method_or_provider")
        host = "api.groq.com" if provider == "groq" else "generativelanguage.googleapis.com"
        if request.url.scheme != "https" or request.url.host != host or request.url.port not in (None, 443):
            self._refuse("unexpected_destination")
        body_bytes = request.content
        bound = len(body_bytes) + FRAMING_ALLOWANCE
        if bound > INPUT_LIMIT[provider]:
            self._refuse("input_allowance")
        try:
            body = json.loads(body_bytes)
        except (ValueError, TypeError):
            self._refuse("invalid_request_body")
        if not isinstance(body, dict):
            self._refuse("invalid_request_body")
        if provider == "groq":
            caps = [body[k] for k in ("max_tokens", "max_completion_tokens") if body.get(k) is not None]
            if (request.url.path != "/openai/v1/chat/completions"
                    or body.get("model") != GROQ_MODEL or body.get("reasoning_effort") != "low"
                    or not caps or any(type(x) is not int or x != OUTPUT_LIMIT for x in caps)
                    or body.get("temperature") != 0 or body.get("n") not in (None, 1)
                    or body.get("stream") not in (None, False)):
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
                    or not isinstance(part["text"], str) or not part["text"] for part in parts
                ):
                    self._refuse("embedding_contract")
        return bound

    def admit(self, request: httpx.Request, provider: str) -> Ticket:
        with self._lock:
            self.ensure_running()
            if self.case is None:
                self._refuse("no_active_case")
            bound = self._validate_request(request, provider)
            charge = RESERVATION[provider]
            if self.per_case[provider] >= PER_CASE_LIMITS[provider]:
                self._refuse("per_case_request_ceiling")
            if self.counts[provider] >= TOTAL_LIMITS[provider] or self.reserved + charge > CAMPAIGN_LIMIT:
                self._refuse("campaign_ceiling")
            ticket = Ticket(sum(self.counts.values()) + 1, provider, self.case)
            self._write({"event": "reserved_before_transport", "case": ticket.case,
                         "index": ticket.index, "provider": provider, "input_allowance_used": bound,
                         "reserved_nanousd": charge})
            self.counts[provider] += 1
            self.per_case[provider] += 1
            self.reserved += charge
            self._inflight[ticket.index] = ticket
            return ticket

    def returned(self, ticket: Ticket, status: int, usage: dict | None = None) -> None:
        with self._lock:
            if self._inflight.pop(ticket.index, None) != ticket:
                self._refuse("unknown_ticket")
            event = {"event": "transport_returned", "index": ticket.index,
                     "provider": ticket.provider, "status": status}
            usage_invalid = False
            if ticket.provider == "groq" and status == 200 and isinstance(usage, dict):
                inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if type(inp) is int and type(out) is int and min(inp, out) >= 0:
                    event.update({"input_tokens": inp, "output_tokens": out})
                    usage_invalid = inp > INPUT_LIMIT["groq"] or out > OUTPUT_LIMIT
            self._write(event)
            if status != 200:
                self.stop("provider_http_error")
            elif usage_invalid:
                self.stop("provider_usage_exceeded_allowance")

    def uncertain(self, ticket: Ticket) -> None:
        with self._lock:
            self._inflight.pop(ticket.index, None)
            self._write({"event": "transport_uncertain", "index": ticket.index})
            self.stop("uncertain_transmission")

    def summary(self) -> dict:
        with self._lock:
            return {"requests_reserved": dict(self.counts), "reserved_nanousd": self.reserved,
                    "pending": len(self._inflight), "stop_reason": self.stop_reason,
                    "completed_cases": list(self.completed)}


class GuardedTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Guard runs immediately before the supplied transport; no implicit network."""

    def __init__(self, campaign: Campaign, provider: str, *,
                 sync: httpx.BaseTransport | None = None,
                 async_: httpx.AsyncBaseTransport | None = None):
        self.campaign, self.provider = campaign, provider
        self.sync, self.async_ = sync, async_

    @staticmethod
    def _usage(response):
        if response.status_code != 200:
            return None
        try:
            body = response.json()
            return body.get("usage") if isinstance(body, dict) else None
        except ValueError:
            return None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.sync is None:
            self.campaign.stop("missing_sync_transport")
            raise ExecutionStopped("missing_sync_transport")
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
            raise ExecutionStopped("missing_async_transport")
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
