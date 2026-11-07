"""Small Redis-backed, shared LLM admission controller.

Reservations are conservative: a failed model call keeps its estimated tokens in
the current minute. A successful call reconciles the estimate with reported usage.
Redis leases prevent overlapping Groq gatekeeper requests across API/worker
processes and expire even if a process crashes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import time
from uuid import uuid4


ADMIT_SCRIPT = """
local prefix = KEYS[1]
local t = redis.call('TIME')
local now_ms = t[1] * 1000 + math.floor(t[2] / 1000)
local minute = math.floor(now_ms / 60000)
local requests = prefix .. ':requests:' .. minute
local tokens = prefix .. ':tokens:' .. minute
local lease = prefix .. ':gatekeeper:lease'
local cooldown = prefix .. ':cooldown'
local rpm = tonumber(ARGV[1])
local tpm = tonumber(ARGV[2])
local estimate = tonumber(ARGV[3])
local owner = ARGV[4]
local lease_ms = tonumber(ARGV[5])
local gatekeeper = ARGV[6] == '1'
local wait_ms = 60000 - (now_ms % 60000)
local cooldown_ms = redis.call('PTTL', cooldown)
if cooldown_ms > 0 then return {0, cooldown_ms, 'cooldown', minute} end
if rpm > 0 and tonumber(redis.call('GET', requests) or '0') >= rpm then
  return {0, wait_ms, 'requests', minute}
end
if tpm > 0 and tonumber(redis.call('GET', tokens) or '0') + estimate > tpm then
  return {0, wait_ms, 'tokens', minute}
end
if gatekeeper and not redis.call('SET', lease, owner, 'NX', 'PX', lease_ms) then
  return {0, math.max(1, redis.call('PTTL', lease)), 'concurrency', minute}
end
redis.call('INCR', requests)
redis.call('PEXPIRE', requests, 120000)
redis.call('INCRBY', tokens, estimate)
redis.call('PEXPIRE', tokens, 120000)
return {1, 0, 'admitted', minute}
"""

RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


@dataclass(frozen=True)
class Admission:
    provider: str
    owner: str
    minute: int
    estimated_tokens: int
    gatekeeper: bool
    wait_ms: float


class ProviderThrottled(Exception):
    def __init__(self, provider: str, reason: str, retry_after_ms: int):
        super().__init__(f"{provider} admission unavailable: {reason}")
        self.provider = provider
        self.reason = reason
        self.retry_after_ms = retry_after_ms


def _prefix(provider: str) -> str:
    return f"{os.getenv('LLM_THROTTLE_NAMESPACE', 'scientific_rag:llm')}:{provider}"


def estimate_tokens(role: str, payload: object) -> int:
    """Conservative budget estimate, not a tokenizer or a billable-token count."""
    input_chars = len(str(payload))
    if role == "expert":
        input_chars += 3500  # Fixed synthesis instructions not present in payload.
    output_reserve = int(os.getenv(f"LLM_{role.upper()}_OUTPUT_RESERVE_TOKENS", "3000" if role == "expert" else "500"))
    return max(1, (input_chars + 3) // 4 + output_reserve)


def _budget(provider: str, kind: str) -> int:
    defaults = {"groq": {"RPM": 20, "TPM": 6000}, "openrouter": {"RPM": 10, "TPM": 0}, "gemini": {"RPM": 0, "TPM": 0}}
    return int(os.getenv(f"LLM_{provider.upper()}_{kind}_BUDGET", str(defaults[provider][kind])))


async def admit(redis, provider: str, role: str, payload: object, timeout_seconds: float) -> Admission | None:
    """Wait briefly for a Groq gatekeeper lease; otherwise fail over promptly."""
    estimate = estimate_tokens(role, payload)
    rpm, tpm = _budget(provider, "RPM"), _budget(provider, "TPM")
    if rpm < 0 or tpm < 0:
        raise ValueError("LLM shared budgets cannot be negative")
    owner = str(uuid4())
    gatekeeper = provider == "groq" and role == "gatekeeper"
    if rpm == 0 and tpm == 0 and not gatekeeper:
        return None
    wait_limit = float(os.getenv("LLM_GROQ_GATEKEEPER_WAIT_SECONDS", "4")) if gatekeeper else 0.0
    stop_at = time.monotonic() + min(max(0.0, wait_limit), max(0.0, timeout_seconds))
    waited_at = time.monotonic()
    prefix = _prefix(provider)
    while True:
        try:
            allowed, retry_ms, reason, minute = await redis.eval(
                ADMIT_SCRIPT, 1, prefix, rpm, tpm, estimate, owner,
                max(1000, round((timeout_seconds + 5) * 1000)), int(gatekeeper),
            )
        except Exception as error:
            raise ProviderThrottled(provider, f"Redis unavailable ({type(error).__name__})", 0) from error
        if isinstance(reason, bytes):
            reason = reason.decode()
        if int(allowed):
            return Admission(provider, owner, int(minute), estimate, gatekeeper, round((time.monotonic() - waited_at) * 1000, 1))
        if reason != "concurrency" or time.monotonic() >= stop_at:
            raise ProviderThrottled(provider, str(reason), int(retry_ms))
        await asyncio.sleep(min(0.2, max(0.0, stop_at - time.monotonic())))


async def release(redis, admission: Admission, actual_tokens: int | None) -> None:
    prefix = _prefix(admission.provider)
    if actual_tokens is not None:
        key = f"{prefix}:tokens:{admission.minute}"
        if await redis.exists(key):
            await redis.incrby(key, actual_tokens - admission.estimated_tokens)
    if admission.gatekeeper:
        await redis.eval(RELEASE_SCRIPT, 1, f"{prefix}:gatekeeper:lease", admission.owner)


async def cool_down(redis, provider: str, seconds: float) -> None:
    await redis.set(f"{_prefix(provider)}:cooldown", "1", px=max(1, round(seconds * 1000)))
