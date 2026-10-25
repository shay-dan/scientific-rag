"""Authentication and in-process abuse controls for a controlled beta."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hmac
from threading import Lock

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    name: str
    role: str


def principal_for_key(raw_key: str, configured_keys: str) -> Principal | None:
    for item in configured_keys.split(","):
        if ":" not in item:
            continue
        identity, key = item.strip().split(":", 1)
        name, _, declared_role = identity.partition("|")
        if hmac.compare_digest(raw_key, key):
            return Principal(name=name, role=declared_role or ("admin" if name == "admin" else "user"))
    return None


def require_api_key(request: Request) -> Principal:
    raw_key = request.headers.get("X-API-Key", "")
    service_key = getattr(request.app.state.settings, "health_coach_api_key", "")
    principal = (Principal("health_coach", "service") if service_key and hmac.compare_digest(raw_key, service_key)
                 else principal_for_key(raw_key, request.app.state.settings.app_api_keys))
    if principal is None:
        raise HTTPException(status_code=401, detail="Valid X-API-Key required")
    return principal


class BetaLimiter:
    """Single-instance limiter. Replace with Redis before running multiple API instances."""
    def __init__(self, requests_per_minute: int, research_runs_per_day: int, deep_research_runs_per_day: int = 1) -> None:
        self.requests_per_minute = requests_per_minute
        self.research_runs_per_day = research_runs_per_day
        self.deep_research_runs_per_day = deep_research_runs_per_day
        self._requests: dict[str, deque[datetime]] = defaultdict(deque)
        self._research: dict[tuple[str, date], int] = defaultdict(int)
        self._deep_research: dict[tuple[str, date], int] = defaultdict(int)
        self._lock = Lock()

    def check_request(self, principal: Principal) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            bucket = self._requests[principal.name]
            while bucket and (now - bucket[0]).total_seconds() >= 60:
                bucket.popleft()
            if len(bucket) >= self.requests_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded; try again in a minute")
            bucket.append(now)

    def consume_research_run(self, principal: Principal, mode: str = "fast") -> None:
        if principal.role == "admin":
            return
        key = (principal.name, datetime.now(timezone.utc).date())
        with self._lock:
            if self._research[key] >= self.research_runs_per_day:
                raise HTTPException(status_code=429, detail="Daily research quota reached")
            if mode == "deep" and self._deep_research[key] >= self.deep_research_runs_per_day:
                raise HTTPException(status_code=429, detail="Daily Deep research quota reached")
            self._research[key] += 1
            if mode == "deep":
                self._deep_research[key] += 1


_REQUEST_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2])) end
if current > tonumber(ARGV[1]) then return {0, current} end
return {1, current}
"""

_RESEARCH_LIMIT_SCRIPT = """
local total = tonumber(redis.call('GET', KEYS[1]) or '0')
local deep = tonumber(redis.call('GET', KEYS[2]) or '0')
if total >= tonumber(ARGV[1]) then return {0, total, 'total'} end
if ARGV[4] == 'deep' and deep >= tonumber(ARGV[2]) then return {0, deep, 'deep'} end
redis.call('INCR', KEYS[1]); redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
if ARGV[4] == 'deep' then redis.call('INCR', KEYS[2]); redis.call('EXPIRE', KEYS[2], tonumber(ARGV[3])) end
return {1, total + 1, 'ok'}
"""


class RedisBetaLimiter:
    """Multi-instance request and daily research quotas backed by Redis Lua."""

    def __init__(self, redis, requests_per_minute: int, research_runs_per_day: int, deep_research_runs_per_day: int = 1) -> None:
        self.redis = redis
        self.requests_per_minute = requests_per_minute
        self.research_runs_per_day = research_runs_per_day
        self.deep_research_runs_per_day = deep_research_runs_per_day

    async def check_request(self, principal: Principal) -> None:
        minute = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        key = f"scientific_rag:rate:request:{principal.name}:{minute}"
        allowed, _ = await self.redis.eval(_REQUEST_LIMIT_SCRIPT, 1, key, self.requests_per_minute, 120)
        if not int(allowed):
            raise HTTPException(status_code=429, detail="Rate limit exceeded; try again in a minute")

    async def consume_research_run(self, principal: Principal, mode: str = "fast") -> None:
        if principal.role == "admin":
            return
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        keys = (
            f"scientific_rag:rate:research:{principal.name}:{day}",
            f"scientific_rag:rate:deep:{principal.name}:{day}",
        )
        allowed, _, reason = await self.redis.eval(
            _RESEARCH_LIMIT_SCRIPT, 2, *keys,
            self.research_runs_per_day, self.deep_research_runs_per_day, 2 * 24 * 3600, mode,
        )
        reason = reason.decode() if isinstance(reason, bytes) else str(reason)
        if not int(allowed):
            detail = "Daily Deep research quota reached" if reason == "deep" else "Daily research quota reached"
            raise HTTPException(status_code=429, detail=detail)
