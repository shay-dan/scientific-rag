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
    principal = principal_for_key(request.headers.get("X-API-Key", ""), request.app.state.settings.app_api_keys)
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
