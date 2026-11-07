"""Atomic Redis-backed spend reservations and hard budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Settings
from .schemas import ResearchMode


_SCALE = 1_000_000
_RESERVE_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return {1, tonumber(redis.call('HGET', KEYS[1], 'amount') or '0'), 'existing'}
end
local amount = tonumber(ARGV[1])
local daily_limit = tonumber(ARGV[2])
local monthly_limit = tonumber(ARGV[3])
local daily_used = tonumber(redis.call('GET', KEYS[2]) or '0')
local monthly_used = tonumber(redis.call('GET', KEYS[3]) or '0')
if daily_limit > 0 and daily_used + amount > daily_limit then
  return {0, daily_used, 'daily'}
end
if monthly_limit > 0 and monthly_used + amount > monthly_limit then
  return {0, monthly_used, 'monthly'}
end
redis.call('INCRBY', KEYS[2], amount)
redis.call('INCRBY', KEYS[3], amount)
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[4]))
redis.call('EXPIRE', KEYS[3], tonumber(ARGV[5]))
redis.call('HSET', KEYS[1], 'amount', amount, 'daily_key', KEYS[2], 'monthly_key', KEYS[3], 'status', 'reserved')
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
return {1, amount, 'reserved'}
"""

_RECONCILE_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then return {0, 0, 'missing'} end
local old_amount = tonumber(redis.call('HGET', KEYS[1], 'amount') or '0')
local daily_key = redis.call('HGET', KEYS[1], 'daily_key')
local monthly_key = redis.call('HGET', KEYS[1], 'monthly_key')
local actual = tonumber(ARGV[1])
local delta = actual - old_amount
if daily_key then redis.call('INCRBY', daily_key, delta) end
if monthly_key then redis.call('INCRBY', monthly_key, delta) end
redis.call('HSET', KEYS[1], 'amount', actual, 'status', ARGV[2])
return {1, delta, ARGV[2]}
"""


class CostBudgetExceeded(ValueError):
    """A new paid run would exceed an operator-defined spend budget."""


@dataclass(frozen=True)
class Reservation:
    amount_usd: float
    status: str

    def as_dict(self) -> dict[str, float | str]:
        return {"amount_usd": self.amount_usd, "status": self.status}


class CostGuard:
    def __init__(self, redis, settings: Settings, *, namespace: str = "scientific_rag:cost") -> None:
        self.redis = redis
        self.settings = settings
        self.namespace = namespace

    def amount_for_mode(self, mode: ResearchMode) -> float:
        return {
            ResearchMode.fast: self.settings.cost_fast_reservation_usd,
            ResearchMode.standard: self.settings.cost_standard_reservation_usd,
            ResearchMode.deep: self.settings.cost_deep_reservation_usd,
        }[mode]

    def _keys(self, reservation_id: str) -> tuple[str, str, str]:
        now = datetime.now(ZoneInfo("Asia/Jerusalem"))
        return (
            f"{self.namespace}:reservation:{reservation_id}",
            f"{self.namespace}:daily:{now:%Y-%m-%d}",
            f"{self.namespace}:monthly:{now:%Y-%m}",
        )

    async def reserve(self, reservation_id: str, amount_usd: float) -> Reservation:
        if not self.settings.cost_tracking_enabled:
            return Reservation(0.0, "disabled")
        if self.redis is None:
            if self.settings.cost_enforce_hard_limit:
                raise CostBudgetExceeded("Cost guard is unavailable; paid work was not started")
            return Reservation(amount_usd, "unavailable")
        keys = self._keys(reservation_id)
        amount = round(amount_usd * _SCALE)
        daily_limit = round(self.settings.cost_daily_budget_usd * _SCALE)
        monthly_limit = round(self.settings.cost_monthly_budget_usd * _SCALE)
        result = await self.redis.eval(
            _RESERVE_SCRIPT, 3, *keys, amount, daily_limit, monthly_limit, 3 * 24 * 3600, 40 * 24 * 3600
        )
        allowed, current, reason = int(result[0]), int(result[1]), _decode(result[2])
        if not allowed:
            limit = self.settings.cost_daily_budget_usd if reason == "daily" else self.settings.cost_monthly_budget_usd
            raise CostBudgetExceeded(
                f"The {reason} cost budget would be exceeded (reserved ${current / _SCALE:.4f} of ${limit:.2f})"
            )
        return Reservation(current / _SCALE, reason)

    async def reserve_research(self, run_id: str, mode: ResearchMode) -> Reservation:
        return await self.reserve(f"research:{run_id}", self.amount_for_mode(mode))

    async def reserve_evaluation(self, run_id: str, case_count: int) -> Reservation:
        amount = self.settings.cost_evaluation_judge_reservation_usd_per_case * max(case_count, 0)
        return await self.reserve(f"evaluation:{run_id}", amount)

    async def reconcile_research(self, run_id: str, actual_usd: float | None, *, status: str = "reconciled") -> Reservation | None:
        if not self.settings.cost_tracking_enabled or self.redis is None or actual_usd is None:
            return None
        key = self._keys(f"research:{run_id}")[0]
        result = await self.redis.eval(_RECONCILE_SCRIPT, 1, key, round(actual_usd * _SCALE), status)
        if not int(result[0]):
            return None
        return Reservation(actual_usd, status)

    async def release_research(self, run_id: str) -> Reservation | None:
        return await self.reconcile_research(run_id, 0.0, status="released")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
