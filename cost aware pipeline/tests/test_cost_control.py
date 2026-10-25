import asyncio
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.config import Settings
from app.cost_control import CostBudgetExceeded, CostGuard
from app.schemas import ResearchMode
from app.services import ResearchService
from scientific_rag.costing import calculate_cost, estimate_text_tokens, rate_for


class AtomicFakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.lock = asyncio.Lock()

    async def eval(self, script, key_count, *items):
        async with self.lock:
            keys = list(items[:key_count])
            args = list(items[key_count:])
            if "daily_limit" in script:
                reservation, daily_key, monthly_key = keys
                if reservation in self.hashes:
                    return [1, self.hashes[reservation]["amount"], b"existing"]
                amount, daily_limit, monthly_limit = map(int, args[:3])
                daily_used = self.values.get(daily_key, 0)
                monthly_used = self.values.get(monthly_key, 0)
                if daily_limit and daily_used + amount > daily_limit:
                    return [0, daily_used, b"daily"]
                if monthly_limit and monthly_used + amount > monthly_limit:
                    return [0, monthly_used, b"monthly"]
                self.values[daily_key] = daily_used + amount
                self.values[monthly_key] = monthly_used + amount
                self.hashes[reservation] = {"amount": amount, "daily_key": daily_key, "monthly_key": monthly_key}
                return [1, amount, b"reserved"]
            reservation = keys[0]
            if reservation not in self.hashes:
                return [0, 0, b"missing"]
            actual = int(args[0])
            data = self.hashes[reservation]
            delta = actual - data["amount"]
            self.values[data["daily_key"]] += delta
            self.values[data["monthly_key"]] += delta
            data["amount"] = actual
            return [1, delta, str(args[1]).encode()]


class PricingTests(unittest.TestCase):
    def test_builtin_groq_rate_prices_usage(self):
        cost, source = calculate_cost("groq", "openai/gpt-oss-120b", 1_000_000, 1_000_000)
        self.assertEqual(cost, 0.75)
        self.assertEqual(source, "builtin-2026-10-24")

    def test_environment_rate_overrides_builtin(self):
        with patch.dict(os.environ, {
            "LLM_GROQ_OPENAI_GPT_OSS_120B_INPUT_USD_PER_MILLION": "1",
            "LLM_GROQ_OPENAI_GPT_OSS_120B_OUTPUT_USD_PER_MILLION": "2",
        }):
            self.assertEqual(rate_for("groq", "openai/gpt-oss-120b").source, "environment")
            self.assertEqual(calculate_cost("groq", "openai/gpt-oss-120b", 500_000, 250_000)[0], 1.0)

    def test_embedding_tokens_are_explicit_estimates(self):
        self.assertEqual(estimate_text_tokens(["abcd", "abcde"]), 3)


class CostGuardTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, **overrides):
        values = dict(cost_daily_budget_usd=0.50, cost_monthly_budget_usd=2.0,
                      cost_fast_reservation_usd=0.05, cost_standard_reservation_usd=0.20,
                      cost_deep_reservation_usd=0.75)
        values.update(overrides)
        return Settings(**values)

    async def test_same_run_reservation_is_idempotent(self):
        redis = AtomicFakeRedis()
        guard = CostGuard(redis, self.settings(), namespace="test:cost")
        first = await guard.reserve_research("same", ResearchMode.fast)
        second = await guard.reserve_research("same", ResearchMode.fast)
        self.assertEqual(first.amount_usd, second.amount_usd)
        self.assertEqual(sum(value for key, value in redis.values.items() if ":daily:" in key), 50_000)

    async def test_concurrent_reservations_cannot_race_past_daily_budget(self):
        redis = AtomicFakeRedis()
        guard = CostGuard(redis, self.settings(), namespace="test:cost")

        async def reserve(index):
            try:
                await guard.reserve_research(str(index), ResearchMode.fast)
                return True
            except CostBudgetExceeded:
                return False

        outcomes = await asyncio.gather(*(reserve(index) for index in range(25)))
        self.assertEqual(sum(outcomes), 10)

    async def test_reconciliation_replaces_reservation_with_actual_cost(self):
        redis = AtomicFakeRedis()
        guard = CostGuard(redis, self.settings(), namespace="test:cost")
        await guard.reserve_research("run", ResearchMode.standard)
        result = await guard.reconcile_research("run", 0.03)
        self.assertEqual(result.amount_usd, 0.03)
        self.assertEqual(sum(value for key, value in redis.values.items() if ":daily:" in key), 30_000)

    async def test_evaluation_reserves_judge_upper_bound_per_case(self):
        redis = AtomicFakeRedis()
        guard = CostGuard(redis, self.settings(cost_evaluation_judge_reservation_usd_per_case=0.02), namespace="test:cost")
        reservation = await guard.reserve_evaluation("evaluation", 4)
        self.assertEqual(reservation.amount_usd, 0.08)

    async def test_missing_redis_fails_closed_when_enforcement_enabled(self):
        guard = CostGuard(None, self.settings(cost_enforce_hard_limit=True))
        with self.assertRaises(CostBudgetExceeded):
            await guard.reserve_research("run", ResearchMode.fast)

    async def test_incomplete_usage_keeps_conservative_reservation(self):
        service = object.__new__(ResearchService)
        service.cost_guard = SimpleNamespace(reconcile_research=AsyncMock())
        record = SimpleNamespace(id="run", model_metadata={"cost_reservation": {"amount_usd": 0.20}})
        metadata = {
            "llm_gateway": {"estimated_cost_usd": None},
            "embedding_usage": {"estimated_cost_usd": 0.01},
        }
        await service._reconcile_cost(record, metadata)
        service.cost_guard.reconcile_research.assert_not_awaited()
        self.assertFalse(metadata["cost_accounting"]["complete"])
        self.assertEqual(metadata["cost_accounting"]["estimated_cost_usd"], 0.01)


if __name__ == "__main__":
    unittest.main()
