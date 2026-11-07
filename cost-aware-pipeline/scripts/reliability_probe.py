"""No-LLM reliability probe for HTTP, PostgreSQL, and atomic Redis budgets."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request
from uuid import uuid4

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import text

from app.config import Settings, get_settings
from app.cost_control import CostBudgetExceeded, CostGuard
from app.db import session_factory


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile_value), len(ordered) - 1)
    return round(ordered[index], 2)


async def timed_many(factory, count: int, concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int):
        async with semaphore:
            started = time.perf_counter()
            try:
                await factory(index)
                return True, (time.perf_counter() - started) * 1000, None
            except Exception as error:
                return False, (time.perf_counter() - started) * 1000, type(error).__name__

    rows = await asyncio.gather(*(one(index) for index in range(count)))
    latencies = [row[1] for row in rows]
    return {
        "requests": count,
        "concurrency": concurrency,
        "successes": sum(1 for row in rows if row[0]),
        "failures": sum(1 for row in rows if not row[0]),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "errors": sorted({row[2] for row in rows if row[2]}),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    async def http_request(_):
        def fetch():
            with urllib.request.urlopen(f"{args.base_url}/healthz", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
        await asyncio.to_thread(fetch)

    async def database_request(_):
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))

    report = {
        "http_health": await timed_many(http_request, args.requests, args.concurrency),
        "postgresql": await timed_many(database_request, args.requests, args.concurrency),
    }

    settings = get_settings()
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    namespace = f"scientific_rag:reliability:{uuid4()}"
    test_settings = Settings(
        cost_daily_budget_usd=0.10,
        cost_monthly_budget_usd=0.10,
        cost_fast_reservation_usd=0.01,
        cost_standard_reservation_usd=0.01,
        cost_deep_reservation_usd=0.01,
    )
    guard = CostGuard(redis, test_settings, namespace=namespace)

    async def reserve(index):
        try:
            await guard.reserve(str(index), 0.01)
        except CostBudgetExceeded:
            return

    redis_report = await timed_many(reserve, 30, 30)
    day_keys = [key async for key in redis.scan_iter(match=f"{namespace}:daily:*")]
    reserved = int(await redis.get(day_keys[0]) or 0) if day_keys else 0
    redis_report["admitted"] = reserved // 10_000
    redis_report["expected_admitted"] = 10
    report["redis_atomic_budget"] = redis_report
    keys = [key async for key in redis.scan_iter(match=f"{namespace}:*")]
    if keys:
        await redis.delete(*keys)
    await redis.aclose()

    report["passed"] = (
        report["http_health"]["failures"] == 0
        and report["postgresql"]["failures"] == 0
        and redis_report["admitted"] == 10
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
