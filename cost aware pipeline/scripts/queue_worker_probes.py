"""Queue and optionally wait for provider-free ARQ worker probe jobs."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import time
from uuid import uuid4

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--enqueue-only", action="store_true")
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    batch_id = args.batch_id or uuid4().hex
    started = time.perf_counter()
    for item in range(args.count):
        await redis.enqueue_job(
            "reliability_probe_job", batch_id, item, args.delay_ms,
            _job_id=f"reliability:{batch_id}:{item}",
        )
    if args.enqueue_only:
        print(f"batch_id={batch_id} queued={args.count}")
        await redis.aclose()
        return 0
    key = f"scientific_rag:reliability:worker:{batch_id}"
    deadline = time.monotonic() + args.timeout
    completed = 0
    while time.monotonic() < deadline:
        completed = int(await redis.scard(key))
        if completed == args.count:
            break
        await asyncio.sleep(0.2)
    elapsed = time.perf_counter() - started
    print(f"batch_id={batch_id} queued={args.count} completed={completed} elapsed_seconds={elapsed:.2f}")
    await redis.delete(key)
    await redis.aclose()
    return 0 if completed == args.count else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
