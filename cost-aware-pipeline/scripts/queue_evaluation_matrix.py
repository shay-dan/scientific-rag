"""Queue the same reviewed evaluation dataset in Fast and Standard modes."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings
from app.db import close_database, session_factory
from app.models import EvaluationCase, EvaluationDataset, EvaluationRun


async def queue_matrix(dataset_path: Path, modes: list[str], case_index: int | None = None) -> list[dict]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if case_index is not None:
        if case_index < 1 or case_index > len(cases):
            raise ValueError(f"case index must be between 1 and {len(cases)}")
        cases = [cases[case_index - 1]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    queued = []
    redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    try:
        for mode in modes:
            async with session_factory() as session:
                dataset = EvaluationDataset(
                    name=f"{payload['name']}-{mode}-{stamp}",
                    created_by="claim-diagnostic-matrix",
                )
                session.add(dataset)
                await session.flush()
                session.add_all([
                    EvaluationCase(
                        dataset_id=dataset.id,
                        question=case["question"],
                        reference_answer=case["reference_answer"],
                        reference_context=case.get("reference_context"),
                        reference_claims=case.get("reference_claims", []),
                        provenance="human",
                    )
                    for case in cases
                ])
                run = EvaluationRun(dataset_id=dataset.id, status="queued", metrics={"mode": mode})
                session.add(run)
                await session.commit()
                await redis.enqueue_job("run_evaluation_job", str(run.id))
                queued.append({
                    "mode": mode,
                    "dataset": dataset.name,
                    "evaluation_run_id": str(run.id),
                    "case_count": len(cases),
                })
    finally:
        await redis.aclose()
        await close_database()
    return queued


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("fast", "standard"), default=["fast", "standard"])
    parser.add_argument("--case-index", type=int, help="Queue only one 1-based case from the dataset")
    args = parser.parse_args()
    print(json.dumps(await queue_matrix(args.dataset, args.modes, args.case_index), indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
