"""Rebuild evaluation-run aggregates from persisted judged results."""

import argparse
import asyncio
from uuid import UUID

from sqlalchemy import select

from app.db import close_database, session_factory
from app.evaluation import summarize_scores
from app.models import EvaluationResult, EvaluationRun


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", type=UUID)
    args = parser.parse_args()
    try:
        async with session_factory() as session:
            for run_id in args.run_ids:
                run = await session.get(EvaluationRun, run_id)
                if run is None:
                    continue
                results = list((await session.execute(
                    select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run_id)
                )).scalars())
                judged = [item.judge_scores for item in results if item.status == "judged"]
                run.metrics = {
                    **run.metrics,
                    **summarize_scores(judged),
                    "total_cases": len(results),
                    "failed_cases": len(results) - len(judged),
                }
            await session.commit()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
