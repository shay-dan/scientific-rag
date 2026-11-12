"""Rebuild evaluation-run aggregates from persisted judged results."""

import argparse
import asyncio
from uuid import UUID

from sqlalchemy import select

from app.db import close_database, session_factory
from app.evaluation import summarize_scores
from app.evaluation_report import build_evaluation_report, case_content_hash, legacy_case_record
from app.models import EvaluationCase, EvaluationResult, EvaluationRun


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
                cases = {case.id: case for case in (await session.execute(
                    select(EvaluationCase).where(EvaluationCase.dataset_id == run.dataset_id))).scalars()}
                records = [legacy_case_record(
                    case_id=str(item.evaluation_case_id),
                    case_hash=case_content_hash(
                        question=cases[item.evaluation_case_id].question,
                        reference_answer=cases[item.evaluation_case_id].reference_answer,
                        reference_context=cases[item.evaluation_case_id].reference_context,
                        reference_claims=cases[item.evaluation_case_id].reference_claims,
                        provenance=cases[item.evaluation_case_id].provenance),
                    case_provenance=cases[item.evaluation_case_id].provenance, status=item.status,
                    judge_scores=item.judge_scores, error=item.error) for item in results]
                provenance = run.metrics.get("provenance") or {"legacy_run": True, "mode": run.metrics.get("mode")}
                report = build_evaluation_report(provenance, records)
                run.metrics = {
                    **run.metrics,
                    **summarize_scores(judged),  # engine-aware (R4): mixed engines yield no single value
                    "total_cases": len(results),
                    "failed_cases": len(results) - len(judged),
                    "unjudged_cases": len(results) - len(judged),
                    "outcomes": report["population"]["outcomes"],
                    "report": report,
                }
            await session.commit()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
