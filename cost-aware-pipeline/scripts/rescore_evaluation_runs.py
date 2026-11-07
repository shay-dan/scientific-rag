"""Re-score persisted evaluation results without running research or mining again."""

import argparse
import asyncio
import json
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.db import close_database, session_factory
from app.evaluation import diagnose_reference_claims, diagnose_retrieved_contexts, score_with_ragas
from app.models import EvaluationCase, EvaluationResult


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", type=UUID)
    args = parser.parse_args()
    settings = get_settings()
    summaries = []
    try:
        async with session_factory() as session:
            results = list((await session.execute(
                select(EvaluationResult).where(EvaluationResult.evaluation_run_id.in_(args.run_ids))
            )).scalars())
            for result in results:
                if not result.answer or not result.retrieved_contexts:
                    summaries.append({"result_id": str(result.id), "status": "skipped_missing_answer_or_context"})
                    continue
                case = await session.get(EvaluationCase, result.evaluation_case_id)
                scores = await asyncio.wait_for(score_with_ragas(
                    question=case.question,
                    answer=result.answer,
                    retrieved_contexts=list(result.retrieved_contexts),
                    reference_answer=case.reference_answer,
                    reference_context=case.reference_context,
                    api_key=settings.google_api_key,
                    judge_model=settings.ragas_judge_model,
                    embedding_model=settings.embedding_model,
                    embedding_dimensions=settings.embedding_dimensions,
                ), timeout=settings.evaluation_case_timeout_seconds)
                if case.reference_claims:
                    scores["claim_diagnostics"] = await asyncio.wait_for(diagnose_reference_claims(
                        question=case.question,
                        reference_claims=list(case.reference_claims),
                        retrieved_contexts=list(result.retrieved_contexts),
                        api_key=settings.google_api_key,
                        judge_model=settings.ragas_judge_model,
                    ), timeout=settings.claim_diagnostic_timeout_seconds)
                scores["retrieval_diagnostics"] = await asyncio.wait_for(diagnose_retrieved_contexts(
                    question=case.question,
                    reference_answer=case.reference_answer,
                    reference_claims=list(case.reference_claims or []),
                    retrieved_contexts=list(result.retrieved_contexts),
                    api_key=settings.google_api_key,
                    judge_model=settings.ragas_judge_model,
                ), timeout=settings.claim_diagnostic_timeout_seconds)
                scores["engine"] = "ragas"
                scores["context_metrics_mode"] = "ragas_reference_answer"
                result.judge_scores = scores
                summaries.append({
                    "result_id": str(result.id),
                    "evaluation_run_id": str(result.evaluation_run_id),
                    "question": case.question,
                    "scores": scores,
                })
                await session.commit()
                print(json.dumps(summaries[-1]), flush=True)
    finally:
        await close_database()
    print(json.dumps({"rescored": len([item for item in summaries if "scores" in item]), "results": summaries}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
