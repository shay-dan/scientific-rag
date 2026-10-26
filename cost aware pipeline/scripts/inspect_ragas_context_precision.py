"""Print Ragas' raw per-context verdicts for one persisted evaluation result."""

import argparse
import asyncio
import json
from uuid import UUID

from app.config import get_settings
from app.db import close_database, session_factory
from app.evaluation import score_ragas_context_precision
from app.models import EvaluationCase, EvaluationResult


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_id", type=UUID)
    parser.add_argument("--claim-aware", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    try:
        async with session_factory() as session:
            result = await session.get(EvaluationResult, args.result_id)
            if result is None:
                raise ValueError("Evaluation result not found")
            case = await session.get(EvaluationCase, result.evaluation_case_id)
            output = {
                "metric": "ContextPrecisionWithReference",
                "note": (
                    "Ragas 0.4 public API returns a score rather than exposing private prompt verdict objects. "
                    "Each score below evaluates one persisted excerpt independently."
                ),
                "claim_aware_requested": args.claim_aware,
                "question": case.question,
                "reference_answer": case.reference_answer,
                "context_scores": [],
            }
            for rank, context in enumerate(result.retrieved_contexts, start=1):
                score = await score_ragas_context_precision(
                    question=case.question,
                    reference_answer=case.reference_answer,
                    retrieved_contexts=[context],
                    api_key=settings.google_api_key,
                    judge_model=settings.ragas_judge_model,
                )
                output["context_scores"].append({
                    "rank": rank,
                    "context_preview": context[:500],
                    "score": score,
                })
            print(json.dumps(output, indent=2), flush=True)
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
