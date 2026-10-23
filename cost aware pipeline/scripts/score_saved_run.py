"""Score an existing research run against a reviewed benchmark case with Ragas.

This is read-only with respect to PostgreSQL. It avoids re-running research and
does not silently substitute fallback scores for Ragas scores.
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.config import get_settings
from app.db import close_database, session_factory
from app.evaluation import diagnose_reference_claims, score_with_ragas
from app.models import ResearchEvidence, ResearchRun


async def score(run_id: UUID, benchmark_path: Path) -> dict:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    async with session_factory() as session:
        run = await session.get(ResearchRun, run_id)
        if run is None or run.status != "completed" or not run.answer:
            raise ValueError("Run not found or not completed")
        case = next((item for item in benchmark["cases"] if item["question"] == run.question), None)
        if case is None:
            raise ValueError("The benchmark has no reviewed case with this exact question")
        contexts = list((await session.execute(
            select(ResearchEvidence.content)
            .where(ResearchEvidence.research_run_id == run.id)
            .order_by(ResearchEvidence.ordinal)
        )).scalars())
        if not contexts:
            raise ValueError("The run has no saved evidence passages to evaluate")
        settings = get_settings()
        scores = await asyncio.wait_for(score_with_ragas(
            question=run.question,
            answer=run.answer,
            retrieved_contexts=contexts,
            reference_answer=case["reference_answer"],
            reference_context=case.get("reference_context"),
            api_key=settings.google_api_key,
            judge_model=settings.ragas_judge_model,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        ), timeout=240)
        claim_diagnostics = None
        reference_claims = case.get("reference_claims") or []
        if reference_claims:
            claim_diagnostics = await asyncio.wait_for(diagnose_reference_claims(
                question=run.question,
                reference_claims=reference_claims,
                retrieved_contexts=contexts,
                api_key=settings.google_api_key,
                judge_model=settings.ragas_judge_model,
            ), timeout=120)
        return {
            "run_id": str(run.id),
            "pipeline_version": run.model_metadata.get("retrieval_version"),
            "question": run.question,
            "reference_dataset": benchmark["name"],
            "retrieved_passages": len(contexts),
            "engine": "ragas",
            "scores": scores,
            "claim_diagnostics": claim_diagnostics,
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID)
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmark-v1.json"))
    args = parser.parse_args()
    try:
        print(json.dumps(await score(args.run_id, args.benchmark), indent=2), flush=True)
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
