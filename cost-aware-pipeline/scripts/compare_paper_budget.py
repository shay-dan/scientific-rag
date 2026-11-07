"""One-off Standard 3-vs-5 paper experiment; does not change app defaults.

Runs fresh research and scores each saved answer against the same committed
Ragas reference. Makes billable API calls and stores the two research runs.
"""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import select

from app import services
from app.config import get_settings
from app.cost_policy import budget_for
from app.db import close_database, session_factory
from app.models import ResearchEvidence, ResearchRun, SourceDocument
from app.schemas import ResearchMode
from scripts.score_saved_run import score


QUESTION = "What are current uses of CRISPR in cancer research?"
BENCHMARK = Path("evaluation/benchmark-v1.json")


async def run_case(papers_per_source: int, redis: Redis) -> dict:
    budget = replace(budget_for(ResearchMode.standard), papers_per_source=papers_per_source)
    service = services.ResearchService(get_settings(), redis)
    started = time.monotonic()
    with patch.object(services, "budget_for", return_value=budget):
        async with session_factory() as session:
            response = await service.run(
                session, QUESTION, ResearchMode.standard, False,
                actor=f"paper-budget-experiment-{papers_per_source}", fresh=True,
            )
    run_id = UUID(str(response.id))
    async with session_factory() as session:
        run = await session.get(ResearchRun, run_id)
        evidence = list((await session.execute(select(ResearchEvidence).where(ResearchEvidence.research_run_id == run_id))).scalars())
        sources = list((await session.execute(select(SourceDocument).where(SourceDocument.research_run_id == run_id))).scalars())
        gateway = run.model_metadata.get("llm_gateway", {})
        result = {
            "run_id": str(run_id), "papers_per_source": papers_per_source,
            "status": run.status, "research_seconds": run.model_metadata.get("duration_seconds"),
            "indexed_papers": len(sources),
            "selected_papers": len({item.metadata_json.get("external_id") or item.metadata_json.get("title") for item in evidence}),
            "selected_passages": len(evidence),
            "paper_titles": [source.title for source in sources],
            "gateway_provider_calls": gateway.get("provider_calls"),
            "gateway_fallbacks": gateway.get("fallback_count"),
            "gateway_throttle_skips": gateway.get("throttle_skips"),
            "gateway_policy_skips": gateway.get("policy_skips"),
            "reported_input_tokens": gateway.get("input_tokens"),
            "reported_output_tokens": gateway.get("output_tokens"),
        }
    print(json.dumps({"stage": "research", **result}), flush=True)
    try:
        evaluation = await score(run_id, BENCHMARK)
        result["ragas"] = evaluation["scores"]
    except Exception as error:
        result["ragas_error_type"] = type(error).__name__
    result["experiment_wall_seconds"] = round(time.monotonic() - started, 1)
    print(json.dumps({"stage": "evaluated", **result}), flush=True)
    return result


async def main() -> None:
    redis = Redis.from_url(get_settings().redis_url)
    try:
        first_started = time.monotonic()
        await run_case(3, redis)
        # Avoid carrying the first run's minute-level provider reservation into
        # the second. External provider traffic can still differ between runs.
        await asyncio.sleep(max(0.0, 65 - (time.monotonic() - first_started)))
        await run_case(5, redis)
    finally:
        await redis.aclose()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
