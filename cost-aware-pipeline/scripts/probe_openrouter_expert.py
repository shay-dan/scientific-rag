"""One read-only, billable OpenRouter Expert-size latency probe for a saved run.

Usage inside the API container: python -m scripts.probe_openrouter_expert RUN_ID
Only aggregate sizes, latency, and error type are printed; no answer is saved.
"""

import asyncio
import sys
import time
from uuid import UUID

from sqlalchemy import select

from app.db import session_factory
from app.models import ResearchEvidence, ResearchRun
from scientific_rag.llm_gateway import create_chat_model, provider_timeout_seconds


async def main(run_id: UUID) -> None:
    async with session_factory() as session:
        run = await session.get(ResearchRun, run_id)
        if run is None or run.status != "completed":
            raise ValueError("A completed research run is required")
        evidence = list((await session.execute(select(ResearchEvidence).where(ResearchEvidence.research_run_id == run_id).order_by(ResearchEvidence.ordinal))).scalars())
    if not evidence:
        raise ValueError("The selected run has no persisted Expert evidence")
    context = "\n\n".join(
        f"[{item.metadata_json.get('citation_id') or f'S{item.ordinal + 1}'}] "
        f"{item.metadata_json.get('title') or 'Source'}\n{item.content}"
        for item in evidence
    )
    prompt = (
        "Answer the question clearly using only the provided research excerpts. "
        "Cite each factual claim with its source label. Distinguish primary studies "
        "from indirect comparisons; include limitations and never invent numbers.\n\n"
        f"QUESTION: {run.question}\n\nSOURCES:\n{context}\n\nANSWER:"
    )
    timeout = provider_timeout_seconds("openrouter")
    started = time.perf_counter()
    try:
        model = create_chat_model("openrouter", "expert", 0.3)
        answer = await asyncio.wait_for(model.ainvoke(prompt), timeout=timeout)
    except Exception as error:
        print(f"provider=openrouter role=expert status=failed error_type={type(error).__name__} "
              f"latency_s={time.perf_counter() - started:.2f} context_chars={len(context)} timeout_s={timeout}")
        return
    usage = getattr(answer, "usage_metadata", None) or {}
    print(f"provider=openrouter role=expert status=ok latency_s={time.perf_counter() - started:.2f} "
          f"context_chars={len(context)} answer_chars={len(str(answer.content))} "
          f"input_tokens={usage.get('input_tokens')} output_tokens={usage.get('output_tokens')} timeout_s={timeout}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.probe_openrouter_expert RUN_ID")
    asyncio.run(main(UUID(sys.argv[1])))
