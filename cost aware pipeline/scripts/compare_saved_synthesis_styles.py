"""Compare three answer styles on exact evidence saved by completed research runs.

No mining or retrieval is performed.  The original answer is compared with the
structured evidence contract and one bounded coverage-completion pass.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document
from sqlalchemy import select

from app.config import get_settings
from app.db import close_database, session_factory
from app.evaluation import EvaluationUsageTracker, diagnose_reference_claims, score_with_ragas
from app.models import ResearchEvidence, ResearchRun
from app.weak_model_synthesis import (
    CoverageCompletion,
    StructuredEvidenceAnswerV2,
    V2RepairBatch,
    build_evidence_blocks,
    coverage_prompt_v2,
    generation_prompt_v2,
    merge_coverage_v2,
    merge_v2_repairs,
    render_answer_v2,
    render_context,
    repair_prompt_v2,
    validate_answer_v2,
)
from src.scientific_rag.costing import calculate_cost, estimate_text_tokens
from src.scientific_rag.llm_gateway import ainvoke_structured, begin_trace, end_trace, summarize_trace


DEFAULT_RUN_IDS = (
    "7e04a182-8ba9-4899-add4-f90246fb2f43",  # ax-02
    "f7512275-5a4e-4883-acc1-b080e4a27b88",  # ax-04
)
PROVIDER = "openrouter"
MODEL = "qwen/qwen3.6-35b-a3b"


@contextmanager
def pinned_expert():
    keys = ("LLM_PROVIDER_ORDER_EXPERT", "LLM_PROVIDER_ORDER", "OPENROUTER_MODEL_EXPERT")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LLM_PROVIDER_ORDER_EXPERT"] = PROVIDER
        os.environ["LLM_PROVIDER_ORDER"] = PROVIDER
        os.environ["OPENROUTER_MODEL_EXPERT"] = MODEL
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def benchmark_case(benchmark: dict, question: str) -> dict:
    case = next((row for row in benchmark["cases"] if row["question"] == question), None)
    if case is None:
        raise ValueError(f"No benchmark case matches saved question: {question}")
    return case


def reference_context(case: dict) -> str | None:
    value = case.get("reference_context")
    if isinstance(value, list):
        return "\n\n".join(
            item.get("excerpt", "") if isinstance(item, dict) else str(item) for item in value
        )
    return value


async def load_saved(run_ids: list[UUID], benchmark: dict) -> list[dict]:
    loaded = []
    async with session_factory() as session:
        for run_id in run_ids:
            run = await session.get(ResearchRun, run_id)
            if run is None or run.status != "completed" or not run.answer:
                raise ValueError(f"Saved run {run_id} is missing or incomplete")
            evidence = list((await session.execute(
                select(ResearchEvidence)
                .where(ResearchEvidence.research_run_id == run.id)
                .order_by(ResearchEvidence.ordinal)
            )).scalars())
            if not evidence:
                raise ValueError(f"Saved run {run_id} has no persisted evidence")
            documents = []
            labels: dict[str, str] = {}
            for row in evidence:
                metadata = dict(row.metadata_json or {})
                identity = str(metadata.get("doc_id") or metadata.get("url") or metadata.get("title") or row.ordinal)
                metadata["citation_id"] = labels.setdefault(identity, f"S{len(labels) + 1}")
                documents.append(Document(page_content=row.content, metadata=metadata))
            loaded.append({
                "run": run,
                "case": benchmark_case(benchmark, run.question),
                "documents": documents,
            })
    return loaded


def preflight(items: list[dict]) -> dict:
    generation_input = 0
    for item in items:
        context = render_context(build_evidence_blocks(item["documents"]))
        # generation + possible local repair + coverage completion
        generation_input += estimate_text_tokens([item["run"].question, context]) * 3
    max_generation_calls = len(items) * 3
    generation_output = max_generation_calls * 4000
    generation_cost, rate_source = calculate_cost(
        PROVIDER, MODEL, generation_input, generation_output
    )
    answer_count = len(items) * 3
    return {
        "saved_runs": len(items),
        "answer_variants": answer_count,
        "new_research_calls": 0,
        "maximum_generation_calls": max_generation_calls,
        "generation_input_tokens_estimated": generation_input,
        "generation_output_tokens_budgeted": generation_output,
        "generation_cost_upper_bound_usd": generation_cost,
        "generation_rate_source": rate_source,
        "ragas_cost_reservation_usd": round(answer_count * 0.10, 2),
        "combined_conservative_upper_bound_usd": (
            round((generation_cost or 0) + answer_count * 0.10, 4)
        ),
    }


async def structured_variants(item: dict) -> list[dict]:
    run = item["run"]
    documents = item["documents"]
    blocks = build_evidence_blocks(documents)
    context = render_context(blocks)
    rows = [{
        "variant": "saved_freeform",
        "answer": run.answer,
        "generation": {"calls": 0, "cost_usd": 0.0, "saved_run_id": str(run.id)},
    }]
    trace_token = begin_trace()
    started = time.perf_counter()
    try:
        with pinned_expert():
            answer = await ainvoke_structured(
                "expert", 0, generation_prompt_v2(run.question, context), StructuredEvidenceAnswerV2
            )
            errors = validate_answer_v2(answer, blocks)
            repair_audit = None
            if errors:
                repairs = await ainvoke_structured(
                    "expert", 0, repair_prompt_v2(run.question, context, answer, errors), V2RepairBatch
                )
                answer, repair_audit, errors = merge_v2_repairs(answer, repairs, blocks)
            if errors:
                raise ValueError(f"Structured answer remains invalid: {errors}")
    finally:
        attempts = end_trace(trace_token)
    rows.append({
        "variant": "structured_v2",
        "answer": render_answer_v2(answer),
        "structured_answer": answer.model_dump(),
        "repair_audit": repair_audit,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "generation": summarize_trace(attempts),
    })

    trace_token = begin_trace()
    started = time.perf_counter()
    try:
        with pinned_expert():
            completion = await ainvoke_structured(
                "expert", 0, coverage_prompt_v2(run.question, context, answer), CoverageCompletion
            )
            covered, coverage_audit, errors = merge_coverage_v2(answer, completion, blocks)
            if errors:
                raise ValueError(f"Coverage additions are invalid: {errors}")
    finally:
        attempts = end_trace(trace_token)
    rows.append({
        "variant": "structured_v2_coverage",
        "answer": render_answer_v2(covered),
        "structured_answer": covered.model_dump(),
        "coverage_audit": coverage_audit,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "generation": summarize_trace(attempts),
    })
    return rows


async def evaluate(item: dict, row: dict) -> None:
    settings = get_settings()
    case = item["case"]
    contexts = [document.page_content for document in item["documents"]]
    tracker = EvaluationUsageTracker()
    row["scores"] = await score_with_ragas(
        question=item["run"].question,
        answer=row["answer"],
        retrieved_contexts=contexts,
        reference_answer=case["reference_answer"],
        reference_context=reference_context(case),
        api_key=settings.google_api_key,
        judge_model=settings.ragas_judge_model,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        usage_tracker=tracker,
    )
    claims = list(case.get("reference_claims") or [])
    if claims:
        row["claim_diagnostics"] = await diagnose_reference_claims(
            question=item["run"].question,
            reference_claims=claims,
            retrieved_contexts=contexts,
            api_key=settings.google_api_key,
            judge_model=settings.ragas_judge_model,
            usage_tracker=tracker,
        )
    row["judge_usage"] = tracker.summary(
        judge_model=settings.ragas_judge_model, embedding_model=settings.embedding_model
    )


async def run(args) -> int:
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    items = await load_saved([UUID(value) for value in args.run_ids.split(",")], benchmark)
    estimate = preflight(items)
    print(json.dumps({"preflight": estimate}, indent=2), flush=True)
    if args.estimate_only:
        return 0
    report = {
        "experiment": "saved-evidence-synthesis-style-comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": estimate,
        "results": [],
    }
    for item in items:
        try:
            variants = await structured_variants(item)
            for row in variants:
                row.update({
                    "run_id": str(item["run"].id),
                    "case_id": item["case"]["id"],
                    "question": item["run"].question,
                    "evidence_passages": len(item["documents"]),
                })
                if not args.skip_scoring:
                    await evaluate(item, row)
                report["results"].append(row)
                write_report(args.output, report)
        except Exception as error:
            report["results"].append({
                "run_id": str(item["run"].id),
                "case_id": item["case"]["id"],
                "question": item["run"].question,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
            write_report(args.output, report)
    write_report(args.output, report)
    return 0


async def async_main(args) -> int:
    try:
        return await run(args)
    finally:
        await close_database()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", default=",".join(DEFAULT_RUN_IDS))
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/pilot-silver-v3.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/saved-evidence-synthesis-comparison.json"))
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
