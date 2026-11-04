"""Three-case cited-limitations and bounded coverage-completion experiment."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.weak_model_synthesis import (
    CoverageCompletion, StructuredEvidenceAnswerV2, V2RepairBatch,
    build_evidence_blocks, coverage_prompt_v2, generation_prompt_v2,
    merge_coverage_v2, merge_v2_repairs, repair_prompt_v2,
    render_answer_v2, render_context, validate_answer_v2,
)
from scripts.run_expert_provider_comparison import prepare
from scientific_rag.costing import calculate_cost, estimate_text_tokens
from scientific_rag.llm_gateway import ainvoke_structured, begin_trace, end_trace, summarize_trace

PROVIDER = "groq"
MODEL = "openai/gpt-oss-120b"
DEFAULT_CASES = ("pm-01", "ax-01", "mx-01")


@contextmanager
def pinned_model():
    keys = ("LLM_PROVIDER_ORDER_EXPERT", "LLM_PROVIDER_ORDER", "GROQ_MODEL_EXPERT")
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LLM_PROVIDER_ORDER_EXPERT"] = PROVIDER
        os.environ["LLM_PROVIDER_ORDER"] = PROVIDER
        os.environ["GROQ_MODEL_EXPERT"] = MODEL
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class Pacer:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.last_call = 0.0

    async def call(self, prompt: str, schema: type):
        wait = self.seconds - (time.monotonic() - self.last_call)
        if wait > 0:
            print(f"pacing {wait:.1f}s", flush=True)
            await asyncio.sleep(wait)
        try:
            return await ainvoke_structured("expert", 0, prompt, schema)
        finally:
            self.last_call = time.monotonic()


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(prepared: list[dict]) -> dict:
    input_tokens = 0
    for item in prepared:
        context = render_context(build_evidence_blocks(item["documents"][:5]))
        base = estimate_text_tokens([item["case"]["question"], context, "x" * 5000])
        input_tokens += base * 3
    maximum_calls = len(prepared) * 3
    output_tokens = maximum_calls * 4000
    cost, source = calculate_cost(PROVIDER, MODEL, input_tokens, output_tokens)
    return {
        "case_count": len(prepared), "maximum_calls": maximum_calls,
        "estimated_input_tokens": input_tokens, "budgeted_output_tokens": output_tokens,
        "maximum_estimated_cost_usd": cost, "rate_source": source,
        "provider": PROVIDER, "model": MODEL, "fallback_disabled": True,
    }


def result_row(item: dict, variant: str, answer: StructuredEvidenceAnswerV2, documents: list,
               started: float, attempts: list, **extra) -> dict:
    return {
        "status": "completed", "case_id": item["case"]["id"],
        "source_group": item["case"]["source_group"], "question": item["case"]["question"],
        "variant": variant, "provider": PROVIDER, "model": MODEL, "fallback_disabled": True,
        "answer": render_answer_v2(answer), "structured_answer_v2": answer.model_dump(),
        "selected_source_ids": [str(doc.metadata["doc_id"]) for doc in documents],
        "selected_context_ids": [doc.metadata["benchmark_context_id"] for doc in documents],
        "context_count": len(documents), "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "llm": summarize_trace(attempts), **extra,
    }


async def generate_structured(item: dict, pacer: Pacer) -> dict:
    documents = item["documents"][:5]
    blocks = build_evidence_blocks(documents)
    context = render_context(blocks)
    started = time.perf_counter()
    token = begin_trace()
    try:
        with pinned_model():
            answer = await pacer.call(
                generation_prompt_v2(item["case"]["question"], context), StructuredEvidenceAnswerV2
            )
            errors = validate_answer_v2(answer, blocks)
            repair_audit = None
            repair_attempted = False
            if errors:
                repair_attempted = True
                batch = await pacer.call(
                    repair_prompt_v2(item["case"]["question"], context, answer, errors), V2RepairBatch
                )
                answer, repair_audit, errors = merge_v2_repairs(answer, batch, blocks)
            if errors:
                raise ValueError(f"V2 answer remains invalid after repair: {errors}")
    finally:
        attempts = end_trace(token)
    return result_row(item, "structured_v2", answer, documents, started, attempts,
                      repair_attempted=repair_attempted, repair_audit=repair_audit)


async def complete_coverage(item: dict, structured: dict, pacer: Pacer) -> dict:
    documents = item["documents"][:5]
    blocks = build_evidence_blocks(documents)
    context = render_context(blocks)
    original = StructuredEvidenceAnswerV2.model_validate(structured["structured_answer_v2"])
    started = time.perf_counter()
    token = begin_trace()
    try:
        with pinned_model():
            completion = await pacer.call(
                coverage_prompt_v2(item["case"]["question"], context, original), CoverageCompletion
            )
            merged, audit, errors = merge_coverage_v2(original, completion, blocks)
            if errors:
                raise ValueError(f"Coverage additions failed validation: {errors}")
    finally:
        attempts = end_trace(token)
    return result_row(item, "structured_v2_coverage", merged, documents, started, attempts,
                      coverage_audit=audit, derived_from_variant="structured_v2")


async def run(args) -> int:
    case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    prepared = prepare(args.dataset, args.comparison, case_ids)
    estimate = preflight(prepared)
    print(json.dumps({"preflight": estimate}, indent=2), flush=True)
    if args.estimate_only:
        return 0
    report = {
        "experiment": "weak-model-v2-cited-limitations-coverage-smoke",
        "created_at": datetime.now(timezone.utc).isoformat(), "case_ids": case_ids,
        "preflight": estimate, "results": [],
    }
    pacer = Pacer(args.pace_seconds)
    for item in prepared:
        print(f"running {item['case']['id']} structured_v2", flush=True)
        try:
            structured = await generate_structured(item, pacer)
            report["results"].append(structured)
            _write(args.output, report)
            print(f"running {item['case']['id']} structured_v2_coverage", flush=True)
            report["results"].append(await complete_coverage(item, structured, pacer))
        except Exception as error:
            report["results"].append({
                "status": "failed", "case_id": item["case"]["id"], "question": item["case"]["question"],
                "variant": "structured_v2_pipeline", "error": f"{type(error).__name__}: {error}",
            })
        _write(args.output, report)
    report["summary"] = {
        "completed": sum(row.get("status") == "completed" for row in report["results"]),
        "failed": sum(row.get("status") == "failed" for row in report["results"]),
    }
    _write(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v3.json")
    parser.add_argument("--comparison", type=Path, default=ROOT / "evaluation" / "real-frozen-v3-12x4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "weak-model-v2-smoke.json")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASES))
    parser.add_argument("--pace-seconds", type=float, default=35.0)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
