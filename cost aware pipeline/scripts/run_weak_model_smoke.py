"""Compare weak-model free-form, structured, and one-repair synthesis."""

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

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.weak_model_synthesis import (
    ClaimRepairBatch,
    StructuredEvidenceAnswer,
    build_evidence_blocks,
    claim_validation_errors,
    generation_prompt,
    merge_local_repairs,
    repair_prompt,
    render_answer,
    render_context,
    validate_answer,
)
from scripts.run_expert_provider_comparison import FixedRetriever, prepare
from app.real_comparison import FrozenResearchAgent
from scientific_rag.costing import calculate_cost, estimate_text_tokens
from scientific_rag.llm_gateway import (
    InvalidStructuredResponse,
    ainvoke_structured,
    begin_trace,
    end_trace,
    summarize_trace,
)


VARIANTS = ("freeform", "structured", "structured_repair")
DEFAULT_CASES = ("pm-01", "ax-01", "mx-01")
PROVIDER = "groq"
MODEL = "openai/gpt-oss-120b"


@contextmanager
def pinned_weak_model():
    keys = ("LLM_PROVIDER_ORDER_EXPERT", "LLM_PROVIDER_ORDER", "GROQ_MODEL_EXPERT")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LLM_PROVIDER_ORDER_EXPERT"] = PROVIDER
        os.environ["LLM_PROVIDER_ORDER"] = PROVIDER
        os.environ["GROQ_MODEL_EXPERT"] = MODEL
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def preflight(prepared: list[dict], seed_results: Path | None = None) -> dict:
    """Conservative maximum: free-form + structured + one repair per case."""
    input_tokens = 0
    maximum_calls = 0
    if seed_results:
        source = json.loads(seed_results.read_text(encoding="utf-8"))
        structured = {row["case_id"]: row for row in source.get("results", []) if row.get("variant") == "structured"}
        for item in prepared:
            row = structured.get(item["case"]["id"])
            if not row or not row.get("structured_answer"):
                continue
            blocks = build_evidence_blocks(item["documents"][:5])
            answer = StructuredEvidenceAnswer.model_validate(row["structured_answer"])
            invalid = claim_validation_errors(answer, blocks)
            if invalid:
                prompt = repair_prompt(item["case"]["question"], render_context(blocks), answer, invalid)
                input_tokens += estimate_text_tokens([prompt, "x" * 1500])
                maximum_calls += 1
    else:
        for item in prepared:
            blocks = build_evidence_blocks(item["documents"][:5])
            context = render_context(blocks)
            base = estimate_text_tokens([item["case"]["question"], context, "x" * 5000])
            input_tokens += base * 3
        maximum_calls = len(prepared) * 3
    # GPT-OSS usage includes reasoning tokens. A previous 1,400-token allowance
    # under-estimated a valid 2,756-token structured response, so budget 4,000
    # output tokens for every possible call.
    output_tokens = maximum_calls * 4000
    cost, source = calculate_cost(PROVIDER, MODEL, input_tokens, output_tokens)
    return {
        "case_count": len(prepared),
        "variants": list(VARIANTS),
        "maximum_calls": maximum_calls,
        "estimated_input_tokens": input_tokens,
        "budgeted_output_tokens": output_tokens,
        "maximum_estimated_cost_usd": cost,
        "rate_source": source,
        "provider": PROVIDER,
        "model": MODEL,
        "fallback_disabled": True,
    }


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _freeform(item: dict) -> dict:
    agent = FrozenResearchAgent(
        FixedRetriever(item["documents"][:5]),
        benchmark_subquestions=[item["case"]["question"]],
        max_synthesis_subquestions=1,
        retrieval_documents=5,
        max_context_documents=5,
    )
    report = await agent.run_research(item["case"]["question"])
    return {
        "answer": report["answer"],
        "contract_valid": bool(report["answer"]),
        "validation_errors": [],
        "repair_attempted": False,
        "documents": report["raw_docs_used"],
    }


async def _structured(item: dict) -> dict:
    documents = item["documents"][:5]
    blocks = build_evidence_blocks(documents)
    context = render_context(blocks)
    errors: list[str] = []
    try:
        answer = await ainvoke_structured(
            "expert", 0, generation_prompt(item["case"]["question"], context), StructuredEvidenceAnswer
        )
        validation = validate_answer(answer, blocks)
        errors = list(validation.errors)
    except InvalidStructuredResponse as error:
        answer = None
        validation = None
        errors = [f"{type(error).__name__}: {error}"]
    valid = bool(answer is not None and validation is not None and validation.valid)
    return {
        "answer": render_answer(answer) if valid else None,
        "structured_answer": answer.model_dump() if answer is not None else None,
        "contract_valid": valid,
        "validation_errors": errors,
        "repair_attempted": False,
        "documents": documents,
    }


async def _repair_saved_structured(item: dict, initial: dict) -> dict:
    """Repair the exact structured candidate; never replace it with a fresh first attempt."""
    documents = item["documents"][:5]
    blocks = build_evidence_blocks(documents)
    context = render_context(blocks)
    raw = initial.get("structured_answer")
    if raw is None:
        raise InvalidStructuredResponse("The structured arm produced no parseable candidate to repair")
    answer = StructuredEvidenceAnswer.model_validate(raw)
    initial_validation = validate_answer(answer, blocks)
    if initial_validation.valid:
        return {
            "answer": render_answer(answer),
            "structured_answer": answer.model_dump(),
            "contract_valid": True,
            "validation_errors": [],
            "initial_validation_errors": [],
            "repair_attempted": False,
            "derived_from_structured": True,
            "documents": documents,
        }
    invalid = claim_validation_errors(answer, blocks)
    repair_batch = await ainvoke_structured(
        "expert", 0,
        repair_prompt(item["case"]["question"], context, answer, invalid),
        ClaimRepairBatch,
    )
    repaired, audit, repaired_validation = merge_local_repairs(answer, repair_batch, blocks)
    return {
        "answer": render_answer(repaired) if repaired_validation.valid else None,
        "structured_answer": repaired.model_dump(),
        "contract_valid": repaired_validation.valid,
        "validation_errors": list(repaired_validation.errors),
        "initial_validation_errors": list(initial_validation.errors),
        "repair_attempted": True,
        "repair_mode": "local_claim_repair",
        "repair_audit": {
            "claims_before": audit.claims_before,
            "claims_after": audit.claims_after,
            "preserved_claim_indices": list(audit.preserved_claim_indices),
            "repaired_claim_indices": list(audit.repaired_claim_indices),
            "dropped_claim_indices": list(audit.dropped_claim_indices),
            "unresolved_reasons": list(audit.unresolved_reasons),
        },
        "derived_from_structured": True,
        "documents": documents,
    }


async def execute(item: dict, variant: str, *, initial_structured: dict | None = None) -> dict:
    started = time.perf_counter()
    token = begin_trace()
    closed = False
    try:
        with pinned_weak_model():
            if variant == "freeform":
                outcome = await _freeform(item)
            elif variant == "structured_repair":
                if initial_structured is None:
                    raise RuntimeError("structured_repair requires the exact structured-arm result")
                outcome = await _repair_saved_structured(item, initial_structured)
            else:
                outcome = await _structured(item)
        attempts = end_trace(token)
        closed = True
        documents = outcome.pop("documents")
        status = "completed" if outcome["contract_valid"] else "failed_validation"
        return {
            "status": status,
            "case_id": item["case"]["id"],
            "source_group": item["case"]["source_group"],
            "question": item["case"]["question"],
            "variant": variant,
            "provider": PROVIDER,
            "model": MODEL,
            "fallback_disabled": True,
            **outcome,
            "selected_source_ids": [str(doc.metadata["doc_id"]) for doc in documents],
            "selected_context_ids": [doc.metadata["benchmark_context_id"] for doc in documents],
            "context_count": len(documents),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm": summarize_trace(attempts),
        }
    except InvalidStructuredResponse as error:
        attempts = end_trace(token)
        closed = True
        return {
            "status": "failed_validation", "case_id": item["case"]["id"],
            "source_group": item["case"]["source_group"], "question": item["case"]["question"],
            "variant": variant, "provider": PROVIDER, "model": MODEL,
            "fallback_disabled": True, "error": f"{type(error).__name__}: {error}",
            "repair_attempted": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm": summarize_trace(attempts),
        }
    except Exception as error:
        attempts = end_trace(token)
        closed = True
        return {
            "status": "failed", "case_id": item["case"]["id"],
            "source_group": item["case"]["source_group"], "question": item["case"]["question"],
            "variant": variant, "provider": PROVIDER, "model": MODEL,
            "fallback_disabled": True, "error": f"{type(error).__name__}: {error}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm": summarize_trace(attempts),
        }
    finally:
        if not closed:
            end_trace(token)


async def run(args) -> int:
    case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    prepared = prepare(args.dataset, args.comparison, case_ids)
    estimate = preflight(prepared, args.seed_results)
    print(json.dumps({"preflight": estimate}, indent=2), flush=True)
    if args.estimate_only:
        return 0
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        seeded = []
        if args.seed_results:
            source = json.loads(args.seed_results.read_text(encoding="utf-8"))
            seeded = [row for row in source.get("results", []) if row.get("variant") in {"freeform", "structured"}]
        report = {
        "experiment": "weak-model-contract-smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": case_ids,
        "variants": variants,
        "preflight": estimate,
        "results": seeded,
        }
    report["preflight"] = estimate
    existing = {(row["case_id"], row["variant"]): row for row in report["results"]}
    execution_count = 0
    for item in prepared:
        for variant in variants:
            key = (item["case"]["id"], variant)
            previous = existing.get(key)
            if previous is not None:
                serialized = json.dumps(previous, ensure_ascii=False).lower()
                provider_failure = any(marker in serialized for marker in ("429", "rate limit", "timeout", "connection"))
                if previous.get("status") == "completed" or not provider_failure:
                    continue
            if execution_count and args.pace_seconds > 0:
                print(f"pacing {args.pace_seconds:.1f}s before next provider call", flush=True)
                await asyncio.sleep(args.pace_seconds)
            print(f"running {item['case']['id']} {variant}", flush=True)
            initial_structured = existing.get((item["case"]["id"], "structured"))
            result = await execute(item, variant, initial_structured=initial_structured)
            report["results"] = [row for row in report["results"] if (row["case_id"], row["variant"]) != key]
            report["results"].append(result)
            existing[key] = result
            execution_count += 1
            _write(args.output, report)
    report["summary"] = {
        "planned": len(prepared) * len(variants),
        "completed": sum(row["status"] == "completed" for row in report["results"]),
        "failed_validation": sum(row["status"] == "failed_validation" for row in report["results"]),
        "failed": sum(row["status"] == "failed" for row in report["results"]),
        "repair_attempts": sum(bool(row.get("repair_attempted")) for row in report["results"]),
    }
    _write(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v3.json")
    parser.add_argument("--comparison", type=Path, default=ROOT / "evaluation" / "real-frozen-v3-12x4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "weak-model-smoke.json")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pace-seconds", type=float, default=35.0)
    parser.add_argument("--seed-results", type=Path, help="Reuse freeform and structured arms from an audited prior run")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
    claim_validation_errors,
    merge_local_repairs,
