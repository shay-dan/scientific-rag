"""Regenerate only rejected Silver cases from already verified source abstracts."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.silver_dataset_validation import SourceDocument, resolve_source_id  # noqa: E402
from scientific_rag.costing import calculate_cost, estimate_text_tokens  # noqa: E402
from scientific_rag.llm_gateway import (  # noqa: E402
    ainvoke_structured,
    begin_trace,
    end_trace,
    model_for,
    summarize_trace,
)


REPAIR_IDS = ("pm-02", "pm-03", "mx-04")


class RepairSupport(BaseModel):
    claim_index: int = Field(ge=1)
    source_id: str


class RepairCandidate(BaseModel):
    question: str = Field(min_length=20, max_length=300)
    reference_answer: str = Field(min_length=100, max_length=2500)
    reference_claims: list[str] = Field(min_length=3, max_length=5)
    claim_support: list[RepairSupport] = Field(min_length=3, max_length=5)


def load_source_cache(path: Path) -> dict[str, SourceDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {source_id: SourceDocument(**item) for source_id, item in raw.items()}


def build_repair_prompt(case: dict[str, Any], documents: dict[str, SourceDocument]) -> str:
    source_ids = sorted({item["source_id"] for item in case.get("reference_context", []) if item["source_id"] in documents})
    sources = "\n\n".join(
        f"SOURCE {source_id}\nTITLE: {documents[source_id].title}\nABSTRACT:\n{documents[source_id].abstract}"
        for source_id in source_ids if documents[source_id].abstract.strip()
    )
    return f"""Create a conservative Silver benchmark case using ONLY the original-source abstracts below.

The old case was rejected because some claims exceeded the evidence. Repair it; do not defend the old wording.
- You may narrow or rewrite the question so every part is answerable from the supplied abstracts.
- Return 3-5 atomic claims. Every claim must be directly entailed by at least one abstract.
- Preserve qualifiers and distinguish reviews from primary experimental evidence.
- Do not add common knowledge, clinical details, mechanisms, numbers, or limitations absent from the abstracts.
- `claim_support` must contain exactly one entry for every claim_index.
- Copy source_id exactly, including the PMID: prefix.
- The answer must contain only the returned supported claims, written readably.

OLD QUESTION:
{case['question']}

VERIFIED ORIGINAL SOURCES:
{sources}
"""


@contextmanager
def groq_only():
    previous = os.environ.get("LLM_PROVIDER_ORDER")
    os.environ["LLM_PROVIDER_ORDER"] = "groq"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LLM_PROVIDER_ORDER", None)
        else:
            os.environ["LLM_PROVIDER_ORDER"] = previous


def validate_candidate(candidate: RepairCandidate, documents: dict[str, SourceDocument]) -> list[str]:
    errors: list[str] = []
    expected = set(range(1, len(candidate.reference_claims) + 1))
    returned = {item.claim_index for item in candidate.claim_support}
    if returned != expected:
        errors.append("claim_support indexes do not match reference_claims")
    for item in candidate.claim_support:
        if resolve_source_id(item.source_id, documents) is None:
            errors.append(f"unknown source id: {item.source_id}")
    return errors


def estimate(prompts: list[str]) -> dict[str, Any]:
    model = model_for("groq", "qa_generator")
    input_tokens = estimate_text_tokens(prompts)
    output_tokens = len(prompts) * 1600
    cost, rate_source = calculate_cost("groq", model, input_tokens, output_tokens)
    return {
        "provider": "groq",
        "model": model,
        "calls": len(prompts),
        "estimated_input_tokens": input_tokens,
        "budgeted_output_tokens": output_tokens,
        "estimated_usd": cost,
        "rate_source": rate_source,
    }


async def run(input_path: Path, cache_path: Path, output_path: Path, audit_path: Path, *, estimate_only: bool) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    documents = load_source_cache(cache_path)
    selected = [case for case in payload.get("cases", []) if case.get("id") in REPAIR_IDS]
    prompts = [build_repair_prompt(case, documents) for case in selected]
    preflight = estimate(prompts)
    print(json.dumps({"repair_preflight": preflight}, indent=2))
    if preflight["estimated_usd"] is None:
        raise RuntimeError("No trusted model rate is configured for repair generation")
    if preflight["estimated_usd"] > float(os.getenv("SILVER_REPAIR_MAX_USD", "0.05")):
        raise RuntimeError("Repair generation exceeds SILVER_REPAIR_MAX_USD")
    if estimate_only:
        return

    replacements: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    trace_token = begin_trace()
    try:
        with groq_only():
            for position, (case, prompt) in enumerate(zip(selected, prompts), 1):
                print(f"Repairing case {position}/{len(selected)}: {case['id']}")
                candidate = await ainvoke_structured("qa_generator", 0.0, prompt, RepairCandidate)
                errors = validate_candidate(candidate, documents)
                if errors:
                    diagnostics.append({"case_id": case["id"], "status": "rejected", "errors": errors})
                    continue
                contexts = []
                support = []
                for item in candidate.claim_support:
                    source_id = resolve_source_id(item.source_id, documents)
                    assert source_id is not None
                    document = documents[source_id]
                    context_id = f"{case['id']}-repair-ctx-{item.claim_index:02d}"
                    contexts.append({
                        "context_id": context_id,
                        "source_id": source_id,
                        "title": document.title,
                        "url": document.url,
                        "excerpt": document.abstract,
                    })
                    support.append({
                        "claim_index": item.claim_index,
                        "claim": candidate.reference_claims[item.claim_index - 1],
                        "context_id": context_id,
                        "source_id": source_id,
                        "mapping_method": "repair_generator_explicit_mapping",
                    })
                replacements[case["id"]] = {
                    **case,
                    "question": candidate.question,
                    "reference_answer": candidate.reference_answer,
                    "reference_claims": candidate.reference_claims,
                    "reference_context": contexts,
                    "claim_support": support,
                    "claim_support_status": "repair_candidate_pending_independent_audit",
                    "generator": f"groq:{model_for('groq', 'qa_generator')}",
                    "validator": "pending_stage3_independent_audit",
                    "validation_score": None,
                    "provenance": "silver_auto_repair_candidate",
                    "human_review": False,
                }
                diagnostics.append({"case_id": case["id"], "status": "generated", "claims": len(candidate.reference_claims)})
    finally:
        trace = end_trace(trace_token)

    if len(replacements) != len(REPAIR_IDS):
        raise RuntimeError(f"Generated {len(replacements)}/{len(REPAIR_IDS)} repair candidates")
    cases = [replacements.get(case["id"], case) for case in payload.get("cases", [])]
    repaired = {
        **payload,
        "dataset_name": "pilot-silver-v1-repair-candidates",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    actual = summarize_trace(trace)
    output_path.write_text(json.dumps(repaired, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps({"preflight": preflight, "actual": actual, "cases": diagnostics}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "candidates": len(replacements), "actual_cost_usd": actual.get("estimated_cost_usd")}, indent=2))


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "silver-source-cache.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1-repair-candidates.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1-repair-audit.json")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.input, args.cache, args.output, args.audit, estimate_only=args.estimate_only))


if __name__ == "__main__":
    main()
