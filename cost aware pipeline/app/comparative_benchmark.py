"""Specification and cost planning for the RAG complexity comparison.

This module deliberately keeps benchmark planning separate from the paid job
queue. It makes it possible to validate a dataset and estimate exposure before
calling any provider.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any

from scientific_rag.costing import calculate_cost, estimate_text_tokens


class RAGVariant(str, Enum):
    NAIVE = "naive"
    HYBRID_BASIC = "hybrid_basic"
    HYBRID_ENHANCED = "hybrid_enhanced"
    AGENTIC_FAST = "agentic_fast"


@dataclass(frozen=True)
class VariantCallBudget:
    """Conservative per-question call assumptions used before execution."""

    generation_calls: int
    embedding_calls: int
    judge_calls: int


VARIANT_BUDGETS: dict[RAGVariant, VariantCallBudget] = {
    RAGVariant.NAIVE: VariantCallBudget(1, 1, 1),
    RAGVariant.HYBRID_BASIC: VariantCallBudget(1, 1, 1),
    RAGVariant.HYBRID_ENHANCED: VariantCallBudget(1, 1, 1),
    # Fast currently reserves up to five generation calls (planner,
    # gatekeeper/repair and synthesis/fallback paths). The actual count is
    # recorded after execution; this is intentionally an upper bound.
    RAGVariant.AGENTIC_FAST: VariantCallBudget(5, 2, 1),
}


REQUIRED_CASE_FIELDS = {"case_id", "cohort", "question", "reference_answer", "reference_context"}
VALID_COHORTS = {"pubmed", "arxiv", "mixed"}


def validate_comparative_dataset(payload: dict[str, Any], *, required_cases: int | None = 100) -> list[str]:
    """Return validation errors without making a provider or database call."""
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return ["dataset must contain a cases list"]
    if required_cases is not None and len(cases) != required_cases:
        errors.append(f"expected exactly {required_cases} cases, found {len(cases)}")
    ids: set[str] = set()
    cohorts: dict[str, int] = {cohort: 0 for cohort in VALID_COHORTS}
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            errors.append(f"case {index} is not an object")
            continue
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            errors.append(f"case {index} missing fields: {sorted(missing)}")
        case_id = str(case.get("case_id", ""))
        if not case_id:
            errors.append(f"case {index} has an empty case_id")
        elif case_id in ids:
            errors.append(f"duplicate case_id: {case_id}")
        ids.add(case_id)
        cohort = case.get("cohort")
        if cohort not in VALID_COHORTS:
            errors.append(f"case {case_id or index} has invalid cohort: {cohort}")
        else:
            cohorts[cohort] += 1
        for field in ("question", "reference_answer", "reference_context"):
            if not str(case.get(field, "")).strip():
                errors.append(f"case {case_id or index} has empty {field}")
        if not case.get("reference_claims"):
            errors.append(f"case {case_id or index} needs reference_claims for claim-level scoring")
        urls = re.findall(r"https?://\S+", str(case.get("reference_context", "")))
        if not urls:
            errors.append(f"case {case_id or index} has no URL in reference_context")
    return errors


def estimate_comparative_plan(
    *,
    case_count: int,
    repetitions: int = 3,
    judge: bool = True,
    input_tokens_per_generation: int = 2500,
    output_tokens_per_generation: int = 700,
    input_tokens_per_embedding: int = 500,
) -> dict[str, Any]:
    """Build a transparent upper-bound estimate for the full comparison."""
    if case_count < 1 or repetitions < 1:
        raise ValueError("case_count and repetitions must be positive")
    variants: dict[str, Any] = {}
    total_cost = 0.0
    unknown_cost = False
    for variant, budget in VARIANT_BUDGETS.items():
        questions = case_count * repetitions
        generation_tokens_in = questions * budget.generation_calls * input_tokens_per_generation
        generation_tokens_out = questions * budget.generation_calls * output_tokens_per_generation
        embedding_tokens = questions * budget.embedding_calls * input_tokens_per_embedding
        generation_cost, generation_rate = calculate_cost("groq", "openai/gpt-oss-120b", generation_tokens_in, generation_tokens_out)
        embedding_cost, embedding_rate = calculate_cost("gemini", "gemini-embedding-2", embedding_tokens)
        judge_cost, judge_rate = calculate_cost("gemini", "gemini-3.5-flash-lite", questions * (input_tokens_per_generation // 2), questions * (output_tokens_per_generation // 2)) if judge else (0.0, "disabled")
        components = [value for value in (generation_cost, embedding_cost, judge_cost) if value is not None]
        if len(components) != 3:
            unknown_cost = True
        estimated = sum(components)
        total_cost += estimated
        call_budget = asdict(budget)
        if not judge:
            call_budget["judge_calls"] = 0
        variants[variant.value] = {
            "questions": questions,
            "upper_bound_calls": call_budget,
            "estimated_tokens": {
                "generation_input": generation_tokens_in,
                "generation_output": generation_tokens_out,
                "embedding_input": embedding_tokens,
            },
            "estimated_cost_usd": round(estimated, 6) if not unknown_cost else None,
            "rates": {"generation": generation_rate, "embedding": embedding_rate, "judge": judge_rate},
        }
    return {
        "case_count": case_count,
        "repetitions": repetitions,
        "total_variant_runs": case_count * repetitions * len(VARIANT_BUDGETS),
        "judge_enabled": judge,
        "variants": variants,
        "estimated_total_cost_usd": round(total_cost, 6) if not unknown_cost else None,
        "estimate_type": "conservative_upper_bound",
        "note": "Actual provider usage may be lower. This estimate excludes external mining and assumes the configured Groq model for generation.",
    }


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float | None:
    """Compute source recall for cases that provide verified source ids."""
    expected = {str(item) for item in expected_ids if str(item).strip()}
    if not expected:
        return None
    retrieved = {str(item) for item in retrieved_ids[:k]}
    return round(len(retrieved & expected) / len(expected), 4)

