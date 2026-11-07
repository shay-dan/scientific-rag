"""Provider-free preparation and validation for the 12-case silver pilot."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


_STOP_WORDS = {
    "about", "after", "also", "among", "being", "between", "from", "have",
    "into", "more", "most", "that", "their", "these", "this", "those",
    "through", "using", "were", "which", "with",
}


def _terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", value.lower())
        if len(token) >= 4 and token not in _STOP_WORDS
    }


def lexical_claim_support(claim: str, excerpt: str) -> float:
    """Return a transparent candidate-mapping score, not a semantic verdict."""
    claim_terms = _terms(claim)
    if not claim_terms:
        return 0.0
    return round(len(claim_terms & _terms(excerpt)) / len(claim_terms), 4)


def add_explicit_claim_support(payload: dict[str, Any]) -> dict[str, Any]:
    """Add stable context IDs and one explicit evidence mapping per claim.

    The original generator independently validated each complete Q&A but failed
    to persist its claim-to-source mapping. This deterministic repair makes the
    mapping explicit while labelling it honestly as lexical candidate mapping.
    """
    repaired = deepcopy(payload)
    for case in repaired.get("cases", []):
        case_id = str(case.get("id") or case.get("case_id") or "case")
        contexts = case.get("reference_context", [])
        for index, row in enumerate(contexts, start=1):
            row["context_id"] = row.get("context_id") or f"{case_id}-ctx-{index:02d}"
        mappings = []
        for claim_index, claim in enumerate(case.get("reference_claims", []), start=1):
            scored = [
                (lexical_claim_support(str(claim), str(row.get("excerpt", ""))), row)
                for row in contexts
            ]
            score, best = max(scored, key=lambda item: item[0], default=(0.0, {}))
            mappings.append(
                {
                    "claim_index": claim_index,
                    "claim": claim,
                    "context_id": best.get("context_id"),
                    "source_id": best.get("source_id"),
                    "lexical_support": score,
                    "mapping_method": "deterministic_lexical_candidate",
                }
            )
        case["claim_support"] = mappings
        case["claim_support_status"] = "explicit_candidate_mapping"
    repaired["claim_support_method"] = "deterministic_lexical_candidate"
    repaired["claim_support_limitations"] = (
        "Mappings restore the omitted claim-to-context links and are backed by the existing "
        "independent whole-answer validator, but lexical overlap is not human or semantic verification."
    )
    return repaired


def validate_pilot_dataset(payload: dict[str, Any], *, expected_cases: int = 12) -> list[str]:
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return ["dataset must contain a cases list"]
    if len(cases) != expected_cases:
        errors.append(f"expected exactly {expected_cases} cases, found {len(cases)}")
    seen_ids: set[str] = set()
    for position, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or "")
        if not case_id:
            errors.append(f"case {position} has no id")
        elif case_id in seen_ids:
            errors.append(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        if case.get("source_group") not in {"pubmed", "arxiv", "mixed"}:
            errors.append(f"case {case_id or position} has invalid source_group")
        if not str(case.get("question", "")).strip():
            errors.append(f"case {case_id or position} has no question")
        if not str(case.get("reference_answer", "")).strip():
            errors.append(f"case {case_id or position} has no reference_answer")
        contexts = case.get("reference_context") or []
        context_ids = {str(row.get("context_id")) for row in contexts if row.get("context_id")}
        if not contexts:
            errors.append(f"case {case_id or position} has no reference_context")
        for row in contexts:
            if not row.get("source_id") or not row.get("excerpt") or not row.get("url"):
                errors.append(f"case {case_id or position} has incomplete context")
        claims = case.get("reference_claims") or []
        mappings = case.get("claim_support") or []
        semantic_audit = case.get("claim_support_status") == "independently_semantically_validated"
        if len(mappings) != len(claims):
            errors.append(f"case {case_id or position} does not map every claim exactly once")
        mapped_indexes = {item.get("claim_index") for item in mappings}
        if mapped_indexes != set(range(1, len(claims) + 1)):
            errors.append(f"case {case_id or position} has invalid claim indexes")
        for mapping in mappings:
            if str(mapping.get("context_id")) not in context_ids:
                errors.append(f"case {case_id or position} maps a claim to missing context")
            if semantic_audit:
                if mapping.get("decision") != "supported" or float(mapping.get("confidence") or 0.0) < 0.75:
                    errors.append(f"case {case_id or position} has an unverified semantic claim mapping")
            elif float(mapping.get("lexical_support") or 0.0) < 0.20:
                errors.append(f"case {case_id or position} has weak claim mapping below 0.20")
        score = case.get("validation_score")
        semantic_case_accepted = semantic_audit and case.get("eligible_for_benchmark") is True
        if not semantic_case_accepted and (score is None or float(score) < 0.80):
            errors.append(f"case {case_id or position} lacks independent validation >= 0.80")
    return errors
