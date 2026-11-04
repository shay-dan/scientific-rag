"""Audit an auto-generated silver dataset without provider calls."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def audit(dataset: Path, report: Path | None = None) -> dict:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    duplicate_ids = [key for key, count in Counter(item.get("id") for item in cases).items() if count > 1]
    source_ids = []
    mapping = []
    claim_scores = []
    invalid_mappings = []
    for case in cases:
        context_ids = {str(row.get("source_id")) for row in case.get("reference_context", [])}
        support_by_index = {item.get("claim_index"): item for item in case.get("claim_support", [])}
        for claim_index, claim in enumerate(case.get("reference_claims", []), start=1):
            claim_terms = {term for term in re.findall(r"[a-z0-9]{4,}", str(claim).lower()) if term not in {"that", "this", "with", "from", "which", "their", "these", "those"}}
            best = 0.0
            for row in case.get("reference_context", []):
                excerpt_terms = set(re.findall(r"[a-z0-9]{4,}", str(row.get("excerpt", "")).lower()))
                if claim_terms:
                    best = max(best, len(claim_terms & excerpt_terms) / len(claim_terms))
            support = support_by_index.get(claim_index)
            mapping.append(bool(support and str(support.get("context_id")) in {
                str(row.get("context_id")) for row in case.get("reference_context", [])
            }))
            claim_scores.append(best)
        case_source_ids = [row.get("source_id") for row in case.get("reference_context", [])]
        source_ids.append((case.get("id"), tuple(case_source_ids)))
        for support in case.get("claim_support", []):
            if support.get("source_id") not in context_ids:
                invalid_mappings.append({"case_id": case.get("id"), "source_id": support.get("source_id")})
    result = {
        "dataset": str(dataset),
        "case_count": len(cases),
        "approved_count": len(cases),
        "rejection_rate": None,
        "duplicate_case_ids": duplicate_ids,
        "duplicate_source_references": sum(len(ids) - len(set(ids)) for _, ids in source_ids),
        "claim_count": sum(len(item.get("reference_claims", [])) for item in cases),
        "claim_mapping_coverage": round(sum(mapping) / len(mapping), 4) if mapping else 0.0,
        "claim_excerpt_lexical_support_mean": round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else 0.0,
        "claims_with_lexical_support_at_least_0_20": sum(score >= 0.20 for score in claim_scores),
        "claims_with_lexical_support_total": len(claim_scores),
        "invalid_claim_source_mappings": invalid_mappings,
        "all_cases_have_context": all(bool(item.get("reference_context")) for item in cases),
        "status": "pass" if len(cases) == 12 and not duplicate_ids and not invalid_mappings and all(mapping) else "fail",
        "note": "Rejection rate is read from the generation report; this dataset contains approved cases only.",
    }
    if report and report.exists():
        generation = json.loads(report.read_text(encoding="utf-8"))
        attempted = len(generation.get("cases", []))
        result["latest_report_case_count"] = attempted
        if attempted >= len(cases):
            result["attempted_cases"] = attempted
            result["rejected_or_failed_cases"] = attempted - len(cases)
            result["rejection_rate"] = round((attempted - len(cases)) / attempted, 4) if attempted else None
        else:
            result["rejection_rate"] = None
            result["note"] += " Latest report is a single-case replacement run, so aggregate rejection rate is unavailable."
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/pilot-silver-v1.json"))
    parser.add_argument("--report", type=Path, default=Path("evaluation/pilot-silver-v1-report.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/pilot-silver-v1-audit.json"))
    args = parser.parse_args()
    result = audit(args.dataset, args.report)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
