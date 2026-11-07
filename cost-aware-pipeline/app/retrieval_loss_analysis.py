"""Provider-free diagnosis of evidence loss between retrieval variants."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _unique(values: list[Any]) -> set[str]:
    return {str(value) for value in values if value}


def _expected_sources(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for context in case.get("reference_context", []):
        source_id = str(context.get("source_id") or "")
        if source_id:
            sources.setdefault(source_id, context)
    return sources


def _result_index(comparison: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["case_id"]), str(row["variant"])): row
        for row in comparison.get("results", [])
        if row.get("status") == "completed"
    }


def _trace(result: dict[str, Any]) -> dict[str, Any]:
    traces = result.get("retrieval_traces") or []
    if len(traces) != 1:
        raise ValueError(
            f"Expected one retrieval trace for {result.get('case_id')} / {result.get('variant')}, "
            f"found {len(traces)}"
        )
    return traces[0]


def _candidate_diagnosis(trace: dict[str, Any], title: str) -> dict[str, Any]:
    matching = [row for row in trace.get("candidates", []) if row.get("title") == title]
    if not matching:
        return {"candidate_present": False, "reason": "not_in_recorded_candidate_trace"}
    best = min(
        matching,
        key=lambda row: (
            row.get("mmr_rank") is None,
            row.get("mmr_rank") or 10**9,
            row.get("vector_rank") or 10**9,
        ),
    )
    return {
        "candidate_present": True,
        "best_vector_rank": best.get("vector_rank"),
        "best_mmr_rank": best.get("mmr_rank"),
        "best_fulltext_rank": best.get("fulltext_rank"),
        "best_rrf_score": best.get("rrf_score"),
        "mmr_effect": best.get("mmr_effect"),
        "selected_candidate_for_title": any(bool(row.get("selected")) for row in matching),
        "rejection_reasons": sorted(
            {str(row["rejection_reason"]) for row in matching if row.get("rejection_reason")}
        ),
    }


def analyze_enhanced_loss(comparison: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Compare saved Basic and Enhanced runs without performing new retrieval."""
    results = _result_index(comparison)
    cases: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    basic_zero_fts = 0
    enhanced_zero_fts = 0
    lost_expected_total = 0
    gained_expected_total = 0
    gained_unexpected_total = 0
    enhanced_duplicate_parent_rejections = 0
    naive_basic_comparable_cases = 0
    naive_basic_identical_context_order_cases = 0
    global_sources: dict[str, dict[str, Any]] = {}
    for dataset_case in dataset.get("cases", []):
        global_sources.update(_expected_sources(dataset_case))

    for case in dataset.get("cases", []):
        case_id = str(case["id"])
        basic = results[(case_id, "hybrid_basic")]
        enhanced = results[(case_id, "hybrid_enhanced")]
        naive = results.get((case_id, "naive"))
        basic_trace = _trace(basic)
        enhanced_trace = _trace(enhanced)
        expected = _expected_sources(case)
        expected_ids = set(expected)
        basic_selected = _unique(basic.get("selected_source_ids", []))
        enhanced_selected = _unique(enhanced.get("selected_source_ids", []))
        basic_hits = basic_selected & expected_ids
        enhanced_hits = enhanced_selected & expected_ids
        lost_expected = sorted(basic_hits - enhanced_hits)
        gained_expected = sorted(enhanced_hits - basic_hits)
        gained_unexpected = sorted((enhanced_selected - basic_selected) - expected_ids)
        lost_expected_total += len(lost_expected)
        gained_expected_total += len(gained_expected)
        gained_unexpected_total += len(gained_unexpected)
        basic_zero_fts += int(int(basic_trace.get("fulltext_count") or 0) == 0)
        enhanced_zero_fts += int(int(enhanced_trace.get("fulltext_count") or 0) == 0)

        for candidate in enhanced_trace.get("candidates", []):
            reason = candidate.get("rejection_reason")
            if reason:
                rejection_counts[str(reason)] += 1
                enhanced_duplicate_parent_rejections += int(reason == "duplicate_parent_context")

        if naive is not None:
            naive_basic_comparable_cases += 1
            naive_basic_identical_context_order_cases += int(
                list(naive.get("selected_context_ids", [])) == list(basic.get("selected_context_ids", []))
            )

        lost_details = []
        for source_id in lost_expected:
            reference = expected[source_id]
            lost_details.append(
                {
                    "source_id": source_id,
                    "title": reference.get("title"),
                    "enhanced_candidate": _candidate_diagnosis(
                        enhanced_trace, str(reference.get("title") or "")
                    ),
                }
            )
        gained_unexpected_details = []
        for source_id in gained_unexpected:
            reference = global_sources.get(source_id, {})
            gained_unexpected_details.append(
                {
                    "source_id": source_id,
                    "title": reference.get("title"),
                    "enhanced_candidate": _candidate_diagnosis(
                        enhanced_trace, str(reference.get("title") or "")
                    ) if reference else {"candidate_present": False, "reason": "source_not_in_reference_catalog"},
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "source_group": case.get("source_group"),
                "question": case.get("question"),
                "expected_source_count": len(expected_ids),
                "basic_recall_at_5": basic.get("recall_at_5"),
                "enhanced_recall_at_5": enhanced.get("recall_at_5"),
                "recall_delta": round(
                    float(enhanced.get("recall_at_5") or 0) - float(basic.get("recall_at_5") or 0), 4
                ),
                "basic_selected_sources": sorted(basic_selected),
                "enhanced_selected_sources": sorted(enhanced_selected),
                "lost_expected_sources": lost_details,
                "gained_expected_sources": gained_expected,
                "gained_unexpected_sources": gained_unexpected,
                "gained_unexpected_source_details": gained_unexpected_details,
                "basic_fulltext_count": int(basic_trace.get("fulltext_count") or 0),
                "enhanced_fulltext_count": int(enhanced_trace.get("fulltext_count") or 0),
                "enhanced_selected_count": int(enhanced_trace.get("selected_count") or 0),
            }
        )

    degraded = [case for case in cases if case["recall_delta"] < 0]
    improved = [case for case in cases if case["recall_delta"] > 0]
    unchanged = [case for case in cases if case["recall_delta"] == 0]
    return {
        "analysis": "saved_trace_enhanced_vs_hybrid_basic",
        "new_retrieval_or_llm_calls": False,
        "case_count": len(cases),
        "summary": {
            "degraded_cases": len(degraded),
            "improved_cases": len(improved),
            "unchanged_cases": len(unchanged),
            "lost_expected_sources": lost_expected_total,
            "gained_expected_sources": gained_expected_total,
            "gained_unexpected_sources": gained_unexpected_total,
            "basic_queries_with_zero_fts_hits": basic_zero_fts,
            "enhanced_queries_with_zero_fts_hits": enhanced_zero_fts,
            "enhanced_duplicate_parent_rejections": enhanced_duplicate_parent_rejections,
            "naive_basic_comparable_cases": naive_basic_comparable_cases,
            "naive_basic_identical_context_order_cases": naive_basic_identical_context_order_cases,
            "enhanced_rejection_reasons": dict(sorted(rejection_counts.items())),
        },
        "findings": [
            "Enhanced and Hybrid Basic use the same query and frozen corpus; their retrieval difference is MMR before RRF plus parent-level deduplication/expansion.",
            "Parent expansion cannot add information in this snapshot because each stored parent contains one benchmark excerpt and has one child.",
            "When PostgreSQL FTS returns zero hits, RRF has only one ranked list: Basic preserves dense rank while Enhanced preserves MMR rank.",
            "Naive and Hybrid Basic returned the same ordered contexts for every comparable case, so their answer-score difference comes from generation/provider variance rather than retrieval.",
            "A source-level recall loss means Enhanced replaced an expected source with diversity that did not improve expected evidence coverage in this pilot.",
        ],
        "cases": cases,
    }
