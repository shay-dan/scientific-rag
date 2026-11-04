"""Provider-free aggregation for the retrieval-only ablation."""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any


STRATEGIES = ("naive", "hybrid_basic", "mmr", "hybrid_mmr", "enhanced")


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def build_retrieval_ablation_report(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["results"]
    indexed = {(row["case_id"], row["strategy"]): row for row in rows}
    case_ids = sorted({row["case_id"] for row in rows})
    aggregate: dict[str, Any] = {}
    cohorts: dict[str, dict[str, Any]] = defaultdict(dict)
    for strategy in STRATEGIES:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        aggregate[strategy] = {
            "cases": len(strategy_rows),
            "mean_source_recall": _mean([float(row["source_recall"]) for row in strategy_rows]),
            "queries_with_fts_hits": sum(int(row.get("fulltext_count") or 0) > 0 for row in strategy_rows),
            "mean_fts_candidates": _mean([float(row.get("fulltext_count") or 0) for row in strategy_rows]),
            "mean_unique_sources": _mean([
                float(len(set(row.get("selected_source_ids", [])))) for row in strategy_rows
            ]),
        }
        for cohort in sorted({str(row["source_group"]) for row in strategy_rows}):
            cohort_rows = [row for row in strategy_rows if row["source_group"] == cohort]
            cohorts[cohort][strategy] = {
                "cases": len(cohort_rows),
                "mean_source_recall": _mean([float(row["source_recall"]) for row in cohort_rows]),
            }

    def paired(left: str, right: str) -> dict[str, int]:
        deltas = [
            round(float(indexed[(case_id, right)]["source_recall"]) - float(indexed[(case_id, left)]["source_recall"]), 4)
            for case_id in case_ids
        ]
        return {
            "improved": sum(delta > 0 for delta in deltas),
            "degraded": sum(delta < 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
        }

    parent_identical = sum(
        list(indexed[(case_id, "hybrid_mmr")].get("selected_context_ids", []))
        == list(indexed[(case_id, "enhanced")].get("selected_context_ids", []))
        for case_id in case_ids
    )
    cases = []
    for case_id in case_ids:
        first = indexed[(case_id, "naive")]
        cases.append({
            "case_id": case_id,
            "source_group": first["source_group"],
            "question": first["question"],
            "source_recall": {
                strategy: indexed[(case_id, strategy)]["source_recall"] for strategy in STRATEGIES
            },
            "fts_candidates": indexed[(case_id, "hybrid_basic")].get("fulltext_count", 0),
        })
    return {
        "experiment": payload.get("experiment"),
        "snapshot_hash": payload.get("snapshot_hash"),
        "case_count": len(case_ids),
        "answer_generation_calls": payload.get("answer_generation_calls", 0),
        "llm_calls": payload.get("llm_calls", 0),
        "embedding_usage": payload.get("embedding_usage", {}),
        "aggregate": aggregate,
        "cohorts": dict(cohorts),
        "paired_changes": {
            "fts_added_to_dense": paired("naive", "hybrid_basic"),
            "fts_added_to_mmr": paired("mmr", "hybrid_mmr"),
        },
        "parent_handling_identical_ordered_contexts": parent_identical,
        "cases": cases,
        "interpretation": {
            "fts_status": "working_for_all_queries",
            "dense_vs_hybrid_basic": "Equal mean expected-source recall in this pilot; lexical retrieval is now active but did not improve the already strong dense baseline.",
            "mmr_effect": "MMR alone reduced expected-source recall; FTS+RRF recovered part, but not all, of that loss.",
            "parent_effect": "No measurable effect in this excerpt snapshot because every parent has one child.",
            "production_decision": "Do not change the production default from this 12-case retrieval-only pilot.",
        },
    }
