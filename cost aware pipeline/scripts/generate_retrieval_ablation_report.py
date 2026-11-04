"""Generate Markdown/JSON reports for the refined retrieval-only ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval_ablation_report import STRATEGIES, build_retrieval_ablation_report


LABELS = {
    "naive": "Dense",
    "hybrid_basic": "Dense + FTS + RRF",
    "mmr": "Dense + MMR",
    "hybrid_mmr": "Dense + MMR + FTS + RRF",
    "enhanced": "Dense + MMR + FTS + RRF + Parent",
}


def markdown(report: dict) -> str:
    lines = [
        "# Refined retrieval-only ablation",
        "",
        "This experiment used the same frozen PostgreSQL/pgvector snapshot for all strategies. It made no answer-generation, Planner, Expert, Gatekeeper, or Judge calls. One query embedding was shared by all five strategies for each question.",
        "",
        "## Aggregate results",
        "",
        "| Strategy | Mean source recall | FTS coverage | Mean FTS candidates | Mean unique sources |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        row = report["aggregate"][strategy]
        lines.append(
            f"| {LABELS[strategy]} | {row['mean_source_recall']:.4f} | "
            f"{row['queries_with_fts_hits']}/{row['cases']} | {row['mean_fts_candidates']:.2f} | "
            f"{row['mean_unique_sources']:.2f} |"
        )
    lines.extend([
        "",
        "## Paired effects",
        "",
        f"- Adding FTS+RRF to Dense: {report['paired_changes']['fts_added_to_dense']['improved']} improved, {report['paired_changes']['fts_added_to_dense']['degraded']} degraded, {report['paired_changes']['fts_added_to_dense']['unchanged']} unchanged.",
        f"- Adding FTS+RRF to MMR: {report['paired_changes']['fts_added_to_mmr']['improved']} improved, {report['paired_changes']['fts_added_to_mmr']['degraded']} degraded, {report['paired_changes']['fts_added_to_mmr']['unchanged']} unchanged.",
        f"- Adding parent handling after Hybrid+MMR returned identical ordered contexts in {report['parent_handling_identical_ordered_contexts']}/{report['case_count']} cases.",
        "",
        "## Per-case source recall",
        "",
        "| Case | Group | Dense | Hybrid Basic | MMR | Hybrid MMR | Enhanced | FTS candidates |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        score = case["source_recall"]
        lines.append(
            f"| {case['case_id']} | {case['source_group']} | {score['naive']:.4f} | "
            f"{score['hybrid_basic']:.4f} | {score['mmr']:.4f} | {score['hybrid_mmr']:.4f} | "
            f"{score['enhanced']:.4f} | {case['fts_candidates']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The refined FTS query builder fixed the zero-hit failure: every question produced lexical candidates. Dense and Hybrid Basic now have equal expected-source recall (0.9514), so this small answer-derived corpus does not yet demonstrate an FTS quality gain, but it does show that lexical retrieval is functioning without harming recall.",
        "",
        "MMR alone reduced recall from 0.9514 to 0.7847. FTS+RRF recovered it to 0.9097, showing that an independent lexical signal can repair some diversity-induced losses. It still remained below the Dense baseline, so MMR should not be enabled unconditionally from this evidence.",
        "",
        "Parent handling changed nothing in all cases. That is expected: this benchmark stores one short excerpt per parent and one child per parent. Parent-child retrieval must be tested on full-length documents with multiple children before it can be judged.",
        "",
        "## Cost and decision",
        "",
        f"- Query embedding calls: {report['embedding_usage'].get('calls')}.",
        f"- Estimated embedding cost: ${float(report['embedding_usage'].get('estimated_cost_usd') or 0):.8f}.",
        "- Answer-generation and judge calls: 0.",
        "- Production default: unchanged. The next valid experiment requires a realistic full-text corpus, not more tuning against these answer-selected excerpts.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    source = ROOT / "evaluation" / "retrieval-ablation-12x5-refined.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = build_retrieval_ablation_report(payload)
    json_path = ROOT / "evaluation" / "retrieval-ablation-12x5-report.json"
    md_path = ROOT / "evaluation" / "retrieval-ablation-12x5-report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "paired_changes": report["paired_changes"]}, indent=2))


if __name__ == "__main__":
    main()
