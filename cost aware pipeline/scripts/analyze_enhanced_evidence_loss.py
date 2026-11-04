"""Explain Enhanced retrieval evidence loss from already-saved traces."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval_loss_analysis import analyze_enhanced_loss


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Why Hybrid Enhanced lost evidence",
        "",
        "This is a provider-free post-hoc analysis of the saved 12-case pilot. It did not run retrieval, research, embeddings, or LLM calls.",
        "",
        "## Aggregate diagnosis",
        "",
        f"- Cases with lower source recall: **{summary['degraded_cases']} / {report['case_count']}**",
        f"- Cases with higher source recall: **{summary['improved_cases']} / {report['case_count']}**",
        f"- Expected sources present in Basic but lost by Enhanced: **{summary['lost_expected_sources']}**",
        f"- Expected sources newly recovered by Enhanced: **{summary['gained_expected_sources']}**",
        f"- Unexpected sources added by Enhanced: **{summary['gained_unexpected_sources']}**",
        f"- Queries with no PostgreSQL FTS hit: Basic **{summary['basic_queries_with_zero_fts_hits']}**, Enhanced **{summary['enhanced_queries_with_zero_fts_hits']}**",
        f"- Candidates rejected as duplicate parents: **{summary['enhanced_duplicate_parent_rejections']}**",
        f"- Naive and Hybrid Basic with identical ordered contexts: **{summary['naive_basic_identical_context_order_cases']} / {summary['naive_basic_comparable_cases']}**",
        "",
        "## Per-case evidence changes",
        "",
        "| Case | Group | Basic recall | Enhanced recall | Delta | Lost expected sources | Added unexpected sources |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['source_group']} | {float(case['basic_recall_at_5']):.4f} | "
            f"{float(case['enhanced_recall_at_5']):.4f} | {case['recall_delta']:+.4f} | "
            f"{len(case['lost_expected_sources'])} | {len(case['gained_unexpected_sources'])} |"
        )
    lines.extend([
        "",
        "## Lost-source trace details",
        "",
    ])
    for case in report["cases"]:
        for lost in case["lost_expected_sources"]:
            candidate = lost["enhanced_candidate"]
            effect = candidate.get("mmr_effect") or {}
            lines.append(
                f"- **{case['case_id']} — {lost['title']}**: "
                f"vector rank `{candidate.get('best_vector_rank')}`, MMR rank `{candidate.get('best_mmr_rank')}`, "
                f"relevance `{effect.get('relevance')}`, redundancy `{effect.get('redundancy_to_prior')}`, "
                f"MMR score `{effect.get('mmr_score_lambda_0_5')}`, FTS rank `{candidate.get('best_fulltext_rank')}`, "
                f"reasons `{', '.join(candidate.get('rejection_reasons', [])) or candidate.get('reason', 'none')}`."
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The saved run did **not** exercise a working lexical+dense fusion: PostgreSQL FTS returned zero candidates for all 12 natural-language queries. Consequently, Hybrid Basic reduced to dense retrieval and returned the same ordered contexts as Naive in every case. Their different answer scores are generation/provider variance, not a retrieval improvement.",
        "",
        "Hybrid Basic and Enhanced start from the same dense candidates. With no FTS list, RRF cannot combine independent signals; Enhanced differs primarily because it runs MMR before RRF, followed by parent-level deduplication.",
        "",
        "In this snapshot, parent expansion is structurally unable to help: each parent is only one short benchmark excerpt and has one child. If duplicate-parent rejections are zero, parent deduplication also did not cause the measured loss. The remaining differentiator is MMR reordering before RRF.",
        "",
        "With `lambda=0.5`, MMR gives relevance and non-redundancy equal weight. The six expected sources that disappeared were originally at dense ranks 2–5 and were moved to MMR ranks 6–12. On this tiny corpus, the diversity penalty therefore replaced answer-bearing sources with diverse but mostly non-reference sources. RRF could not correct that movement because it received no FTS-ranked list.",
        "",
        "## Next controlled check",
        "",
        "First repair the FTS query builder and add a regression test proving non-zero lexical hits for representative PubMed, arXiv, and mixed questions. Then run a retrieval-only ablation on a realistic full-text corpus: Dense, Dense+FTS+RRF, Dense+MMR, and Dense+FTS+RRF+parent handling. It requires no answer generation and isolates each component.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    comparison = json.loads((ROOT / "evaluation" / "real-frozen-12x4-scored.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "evaluation" / "pilot-silver-v1.json").read_text(encoding="utf-8"))
    report = analyze_enhanced_loss(comparison, dataset)
    json_path = ROOT / "evaluation" / "enhanced-evidence-loss-analysis.json"
    md_path = ROOT / "evaluation" / "enhanced-evidence-loss-analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
