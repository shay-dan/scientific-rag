"""Generate the provider-free final report for the 12-case real comparison."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.comparative_benchmark import estimate_comparative_plan


VARIANT_ORDER = ("naive", "hybrid_basic", "hybrid_enhanced", "agentic_fast")
JUDGE_METRICS = (
    "answer_correctness",
    "answer_relevancy",
    "citation_correctness",
    "citation_completeness",
    "overall_quality",
)


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stddev": round(statistics.pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p95": round(ordered[p95_index], 4),
    }


def aggregate(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    output = {
        "runs": len(rows),
        "completed": len(completed),
        "success_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
        "recall_at_5": _distribution([float(row["recall_at_5"]) for row in completed]),
        "latency_seconds": _distribution([float(row["latency_ms"]) / 1000 for row in completed]),
        "unique_sources": _distribution([float(row["source_count"]) for row in completed]),
        "provider_calls": sum(int(row["llm"].get("provider_calls") or 0) for row in completed),
        "fallback_count": sum(int(row["llm"].get("fallback_count") or 0) for row in completed),
        "runs_with_fallback": sum(bool(row["llm"].get("fallback_count")) for row in completed),
        "judge": {},
    }
    for metric in JUDGE_METRICS:
        output["judge"][metric] = _distribution([float(row["judge"][metric]) for row in completed])
    known_costs = [row.get("estimated_cost_usd") for row in completed if row.get("estimated_cost_usd") is not None]
    output["research_cost"] = {
        "known_result_count": len(known_costs),
        "unknown_result_count": len(completed) - len(known_costs),
        "known_cost_usd": round(sum(known_costs), 8),
        "complete": len(known_costs) == len(completed),
    }
    return output


def build_report(
    comparison: dict,
    dataset: dict,
    manifest: dict,
    evidence_analysis: dict | None = None,
    retrieval_followup: dict | None = None,
) -> dict:
    cohort_by_case = {case["id"]: case["source_group"] for case in dataset["cases"]}
    variants = {
        variant: aggregate([row for row in comparison["results"] if row["variant"] == variant])
        for variant in VARIANT_ORDER
    }
    cohorts = {
        cohort: {
            variant: aggregate([
                row for row in comparison["results"]
                if row["variant"] == variant and cohort_by_case[row["case_id"]] == cohort
            ])
            for variant in VARIANT_ORDER
        }
        for cohort in ("pubmed", "arxiv", "mixed")
    }
    winner_counts = {variant: 0 for variant in VARIANT_ORDER}
    for case_id in cohort_by_case:
        rows = [row for row in comparison["results"] if row["case_id"] == case_id]
        best = max(float(row["judge"]["overall_quality"]) for row in rows)
        for row in rows:
            if float(row["judge"]["overall_quality"]) == best:
                winner_counts[row["variant"]] += 1
    research_planning_estimate = estimate_comparative_plan(case_count=12, repetitions=1, judge=False)["estimated_total_cost_usd"]
    judge_measured_partial = comparison["judge_summary"]["estimated_cost_usd"]
    snapshot_cost = manifest.get("embedding_usage", {}).get("estimated_cost_usd", manifest.get("estimated_embedding_cost_usd"))
    return {
        "experiment": "real-frozen-12x4",
        "validity": "exploratory_silver_pilot_not_baseline",
        "decision_status": "insufficient_for_production_default_change",
        "snapshot_hash": comparison["snapshot_hash"],
        "case_count": 12,
        "variant_run_count": 48,
        "human_review": False,
        "judge": {
            "type": "independent_custom_llm_as_judge_not_ragas",
            "model": "gemini-2.5-flash",
            "calls": 48,
            "limitations": "One automated judge, no human review; reported usage omits separately billed thinking tokens when the provider reports them outside candidate tokens.",
        },
        "metric_definitions": {
            "recall_at_5": "Unique expected source IDs found among the returned contexts divided by unique expected source IDs. Agentic Fast returned at most four contexts, so the name @5 does not imply five returned contexts for every arm.",
            "reported_stddev": "Population standard deviation across different questions, not repeated-run model variance.",
            "judge_scores": "Scores from the custom Gemini judge prompt; they are not Ragas metrics and are not human-verified accuracy percentages.",
        },
        "variants": variants,
        "cohorts": cohorts,
        "overall_quality_winner_counts_including_ties": winner_counts,
        "costs": {
            "snapshot_embedding_estimated_from_measured_calls_usd": snapshot_cost,
            "judge_measured_partial_usd": judge_measured_partial,
            "judge_cost_complete": False,
            "judge_cost_incomplete_reason": "The saved accounting includes prompt and candidate tokens but not separately reported thinking tokens.",
            "research_actual_usd": None,
            "research_actual_reason": "Some generation providers did not return complete token usage.",
            "research_preflight_planning_estimate_usd": research_planning_estimate,
            "research_preflight_is_upper_bound": False,
            "partial_measured_plus_planning_estimate_usd": round(float(snapshot_cost) + float(judge_measured_partial) + float(research_planning_estimate), 8),
        },
        "interpretation": {
            "descriptive_leader_on_this_single_run": "hybrid_basic",
            "production_default_recommendation": "No production default change from this pilot.",
            "hybrid_basic_finding": "FTS returned zero candidates for every saved query, so Hybrid Basic reduced to dense-only retrieval and returned the same ordered contexts as Naive. Score differences between those arms are generation/provider variance, not evidence of hybrid-search benefit.",
            "agentic_finding": "The restricted frozen-corpus Agentic Fast arm did not show a mean advantage here; this is not a test of live mining or the full production orchestration path.",
            "enhanced_finding": "Investigate MMR separately. Parent expansion had no information advantage because every benchmark parent held one short excerpt and one child.",
        },
        "enhanced_evidence_loss_analysis": evidence_analysis.get("summary") if evidence_analysis else None,
        "retrieval_only_followup": {
            "aggregate": retrieval_followup.get("aggregate"),
            "paired_changes": retrieval_followup.get("paired_changes"),
            "parent_handling_identical_ordered_contexts": retrieval_followup.get("parent_handling_identical_ordered_contexts"),
        } if retrieval_followup else None,
        "limitations": [
            "The 12 cases and references are silver and have no human review. Claim-to-context mappings were repaired using lexical overlap, which is candidate mapping rather than semantic verification.",
            "There is one execution per case and variant, so model-output variance across repeated runs is not measured.",
            "The corpus contains 31 sources and 51 excerpts, all shorter than one child chunk; parent expansion therefore has no information advantage in this pilot.",
            "Questions were generated from the same source collection, favoring direct retrieval and reducing the need for external multi-step research.",
            "PostgreSQL FTS returned zero candidates for all 12 queries, so this run did not measure a functioning dense-plus-lexical fusion.",
            "Burst execution triggered Groq rate limits and OpenRouter timeouts, so latency includes provider-availability effects and is not normal single-user latency.",
            "Provider fallback changed the answer-generating model across runs, confounding architecture and model effects.",
            "Agentic Fast had a four-context retrieval budget while the non-agentic arms had five; source recall is therefore not a perfectly equal-budget comparison.",
            "Agentic Fast used production planner/gatekeeper and synthesis components but deliberately bypassed live source mining, cache routing, and the complete production orchestrator.",
            "Gemini 2.5 Flash is an automated judge, not human ground truth; scores should be treated as comparative evidence, not absolute accuracy.",
            "Research cost is incomplete and the preflight estimate is not a guaranteed upper bound.",
        ],
    }


def markdown(report: dict) -> str:
    lines = [
        "# Real 12-case RAG comparison",
        "",
        "This is an **exploratory silver pilot, not a production baseline**. It replaces the invalid in-memory comparison and uses one shared PostgreSQL/pgvector snapshot, but important controls remain incomplete.",
        "",
        "## Main results",
        "",
        "| Variant | Success | Recall@5 | Answer correctness | Citation correctness | Overall quality | Mean latency | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        row = report["variants"][variant]
        lines.append(
            f"| {variant} | {row['success_rate']:.0%} | {row['recall_at_5']['mean']:.4f} ± {row['recall_at_5']['stddev']:.4f} | "
            f"{row['judge']['answer_correctness']['mean']:.4f} | {row['judge']['citation_correctness']['mean']:.4f} | "
            f"{row['judge']['overall_quality']['mean']:.4f} | {row['latency_seconds']['mean']:.2f}s | {row['fallback_count']} |"
        )
    lines.extend([
        "",
        "## What it means",
        "",
        "Hybrid Basic had the highest descriptive answer-score mean, but it returned the **same ordered contexts as Naive for all 12 questions** because PostgreSQL FTS produced zero hits throughout the run. Therefore its answer-score advantage is generation/provider variance, not demonstrated hybrid-retrieval benefit.",
        "",
        "This does **not** prove that Agentic RAG is unnecessary. The benchmark removed external mining and used short answer-selected excerpts. The Agentic arm exercised planner/gatekeeper and synthesis components, but not the complete production research path.",
        "",
        "The reported standard deviation is variation across different questions, not variation from repeated runs. The judge scores are custom LLM-as-judge scores, not Ragas scores or human-verified accuracy percentages.",
    ])
    enhanced = report.get("enhanced_evidence_loss_analysis")
    if enhanced:
        lines.extend([
            "",
            "## Enhanced evidence-loss diagnosis",
            "",
            f"- Lower source recall in **{enhanced['degraded_cases']} / {report['case_count']}** cases; no case improved.",
            f"- **{enhanced['lost_expected_sources']}** expected sources were lost and **{enhanced['gained_unexpected_sources']}** unexpected source selections were added.",
            f"- FTS returned zero hits for **{enhanced['basic_queries_with_zero_fts_hits']} / {report['case_count']}** Basic queries and **{enhanced['enhanced_queries_with_zero_fts_hits']} / {report['case_count']}** Enhanced queries.",
            f"- Parent deduplication rejected **{enhanced['enhanced_duplicate_parent_rejections']}** candidates, so it did not cause the loss.",
            f"- Naive and Hybrid Basic returned identical ordered contexts in **{enhanced['naive_basic_identical_context_order_cases']} / {enhanced['naive_basic_comparable_cases']}** cases.",
            "",
            "The saved traces isolate MMR as the effective differentiator: answer-bearing sources originally at dense ranks 2–5 moved to MMR ranks 6–12. See `enhanced-evidence-loss-analysis.md` for per-case details.",
        ])
    followup = report.get("retrieval_only_followup")
    if followup:
        aggregate = followup["aggregate"]
        paired = followup["paired_changes"]
        lines.extend([
            "",
            "## Retrieval-only follow-up after the FTS fix",
            "",
            f"- PostgreSQL FTS returned candidates for **{aggregate['hybrid_basic']['queries_with_fts_hits']}/{aggregate['hybrid_basic']['cases']}** questions.",
            f"- Dense recall: **{aggregate['naive']['mean_source_recall']:.4f}**; Dense+FTS+RRF recall: **{aggregate['hybrid_basic']['mean_source_recall']:.4f}**.",
            f"- MMR recall: **{aggregate['mmr']['mean_source_recall']:.4f}**; MMR+FTS+RRF recall: **{aggregate['hybrid_mmr']['mean_source_recall']:.4f}**.",
            f"- Adding FTS to MMR improved **{paired['fts_added_to_mmr']['improved']}** cases, degraded **{paired['fts_added_to_mmr']['degraded']}**, and left **{paired['fts_added_to_mmr']['unchanged']}** unchanged.",
            f"- Parent handling returned identical ordered contexts in **{followup['parent_handling_identical_ordered_contexts']}/{report['case_count']}** cases.",
            "",
            "The FTS zero-hit defect is fixed. The follow-up still does not justify unconditional MMR or a production-default change; see `retrieval-ablation-12x5-report.md`.",
        ])
    lines.extend([
        "",
        "## Cost",
        "",
        f"- Snapshot embedding estimate from measured calls: ${report['costs']['snapshot_embedding_estimated_from_measured_calls_usd']:.6f}",
        f"- Judge measured partial cost: ${report['costs']['judge_measured_partial_usd']:.6f} (thinking-token cost is not included)",
        f"- Research preflight planning estimate: ${report['costs']['research_preflight_planning_estimate_usd']:.5f} (not a guaranteed upper bound)",
        f"- Partial measured amounts plus planning estimate: ${report['costs']['partial_measured_plus_planning_estimate_usd']:.6f}; this is not an actual total.",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend([
        "",
        "## Recommendation",
        "",
        "Do not change the production default from this pilot. FTS query construction is now repaired and the retrieval-only ablation isolated MMR and parent handling. Next, strengthen source/claim validation, use a realistic full-text frozen corpus, pin one answer model, equalize context budgets, and repeat the same 12 cases before considering wider expansion.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    comparison = json.loads((ROOT / "evaluation" / "real-frozen-12x4-scored.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "evaluation" / "pilot-silver-v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "evaluation" / "frozen-12-corpus-manifest.json").read_text(encoding="utf-8"))
    analysis_path = ROOT / "evaluation" / "enhanced-evidence-loss-analysis.json"
    evidence_analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else None
    followup_path = ROOT / "evaluation" / "retrieval-ablation-12x5-report.json"
    retrieval_followup = json.loads(followup_path.read_text(encoding="utf-8")) if followup_path.exists() else None
    report = build_report(comparison, dataset, manifest, evidence_analysis, retrieval_followup)
    (ROOT / "evaluation" / "real-frozen-12x4-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "evaluation" / "real-frozen-12x4-report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "written", "descriptive_leader": report["interpretation"]["descriptive_leader_on_this_single_run"], "costs": report["costs"]}, indent=2))


if __name__ == "__main__":
    main()
