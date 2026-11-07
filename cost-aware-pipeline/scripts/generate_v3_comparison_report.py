"""Generate a self-contained report for the repaired Silver v3 comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics


VARIANTS = ("naive", "hybrid_basic", "hybrid_enhanced", "agentic_fast")
METRICS = (
    "answer_correctness",
    "answer_relevancy",
    "citation_correctness",
    "citation_completeness",
    "overall_quality",
)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stddev": round(statistics.pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def aggregate(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("judge")]
    provider_attempts: Counter[str] = Counter()
    provider_successes: Counter[str] = Counter()
    known_research_cost = 0.0
    unknown_cost_attempts = 0
    for row in completed:
        for provider, stats in row.get("llm", {}).get("by_provider", {}).items():
            provider_attempts[provider] += int(stats.get("attempts") or 0)
            provider_successes[provider] += int(stats.get("successes") or 0)
        for model_stats in row.get("llm", {}).get("by_model", {}).values():
            cost = model_stats.get("estimated_cost_usd")
            if cost is None:
                unknown_cost_attempts += int(model_stats.get("attempts") or 0)
            else:
                known_research_cost += float(cost)
        embedding_cost = row.get("embedding", {}).get("estimated_cost_usd")
        if embedding_cost is not None:
            known_research_cost += float(embedding_cost)

    return {
        "runs": len(rows),
        "completed": len(completed),
        "success_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
        "recall_at_5": distribution([float(row["recall_at_5"]) for row in completed]),
        "latency_seconds": distribution([float(row["latency_ms"]) / 1000 for row in completed]),
        "unique_sources": distribution([float(row["source_count"]) for row in completed]),
        "judge": {
            metric: distribution([float(row["judge"][metric]) for row in completed])
            for metric in METRICS
        },
        "fallbacks": sum(int(row.get("llm", {}).get("fallback_count") or 0) for row in completed),
        "runs_with_fallback": sum(bool(row.get("llm", {}).get("fallback_count")) for row in completed),
        "provider_attempts": dict(provider_attempts),
        "provider_successes": dict(provider_successes),
        "known_research_cost_lower_bound_usd": round(known_research_cost, 8),
        "unknown_cost_attempts": unknown_cost_attempts,
    }


def build_report(comparison: dict, dataset: dict, manifest: dict) -> dict:
    cases = {case["id"]: case for case in dataset["cases"]}
    by_variant = {
        variant: aggregate([row for row in comparison["results"] if row["variant"] == variant])
        for variant in VARIANTS
    }
    by_cohort = {
        cohort: {
            variant: aggregate([
                row for row in comparison["results"]
                if row["variant"] == variant and cases[row["case_id"]]["source_group"] == cohort
            ])
            for variant in VARIANTS
        }
        for cohort in sorted({case["source_group"] for case in cases.values()})
    }

    winners: Counter[str] = Counter()
    unique_winners: Counter[str] = Counter()
    per_case = []
    for case_id, case in cases.items():
        rows = [row for row in comparison["results"] if row["case_id"] == case_id]
        best = max(float(row["judge"]["overall_quality"]) for row in rows)
        tied = [row["variant"] for row in rows if float(row["judge"]["overall_quality"]) == best]
        winners.update(tied)
        if len(tied) == 1:
            unique_winners[tied[0]] += 1
        per_case.append({
            "case_id": case_id,
            "cohort": case["source_group"],
            "question": case["question"],
            "best_overall_quality": best,
            "winning_variants": tied,
            "scores": {
                row["variant"]: {
                    "recall_at_5": row["recall_at_5"],
                    "overall_quality": row["judge"]["overall_quality"],
                    "latency_seconds": round(float(row["latency_ms"]) / 1000, 3),
                }
                for row in rows
            },
        })

    descriptive_leader = max(
        VARIANTS,
        key=lambda variant: by_variant[variant]["judge"]["overall_quality"]["mean"],
    )
    snapshot_cost = manifest.get("embedding_usage", {}).get(
        "estimated_cost_usd", manifest.get("estimated_embedding_cost_usd")
    )
    judge_summary = comparison.get("judge_summary", {})
    known_research_lower_bound = sum(
        stats["known_research_cost_lower_bound_usd"] for stats in by_variant.values()
    )
    return {
        "experiment": "real-frozen-v3-12x4",
        "validity": "exploratory_repaired_silver_pilot_not_production_baseline",
        "created_at": comparison.get("created_at"),
        "snapshot_hash": comparison["snapshot_hash"],
        "case_count": len(cases),
        "run_count": len(comparison["results"]),
        "human_review": False,
        "judge": {
            "model": "gemini-2.5-flash",
            "calls": judge_summary.get("judged"),
            "failed": judge_summary.get("failed"),
            "measured_cost_usd": judge_summary.get("estimated_cost_usd"),
            "cost_complete": judge_summary.get("cost_complete"),
        },
        "variants": by_variant,
        "cohorts": by_cohort,
        "winner_counts_including_ties": dict(winners),
        "unique_winner_counts": dict(unique_winners),
        "per_case": per_case,
        "descriptive_leader": descriptive_leader,
        "costs": {
            "snapshot_embedding_usd": snapshot_cost,
            "judge_usd": judge_summary.get("estimated_cost_usd"),
            "known_research_lower_bound_usd": round(known_research_lower_bound, 8),
            "research_cost_complete": all(
                stats["unknown_cost_attempts"] == 0 for stats in by_variant.values()
            ),
            "warning": "The research amount is a lower bound when a provider returned no usage metadata.",
        },
        "interpretation": {
            "descriptive_finding": (
                f"{descriptive_leader} has the highest mean overall-quality score in this single execution."
            ),
            "decision": "Do not change the production default from this pilot alone.",
            "why": (
                "There is one run per question/configuration, no human review, and provider fallback changed "
                "the answer model across runs. Agentic Fast also had a smaller retrieval budget than the other arms."
            ),
        },
        "limitations": [
            "The 12 references are automatically repaired and independently model-audited Silver data, not human-reviewed Gold data.",
            "There is one execution per case and variant, so repeated-run variance is not measured.",
            "Provider fallback changed the answer-generating model and confounds architecture with model availability.",
            "Agentic Fast returned at most four contexts while the other variants returned five.",
            "External mining was disabled to keep one frozen corpus; this does not test the full live Agentic workflow.",
            "The questions were generated from the frozen evidence, which favors direct retrieval.",
            "The judge is an automated Gemini judge and not a human accuracy assessment.",
            "Research cost is a measured lower bound where providers omitted usage metadata.",
        ],
    }


def markdown(report: dict) -> str:
    lines = [
        "# Repaired Silver v3: real frozen RAG comparison",
        "",
        "This is an **exploratory 12-question Silver pilot**, not a production baseline.",
        "",
        "## Results",
        "",
        "| Variant | Success | Recall@5 | Correctness | Relevancy | Citation correctness | Citation completeness | Overall quality | Mean latency | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = report["variants"][variant]
        judge = row["judge"]
        lines.append(
            f"| {variant} | {row['success_rate']:.0%} | {row['recall_at_5']['mean']:.4f} ± {row['recall_at_5']['stddev']:.4f} | "
            f"{judge['answer_correctness']['mean']:.4f} | {judge['answer_relevancy']['mean']:.4f} | "
            f"{judge['citation_correctness']['mean']:.4f} | {judge['citation_completeness']['mean']:.4f} | "
            f"{judge['overall_quality']['mean']:.4f} ± {judge['overall_quality']['stddev']:.4f} | "
            f"{row['latency_seconds']['mean']:.2f}s | {row['fallbacks']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Descriptive leader: **{report['descriptive_leader']}**.",
        f"- {report['interpretation']['decision']}",
        f"- {report['interpretation']['why']}",
        "",
        "The comparison suggests that the fixed Hybrid Basic pipeline is the strongest candidate for simple questions in this small frozen-corpus test. Agentic Fast did not earn back its additional orchestration complexity here, but this benchmark deliberately disabled live mining and therefore does not test its main advantage: finding missing evidence dynamically.",
        "",
        "## Cohort breakdown",
        "",
        "| Cohort | Variant | Recall@5 | Overall quality | Mean latency |",
        "|---|---|---:|---:|---:|",
    ])
    for cohort, variants in report["cohorts"].items():
        for variant in VARIANTS:
            row = variants[variant]
            lines.append(
                f"| {cohort} | {variant} | {row['recall_at_5']['mean']:.4f} | "
                f"{row['judge']['overall_quality']['mean']:.4f} | {row['latency_seconds']['mean']:.2f}s |"
            )
    lines.extend([
        "",
        "## Provider behavior",
        "",
        "| Variant | Groq attempts/successes | OpenRouter attempts/successes | Gemini attempts/successes | Runs with fallback |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        row = report["variants"][variant]
        attempts = row["provider_attempts"]
        successes = row["provider_successes"]
        lines.append(
            f"| {variant} | {attempts.get('groq', 0)}/{successes.get('groq', 0)} | "
            f"{attempts.get('openrouter', 0)}/{successes.get('openrouter', 0)} | "
            f"{attempts.get('gemini', 0)}/{successes.get('gemini', 0)} | {row['runs_with_fallback']}/{row['runs']} |"
        )
    lines.extend([
        "",
        "Agentic Fast includes Planner and Gatekeeper calls in addition to Expert synthesis, so its provider-attempt count is intentionally not directly comparable with the three single-stage pipelines.",
        "",
        "## Cost accounting",
        "",
        f"- Snapshot embeddings: ${float(report['costs']['snapshot_embedding_usd'] or 0):.6f}",
        f"- Automated judge: ${float(report['costs']['judge_usd'] or 0):.6f}",
        f"- Known research lower bound: ${report['costs']['known_research_lower_bound_usd']:.6f}",
        f"- Research cost complete: **{report['costs']['research_cost_complete']}**",
        "- Missing research usage is concentrated in rejected/rate-limited Groq attempts and the free OpenRouter route; the reported research amount remains a conservative measured lower bound.",
        "",
        "## Winner counts",
        "",
    ])
    for variant in VARIANTS:
        lines.append(
            f"- {variant}: {report['winner_counts_including_ties'].get(variant, 0)} wins including ties; "
            f"{report['unique_winner_counts'].get(variant, 0)} unique wins."
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend([
        "",
        "## Recommended next experiment",
        "",
        "Pin one answer model, equalize every arm to five contexts, and repeat the same 12 questions three times. Then analyze simple versus multi-hop questions separately. Only after those controls pass should the benchmark expand toward 100 questions.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_report(comparison, dataset, manifest)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "descriptive_leader": report["descriptive_leader"],
        "json": str(args.json_output),
        "markdown": str(args.markdown_output),
    }, indent=2))


if __name__ == "__main__":
    main()
