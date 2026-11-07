"""Run a frozen Fast benchmark repeatedly and persist a statistical report.

This command makes paid provider calls. It performs a conservative budget
preflight before creating any datasets or jobs.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.cost_control import CostGuard
from app.db import close_database, session_factory
from app.models import EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchRun, SourceDocument
from app.schemas import EvaluationDatasetInput, ResearchMode


TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "cancelled"}
QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.65,
    "context_recall": 0.65,
    "success_rate": 0.95,
}


def select_matrix_cases(payload: dict) -> dict:
    """Select the frozen 3 PubMed + 3 arXiv + 4 mixed baseline matrix."""
    case_ids = payload.get("matrix_case_ids", [])
    if not case_ids:
        raise ValueError("The benchmark must define matrix_case_ids")
    case_by_id = {case["case_id"]: case for case in payload["cases"]}
    if len(case_by_id) != len(payload["cases"]):
        raise ValueError("Benchmark case_id values must be unique")
    missing = [case_id for case_id in case_ids if case_id not in case_by_id]
    if missing:
        raise ValueError(f"Unknown matrix case ids: {missing}")
    selected = [case_by_id[case_id] for case_id in case_ids]
    counts = Counter(case["cohort"] for case in selected)
    if counts != Counter({"pubmed": 3, "arxiv": 3, "mixed": 4}):
        raise ValueError(f"Fast baseline must contain 3 PubMed, 3 arXiv and 4 mixed cases; got {dict(counts)}")
    return {**payload, "name": payload.get("matrix_name", payload["name"]), "cases": selected}


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4) if values else None,
        "median": round(statistics.median(values), 4) if values else None,
        "stddev_population": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0 if values else None,
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


async def current_reserved_usd(redis) -> float:
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    raw = await redis.get(f"scientific_rag:cost:daily:{now:%Y-%m-%d}")
    return (int(raw) / 1_000_000) if raw else 0.0


async def create_and_queue(payload: dict, repetition: int, redis) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    async with session_factory() as session:
        dataset = EvaluationDataset(
            name=f"{payload['name']}-fast-r{repetition}-{stamp}",
            created_by="frozen-fast-benchmark",
        )
        session.add(dataset)
        await session.flush()
        session.add_all([
            EvaluationCase(
                dataset_id=dataset.id,
                question=case["question"],
                reference_answer=case["reference_answer"],
                reference_context=case.get("reference_context"),
                reference_claims=case.get("reference_claims", []),
                provenance="human",
            )
            for case in payload["cases"]
        ])
        run = EvaluationRun(dataset_id=dataset.id, status="queued", metrics={"mode": "fast", "repetition": repetition})
        session.add(run)
        await session.flush()
        reservation = await CostGuard(redis, get_settings()).reserve_evaluation(str(run.id), len(payload["cases"]))
        run.metrics = {
            **run.metrics,
            "cost_reservation": {**reservation.as_dict(), "scope": "judge_upper_bound", "case_count": len(payload["cases"])},
        }
        await session.commit()
        await redis.enqueue_job("run_evaluation_job", str(run.id), _job_id=f"evaluation:{run.id}")
        return str(run.id)


async def wait_for_run(run_id: str, poll_seconds: int) -> str:
    last_progress = None
    while True:
        async with session_factory() as session:
            run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == run_id))).scalar_one()
            progress = (
                run.status,
                run.metrics.get("phase"),
                run.metrics.get("research_completed", 0),
                run.metrics.get("evaluation_completed", 0),
            )
            if progress != last_progress:
                print(
                    f"run={run_id} status={progress[0]} phase={progress[1]} "
                    f"research={progress[2]}/{run.metrics.get('total_cases', 30)} "
                    f"judged={progress[3]}/{run.metrics.get('total_cases', 30)}",
                    flush=True,
                )
                last_progress = progress
            if run.status in TERMINAL_STATUSES:
                return run.status
        await asyncio.sleep(poll_seconds)


async def collect_report(payload: dict, run_ids: list[str]) -> dict:
    case_meta = {case["question"]: case for case in payload["cases"]}
    metric_values = {name: [] for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    repeat_metric_values = {name: [] for name in metric_values}
    research_latencies: list[float] = []
    research_costs: list[float] = []
    source_keys: set[str] = set()
    source_counts: list[float] = []
    fallback_count = successful_llm_calls = 0
    statuses = Counter()
    cohorts: dict[str, dict[str, list[float]]] = {}
    repeats = []

    async with session_factory() as session:
        for run_id in run_ids:
            run = (await session.execute(select(EvaluationRun).where(EvaluationRun.id == run_id))).scalar_one()
            results = list((await session.execute(
                select(EvaluationResult, EvaluationCase, ResearchRun)
                .join(EvaluationCase, EvaluationResult.evaluation_case_id == EvaluationCase.id)
                .outerjoin(ResearchRun, EvaluationResult.research_run_id == ResearchRun.id)
                .where(EvaluationResult.evaluation_run_id == run.id)
            )).all())
            repeat_scores = {name: [] for name in metric_values}
            for result, case, research in results:
                statuses[result.status] += 1
                metadata = case_meta.get(case.question, {})
                cohort = metadata.get("cohort", "unknown")
                cohort_scores = cohorts.setdefault(cohort, {name: [] for name in metric_values})
                if result.status == "judged":
                    for name in metric_values:
                        value = result.judge_scores.get(name)
                        if value is not None:
                            number = float(value)
                            metric_values[name].append(number)
                            repeat_scores[name].append(number)
                            cohort_scores[name].append(number)
                if research is None:
                    continue
                if research.created_at and research.completed_at:
                    research_latencies.append((research.completed_at - research.created_at).total_seconds())
                accounting = research.model_metadata.get("cost_accounting", {})
                if accounting.get("estimated_cost_usd") is not None:
                    research_costs.append(float(accounting["estimated_cost_usd"]))
                gateway = research.model_metadata.get("llm_gateway", {})
                fallback_count += int(gateway.get("fallback_count", 0))
                successful_llm_calls += int(gateway.get("successful_calls", 0))
                sources = list((await session.execute(
                    select(SourceDocument).where(SourceDocument.research_run_id == research.id)
                )).scalars())
                source_counts.append(float(len(sources)))
                for source in sources:
                    source_keys.add(source.source_url or f"{source.source}:{source.external_id or source.title}")
            repeat_summary = {name: distribution(values)["mean"] for name, values in repeat_scores.items()}
            for name, value in repeat_summary.items():
                if value is not None:
                    repeat_metric_values[name].append(float(value))
            repeats.append({
                "evaluation_run_id": str(run.id),
                "status": run.status,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "metrics": repeat_summary,
                "failed_cases": run.metrics.get("failed_cases", 0),
                "judge_cost_upper_bound_usd": run.metrics.get("cost_reservation", {}).get("amount_usd"),
            })

    total_cases = sum(statuses.values())
    successful_cases = statuses["judged"]
    # Timeout strings are summarized separately below without changing the
    # success definition (a case is successful only when it was judged).
    async with session_factory() as session:
        errors = list((await session.execute(
            select(EvaluationResult.error).where(EvaluationResult.evaluation_run_id.in_(run_ids))
        )).scalars())
    timeout_cases = sum("timeout" in (error or "").lower() or "timed out" in (error or "").lower() for error in errors)
    metric_summary = {name: distribution(values) for name, values in metric_values.items()}
    across_repeats = {name: distribution(values) for name, values in repeat_metric_values.items()}
    success_rate = successful_cases / total_cases if total_cases else 0.0
    gates = {
        name: {
            "threshold": threshold,
            "observed": success_rate if name == "success_rate" else metric_summary[name]["mean"],
            "passed": (success_rate if name == "success_rate" else metric_summary[name]["mean"] or 0) >= threshold,
        }
        for name, threshold in QUALITY_THRESHOLDS.items()
    }
    judge_upper_bound = sum(float(item.get("judge_cost_upper_bound_usd") or 0) for item in repeats)
    return {
        "benchmark": payload["name"],
        "benchmark_version": payload.get("version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fast",
        "configuration_repetitions": len(run_ids),
        "cases_per_repetition": len(payload["cases"]),
        "case_statuses": dict(statuses),
        "success_rate": round(success_rate, 4),
        "timeout_count": timeout_cases,
        "metrics_all_cases": metric_summary,
        "metrics_across_repetition_means": across_repeats,
        "cohorts": {
            cohort: {name: distribution(values) for name, values in scores.items()}
            for cohort, scores in cohorts.items()
        },
        "research_latency_seconds": distribution(research_latencies),
        "research_cost_usd": {
            **distribution(research_costs),
            "total_measured_or_estimated": round(sum(research_costs), 6),
            "ragas_judge_upper_bound": round(judge_upper_bound, 6),
            "note": "Research cost uses provider usage plus estimated embedding tokens. Ragas judge cost is a conservative reservation, not measured token billing.",
        },
        "sources": {
            "unique_across_matrix": len(source_keys),
            "per_case": distribution(source_counts),
        },
        "fallbacks": {
            "count": fallback_count,
            "successful_llm_calls": successful_llm_calls,
            "fallback_events_per_successful_call": round(fallback_count / successful_llm_calls, 4) if successful_llm_calls else None,
        },
        "regression_gates": gates,
        "production_comparison": {
            "status": "baseline_candidate",
            "maximum_allowed_metric_drop": 0.05,
            "latency_cost_growth_limit_percent": None,
            "note": "Freeze this report as production baseline after review. A latency/cost percentage gate still requires an operator-selected value.",
        },
        "repetitions": repeats,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=project_root / "evaluation" / "benchmark-fast-v2.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the dataset and print the budget calculation without creating paid jobs",
    )
    parser.add_argument("--report-run", action="append", default=[], help="Build a report for an existing evaluation run id; skips paid work")
    args = parser.parse_args()
    bank_payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    if len(bank_payload.get("cases", [])) != 30:
        raise ValueError("The frozen benchmark bank must contain exactly 30 reviewed cases")
    payload = select_matrix_cases(bank_payload)
    EvaluationDatasetInput.model_validate({"name": payload["name"], "cases": payload["cases"]})
    if len(payload["cases"]) != 10:
        raise ValueError("The Fast baseline matrix must contain exactly 10 cases")
    if args.repetitions < 1:
        raise ValueError("repetitions must be at least one")

    redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    run_ids = list(args.report_run)
    try:
        if not run_ids:
            settings = get_settings()
            conservative_required = len(payload["cases"]) * args.repetitions * (
                settings.cost_fast_reservation_usd + settings.cost_evaluation_judge_reservation_usd_per_case
            )
            currently_reserved = await current_reserved_usd(redis)
            available = max(settings.cost_daily_budget_usd - currently_reserved, 0)
            print(json.dumps({
                "budget_preflight": {
                    "conservative_required_usd": round(conservative_required, 2),
                    "currently_reserved_today_usd": round(currently_reserved, 6),
                    "configured_daily_budget_usd": settings.cost_daily_budget_usd,
                    "available_usd": round(available, 6),
                }
            }, indent=2), flush=True)
            if args.preflight_only:
                return 0
            if settings.cost_enforce_hard_limit and conservative_required > available:
                raise RuntimeError(
                    f"Benchmark requires up to ${conservative_required:.2f} in conservative reservations, "
                    f"but only ${available:.2f} remains in today's configured budget. "
                    "Raise COST_DAILY_BUDGET_USD explicitly, restart API/worker, and rerun."
                )
            for repetition in range(1, args.repetitions + 1):
                run_id = await create_and_queue(payload, repetition, redis)
                run_ids.append(run_id)
                status = await wait_for_run(run_id, args.poll_seconds)
                if status == "failed":
                    print(f"repetition={repetition} failed; continuing so the failure rate is measured", flush=True)
        report = await collect_report(payload, run_ids)
        output = args.output or project_root / "evaluation" / "results" / f"fast-v2-{datetime.now():%Y%m%d-%H%M%S}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"report={output}", flush=True)
        return 0 if all(item["passed"] for item in report["regression_gates"].values()) else 2
    finally:
        await redis.aclose()
        await close_database()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
