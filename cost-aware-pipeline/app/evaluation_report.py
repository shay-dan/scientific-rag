"""R4: evaluation case records, outcome taxonomy, and deterministic reports.

Standard library only. Everything here is a pure function of persisted per-case
records and run provenance, so reports can be regenerated offline without any
model or network call.

Rules enforced here:
- every case stays in the population with exactly one outcome category;
- completion/coverage is reported separately from scientific-quality metrics;
- quality metrics are conditional on scored cases and never invent a score
  (failures and abstentions are excluded, not zero-filled);
- scores from different evaluator engines, judge models or rubric versions are
  never averaged together;
- every metric carries its sample size and, where meaningful, a confidence interval.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from typing import Any, Iterable

# v2: outcomes not_executed and research_interrupted (population is the full dataset,
# not only cases with result rows); judge cost completeness statuses; git SHA source.
REPORT_SCHEMA = "scientific-rag-evaluation-report/v2"
CASE_RECORD_SCHEMA = "scientific-rag-evaluation-case/v1"
QUALITY_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_SEED = 20261112
SMALL_SAMPLE = 10
Z_95 = 1.959963984540054

# Outcome taxonomy. Exactly one category per case.
OUTCOMES = (
    "judged",                    # answered and scored by an evaluator
    "insufficient_evidence",     # the pipeline abstained: no adequate evidence found
    "acquisition_unavailable",   # external literature source unavailable
    "screening_failed",          # abstention caused only by relevance-screening failures
    "research_failed",           # research raised an error
    "research_timeout",          # research timed out
    "judge_failed",              # every evaluator engine failed
    "judge_timeout",             # the last evaluator engine timed out
    "invalid_judge_output",      # the last evaluator engine returned invalid output
    "missing_assessment",        # answered but never assessed (e.g. interrupted run)
    "research_interrupted",      # research started but never finished (worker interrupted)
    "not_executed",              # never started (run interrupted or stopped before this case)
)
ANSWERED = ("judged", "judge_failed", "judge_timeout", "invalid_judge_output", "missing_assessment")
ABSTAINED = ("insufficient_evidence",)
ACQUISITION_FAILURES = ("acquisition_unavailable", "screening_failed")
RESEARCH_FAILURES = ("research_failed", "research_timeout")
JUDGE_FAILURES = ("judge_failed", "judge_timeout", "invalid_judge_output", "missing_assessment")
NOT_COMPLETED = ("research_interrupted", "not_executed")

# Case provenance. Only explicitly declared values are trusted.
CASE_PROVENANCE = ("human_reviewed", "human_authored", "synthetic", "model_generated", "unspecified")
LEGACY_PROVENANCE = {"human": "legacy_human_unverified"}

_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bsk-[0-9A-Za-z_\-]{16,}"),
    re.compile(r"\bgsk_[0-9A-Za-z]{16,}"),
    re.compile(r"(?i)\bbearer\s+[0-9A-Za-z._\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|key|token|secret|password)=[^\s&]+"),
)


# --- helpers -------------------------------------------------------------------------------------
def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sanitize_error(text: str | None, limit: int = 300) -> str | None:
    """Short, credential-free error detail suitable for evaluation artifacts."""
    if not text:
        return None
    cleaned = str(text)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[redacted]", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit]


def normalize_case_provenance(value: str | None) -> str:
    """Declared provenance as stored; legacy 'human' is surfaced as unverified."""
    if value in CASE_PROVENANCE:
        return str(value)
    if value in LEGACY_PROVENANCE:
        return LEGACY_PROVENANCE[value]
    return "unspecified"


def infer_case_provenance(case: dict, dataset: dict) -> str:
    """Provenance for a case loaded from a dataset file, from what the file declares.

    Never defaults to a human label: a case is human-reviewed or human-authored only
    when the file says so; model-generated silver data is labeled as such.
    """
    declared = case.get("case_provenance") or dataset.get("case_provenance")
    if declared in CASE_PROVENANCE:
        return declared
    reviewed = case.get("human_review", dataset.get("human_review"))
    if isinstance(reviewed, dict):
        reviewed = reviewed.get("reviewed") or reviewed.get("status") == "reviewed"
    raw = " ".join(str(value) for value in (case.get("provenance"), dataset.get("provenance"), case.get("generator")) if value)
    if reviewed is True:
        return "human_reviewed"
    if reviewed is False or re.search(r"silver|auto|generated|synthetic|model", raw, re.IGNORECASE):
        return "synthetic" if re.search(r"synthetic|perturb", raw, re.IGNORECASE) else "model_generated"
    if re.search(r"human[- ]authored", str(dataset.get("review_policy", "")), re.IGNORECASE):
        return "human_authored"
    return "unspecified"


def case_content_hash(*, question: str, reference_answer: str, reference_context: str | None,
                      reference_claims: Iterable[str] | None, provenance: str | None) -> str:
    return sha256_hex({
        "question": question, "reference_answer": reference_answer,
        "reference_context": reference_context or None,
        "reference_claims": list(reference_claims or []), "provenance": provenance,
    })


def dataset_content_hash(case_hashes: Iterable[str]) -> str:
    """Order-independent dataset identity: hash of the sorted case hashes."""
    return sha256_hex(sorted(case_hashes))


def valid_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def split_scores(raw: dict | None) -> tuple[dict[str, float], list[str], list[str]]:
    """(valid scores, missing metric names, invalid metric names) for the quality metrics."""
    raw = raw or {}
    valid, missing, invalid = {}, [], []
    for metric in QUALITY_METRICS:
        if metric not in raw or raw[metric] is None:
            missing.append(metric)
            continue
        number = valid_score(raw[metric])
        if number is None:
            invalid.append(metric)
        else:
            valid[metric] = number
    return valid, missing, invalid


# --- discovery details from research traces -----------------------------------------------------
def discovery_details(model_metadata: dict | None) -> dict:
    """Acquisition and screening facts the current pipeline actually records.

    Nothing is inferred beyond the stored traces; absent traces yield ``observed: False``.
    """
    metadata = model_metadata or {}
    sufficiency = metadata.get("source_sufficiency") or {}
    inspectors = [*(metadata.get("pubmed_inspector") or []), *(metadata.get("arxiv_inspector") or [])]
    candidates = [candidate for trace in inspectors for candidate in (trace.get("candidates") or [])]
    decisions = [str(candidate.get("decision") or "") for candidate in candidates]
    errors = [str(error) for trace in inspectors for error in (trace.get("errors") or [])]
    source_errors = {source: len(details.get("errors") or [])
                     for source, details in sorted((metadata.get("source_diagnostics") or {}).items())}
    return {
        "observed": bool(sufficiency or inspectors or source_errors),
        "evidence_outcome": sufficiency.get("outcome"),
        "external_unavailable": any(bool(trace.get("external_unavailable")) for trace in inspectors)
        or sufficiency.get("outcome") == "external_source_unavailable",
        "candidates": len(candidates),
        "accepted_candidates": sum(decision.startswith("retained_") for decision in decisions),
        "screening_failures": sum(decision == "screening_failed_not_indexed" for decision in decisions),
        "rejected_candidates": sum(decision.startswith("rejected") for decision in decisions),
        # Full text was unavailable and only the abstract was eligible (PMC fetch/identity error).
        "full_text_fallbacks": max(sum(bool(candidate.get("pmc_fetch_error")) for candidate in candidates),
                                   sum("PMC full text skipped" in error for error in errors)),
        "pdf_fallbacks": None,  # PDF ingestion is not enabled; R9-F will populate this.
        "source_errors": source_errors,
    }


def classify_abstention(discovery: dict) -> str:
    """Category for a research run that returned ``insufficient_evidence``."""
    if discovery.get("evidence_outcome") == "external_source_unavailable" or discovery.get("external_unavailable"):
        return "acquisition_unavailable"
    failures = int(discovery.get("screening_failures") or 0)
    if failures and not discovery.get("accepted_candidates") and not discovery.get("rejected_candidates"):
        return "screening_failed"
    return "insufficient_evidence"


def generator_models(llm_diagnostics: dict | None) -> list[str]:
    """provider:model pairs that generated research for a case, from the gateway trace."""
    by_model = (llm_diagnostics or {}).get("by_model") or {}
    return sorted(str(key) for key in by_model)


def provider_failures(llm_diagnostics: dict | None) -> dict[str, int]:
    by_provider = (llm_diagnostics or {}).get("by_provider") or {}
    return {provider: int(stats.get("failures") or 0) for provider, stats in sorted(by_provider.items())}


# --- case records -------------------------------------------------------------------------------
def build_case_record(*, case_id: str, case_hash: str, case_provenance: str | None, category: str,
                      stage: str | None = None, error_type: str | None = None, error: str | None = None,
                      research: dict | None = None, discovery: dict | None = None, judge: dict | None = None,
                      raw_scores: dict | None = None, inferred: bool = False) -> dict:
    if category not in OUTCOMES:
        raise ValueError(f"Unknown evaluation outcome: {category}")
    scores, missing, invalid = split_scores(raw_scores) if category == "judged" else ({}, [], [])
    return {
        "schema": CASE_RECORD_SCHEMA,
        "case_id": str(case_id),
        "case_hash": case_hash,
        "case_provenance": normalize_case_provenance(case_provenance),
        "outcome": {"category": category, "stage": stage, "error_type": error_type,
                    "detail": sanitize_error(error), "inferred_from_legacy": inferred},
        "research": research or {},
        "discovery": discovery or {"observed": False},
        "judge": judge or {},
        "scores": scores,
        "missing_metrics": missing,
        "invalid_metrics": invalid,
    }


def legacy_case_record(*, case_id: str, case_hash: str, case_provenance: str | None, status: str,
                       judge_scores: dict | None, error: str | None) -> dict:
    """Best-effort record for results persisted before R4 (marked as inferred)."""
    judge_scores = judge_scores or {}
    if isinstance(judge_scores.get("case_record"), dict):
        return judge_scores["case_record"]
    # "completed" = research answered, judging never recorded; "running" = research never finished.
    category = {"judged": "judged", "insufficient_evidence": "insufficient_evidence", "failed": "research_failed",
                "judge_failed": "judge_failed", "completed": "missing_assessment", "queued": "not_executed",
                }.get(status, "research_interrupted")
    engine = str(judge_scores.get("engine") or "unknown")
    judge = {"engine": engine, "model": judge_scores.get("judge_usage", {}).get("judge", {}).get("model"),
             "rubric_version": "pre-r4-unversioned"} if category == "judged" else {}
    return build_case_record(case_id=case_id, case_hash=case_hash, case_provenance=case_provenance,
                             category=category, stage=None, error=error, judge=judge,
                             raw_scores=judge_scores, inferred=True)


def not_executed_record(*, case_id: str, case_hash: str, case_provenance: str | None) -> dict:
    """Record for a dataset case that has no result row: it was never started."""
    return build_case_record(case_id=case_id, case_hash=case_hash, case_provenance=case_provenance,
                             category="not_executed", stage=None)


# --- statistics ----------------------------------------------------------------------------------
def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def bootstrap_mean_interval(values: list[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES,
                            level: float = 0.95) -> tuple[float, float] | None:
    """Seeded percentile bootstrap for the mean. ``None`` below two observations."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    alpha = (1 - level) / 2
    low = means[int(math.floor(alpha * (resamples - 1)))]
    high = means[int(math.ceil((1 - alpha) * (resamples - 1)))]
    return round(low, 4), round(high, 4)


def _rate(successes: int, total: int) -> dict:
    return {"numerator": successes, "denominator": total,
            "rate": round(successes / total, 4) if total else None,
            "ci95_wilson": list(wilson_interval(successes, total) or []) or None}


def _metric_seed(seed: int, *parts: str) -> int:
    return int(hashlib.sha256(f"{seed}|{'|'.join(parts)}".encode("utf-8")).hexdigest()[:12], 16)


# --- aggregation --------------------------------------------------------------------------------
def engine_key(judge: dict) -> tuple[str, str, str]:
    return (str(judge.get("engine") or "unknown"), str(judge.get("model") or "unknown"),
            str(judge.get("rubric_version") or "unknown"))


def build_evaluation_report(provenance: dict, records: list[dict], *, seed: int = DEFAULT_SEED) -> dict:
    """Deterministic report from run provenance and per-case records."""
    ordered = sorted(records, key=lambda record: (record["case_id"], record.get("case_hash", "")))
    total = len(ordered)
    counts = {category: 0 for category in OUTCOMES}
    for record in ordered:
        counts[record["outcome"]["category"]] += 1

    def group_count(categories: tuple[str, ...]) -> int:
        return sum(counts[category] for category in categories)

    answered = group_count(ANSWERED)
    population = {
        "total_cases": total,
        "outcomes": counts,
        "answered": answered,
        "abstained": group_count(ABSTAINED),
        "acquisition_failures": group_count(ACQUISITION_FAILURES),
        "research_failures": group_count(RESEARCH_FAILURES),
        "judge_failures": group_count(JUDGE_FAILURES),
        "not_completed": group_count(NOT_COMPLETED),
        "inferred_from_legacy": sum(bool(record["outcome"].get("inferred_from_legacy")) for record in ordered),
    }
    completion = {
        "answer_rate": _rate(answered, total),
        "abstention_rate": _rate(population["abstained"], total),
        "acquisition_failure_rate": _rate(population["acquisition_failures"], total),
        "research_failure_rate": _rate(population["research_failures"], total),
        "judge_coverage_of_answered": _rate(counts["judged"], answered),
        "judged_rate": _rate(counts["judged"], total),
        "not_completed_rate": _rate(population["not_completed"], total),
    }

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in ordered:
        if record["outcome"]["category"] == "judged":
            groups.setdefault(engine_key(record.get("judge") or {}), []).append(record)
    quality_groups = []
    for (engine, model, rubric), members in sorted(groups.items()):
        metrics = {}
        for metric in QUALITY_METRICS:
            values = [record["scores"][metric] for record in members if metric in record["scores"]]
            metrics[metric] = {
                "n": len(values),
                "missing": sum(metric in record.get("missing_metrics", []) for record in members),
                "invalid": sum(metric in record.get("invalid_metrics", []) for record in members),
                "mean": round(sum(values) / len(values), 4) if values else None,
                "ci95_bootstrap": list(bootstrap_mean_interval(
                    values, seed=_metric_seed(seed, engine, model, rubric, metric)) or []) or None,
                "small_sample": len(values) < SMALL_SAMPLE,
            }
        quality_groups.append({
            "engine": engine, "judge_model": model, "rubric_version": rubric,
            "cases": len(members), "fallback_cases": sum(bool((record.get("judge") or {}).get("fallback")) for record in members),
            "conditional_on": "cases answered by the pipeline and scored by this evaluator engine",
            "metrics": metrics,
        })
    homogeneous = len(quality_groups) == 1
    headline = quality_groups[0] if homogeneous else None

    warnings = []
    if len(quality_groups) > 1:
        warnings.append("Scores come from more than one evaluator engine, judge model or rubric version; "
                        "they are reported per group and no single headline score exists.")
    if any(group["fallback_cases"] for group in quality_groups):
        warnings.append("Some cases were scored by the fallback judge after the primary engine failed.")
    if any(metric["small_sample"] for group in quality_groups for metric in group["metrics"].values()):
        warnings.append(f"At least one metric has fewer than {SMALL_SAMPLE} scored cases; "
                        "its interval is wide and differences are inconclusive.")
    if population["inferred_from_legacy"]:
        warnings.append("Some outcomes were inferred from pre-R4 result statuses.")
    if population["not_completed"]:
        warnings.append(f"{population['not_completed']} case(s) were never executed or were interrupted; "
                        "the run is incomplete and they carry no outcome beyond that.")
    if str(provenance.get("git_sha") or "unknown") == "unknown":
        warnings.append("Git SHA unknown: reproducible code attribution is unavailable for this run.")
    unspecified = sum(record.get("case_provenance") in {"unspecified", "legacy_human_unverified"} for record in ordered)
    if unspecified:
        warnings.append(f"{unspecified} case(s) have unspecified or unverified provenance; they are not gold labels.")

    costs = _cost_summary(ordered, provenance)
    return {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "provenance": provenance,
        "population": population,
        "completion": completion,
        "quality": {
            "homogeneous": homogeneous,
            "headline": {"engine": headline["engine"], "judge_model": headline["judge_model"],
                         "rubric_version": headline["rubric_version"]} if headline else None,
            "groups": quality_groups,
            "statement": "Quality metrics are conditional on scored cases. Failures and abstentions are "
                         "excluded from them, not scored as zero; see completion for coverage.",
        },
        "discovery": _discovery_summary(ordered),
        "costs": costs,
        "warnings": warnings,
        "significance": "No significance testing is performed; compare runs only when engines, judge models, "
                        "rubric versions and dataset hashes match, and treat overlapping intervals as inconclusive.",
    }


def _discovery_summary(records: list[dict]) -> dict:
    observed = [record.get("discovery") or {} for record in records if (record.get("discovery") or {}).get("observed")]
    return {
        "cases_with_traces": len(observed),
        "evidence_outcomes": dict(sorted(_count(str(item.get("evidence_outcome")) for item in observed).items())),
        "cases_with_screening_failures": sum(bool(item.get("screening_failures")) for item in observed),
        "screening_failures": sum(int(item.get("screening_failures") or 0) for item in observed),
        "cases_with_full_text_fallback": sum(bool(item.get("full_text_fallbacks")) for item in observed),
        "pdf_fallbacks": "not_applicable (PDF ingestion is not enabled)",
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _cost_summary(records: list[dict], provenance: dict) -> dict:
    research_known, research_complete, research_missing = 0.0, 0, 0
    judge_costs: dict[str, float] = {}
    judge_lower_bounds: dict[str, float] = {}
    judge_status_counts: dict[str, int] = {}
    judge_unknown = 0
    provider_failure_totals: dict[str, int] = {}
    judge_attempts = judge_failures = judge_uncertain = 0
    for record in records:
        research = record.get("research") or {}
        if research.get("cost_usd") is not None:
            research_known += float(research["cost_usd"])
            research_complete += bool(research.get("cost_complete"))
        elif research:
            research_missing += 1
        for provider, failures in (research.get("provider_failures") or {}).items():
            provider_failure_totals[provider] = provider_failure_totals.get(provider, 0) + int(failures)
        for engine_name, usage in sorted(((record.get("judge") or {}).get("usage") or {}).items()):
            judge_summary = (usage or {}).get("judge") or {}
            judge_attempts += int(judge_summary.get("attempts") or 0)
            judge_failures += int(judge_summary.get("failures") or 0)
            judge_uncertain += int(judge_summary.get("uncertain_calls") or 0)
            # Usage summaries written before the R4 review carry no completeness status;
            # their usage_actual flag was unreliable, so their cost is unverified.
            status = str(judge_summary.get("cost_status") or "unverified_legacy")
            judge_status_counts[status] = judge_status_counts.get(status, 0) + 1
            if status in {"complete", "no_calls"} and judge_summary.get("cost_usd") is not None:
                judge_costs[engine_name] = round(judge_costs.get(engine_name, 0.0) + float(judge_summary["cost_usd"]), 8)
            else:
                judge_unknown += 1
                lower = judge_summary.get("known_cost_lower_bound_usd")
                if lower is not None:
                    judge_lower_bounds[engine_name] = round(judge_lower_bounds.get(engine_name, 0.0) + float(lower), 8)
    return {
        "research_cost_known_usd": round(research_known, 8),
        "research_cases_with_complete_cost": research_complete,
        "research_cases_without_cost": research_missing,
        # Judge costs: confirmed only where every call reported usage.
        "judge_cost_known_usd_by_engine": dict(sorted(judge_costs.items())),
        "judge_cost_lower_bound_usd_by_engine": dict(sorted(judge_lower_bounds.items())),
        "judge_usage_records_without_cost": judge_unknown,
        "judge_cost_status_counts": dict(sorted(judge_status_counts.items())),
        "judge_cost_complete": judge_unknown == 0,
        "judge_attempts": judge_attempts,
        "judge_failures": judge_failures,
        "judge_uncertain_calls": judge_uncertain,
        "research_provider_failures": dict(sorted(provider_failure_totals.items())),
        "reservation": provenance.get("cost_reservation"),
        "note": "Missing usage is never treated as zero cost. Unconfirmed judge costs are reported as lower "
                "bounds; research costs are lower bounds unless complete; reservations are budgets, not charges.",
    }


# --- rendering ----------------------------------------------------------------------------------
def render_json(report: dict, records: list[dict] | None = None) -> str:
    payload = dict(report)
    if records is not None:
        payload["cases"] = sorted(records, key=lambda record: (record["case_id"], record.get("case_hash", "")))
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _interval(values: list | None) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]" if values else "—"


def render_markdown(report: dict) -> str:
    provenance = report.get("provenance") or {}
    dataset = provenance.get("dataset") or {}
    population = report["population"]
    lines = [
        "# Evaluation report",
        "",
        f"- Report schema: `{report['schema']}`; seed `{report['seed']}`; bootstrap resamples {report['bootstrap_resamples']}",
        f"- Git SHA: `{provenance.get('git_sha', 'unknown')}`; pipeline version: `{provenance.get('pipeline_version', 'unknown')}`",
        f"- Dataset: `{dataset.get('name', 'unknown')}` (version {_fmt(dataset.get('version'))}, "
        f"content hash `{str(dataset.get('content_hash', 'unknown'))[:16]}…`, {_fmt(dataset.get('case_count'))} cases)",
        f"- Case provenance: {', '.join(f'{key}={value}' for key, value in sorted((dataset.get('case_provenance') or {}).items())) or '—'}",
        f"- Mode: `{provenance.get('mode', 'unknown')}`; started {provenance.get('started_at', '—')}",
        "",
        "## Population (every case, exactly one outcome)",
        "",
        "| Outcome | Cases |",
        "|---|---:|",
    ]
    lines += [f"| {category} | {count} |" for category, count in population["outcomes"].items()]
    lines += [f"| **total** | **{population['total_cases']}** |", "", "## Completion and coverage", "",
              "| Rate | Numerator / denominator | Rate | 95% Wilson CI |", "|---|---:|---:|---|"]
    for name, rate in report["completion"].items():
        lines.append(f"| {name} | {rate['numerator']} / {rate['denominator']} | {_fmt(rate['rate'])} | {_interval(rate['ci95_wilson'])} |")
    lines += ["", "## Quality (conditional on scored cases; per evaluator engine)", "", report["quality"]["statement"], ""]
    if not report["quality"]["groups"]:
        lines.append("No case was scored.")
    for group in report["quality"]["groups"]:
        lines += [f"### Engine `{group['engine']}` · judge `{group['judge_model']}` · rubric `{group['rubric_version']}`",
                  "", f"Scored cases: {group['cases']} (fallback: {group['fallback_cases']})", "",
                  "| Metric | n | Missing | Invalid | Mean | 95% bootstrap CI |", "|---|---:|---:|---:|---:|---|"]
        for metric, stats in group["metrics"].items():
            flag = " (small sample)" if stats["small_sample"] else ""
            lines.append(f"| {metric}{flag} | {stats['n']} | {stats['missing']} | {stats['invalid']} | "
                         f"{_fmt(stats['mean'])} | {_interval(stats['ci95_bootstrap'])} |")
        lines.append("")
    costs = report["costs"]
    lines += ["## Costs and provider failures", "",
              f"- Research cost known: ${costs['research_cost_known_usd']:.6f} "
              f"({costs['research_cases_with_complete_cost']} complete, {costs['research_cases_without_cost']} without cost)",
              f"- Judge cost confirmed by engine: {', '.join(f'{k}=${v:.6f}' for k, v in costs['judge_cost_known_usd_by_engine'].items()) or '—'}"
              f" ({costs['judge_usage_records_without_cost']} usage records without confirmed cost; lower bounds: "
              f"{', '.join(f'{k}≥${v:.6f}' for k, v in costs['judge_cost_lower_bound_usd_by_engine'].items()) or '—'})",
              f"- Judge cost status counts: {', '.join(f'{k}={v}' for k, v in costs['judge_cost_status_counts'].items()) or '—'}",
              f"- Judge attempts / failures / uncertain: {costs['judge_attempts']} / {costs['judge_failures']} / {costs['judge_uncertain_calls']}",
              f"- Research provider failures: {', '.join(f'{k}={v}' for k, v in costs['research_provider_failures'].items()) or 'none'}",
              f"- {costs['note']}", "", "## Warnings", ""]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["- None."]
    lines += ["", f"_{report['significance']}_", ""]
    return "\n".join(lines)


CSV_COLUMNS = ("case_id", "case_hash", "case_provenance", "outcome", "stage", "error_type", "inferred_from_legacy",
               "engine", "judge_model", "rubric_version", "fallback_from", *QUALITY_METRICS,
               "missing_metrics", "invalid_metrics", "evidence_outcome", "screening_failures")


def render_csv(records: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in sorted(records, key=lambda item: (item["case_id"], item.get("case_hash", ""))):
        judge = record.get("judge") or {}
        discovery = record.get("discovery") or {}
        writer.writerow({
            "case_id": record["case_id"], "case_hash": record.get("case_hash"),
            "case_provenance": record.get("case_provenance"), "outcome": record["outcome"]["category"],
            "stage": record["outcome"].get("stage"), "error_type": record["outcome"].get("error_type"),
            "inferred_from_legacy": record["outcome"].get("inferred_from_legacy"),
            "engine": judge.get("engine"), "judge_model": judge.get("model"), "rubric_version": judge.get("rubric_version"),
            "fallback_from": (judge.get("fallback") or {}).get("from_engine"),
            **{metric: record["scores"].get(metric) for metric in QUALITY_METRICS},
            "missing_metrics": ";".join(record.get("missing_metrics", [])),
            "invalid_metrics": ";".join(record.get("invalid_metrics", [])),
            "evidence_outcome": discovery.get("evidence_outcome"),
            "screening_failures": discovery.get("screening_failures"),
        })
    return buffer.getvalue()


# --- provenance ---------------------------------------------------------------------------------
def resolve_git_provenance(env: dict | None = None, repo_root: str | None = None) -> dict:
    """Commit of the running code and where it came from; never guessed.

    Sources, in order: ``SCIENTIFIC_RAG_GIT_SHA`` (image build argument ``GIT_SHA``),
    then the local ``.git`` checkout. ``GITHUB_SHA`` is deliberately not used: in CI it
    can name a different commit (for example a pull request's merge ref) than the code
    actually built. If nothing is available the SHA stays ``unknown``.
    """
    import os
    from pathlib import Path

    env = os.environ if env is None else env
    value = str(env.get("SCIENTIFIC_RAG_GIT_SHA") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return {"sha": value, "source": "build_arg"}
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        sha = None
        if head.startswith("ref: "):
            ref = head[5:]
            ref_file = root / ".git" / ref
            if ref_file.exists():
                sha = ref_file.read_text(encoding="utf-8").strip()
            else:
                packed = root / ".git" / "packed-refs"
                if packed.exists():
                    sha = next((line.split(" ", 1)[0] for line in packed.read_text(encoding="utf-8").splitlines()
                                if line.endswith(" " + ref)), None)
        else:
            sha = head
        if sha and re.fullmatch(r"[0-9a-f]{40}", sha):
            # A working tree may have uncommitted changes; .git cannot tell us.
            return {"sha": sha, "source": "local_git_checkout_uncommitted_changes_not_detected"}
    except OSError:
        pass
    return {"sha": "unknown", "source": "unavailable"}


def resolve_git_sha(env: dict | None = None, repo_root: str | None = None) -> str:
    return resolve_git_provenance(env, repo_root)["sha"]


def library_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_run_provenance(*, run_id: str, dataset_id: str, dataset_name: str, case_hashes: list[str],
                         case_provenance: list[str], mode: str, pipeline_version: str,
                         semantic_verification_mode: str, judge_model: str, embedding_model: str,
                         prompt_versions: dict, ragas_rubric_version: str, fallback_chain: list[str],
                         timeouts: dict, started_at: str, cost_reservation: dict | None,
                         git_sha: str | None = None) -> dict:
    counts: dict[str, int] = {}
    for value in case_provenance:
        key = normalize_case_provenance(value)
        counts[key] = counts.get(key, 0) + 1
    git = {"sha": git_sha, "source": "explicit"} if git_sha else resolve_git_provenance()
    return {
        "report_schema": REPORT_SCHEMA,
        "evaluation_run_id": run_id,
        "git_sha": git["sha"],
        "git_sha_source": git["source"],
        "pipeline_version": pipeline_version,
        "semantic_verification_mode": semantic_verification_mode,
        "mode": mode,
        "dataset": {
            "id": dataset_id, "name": dataset_name,
            "version": None,  # not stored by the dataset schema; identify the dataset by content_hash
            "content_hash": dataset_content_hash(case_hashes),
            "case_count": len(case_hashes),
            "case_provenance": dict(sorted(counts.items())),
        },
        "generator": {"source": "per-case llm_gateway traces (see case records)", "providers_models": []},
        "judge": {
            "engines": list(fallback_chain),
            "fallback_chain": list(fallback_chain),
            "ragas_model": judge_model,
            "fallback_model": judge_model,
            "embedding_model": embedding_model,
            "ragas_version": library_version("ragas"),
            "ragas_rubric_version": ragas_rubric_version,
            "prompt_versions": dict(sorted(prompt_versions.items())),
        },
        "profile": {"timeouts_seconds": dict(sorted(timeouts.items()))},
        "started_at": started_at,
        "completed_at": None,
        "cost_reservation": cost_reservation,
    }
