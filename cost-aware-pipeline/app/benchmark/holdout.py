"""R6A: owner-controlled sealed-holdout execution path.

The sealed holdout never enters the repository or an implementation agent's
workspace. The public manifest carries only a *commitment* (a SHA-256 over the
holdout files' hashes) so anyone can later check that the holdout used is the one
that was frozen. The owner runs ``scripts/benchmark_holdout_owner.py`` on a machine
holding the files; it:

1. refuses a holdout directory inside the repository working tree;
2. verifies the files against the commitment before any evaluation;
3. evaluates a predictions/run file produced by the system under test;
4. runs full contract, reference, label-mapping and split validation and refuses
   to score if any error is found (only issue codes and counts are reported);
5. builds the output from an explicit **allowlist** of aggregate metric and
   provenance fields: unknown fields, free text and per-item rows are dropped
   wherever they appear, so no question text, claim text, source label or
   per-item evaluation data can leave, and
6. appends an entry to a holdout evaluation log (every scored holdout use is logged).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import re

from ..evaluation_report import resolve_git_provenance, sha256_hex
from .adapters import file_sha256, load_benchmark
from .contracts import validate_benchmark
from .splits import check_splits

LOG_SCHEMA = "scientific-rag-holdout-log/v1"


class HoldoutRefused(RuntimeError):
    pass


def holdout_commitment(holdout_dir: Path, manifest_name: str = "manifest.json") -> str:
    """SHA-256 over the sorted (relative path, file hash) pairs of every holdout file."""
    root = holdout_dir.resolve()
    pairs = sorted((str(path.relative_to(root)), file_sha256(path)) for path in root.rglob("*") if path.is_file())
    if not any(name == manifest_name for name, _ in pairs):
        raise HoldoutRefused(f"holdout directory has no {manifest_name}")
    return sha256_hex(pairs)


def check_location(holdout_dir: Path, repo_root: Path) -> None:
    holdout, repo = holdout_dir.resolve(), repo_root.resolve()
    if holdout == repo or repo in holdout.parents:
        raise HoldoutRefused("the sealed holdout must live outside the repository working tree")


# --- allowlist output -----------------------------------------------------------------------
# Leaf kinds: "num" (int/float/bool/None), "id" (short identifier string), "ids" (list of ids),
# "nums" (list of numbers). A dict spec maps allowed keys to specs. Dynamic keys are allowed
# only through ``Keys``: a fixed vocabulary of harness-defined names (verdicts, outcomes,
# metric names, evidence types, approved error categories) or a digit-only K. Report-supplied
# keys outside the vocabulary are dropped, so no content can be encoded in a key.
from ..verifier_evaluation import NO_VALID, PREDICTION_OUTCOMES, VERDICTS
from .contracts import EVIDENCE_TYPES, LABEL_PROVENANCE, REVIEWER_STATUS

_ID = re.compile(r"^[A-Za-z0-9_.:+@/\-]{1,96}$")
_K = re.compile(r"^[1-9][0-9]{0,3}$")


class Keys:
    """Dict spec whose keys must come from a fixed vocabulary (or match a strict pattern)."""

    def __init__(self, allowed, spec):
        self.allowed, self.spec = allowed, spec

    def permits(self, key) -> bool:
        if not isinstance(key, str):
            return False
        return bool(self.allowed.match(key)) if isinstance(self.allowed, re.Pattern) else key in self.allowed


# Error categories are curator-defined; only this approved vocabulary is reported from a
# holdout. Others are withheld (their items still count in every other aggregate).
APPROVED_ERROR_CATEGORIES = frozenset({
    "none", "numerical", "causal", "population", "negation", "scope", "contradiction", "dosage",
    "temporal", "entity", "comparison", "hedging", "citation", "other",
})
_RATE = {"numerator": "num", "denominator": "num", "rate": "num", "ci95_wilson": "nums"}
_MEAN = {"n": "num", "mean": "num", "ci95_bootstrap": "nums", **_RATE}
_CLASS = {key: "num" for key in ("true_positives", "predicted", "support_assessed", "support_all", "precision",
                                 "recall", "recall_all_items", "f1", "f1_all_items")}
_COVERAGE = ("assessed_rate", "missing_assessment_rate", "invalid_output_rate", "duplicate_assessment_rate",
             "provider_failure_rate")
_ANSWER_OUTCOMES = ("complete_answer", "partial_answer", "no_supported_claims", "failed_closed")
_SIM_TOTALS = ("supported_retained", "supported_claims_retained", "supported_limitations_retained",
               "unsupported_retained", "critical_contradictions_retained", "supported_removed",
               "supported_lost_to_fail_closed", "essential_lost", "claims_removed", "limitations_removed",
               "targets_lost_to_fail_closed")
_SIM_RATES = ("claim_coverage", "supported_target_coverage", "answer_coverage", "answers_without_usable_claims",
              "incomplete_answer_rate", "answers_losing_oracle_support", "unsupported_retention")
_RESULT_STATUS = ("ok", "missing", "failed", "duplicate")
_RETRIEVAL_METRICS = ("paper_recall", "paper_precision", "paper_ndcg", "passage_recall", "passage_precision",
                      "paper_success")
VERIFIER_ALLOWLIST = {
    "schema": "id",
    "provenance": {
        "dataset": {"dataset_id": "id", "version": "id", "content_hash": "id", "items": "num", "answers": "num",
                    "label_provenance": Keys(frozenset(LABEL_PROVENANCE), "num"),
                    "reviewer_status": Keys(frozenset(REVIEWER_STATUS), "num"),
                    "label_mapping_version": "id"},
        "predictor": {"name": "id", "kind": "id", "version": "id", "provider": "id", "model": "id"},
        "predictions_hash": "id", "git_sha": "id", "git_sha_source": "id", "label_scheme": "id",
        "simulation_policy": "id",
    },
    "population": {"items": "num", "outcomes": Keys(frozenset(PREDICTION_OUTCOMES), "num"),
                   "unknown_item_predictions": "num", "unattributed_predictions": "num", "prediction_records": "num"},
    "coverage": {name: _RATE for name in _COVERAGE},
    "confusion_matrix": {"columns": "ids",
                         "counts": Keys(frozenset(VERDICTS), Keys(frozenset((*VERDICTS, NO_VALID)), "num"))},
    "classes": Keys(frozenset(VERDICTS), _CLASS),
    "summary": {"accuracy_assessed": _RATE, "macro_f1": "num", "macro_f1_all_items": "num",
                "balanced_accuracy": "num", "balanced_accuracy_all_items": "num", "classes_in_macro": "ids",
                "classes_in_macro_all_items": "ids", "classes_without_expected_items": "ids"},
    "safety": {"unsupported_false_acceptance": _RATE, "critical_contradiction_acceptance": _RATE,
               "supported_false_rejection": _RATE, "non_supported_without_valid_prediction": "num",
               "supported_without_valid_prediction": "num"},
    "by_error_category": Keys(APPROVED_ERROR_CATEGORIES,
                              {"items": "num", "assessed": "num", "accuracy": _RATE, "predicted_supported": _RATE}),
    "by_evidence_type": Keys(frozenset((*EVIDENCE_TYPES, "mixed", "unspecified")), {"items": "num", "accuracy": _RATE}),
    "enforcement_simulation": {"policy": "id", "answers": "num",
                               "outcomes": Keys(frozenset(_ANSWER_OUTCOMES), "num"),
                               "oracle_outcomes": Keys(frozenset(_ANSWER_OUTCOMES), "num"),
                               "totals": Keys(frozenset(_SIM_TOTALS), "num"),
                               **{name: _RATE for name in _SIM_RATES}},
    "cost": {"cost_status": "id", "cost_usd": "num", "known_cost_lower_bound_usd": "num"},
}
RETRIEVAL_ALLOWLIST = {
    "schema": "id", "split": "id",
    "provenance": {"benchmark_id": "id", "benchmark_version": "id", "pack_id": "id", "pack_hash": "id",
                   "pack_identity": "id", "run_id": "id", "run_hash": "id", "seed": "num", "ks": "nums",
                   "system": {"name": "id", "version": "id", "kind": "id"}},
    "population": {"questions": "num", "evaluable": "num", "result_status": Keys(frozenset(_RESULT_STATUS), "num"),
                   "unknown_result_questions": "num", "unknown_ranked_evidence": "num"},
    "aggregates": Keys(frozenset(("completed", "all_questions")),
                       {"questions": "num", "by_k": Keys(_K, Keys(frozenset(_RETRIEVAL_METRICS), _MEAN))}),
}
ALLOWLISTS = {"scientific-rag-verifier-report/v1": VERIFIER_ALLOWLIST,
              "scientific-rag-retrieval-report/v1": RETRIEVAL_ALLOWLIST}
UNKNOWN_SCHEMA = {"schema": "unknown", "note": "unknown report schema"}
_DROP = object()


def _allow(value, spec):
    if spec == "num":
        return value if value is None or isinstance(value, (bool, int, float)) else _DROP
    if spec == "id":
        return value if value is None or (isinstance(value, str) and _ID.match(value)) else _DROP
    if spec in ("ids", "nums"):
        if not isinstance(value, list):
            return _DROP
        kept = [_allow(item, "num" if spec == "nums" else "id") for item in value]
        return _DROP if any(item is _DROP for item in kept) else kept
    if isinstance(spec, (dict, Keys)):
        if not isinstance(value, dict):
            return _DROP
        out = {}
        for key, item in value.items():
            if isinstance(spec, Keys):
                if not spec.permits(key):
                    continue
                sub = spec.spec
            else:
                sub = spec.get(key) if isinstance(key, str) else None
                if sub is None:
                    continue
            filtered = _allow(item, sub)
            if filtered is not _DROP:
                out[key] = filtered
        return out
    return _DROP


def aggregate_only(report: dict) -> dict:
    """Owner-safe summary built from the allowlist for the report's schema.

    An unrecognized schema yields a fixed value; the report's own schema string is
    never echoed. Fields and keys not on the allowlist, free-text strings and per-item
    rows are never copied.
    """
    schema = report.get("schema") if isinstance(report, dict) else None
    spec = ALLOWLISTS.get(schema) if isinstance(schema, str) else None
    if spec is None:
        return dict(UNKNOWN_SCHEMA)
    return _allow(report, spec)


def validate_holdout(benchmark) -> list:
    """Full contract, reference, label-mapping and split validation of a holdout bundle."""
    issues = validate_benchmark(benchmark, allow_holdout=True) + check_splits(benchmark)
    from .contracts import Issue

    for question in benchmark.of("question"):
        if isinstance(question, dict) and question.get("split") != "holdout":
            issues.append(Issue("error", "holdout_non_holdout_question", "question", str(question.get("question_id")),
                                "a holdout bundle may contain only holdout questions"))
    return sorted(set(issues))


def run_owner_holdout(*, holdout_dir: Path, expected_commitment: str, repo_root: Path, evaluate,
                      log_path: Path, evaluator: str, run_label: str) -> dict:
    """Verify, evaluate and log one holdout use. ``evaluate(benchmark) -> report``."""
    check_location(holdout_dir, repo_root)
    actual = holdout_commitment(holdout_dir)
    if actual != expected_commitment:
        raise HoldoutRefused("holdout files do not match the published commitment")
    benchmark, load_issues = load_benchmark(holdout_dir / "manifest.json")
    errors = [issue for issue in load_issues + validate_holdout(benchmark) if issue.severity == "error"]
    if errors:
        codes: dict[str, int] = {}
        for issue in errors:
            codes[issue.code] = codes.get(issue.code, 0) + 1
        # Only issue codes and counts: no record IDs or content leave the owner path.
        raise HoldoutRefused("holdout failed validation; nothing was scored: "
                             + ", ".join(f"{code}×{count}" for code, count in sorted(codes.items())))
    summary = aggregate_only(evaluate(benchmark))
    entry = {
        "schema": LOG_SCHEMA,
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluator": evaluator, "run_label": run_label,
        "holdout_commitment": actual,
        "benchmark_id": benchmark.manifest.get("benchmark_id"), "benchmark_version": benchmark.manifest.get("version"),
        "git": resolve_git_provenance(),
        "report_schema": summary.get("schema"),
        "summary_hash": sha256_hex(summary),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"log_entry": entry, "summary": summary}
