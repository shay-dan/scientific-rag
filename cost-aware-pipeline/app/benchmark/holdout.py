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
# "nums" (list of numbers). A dict spec maps allowed keys to specs; "*" allows any
# identifier-like key with the given spec. Anything not allowed is dropped.
_ID = re.compile(r"^[A-Za-z0-9_.:+@/\-]{1,96}$")
_RATE = {"numerator": "num", "denominator": "num", "rate": "num", "ci95_wilson": "nums"}
_MEAN = {"n": "num", "mean": "num", "ci95_bootstrap": "nums", **_RATE}
_CLASS = {key: "num" for key in ("true_positives", "predicted", "support_assessed", "support_all", "precision",
                                 "recall", "recall_all_items", "f1", "f1_all_items")}
VERIFIER_ALLOWLIST = {
    "schema": "id",
    "provenance": {
        "dataset": {"dataset_id": "id", "version": "id", "content_hash": "id", "items": "num", "answers": "num",
                    "label_provenance": {"*": "num"}, "reviewer_status": {"*": "num"},
                    "label_mapping_version": "id"},
        "predictor": {"name": "id", "kind": "id", "version": "id", "provider": "id", "model": "id"},
        "predictions_hash": "id", "git_sha": "id", "git_sha_source": "id", "label_scheme": "id",
        "simulation_policy": "id",
    },
    "population": {"items": "num", "outcomes": {"*": "num"}, "unknown_item_predictions": "num",
                   "unattributed_predictions": "num", "prediction_records": "num"},
    "coverage": {"*": _RATE},
    "confusion_matrix": {"columns": "ids", "counts": {"*": {"*": "num"}}},
    "classes": {"*": _CLASS},
    "summary": {"accuracy_assessed": _RATE, "macro_f1": "num", "macro_f1_all_items": "num",
                "balanced_accuracy": "num", "balanced_accuracy_all_items": "num", "classes_in_macro": "ids",
                "classes_in_macro_all_items": "ids", "classes_without_expected_items": "ids"},
    "safety": {"*": "num", "unsupported_false_acceptance": _RATE, "critical_contradiction_acceptance": _RATE,
               "supported_false_rejection": _RATE},
    "by_error_category": {"*": {"items": "num", "assessed": "num", "accuracy": _RATE, "predicted_supported": _RATE}},
    "by_evidence_type": {"*": {"items": "num", "accuracy": _RATE}},
    "enforcement_simulation": {"policy": "id", "answers": "num", "outcomes": {"*": "num"},
                               "oracle_outcomes": {"*": "num"}, "totals": {"*": "num"}, "*": _RATE},
    "cost": {"cost_status": "id", "cost_usd": "num", "known_cost_lower_bound_usd": "num"},
}
RETRIEVAL_ALLOWLIST = {
    "schema": "id", "split": "id",
    "provenance": {"benchmark_id": "id", "benchmark_version": "id", "pack_id": "id", "pack_hash": "id",
                   "pack_identity": "id", "run_id": "id", "run_hash": "id", "seed": "num", "ks": "nums",
                   "system": {"name": "id", "version": "id", "kind": "id"}},
    "population": {"questions": "num", "evaluable": "num", "result_status": {"*": "num"},
                   "unknown_result_questions": "num", "unknown_ranked_evidence": "num"},
    "aggregates": {"*": {"questions": "num", "by_k": {"*": {"*": _MEAN}}}},
}
ALLOWLISTS = {"scientific-rag-verifier-report/v1": VERIFIER_ALLOWLIST,
              "scientific-rag-retrieval-report/v1": RETRIEVAL_ALLOWLIST}


def _allow(value, spec):
    if spec == "num":
        return value if value is None or isinstance(value, (bool, int, float)) else _DROP
    if spec == "id":
        return value if value is None or (isinstance(value, str) and _ID.match(value)) else _DROP
    if spec in ("ids", "nums"):
        if not isinstance(value, list):
            return _DROP
        kept = [item for item in (_allow(item, spec[:-1] if spec == "nums" else "id") for item in value)]
        return _DROP if any(item is _DROP for item in kept) else kept
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            return _DROP
        out = {}
        for key, item in value.items():
            sub = spec.get(key, spec.get("*") if isinstance(key, str) and _ID.match(key) else None)
            if sub is None:
                continue
            filtered = _allow(item, sub)
            if filtered is not _DROP:
                out[key] = filtered
        return out
    return _DROP


_DROP = object()


def aggregate_only(report: dict) -> dict:
    """Owner-safe summary built from the allowlist for the report's schema.

    Unknown report schemas yield only the schema name. Fields not on the
    allowlist, free-text strings and per-item rows are never copied.
    """
    spec = ALLOWLISTS.get(report.get("schema")) if isinstance(report, dict) else None
    if spec is None:
        return {"schema": report.get("schema") if isinstance(report, dict) else None, "note": "unknown report schema"}
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
