"""R6A: owner-controlled sealed-holdout execution path.

The sealed holdout never enters the repository or an implementation agent's
workspace. The public manifest carries only a *commitment* (a SHA-256 over the
holdout files' hashes) so anyone can later check that the holdout used is the one
that was frozen. The owner runs ``scripts/benchmark_holdout_owner.py`` on a machine
holding the files; it:

1. refuses a holdout directory inside the repository working tree;
2. verifies the files against the commitment before any evaluation;
3. evaluates a predictions/run file produced by the system under test;
4. emits **aggregate metrics only** (no question text, claim text, labels or
   per-item rows), and
5. appends an entry to a holdout evaluation log (every holdout use is logged).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..evaluation_report import resolve_git_provenance, sha256_hex
from .adapters import file_sha256, load_benchmark

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


def aggregate_only(report: dict) -> dict:
    """Strip every per-item or per-question field from a verifier or retrieval report."""
    redacted = json.loads(json.dumps(report))
    redacted.pop("per_question", None)
    simulation = redacted.get("enforcement_simulation")
    if isinstance(simulation, dict):
        simulation.pop("per_answer", None)
    population = redacted.get("population")
    if isinstance(population, dict):
        population.pop("without_relevant_labels", None)
    return redacted


def run_owner_holdout(*, holdout_dir: Path, expected_commitment: str, repo_root: Path, evaluate,
                      log_path: Path, evaluator: str, run_label: str) -> dict:
    """Verify, evaluate and log one holdout use. ``evaluate(benchmark) -> report``."""
    check_location(holdout_dir, repo_root)
    actual = holdout_commitment(holdout_dir)
    if actual != expected_commitment:
        raise HoldoutRefused("holdout files do not match the published commitment")
    benchmark, load_issues = load_benchmark(holdout_dir / "manifest.json")
    if any(issue.severity == "error" for issue in load_issues):
        raise HoldoutRefused("holdout files failed to load: " + "; ".join(i.code for i in load_issues))
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
