"""R6A benchmark-integration infrastructure (offline, standard library only).

Not imported by the API, worker or research pipeline. Modules:
  contracts   record schemas, validation, gold eligibility, hashing
  adapters    JSONL/CSV/JSON loading via declarative adapters; manifest file hashes
  splits      split integrity and question/paper/near-duplicate leakage checks
  retrieval   frozen evidence packs, lexical baseline, retrieval-only metrics
  bridges     offline synthesis replay inputs; conversion to the R5 verifier dataset
  discovery   discovery-run contract and stage-by-stage evaluation (no R9-D changes)
  holdout     owner-controlled sealed-holdout path (commitment, aggregate-only, log)
"""
from __future__ import annotations

import json

from ..evaluation_report import resolve_git_provenance
from .contracts import RECORD_TYPES, Benchmark, Issue, benchmark_hash, label_summary, validate_benchmark
from .splits import check_splits, split_summary

VALIDATION_SCHEMA = "scientific-rag-benchmark-validation/v1"


def validation_report(benchmark: Benchmark, load_issues: list[Issue] | None = None, *,
                      allow_holdout: bool = False, git_provenance: dict | None = None) -> dict:
    issues = sorted(set((load_issues or []) + validate_benchmark(benchmark, allow_holdout=allow_holdout)
                        + check_splits(benchmark)))
    errors = [issue for issue in issues if issue.severity == "error"]
    manifest = benchmark.manifest
    return {
        "schema": VALIDATION_SCHEMA,
        "benchmark": {"benchmark_id": manifest.get("benchmark_id"), "version": manifest.get("version"),
                      "curation": manifest.get("curation"), "holdout_commitment": (manifest.get("holdout") or {}).get("commitment")},
        "status": "invalid" if errors else "valid",
        "hashes": benchmark_hash(benchmark),
        "counts": {record_type: len(benchmark.of(record_type)) for record_type in RECORD_TYPES},
        "splits": split_summary(benchmark),
        "labels": label_summary(benchmark),
        "issue_counts": {"errors": len(errors), "warnings": len(issues) - len(errors)},
        "issues": [issue.as_dict() for issue in issues],
        "git": git_provenance or resolve_git_provenance(),
    }


def render_json(report: dict) -> str:
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def render_validation_markdown(report: dict) -> str:
    benchmark, labels = report["benchmark"], report["labels"]
    lines = [f"# Benchmark validation: {report['status'].upper()}", "",
             f"- Benchmark `{benchmark['benchmark_id']}` version `{benchmark['version']}`; content hash "
             f"`{report['hashes']['content_hash'][:16]}…`",
             f"- Records: {', '.join(f'{k}={v}' for k, v in report['counts'].items())}",
             f"- Questions by split: {report['splits']['questions_by_split']}",
             f"- Claims: {labels['claims']['gold_eligible']} gold-eligible, {labels['claims']['provisional']} provisional; "
             f"relevance: {labels['relevance']['gold_eligible']} gold-eligible, {labels['relevance']['provisional']} provisional",
             f"- Issues: {report['issue_counts']['errors']} errors, {report['issue_counts']['warnings']} warnings", "",
             "| Severity | Code | Record | ID | Message |", "|---|---|---|---|---|"]
    lines += [f"| {i['severity']} | {i['code']} | {i['record_type']} | {i['record_id']} | {i['message']} |"
              for i in report["issues"]] or ["| — | — | — | — | — |"]
    lines += ["", "_Gold-eligible means human_reviewed and reviewed/adjudicated; nothing else is promoted._", ""]
    return "\n".join(lines)
