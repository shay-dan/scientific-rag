"""Regenerate an evaluation report offline from an exported evaluation run (R4).

Input: the JSON from ``GET /ui/evaluations/{id}/export`` (its ``report.cases``) or
from ``GET /v1/evaluations/runs/{id}`` (rebuilt from ``results``). No model,
database or network call is made; identical input and seed give identical output.

Usage:
    python scripts/build_evaluation_report.py export.json --out-dir reports/run-x [--seed N]
Writes report.json, report.md and cases.csv.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation_report import (  # noqa: E402
    DEFAULT_SEED,
    build_evaluation_report,
    legacy_case_record,
    render_csv,
    render_json,
    render_markdown,
)


def load_records(payload: dict) -> tuple[dict, list[dict]]:
    report = payload.get("report") or {}
    records = report.get("cases")
    if records is None:
        records = []
        for index, item in enumerate(payload.get("results", []), start=1):
            scores = item.get("scores") or {}
            records.append(scores["case_record"] if isinstance(scores.get("case_record"), dict) else legacy_case_record(
                case_id=str(item.get("case_id") or f"case-{index:04d}"), case_hash="unknown",
                case_provenance=None, status=str(item.get("status")), judge_scores=scores, error=item.get("error")))
    provenance = report.get("provenance") or {"legacy_run": True, "evaluation_run_id": payload.get("id"),
                                               "mode": payload.get("mode")}
    expected = (report.get("population") or {}).get("total_cases")
    if expected is not None and expected != len(records):
        raise SystemExit(f"Export has {len(records)} case records but the run population is {expected}; "
                         "use the UI export, which includes every dataset case.")
    return provenance, list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    provenance, records = load_records(json.loads(args.export.read_text(encoding="utf-8")))
    report = build_evaluation_report(provenance, records, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(render_json(report, records), encoding="utf-8")
    (args.out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (args.out_dir / "cases.csv").write_text(render_csv(records), encoding="utf-8")
    print(json.dumps({"cases": len(records), "outcomes": report["population"]["outcomes"],
                      "out_dir": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
