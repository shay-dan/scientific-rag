"""Run the tested four-arm comparison against one materialized frozen corpus."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.benchmark_dataset import validate_pilot_dataset
from app.comparative_benchmark import RAGVariant, recall_at_k
from app.config import get_settings
from app.db import session_factory
from app.real_comparison import execute_variant


def _write_checkpoint(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_cases(dataset: dict, *, case_limit: int | None = None, case_ids: str | None = None) -> list[dict]:
    cases = list(dataset["cases"])
    if case_ids:
        requested = [item.strip() for item in case_ids.split(",") if item.strip()]
        by_id = {case["id"]: case for case in cases}
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown case ids: {missing}")
        return [by_id[case_id] for case_id in requested]
    return cases[:case_limit] if case_limit else cases


async def run(args) -> int:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    errors = validate_pilot_dataset(dataset)
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, indent=2))
        return 2
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("research_run_id"):
        print("Frozen corpus has not been materialized. Run build_frozen_benchmark_snapshot.py --materialize after tests pass.")
        return 2
    cases = select_cases(dataset, case_limit=args.case_limit, case_ids=args.case_ids)
    variants = [RAGVariant(item) for item in args.variants.split(",")]
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists() else {
        "experiment": "real-frozen-12-comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_hash": manifest["snapshot_hash"],
        "snapshot_run_id": manifest["research_run_id"],
        "case_count": len(cases),
        "variants": [item.value for item in variants],
        "results": [],
    }
    completed_keys = {(row["case_id"], row["variant"]) for row in report["results"] if row.get("status") == "completed"}
    settings = get_settings()
    for case in cases:
        for variant in variants:
            key = (case["id"], variant.value)
            if key in completed_keys:
                continue
            print(f"running {case['id']} {variant.value}", flush=True)
            try:
                async with session_factory() as session:
                    result = await execute_variant(session, settings, UUID(manifest["research_run_id"]), case["question"], variant)
                expected = list(dict.fromkeys(row["source_id"] for row in case["reference_context"]))
                result.update({
                    "case_id": case["id"],
                    "question": case["question"],
                    "expected_source_count": len(expected),
                    "recall_at_5": recall_at_k(result["selected_source_ids"], expected, k=5),
                })
            except Exception as error:
                result = {"case_id": case["id"], "question": case["question"], "variant": variant.value, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            report["results"] = [row for row in report["results"] if (row["case_id"], row["variant"]) != key]
            report["results"].append(result)
            _write_checkpoint(args.output, report)
    completed = [row for row in report["results"] if row.get("status") == "completed"]
    costs = [row.get("estimated_cost_usd") for row in completed]
    report["summary"] = {
        "planned_runs": len(cases) * len(variants),
        "completed_runs": len(completed),
        "failed_runs": len(report["results"]) - len(completed),
        "known_total_cost_usd": round(sum(costs), 8) if costs and all(value is not None for value in costs) else None,
        "cost_complete": bool(costs) and all(value is not None for value in costs),
    }
    _write_checkpoint(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed_runs"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation" / "frozen-12-corpus-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "real-frozen-12-comparison.json")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--case-ids", help="Comma-separated explicit case IDs; preferred for representative smoke tests")
    parser.add_argument("--variants", default=",".join(item.value for item in RAGVariant))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
