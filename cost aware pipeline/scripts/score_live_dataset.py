"""Score completed runs from a live-dataset research output incrementally."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.db import close_database
from scripts.score_saved_run import score


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def main_async(args) -> None:
    research = json.loads(args.research.read_text(encoding="utf-8"))
    eligible = [row for row in research["results"] if row.get("status") == "completed" and row.get("run_id")]
    report = {
        "research_file": str(args.research),
        "dataset": str(args.dataset),
        "research_case_count": len(research["results"]),
        "completed_research_count": len(eligible),
        "insufficient_evidence_count": sum(row.get("status") == "insufficient_evidence" for row in research["results"]),
        "judge_reservation_usd": round(len(eligible) * 0.10, 2),
        "results": [],
    }
    for index, row in enumerate(eligible, start=1):
        print(f"[{index}/{len(eligible)}] scoring {row['case_id']}", flush=True)
        try:
            result = await score(row["run_id"], args.dataset)
            result["case_id"] = row["case_id"]
            result["source_group"] = row.get("source_group")
            report["results"].append({"status": "scored", **result})
        except Exception as error:
            report["results"].append({
                "status": "failed", "case_id": row["case_id"],
                "run_id": row["run_id"], "error": f"{type(error).__name__}: {error}",
            })
        write(args.output, report)

    scored = [row for row in report["results"] if row["status"] == "scored"]
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    report["summary"] = {
        "scored_count": len(scored),
        "judge_failure_count": len(report["results"]) - len(scored),
        "research_completion_rate": len(eligible) / len(research["results"]) if research["results"] else 0.0,
        **{
            metric: sum(row["scores"][metric] for row in scored) / len(scored) if scored else None
            for metric in metric_names
        },
    }
    write(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


async def run_and_close(args) -> None:
    try:
        await main_async(args)
    finally:
        await close_database()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_and_close(args))


if __name__ == "__main__":
    main()
