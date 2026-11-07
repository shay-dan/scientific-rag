"""Plan and validate the Naive/Hybrid/Agentic RAG comparison.

The command is intentionally dry-run by default. It does not create an
evaluation job and does not call an LLM. Use ``--estimate-only`` for the cost
sheet and ``--validate`` for a supplied 100-case frozen dataset. The paid
execution adapter will be enabled only after the dataset passes validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.comparative_benchmark import estimate_comparative_plan, validate_comparative_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Frozen 100-case JSON dataset")
    parser.add_argument("--case-count", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--estimate-only", action="store_true", help="Print a provider-free upper-bound cost estimate")
    parser.add_argument("--no-judge", action="store_true", help="Exclude answer-judge calls from the estimate")
    parser.add_argument("--validate", action="store_true", help="Validate the supplied dataset without making paid calls")
    parser.add_argument("--output", type=Path, help="Write the plan JSON to this path")
    args = parser.parse_args()

    report = {"comparison": {
        "variants": ["naive", "hybrid_basic", "hybrid_enhanced", "agentic_fast"],
        "answer_model_policy": "same configured primary generation route where possible",
        "retrieval_model_policy": "same embedding model and frozen corpus",
        "human_review": False,
        "ground_truth_policy": "manual or independently verified source excerpts are required; no invented references",
    }}
    if args.estimate_only or not args.dataset:
        report["cost_plan"] = estimate_comparative_plan(
            case_count=args.case_count,
            repetitions=args.repetitions,
            judge=not args.no_judge,
        )

    if args.dataset:
        payload = json.loads(args.dataset.read_text(encoding="utf-8"))
        errors = validate_comparative_dataset(payload, required_cases=args.case_count)
        report["dataset"] = {"path": str(args.dataset), "case_count": len(payload.get("cases", [])), "errors": errors, "valid": not errors}
        if not errors and args.validate:
            report["dataset"]["status"] = "ready_for_execution_adapter"
        elif errors:
            report["dataset"]["status"] = "blocked_until_fixed"

    if not args.estimate_only and args.dataset is None:
        report["execution"] = {
            "status": "not_started",
            "reason": "A 100-case frozen dataset with verified references is required before paid execution.",
        }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"plan={args.output}")
    dataset_errors = report.get("dataset", {}).get("errors", [])
    return 2 if dataset_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

