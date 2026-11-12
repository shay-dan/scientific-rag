"""Evaluate a semantic-verifier predictor offline (R5). Never calls a model.

Usage:
    python scripts/evaluate_verifier.py --dataset DATASET.json --baseline structural_only --out-dir OUT
    python scripts/evaluate_verifier.py --dataset DATASET.json --predictions PREDICTIONS.json --out-dir OUT

Writes report.json, report.md and errors.csv (misclassified and unassessed items).
Identical inputs give byte-identical outputs. Pass --git-sha to pin the recorded
commit; otherwise it is resolved as in R4 (build argument or local .git, else unknown).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.verifier_evaluation import (  # noqa: E402
    BASELINES,
    evaluate,
    render_errors_csv,
    render_json,
    render_markdown,
    validate_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--baseline", choices=sorted(BASELINES))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--git-sha", help="40-character commit SHA to record (default: resolve)")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.baseline:
        predictions = BASELINES[args.baseline](validate_dataset(dataset))
    else:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    git = {"sha": args.git_sha, "source": "explicit"} if args.git_sha else None
    report = evaluate(dataset, predictions, git_provenance=git)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(render_json(report), encoding="utf-8")
    (args.out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (args.out_dir / "errors.csv").write_text(render_errors_csv(dataset, predictions), encoding="utf-8")
    print(json.dumps({"items": report["population"]["items"], "outcomes": report["population"]["outcomes"],
                      "macro_f1": report["summary"]["macro_f1"], "out_dir": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
