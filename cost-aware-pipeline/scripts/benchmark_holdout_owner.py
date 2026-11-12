"""Owner-only: evaluate the sealed holdout outside the repository (R6A).

Run by the evaluation owner on a machine that holds the holdout. The holdout
directory must be outside the repository and match the commitment published in the
public manifest. Output is aggregate-only, and every use is appended to a log.

    python scripts/benchmark_holdout_owner.py \\
        --holdout-dir /secure/holdout --commitment <sha256> \\
        --kind verifier --predictions predictions.json \\
        --log /secure/holdout-log.jsonl --evaluator "<name>" --run-label "<label>" --out summary.json

Kinds: verifier (R5 predictions file) or retrieval (retrieval run file).
Implementation agents must not run this script or receive holdout files.
"""
import argparse
import json
from pathlib import Path
import sys

_app_root = Path(__file__).resolve().parents[1]
# The git checkout root when run from a clone; the application directory inside an image.
repo_root = _app_root.parent if (_app_root.parent / ".git").exists() else _app_root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import render_json  # noqa: E402
from app.benchmark.bridges import to_verifier_dataset  # noqa: E402
from app.benchmark.holdout import run_owner_holdout  # noqa: E402
from app.benchmark.retrieval import build_evidence_pack, evaluate_retrieval  # noqa: E402
from app.benchmark.splits import filter_split  # noqa: E402
from app.verifier_evaluation import evaluate as evaluate_verifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--commitment", required=True)
    parser.add_argument("--kind", choices=("verifier", "retrieval"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))

    def evaluate(benchmark):
        if args.kind == "verifier":
            return evaluate_verifier(to_verifier_dataset(benchmark, split="holdout"), predictions)
        pack = build_evidence_pack(benchmark, split="holdout")
        return evaluate_retrieval(filter_split(benchmark, "holdout"), pack, predictions)

    result = run_owner_holdout(holdout_dir=args.holdout_dir, expected_commitment=args.commitment,
                               repo_root=repo_root, evaluate=evaluate, log_path=args.log,
                               evaluator=args.evaluator, run_label=args.run_label)
    args.out.write_text(render_json(result["summary"]), encoding="utf-8")
    print(json.dumps(result["log_entry"], sort_keys=True))


if __name__ == "__main__":
    main()
