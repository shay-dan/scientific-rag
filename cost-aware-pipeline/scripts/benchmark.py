"""Offline benchmark tooling (R6A). Never calls a model or searches for literature.

Subcommands (all take a benchmark manifest; outputs are deterministic):
  validate          contract, reference, label-mapping, provenance and split-leakage checks
  build-pack        freeze one split's evidence pack
  retrieval-eval    evaluate a retrieval run (or the lexical baseline) against a pack
  verifier-dataset  export one split as an R5 verifier dataset (evaluate with evaluate_verifier.py)
  discovery-eval    evaluate a discovery run stage by stage

Holdout material is refused here; the owner evaluates the sealed holdout with
scripts/benchmark_holdout_owner.py outside the repository.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import render_json, render_validation_markdown, validation_report  # noqa: E402
from app.benchmark.adapters import load_benchmark  # noqa: E402
from app.benchmark.bridges import to_verifier_dataset  # noqa: E402
from app.benchmark.discovery import evaluate_discovery  # noqa: E402
from app.benchmark.retrieval import (  # noqa: E402
    DEFAULT_KS, build_evidence_pack, evaluate_retrieval, lexical_baseline_run, render_retrieval_markdown,
)
from app.benchmark.splits import filter_split  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_valid(manifest: Path):
    benchmark, issues = load_benchmark(manifest)
    report = validation_report(benchmark, issues)
    if report["status"] != "valid":
        raise SystemExit(f"benchmark is invalid ({report['issue_counts']['errors']} errors); run `validate` first")
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "build-pack", "retrieval-eval", "verifier-dataset", "discovery-eval"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True, help="output file (JSON) or directory")
        if name != "validate":
            command.add_argument("--split", choices=("development", "validation"), required=True)
        if name == "retrieval-eval":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--run", type=Path)
            source.add_argument("--baseline", choices=("lexical_bm25",))
            command.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
        if name == "discovery-eval":
            command.add_argument("--run", type=Path, required=True)
            command.add_argument("--rank-cutoff", type=int, default=20)
    args = parser.parse_args()

    if args.command == "validate":
        benchmark, issues = load_benchmark(args.manifest)
        report = validation_report(benchmark, issues)
        _write(args.out / "validation.json", render_json(report))
        _write(args.out / "validation.md", render_validation_markdown(report))
        print(json.dumps({"status": report["status"], **report["issue_counts"]}, sort_keys=True))
        raise SystemExit(0 if report["status"] == "valid" else 1)
    benchmark = _load_valid(args.manifest)
    if args.command == "build-pack":
        _write(args.out, render_json(build_evidence_pack(benchmark, split=args.split)))
    elif args.command == "retrieval-eval":
        pack = build_evidence_pack(benchmark, split=args.split)
        run = lexical_baseline_run(pack) if args.baseline else json.loads(args.run.read_text(encoding="utf-8"))
        report = evaluate_retrieval(filter_split(benchmark, args.split), pack, run,
                                    ks=tuple(int(k) for k in args.ks.split(",")))
        _write(args.out / "retrieval.json", render_json(report))
        _write(args.out / "retrieval.md", render_retrieval_markdown(report))
    elif args.command == "verifier-dataset":
        _write(args.out, render_json(to_verifier_dataset(benchmark, split=args.split)))
    elif args.command == "discovery-eval":
        report = evaluate_discovery(benchmark, json.loads(args.run.read_text(encoding="utf-8")),
                                    split=args.split, rank_cutoff=args.rank_cutoff)
        _write(args.out, render_json(report))


if __name__ == "__main__":
    main()
