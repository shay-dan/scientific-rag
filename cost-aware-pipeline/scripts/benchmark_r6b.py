"""R6B: import an independent development handoff and export unscored offline predictions.

Never calls a model, never searches PubMed/arXiv, never reads reference labels. The
handoff archive and every output derived from it stay outside the repository
(the repository is public; handoff text must not be committed).

Subcommands:
  import         verify the archive (manifest, hashes, counts, label-free contract) and
                 write the import/validation report; --inputs-out also writes the
                 normalized unscored inputs (JSONL) for inspection
  run-baselines  run the enforced stage protocol for one handoff folder
                 (open discovery frozen before fixed evidence is loaded) with the offline
                 reference baselines, and write the portable prediction export
  verify-export  check an export's contract and hashes before it is shared
  score          owner-side only: score an export against reference labels the owner
                 holds; refuses without references (``NoReferenceLabels``)
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark.handoff import HandoffRefused, import_handoff, render_json, write_jsonl  # noqa: E402
from app.benchmark.owner_scoring import NoReferenceLabels, score_export  # noqa: E402
from app.benchmark.prediction_export import build_export, offline_baseline_records, verify_export  # noqa: E402
from app.benchmark.protocol import LeakageError, assert_no_source_identity, session_for_group  # noqa: E402
from app.evaluation_report import resolve_git_provenance  # noqa: E402

PREDICTOR = {"name": "offline_reference_baselines", "kind": "baseline", "version": "r6b-v1",
             "components": {"fixed_evidence_retrieval": "R6A lexical BM25 within each fixed pack (offline reference; "
                                                        "not the production FTS + pgvector + RRF + MMR retriever)",
                            "claim_verification": "R5 lexical_numeric_baseline",
                            "open_discovery": "not executed (live search not authorized)",
                            "fixed_evidence_answer": "not executed (model not authorized)"}}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _import(path: Path):
    try:
        return import_handoff(path)
    except HandoffRefused as exc:
        raise SystemExit(f"handoff refused: {exc}")


def cmd_import(args) -> int:
    _, inputs, report = _import(args.handoff)
    _write(args.out, render_json(report))
    if args.inputs_out:
        for path, items in inputs.items():
            _write(args.inputs_out / (path.replace("/", "__")), write_jsonl(items))
    accepted = sum(f["accepted"] for f in report["files"].values())
    rejected = sum(f["rejected"] for f in report["files"].values())
    print(f"{report['status']}: {accepted} accepted, {rejected} rejected")
    return 0 if report["status"] == "imported" else 1


def cmd_run_baselines(args) -> int:
    _, inputs, report = _import(args.handoff)
    if report["status"] != "imported" and not args.allow_rejections:
        raise SystemExit("import has rejected records; fix the handoff or pass --allow-rejections")
    session = session_for_group(inputs, args.group)
    fixed = [item for path, items in inputs.items() if path.split("/")[0] == args.group
             for item in items if item["kind"] != "open_discovery_question"]
    try:
        audit = assert_no_source_identity(session.discovery_inputs, fixed)   # counts only
    except LeakageError as exc:
        raise SystemExit(f"refused: {exc}")
    records, protocol = offline_baseline_records(session)
    protocol["discovery_input_audit"] = audit
    git = resolve_git_provenance()
    export = build_export(records=records, protocol=protocol, import_report=report,
                          run_id=args.run_id or f"r6b-offline-{args.group}", git=git, predictor=PREDICTOR)
    verify_export(export)
    _write(args.out, render_json(export))
    print(json.dumps({"export_hash": export["export_hash"], "counts": export["counts"]}, sort_keys=True))
    return 0


def cmd_verify(args) -> int:
    verify_export(json.loads(args.export.read_text(encoding="utf-8")))
    print("export verified")
    return 0


def cmd_score(args) -> int:
    export = json.loads(args.export.read_text(encoding="utf-8"))
    if not args.references:
        raise SystemExit("refusing to score: no reference labels supplied (owner-side only)")
    references = json.loads(args.references.read_text(encoding="utf-8"))
    id_map = json.loads(args.id_map.read_text(encoding="utf-8")) if args.id_map else None
    try:
        scores = score_export(export, references, id_map=id_map)
    except NoReferenceLabels as exc:
        raise SystemExit(f"refusing to score: {exc}")
    _write(args.out, render_json(scores))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import"); p.add_argument("handoff", type=Path); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--inputs-out", type=Path); p.set_defaults(func=cmd_import)
    p = sub.add_parser("run-baselines"); p.add_argument("handoff", type=Path)
    p.add_argument("--group", required=True, help="handoff folder, e.g. pilot_eval_inputs or eval_inputs")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--run-id")
    p.add_argument("--allow-rejections", action="store_true"); p.set_defaults(func=cmd_run_baselines)
    p = sub.add_parser("verify-export"); p.add_argument("export", type=Path); p.set_defaults(func=cmd_verify)
    p = sub.add_parser("score"); p.add_argument("export", type=Path); p.add_argument("--references", type=Path)
    p.add_argument("--id-map", type=Path); p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
