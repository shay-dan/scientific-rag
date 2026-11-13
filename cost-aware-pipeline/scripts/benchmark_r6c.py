"""R6C: opt-in evaluation of the production semantic verifier and hybrid retriever.

Two independent adapters, each with its own commands and its own single-task export
(``scientific-rag-benchmark-predictions/v1``); claim-verification and retrieval records are
never combined. Inputs come only from the label-free development handoff, through the R6B
stage protocol (open discovery frozen as not executed before fixed evidence is loaded).
The handoff and all outputs stay outside the repository.

  verifier-dry-run   plan every production verifier call; estimate tokens and a planning
                     cost bound for the configured gateway route, and report whether a live
                     run would be eligible after authorization. No provider call.
  verifier-run       call the production verifier with a provider-enforced output-token
                     limit (R6D). Refused without an authorization file.
  retrieval-dry-run  count production chunks; estimate embedding tokens and cost.
                     No provider or database call.
  retrieval-run      run the production hybrid retriever against a dedicated evaluation
                     database. Refused without an authorization file.

Pilot selection is deterministic and label-blind: ``--limit N`` takes the first N inputs
by opaque input ID; ``--input-ids FILE`` takes a listed subset.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.benchmark.handoff import HandoffRefused, import_handoff, render_json  # noqa: E402
from app.benchmark.prediction_export import build_export, verify_export  # noqa: E402
from app.benchmark.protocol import session_for_group  # noqa: E402
from app.evaluation_report import resolve_git_provenance  # noqa: E402

DISCOVERY_REASON = "open discovery is not part of R6C; live PubMed/arXiv search is not authorized"


def fixed_stage(handoff: Path, group: str):
    """Import, freeze discovery as not executed, then load fixed evidence (R6B protocol)."""
    try:
        _, inputs, report = import_handoff(handoff)
    except HandoffRefused as exc:
        raise SystemExit(f"handoff refused: {exc}")
    if report["status"] != "imported":
        raise SystemExit("import has rejected records; R6C runs only on a cleanly imported handoff")
    session = session_for_group(inputs, group)
    frozen_at = session.skip_discovery(DISCOVERY_REASON)
    fixed = session.load_fixed_evidence()
    _, verified = session.frozen_discovery()
    session.mark_exported()
    protocol = {"discovery_hash_at_freeze": frozen_at, "discovery_frozen_hash": verified,
                "fixed_evidence_loaded_after_discovery": True, "discovery_status": "not_executed"}
    return fixed, report, protocol


def select(items: list[dict], args) -> list[dict]:
    items = sorted(items, key=lambda item: item["input_id"])
    if args.input_ids:
        wanted = set(Path(args.input_ids).read_text(encoding="utf-8").split())
        unknown = wanted - {item["input_id"] for item in items}
        if unknown:
            raise SystemExit(f"{len(unknown)} requested input ID(s) are not in this handoff group")
        items = [item for item in items if item["input_id"] in wanted]
    if args.limit is not None:
        items = items[:args.limit]
    return items


def single_task_export(records, task, *, protocol, report, run_id, predictor) -> dict:
    if any(record["task"] != task for record in records):
        raise SystemExit(f"refusing to mix tasks in a {task} export")
    export = build_export(records=records, protocol=protocol, import_report=report, run_id=run_id,
                          git=resolve_git_provenance(), predictor=predictor)
    verify_export(export)
    return export


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def _authorization(path) -> dict:
    """Strict parse (no NaN/Infinity); field checks happen in the adapter before any call."""
    from app.benchmark.production_verifier_replay import AuthorizationError, parse_authorization
    if not path:
        raise SystemExit("refused: paid provider calls need --authorization (explicit owner approval)")
    try:
        return parse_authorization(Path(path).read_text(encoding="utf-8"))
    except AuthorizationError as exc:
        raise SystemExit(f"refused: {exc}")


def cmd_verifier_dry_run(args) -> int:
    from app.benchmark import production_verifier_replay as verifier
    fixed, report, _ = fixed_stage(args.handoff, args.group)
    plan = verifier.dry_run(select(verifier.claim_inputs(fixed), args), max_output_tokens=args.max_output_tokens)
    plan["dataset_hash"] = report["handoff"]["dataset_hash"]
    _write(args.out, plan)
    print(json.dumps({k: plan[k] for k in ("planned_calls", "estimated_input_tokens", "planning_cost_bound_usd_any_route",
                                           "planning_cost_bound_note", "live_execution")}))
    return 0


def cmd_verifier_run(args) -> int:
    from app.benchmark import production_verifier_replay as verifier
    from app.services import PIPELINE_VERSION
    authorization = _authorization(args.authorization)
    fixed, report, protocol = fixed_stage(args.handoff, args.group)
    items = select(verifier.claim_inputs(fixed), args)
    route = verifier.gateway_route()
    try:
        records, run = asyncio.run(verifier.run_replay(items, authorization=authorization,
                                                       dataset_hash=report["handoff"]["dataset_hash"], route=route))
        # run_replay passes the authorized max_output_tokens to the gateway, which enforces it or refuses
    except verifier.AuthorizationError as exc:
        raise SystemExit(f"refused: {exc}")
    predictor = {"name": verifier.ADAPTER, "kind": "production_component", "version": PIPELINE_VERSION,
                 "components": {"claim_verification": "app.claim_verification.verify_claims (production, unchanged)"},
                 "configuration": verifier.configuration(), "route": route, "execution": run}
    _write(args.out, single_task_export(records, "claim_verification", protocol=protocol, report=report,
                                        run_id=args.run_id or f"r6c-verifier-{args.group}", predictor=predictor))
    return 0


def _settings():
    from app.config import Settings
    return Settings()


def cmd_retrieval_dry_run(args) -> int:
    from app.benchmark import production_retrieval_replay as retrieval
    fixed, report, _ = fixed_stage(args.handoff, args.group)
    plan = retrieval.dry_run(select(retrieval.retrieval_items(fixed), args), _settings(), research_mode=args.research_mode)
    plan["dataset_hash"] = report["handoff"]["dataset_hash"]
    _write(args.out, plan)
    print(json.dumps({k: plan[k] for k in ("child_chunks", "estimated_embedding_tokens", "estimated_cost_usd",
                                           "reserved_cost_usd")}))
    return 0


def cmd_retrieval_run(args) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from app.benchmark import production_retrieval_replay as retrieval
    from app.services import PIPELINE_VERSION
    authorization = _authorization(args.authorization)
    fixed, report, protocol = fixed_stage(args.handoff, args.group)
    items = select(retrieval.retrieval_items(fixed), args)
    settings = _settings()
    # the effective application database (env, .env or default), not only the process environment
    application_url = retrieval.effective_application_database_url(settings)
    try:   # fail closed before any connection
        retrieval.check_database_urls(evaluation_url=args.evaluation_database_url, application_url=application_url)
    except retrieval.IsolationError as exc:
        raise SystemExit(f"refused: {exc}")

    async def run():
        engine = create_async_engine(retrieval.async_database_url(args.evaluation_database_url))
        try:
            return await retrieval.run_retrieval(
                items, session_factory=async_sessionmaker(engine, expire_on_commit=False), settings=settings,
                authorization=authorization, dataset_hash=report["handoff"]["dataset_hash"],
                evaluation_url=args.evaluation_database_url, application_url=application_url,
                research_mode=args.research_mode)
        finally:
            await engine.dispose()
    try:
        records, execution = asyncio.run(run())
    except (retrieval.AuthorizationError, retrieval.IsolationError) as exc:
        raise SystemExit(f"refused: {exc}")
    predictor = {"name": retrieval.ADAPTER, "kind": "production_component", "version": PIPELINE_VERSION,
                 "components": {"fixed_evidence_retrieval": "app.pgvector_retrieval.PostgresVectorService "
                                                             "(production, unchanged; strategy enhanced)"},
                 "configuration": execution["configuration"], "execution": execution}
    _write(args.out, single_task_export(records, "fixed_evidence_retrieval", protocol=protocol, report=report,
                                        run_id=args.run_id or f"r6c-retrieval-{args.group}", predictor=predictor))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, paid in (("verifier-dry-run", cmd_verifier_dry_run, False), ("verifier-run", cmd_verifier_run, True),
                             ("retrieval-dry-run", cmd_retrieval_dry_run, False),
                             ("retrieval-run", cmd_retrieval_run, True)):
        p = sub.add_parser(name)
        p.add_argument("handoff", type=Path)
        p.add_argument("--group", required=True, help="handoff folder: pilot_eval_inputs or eval_inputs")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--input-ids")
        if name.startswith("retrieval"):
            p.add_argument("--research-mode", default="standard", choices=("fast", "standard", "deep"))
        if name == "verifier-dry-run":
            p.add_argument("--max-output-tokens", type=int, default=2000,
                           help="output-token limit a live run would request (planning only)")
        if paid:
            p.add_argument("--authorization", help="owner-approved paid-evaluation authorization JSON")
            p.add_argument("--run-id")
        if name == "retrieval-run":
            p.add_argument("--evaluation-database-url", required=True)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
