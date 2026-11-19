"""R7D fixed-evidence paired generation evaluation.

``dry-run`` is offline and always makes zero provider calls. ``incremental-run`` is the only
execution path (R7D-C): it requires a v2 config, an exact owner authorization and the private
cumulative program ledger (``ledger-*`` commands). ``paid-run`` is refused. ``auth-preflight``
is an opt-in authenticated model listing that sends no inference request.
Private outputs must be written outside the public repository.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.benchmark import program_ledger as programs  # noqa: E402
from app.benchmark.generation_evaluation import (  # noqa: E402
    CONFIG_SCHEMA_V2,
    AuthorizationError,
    ManifestError,
    build_blind_scoring_packet,
    build_public_report,
    canonical_sha256,
    dry_run,
    unblind_owner_scores,
)
from app.benchmark.incremental_generation_evaluation import (  # noqa: E402
    AtomicCheckpoint,
    CampaignLocked,
    CheckpointError,
    run_incremental_evaluation,
)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON literal {value} is not permitted")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _private(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return path
    raise ValueError(f"{label} must stay outside the public repository")


def _dry_run(args) -> int:
    plan = dry_run(_load(_private(args.manifest, "manifest")), _load(args.config))
    if args.program_ledger is not None:
        # Informational: worst case against the cumulative program allowance (zero provider calls).
        baseline = programs.program_baseline(programs.ProgramLedgerStore(
            _private(args.program_ledger, "program ledger")).load())
        warnings = []
        if plan["max_physical_calls"] > baseline["remaining"]["calls"]:
            warnings.append("worst-case requests exceed the remaining program request allowance; "
                            "execution stops at the ceiling before every pair can complete")
        if (plan["planning_cost_bound_usd"] or 0) > baseline["remaining"]["cost_usd"]:
            warnings.append("worst-case reservation exceeds the remaining program cost allowance")
        plan["program"] = {"program_id": baseline["program_id"], "baseline_sha256": baseline["sha256"],
                           "baseline": baseline["budget"], "remaining": baseline["remaining"], "warnings": warnings}
    _write(args.out, plan)
    print(json.dumps({key: plan[key] for key in (
        "provider_calls_made", "scenarios", "max_physical_calls", "planning_cost_bound_usd",
        "eligible_after_exact_owner_authorization", "blockers",
    )} | ({"program": plan["program"]} if "program" in plan else {}), sort_keys=True))
    return 0


def _paid_run(args) -> int:
    # R7D-C: the one-shot runner has no cumulative program accounting; a fresh one-shot run
    # could silently reset the benchmark-program ceilings. Staged execution is the only path.
    raise AuthorizationError("paid-run is superseded by incremental-run, which carries the cumulative "
                             "program ledger")


def _incremental_run(args) -> int:
    if args.authorization is None:
        raise AuthorizationError("incremental-run requires --authorization from an exact owner approval")
    manifest = _load(_private(args.manifest, "manifest"))
    config = _load(_private(args.config, "config"))
    if config.get("schema") != CONFIG_SCHEMA_V2 or args.program_ledger is None:
        raise AuthorizationError("incremental-run requires a v2 config and --program-ledger (cumulative "
                                 "accounting); v1 configurations are historical")
    authorization = _load(_private(args.authorization, "authorization"))
    store = AtomicCheckpoint(_private(args.private_checkpoint, "private checkpoint"))
    ledger_store = programs.ProgramLedgerStore(_private(args.program_ledger, "program ledger"))
    with store.locked():
        checkpoint = asyncio.run(run_incremental_evaluation(
            manifest, config, authorization=authorization, checkpoint_store=store,
            pairs=args.pairs, resume_after_rate_limit=args.resume_after_rate_limit,
            wait_for_token_reset=args.wait_for_token_reset, program_ledger=ledger_store,
        ))
    records = checkpoint["records"]
    accounting = programs.accounting_summary(ledger_store.load(), checkpoint, checkpoint_path=store.path)
    print(json.dumps({
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "completed_pairs": sum(record.get("status") == "completed" for record in records),
        "physical_requests": checkpoint["execution"]["budget"]["provider_calls_charged"],
        "current_experiment": accounting["current_experiment"],
        "cumulative": accounting["cumulative"],
        "stopped": checkpoint["execution"].get("stopped"),
        "stop_code": checkpoint["execution"].get("stop_code"),
        "pending_arm": ({key: checkpoint["active_arm"].get(key) for key in ("scenario_id", "prompt_version", "phase")}
                        if checkpoint.get("active_arm") else None),
    }, sort_keys=True))
    return 0


def _ledger(args) -> programs.ProgramLedgerStore:
    return programs.ProgramLedgerStore(_private(args.ledger, "program ledger"))


def _ledger_init(args) -> int:
    import time
    value = programs.initialize_program_ledger(_ledger(args), program_id=args.program_id, max_calls=args.max_calls,
                                               max_cost_usd=args.max_cost_usd, now=time.time())
    print(json.dumps({"program_id": value["program_id"], "ceilings": value["ceilings"],
                      "baseline": programs.program_baseline(value)}, sort_keys=True))
    return 0


def _ledger_import(args) -> int:
    entry = programs.import_campaign(_ledger(args), _private(args.checkpoint, "checkpoint"))
    print(json.dumps({"imported": entry, "baseline": programs.program_baseline(_ledger(args).load())},
                     sort_keys=True))
    return 0


def _ledger_close(args) -> int:
    entry = programs.close_campaign(_ledger(args), AtomicCheckpoint(_private(args.checkpoint, "checkpoint")))
    print(json.dumps({"closed": entry, "baseline": programs.program_baseline(_ledger(args).load())},
                     sort_keys=True))
    return 0


def _ledger_close_lost(args) -> int:
    entry = programs.close_lost_campaign(_ledger(args), args.campaign_id)
    print(json.dumps({"closed": entry, "baseline": programs.program_baseline(_ledger(args).load())},
                     sort_keys=True))
    return 0


def _ledger_diagnostic(args) -> int:
    entry = programs.record_diagnostic(_ledger(args), _private(args.source, "diagnostic source"), label=args.label,
                                       requests=args.requests, estimated_cost_usd=args.estimated_cost_usd,
                                       uncertain_reserved_usd=args.uncertain_reserved_usd)
    print(json.dumps({"diagnostic": entry}, sort_keys=True))
    return 0


def _ledger_summary(args) -> int:
    ledger = _ledger(args).load()
    path = _private(args.checkpoint, "checkpoint") if args.checkpoint else None
    checkpoint = AtomicCheckpoint(path).load() if path else None
    print(json.dumps({"baseline": programs.program_baseline(ledger),
                      "accounting": programs.accounting_summary(ledger, checkpoint, checkpoint_path=path)},
                     sort_keys=True))
    return 0


def _auth_preflight(args) -> int:
    import os
    from app.benchmark.provider_preflight import authenticated_model_preflight
    config = _load(_private(args.config, "config"))
    result = asyncio.run(authenticated_model_preflight(config["provider"], config["model"],
                                                       api_key=os.environ.get("GROQ_API_KEY")))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["authenticated"] and result["model_listed"] else 1


def _blind_packet(args) -> int:
    manifest = _load(_private(args.manifest, "manifest"))
    payload = _load(_private(args.private_records, "private records"))
    packet, key = build_blind_scoring_packet(manifest, payload["records"])
    _write(_private(args.private_packet, "blind packet"), packet)
    _write(_private(args.private_key, "blind key"), key)
    return 0


def _public_report(args) -> int:
    payload = _load(_private(args.private_records, "private records"))
    if bool(args.scores) != bool(args.private_key):
        raise ValueError("--scores and --private-key must be supplied together")
    if args.scores:
        key = _load(_private(args.private_key, "blind key"))
        if key.get("manifest_sha256") != payload.get("manifest_sha256"):
            raise ValueError("blind key does not match the run manifest")
        scores = unblind_owner_scores(_load(_private(args.scores, "owner scores")), key)
    else:
        scores = {}
    report = build_public_report(
        payload["records"], scores=scores, manifest_sha256=payload["manifest_sha256"],
        config_sha256=payload["config_sha256"],
    )
    _write(args.out, report)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run", help="validate and plan; zero provider calls")
    dry.add_argument("--manifest", type=Path, required=True)
    dry.add_argument("--config", type=Path, required=True)
    dry.add_argument("--out", type=Path, required=True)
    dry.add_argument("--program-ledger", type=Path, help="optional: report the cumulative program allowance")
    dry.set_defaults(func=_dry_run)
    paid = commands.add_parser("paid-run", help="superseded by incremental-run (R7D-C); always refused")
    paid.add_argument("--manifest", type=Path, required=True)
    paid.add_argument("--config", type=Path, required=True)
    paid.add_argument("--authorization", type=Path)
    paid.add_argument("--private-records", type=Path, required=True)
    paid.set_defaults(func=_paid_run)
    incremental = commands.add_parser(
        "incremental-run", help="authorized staged execution with an atomic private checkpoint")
    incremental.add_argument("--manifest", type=Path, required=True)
    incremental.add_argument("--config", type=Path, required=True)
    incremental.add_argument("--authorization", type=Path)
    incremental.add_argument("--private-checkpoint", type=Path, required=True)
    incremental.add_argument("--pairs", type=int, required=True)
    incremental.add_argument("--program-ledger", type=Path,
                             help="private cumulative program ledger (required; v2 configs only)")
    incremental.add_argument(
        "--resume-after-rate-limit", action="store_true",
        help="explicitly continue unexecuted work after a recorded capacity reset has elapsed")
    incremental.add_argument(
        "--wait-for-token-reset", action="store_true",
        help="opt-in: wait in-process for a known per-minute token reset of at most 65 seconds; "
             "never for request-per-day or unknown limits")
    incremental.set_defaults(func=_incremental_run)
    init = commands.add_parser("ledger-init", help="create a private program ledger with fixed ceilings")
    init.add_argument("--ledger", type=Path, required=True)
    init.add_argument("--program-id", required=True)
    init.add_argument("--max-calls", type=int, required=True)
    init.add_argument("--max-cost-usd", type=float, required=True)
    init.set_defaults(func=_ledger_init)
    imported = commands.add_parser("ledger-import", help="record a pre-ledger campaign checkpoint (read-only)")
    imported.add_argument("--ledger", type=Path, required=True)
    imported.add_argument("--checkpoint", type=Path, required=True)
    imported.set_defaults(func=_ledger_import)
    close = commands.add_parser("ledger-close", help="close the open campaign, keeping its accounting")
    close.add_argument("--ledger", type=Path, required=True)
    close.add_argument("--checkpoint", type=Path, required=True)
    close.set_defaults(func=_ledger_close)
    lost = commands.add_parser("ledger-close-lost",
                               help="close an open campaign whose authoritative checkpoint is missing, from its "
                                    "last ledger mark")
    lost.add_argument("--ledger", type=Path, required=True)
    lost.add_argument("--campaign-id", required=True)
    lost.set_defaults(func=_ledger_close_lost)
    diagnostic = commands.add_parser("ledger-record-diagnostic",
                                     help="record separate diagnostic requests (never benchmark pairs)")
    diagnostic.add_argument("--ledger", type=Path, required=True)
    diagnostic.add_argument("--source", type=Path, required=True)
    diagnostic.add_argument("--label", required=True)
    diagnostic.add_argument("--requests", type=int, required=True)
    diagnostic.add_argument("--estimated-cost-usd", type=float, required=True)
    diagnostic.add_argument("--uncertain-reserved-usd", type=float, required=True)
    diagnostic.set_defaults(func=_ledger_diagnostic)
    summary = commands.add_parser("ledger-summary", help="print the five accounting categories")
    summary.add_argument("--ledger", type=Path, required=True)
    summary.add_argument("--checkpoint", type=Path)
    summary.set_defaults(func=_ledger_summary)
    auth = commands.add_parser("auth-preflight",
                               help="opt-in: one authenticated model-listing request; no inference")
    auth.add_argument("--config", type=Path, required=True)
    auth.set_defaults(func=_auth_preflight)
    blind = commands.add_parser("blind-packet", help="build private owner-scoring packet and separate key")
    blind.add_argument("--manifest", type=Path, required=True)
    blind.add_argument("--private-records", type=Path, required=True)
    blind.add_argument("--private-packet", type=Path, required=True)
    blind.add_argument("--private-key", type=Path, required=True)
    blind.set_defaults(func=_blind_packet)
    report = commands.add_parser("public-report", help="build deterministic text-free aggregate report")
    report.add_argument("--private-records", type=Path, required=True)
    report.add_argument("--scores", type=Path)
    report.add_argument("--private-key", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.set_defaults(func=_public_report)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (AuthorizationError, CampaignLocked, CheckpointError, ManifestError, ValueError,
            programs.ProgramLedgerError) as error:
        raise SystemExit(f"refused: {error}") from None


if __name__ == "__main__":
    raise SystemExit(main())
