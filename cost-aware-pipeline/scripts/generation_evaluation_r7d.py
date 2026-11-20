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
import os
import stat
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
    create_blind_key,
    validate_blind_key,
    SCORE_METRICS,
    dry_run,
    unblind_owner_scores,
)
from app.benchmark.incremental_generation_evaluation import (  # noqa: E402
    AtomicCheckpoint,
    CampaignLocked,
    CheckpointError,
    run_incremental_evaluation,
    verify_checkpoint,
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


def _predecessor_inputs(args):
    paths = (args.predecessor_checkpoint, args.predecessor_config,
             args.predecessor_authorization, args.predecessor_plan)
    if not any(paths):
        return None
    if not all(paths):
        raise ValueError("all four predecessor paths are required")
    return (AtomicCheckpoint(_private(paths[0], "predecessor checkpoint")),
            *(_load(_private(p, "predecessor input")) for p in paths[1:]))


def _continuation_plan(args) -> int:
    from app.benchmark.generation_continuation import build_continuation_contract
    predecessor = _predecessor_inputs(args)
    if predecessor is None:
        raise ValueError("continuation-plan requires all predecessor inputs")
    old_store, old_config, old_auth, old_plan = predecessor
    manifest = _load(_private(args.manifest, "manifest"))
    config = _load(_private(args.config, "config"))
    ledger = programs.ProgramLedgerStore(_private(args.program_ledger, "program ledger"))
    contract = build_continuation_contract(
        manifest=manifest, config=config, key=_read_blind_key(args.private_key),
        program_ledger=ledger, predecessor_store=old_store, predecessor_config=old_config,
        predecessor_authorization=old_auth, predecessor_plan=old_plan)
    print(json.dumps({"continuation": contract, "config_sha256": canonical_sha256(config),
                      "program_baseline": programs.program_baseline(ledger.load()),
                      "maximum_new_requests": 2 * contract["remaining_arm_count"],
                      "provider_calls_made": 0}, sort_keys=True))
    return 0


def _incremental_run(args) -> int:
    if args.authorization is None:
        raise AuthorizationError("incremental-run requires --authorization from an exact owner approval")
    manifest = _load(_private(args.manifest, "manifest"))
    config = _load(_private(args.config, "config"))
    if config.get("schema") != CONFIG_SCHEMA_V2 or args.program_ledger is None:
        raise AuthorizationError("incremental-run requires a v2 config and --program-ledger (cumulative "
                                 "accounting); v1 configurations are historical")
    authorization = _load(_private(args.authorization, "authorization"))
    blind_key = _read_blind_key(args.private_key)
    validate_blind_key(manifest, blind_key, authorization.get("blind_key_sha256"))
    store = AtomicCheckpoint(_private(args.private_checkpoint, "private checkpoint"))
    ledger_store = programs.ProgramLedgerStore(_private(args.program_ledger, "program ledger"))
    predecessor = _predecessor_inputs(args)
    with store.locked():
        checkpoint = asyncio.run(run_incremental_evaluation(
            manifest, config, authorization=authorization, checkpoint_store=store,
            pairs=args.pairs, resume_after_rate_limit=args.resume_after_rate_limit,
            wait_for_token_reset=args.wait_for_token_reset, program_ledger=ledger_store, blind_key=blind_key,
            predecessor=predecessor,
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


def _read_blind_key(path: Path | None) -> dict:
    if path is None:
        raise ValueError("a precommitted --private-key is required")
    path = _private(path, "blind key")
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("blind key missing or permissions are not owner-only")
    return _load(path)


def _write_once(path: Path, value: dict) -> None:
    """Exclusive owner-only artifact creation. Never replace an existing key or score freeze."""
    path = _private(path, "private artifact")
    if not path.parent.is_dir() or stat.S_IMODE(path.parent.stat().st_mode) & 0o077:
        raise ValueError("private artifact directory must be owner-only")
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _blind_init(args) -> int:
    manifest = _load(_private(args.manifest, "manifest"))
    key = create_blind_key(manifest)
    _write_once(args.private_key, key)
    print(json.dumps({"blind_key_sha256": canonical_sha256(key),
                      "manifest_sha256": manifest["manifest_sha256"], "provider_calls_made": 0}))
    return 0


def _report_inputs(args, payload):
    from app.benchmark.generation_continuation import reporting_records
    if not all((args.manifest, args.config, args.authorization, args.private_key)):
        raise ValueError("bound report requires manifest, config, authorization and private key")
    manifest = _load(_private(args.manifest, "manifest"))
    config = _load(_private(args.config, "config"))
    authorization = _load(_private(args.authorization, "authorization"))
    verify_checkpoint(payload, manifest, config, authorization)
    if payload.get("active_arm") is not None:
        raise ValueError("checkpoint has pending work")
    key = _read_blind_key(args.private_key)
    validate_blind_key(manifest, key, authorization.get("blind_key_sha256"))
    records = (reporting_records(manifest, config, payload)
               if getattr(args, "allow_partial", False) else payload["records"])
    return manifest, authorization, key, records


def _bound_packet(args, payload):
    manifest, authorization, key, records = _report_inputs(args, payload)
    return build_blind_scoring_packet(manifest, records, key=key,
                                      commitment=authorization.get("blind_key_sha256"))


def _blind_packet(args) -> int:
    payload = AtomicCheckpoint(_private(args.private_records, "private records")).load()
    packet, _ = _bound_packet(args, payload)
    _write_once(args.private_packet, packet)
    print(json.dumps({"packet_sha256": canonical_sha256(packet), "provider_calls_made": 0}))
    return 0


def _score_freeze(packet, scores):
    if packet.get("schema") != "scientific-rag-generation-eval-blind-scoring/v1":
        raise ValueError("invalid blind packet")
    scorable = {row["scenario_id"] for row in packet["cases"] if row["scorable"]}
    if set(scores) != scorable:
        raise ValueError("scores must cover exactly the scorable cases")
    for score in scores.values():
        if (not isinstance(score, dict) or set(score) != {"winner", "metrics"}
                or score["winner"] not in ("A", "B", "tie")
                or not isinstance(score["metrics"], dict)
                or set(score["metrics"]) - SCORE_METRICS):
            raise ValueError("invalid blind score")
        for values in score["metrics"].values():
            if (not isinstance(values, dict) or set(values) != {"A", "B"}
                    or any(type(v) not in (bool, int) or not 0 <= v <= 2 for v in values.values())):
                raise ValueError("invalid bounded arm scores")
    return {"schema": "scientific-rag-generation-eval-score-freeze/v1",
            "packet_sha256": canonical_sha256(packet), "scores": scores}


def _freeze_scores(args) -> int:
    packet = _load(_private(args.private_packet, "blind packet"))
    scores = _load(_private(args.scores, "blind scores"))
    frozen = _score_freeze(packet, scores)
    _write_once(args.frozen_scores, frozen)
    print(json.dumps({"frozen_scores_sha256": canonical_sha256(frozen), "provider_calls_made": 0}))
    return 0


def _public_report(args) -> int:
    payload = AtomicCheckpoint(_private(args.private_records, "private records")).load()
    if args.scores:
        if not all((args.private_key, args.manifest, args.config, args.authorization, args.frozen_scores_sha256)):
            raise ValueError("scored report requires authorized key, manifest/config and externally recorded score freeze")
        packet, key = _bound_packet(args, payload)
        frozen = _load(_private(args.scores, "frozen scores"))
        if (canonical_sha256(frozen) != args.frozen_scores_sha256
                or frozen != _score_freeze(packet, frozen.get("scores", {}))):
            raise ValueError("frozen scores or scoring packet mismatch")
        manifest = _load(_private(args.manifest, "manifest"))
        authorization = _load(_private(args.authorization, "authorization"))
        scores = unblind_owner_scores(frozen["scores"], key, manifest=manifest,
                                      commitment=authorization.get("blind_key_sha256"))
    else:
        scores = {}
    records = payload["records"]
    bound_report = getattr(args, "allow_partial", False) or "continuation" in payload
    if bound_report:
        manifest, _, _, records = _report_inputs(args, payload)
        if len(records) != len(manifest["scenarios"]):
            raise ValueError("incomplete successor report requires --allow-partial")
    report = build_public_report(
        records, scores=scores, manifest_sha256=payload["manifest_sha256"],
        config_sha256=payload["config_sha256"],
    )
    if bound_report:
        report["planned_scenarios"] = 20
        report["attempted_scenarios"] = sum(bool(r["arms"]) for r in payload["records"])
        report["not_attempted_scenarios"] = 20 - report["attempted_scenarios"]
        report["scorable_pairs"] = sum(r["scorable"] for r in report["scenarios"])
        report["execution_stop_code"] = payload["execution"].get("stop_code")
        if "continuation" in payload:
            report["continuation"] = payload["continuation"]
            report["successor_code_sha"] = payload["code_sha"]
            report["successor_call_count"] = len(payload["execution"]["calls"])
    _write(args.out, report)
    return 0


def _add_predecessor_arguments(command):
    for field in ("checkpoint", "config", "authorization", "plan"):
        command.add_argument("--predecessor-" + field, type=Path)


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
    incremental.add_argument("--private-key", type=Path, help="precommitted private blind allocation (required)")
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
    _add_predecessor_arguments(incremental)
    incremental.set_defaults(func=_incremental_run)
    continuation = commands.add_parser("continuation-plan", help="offline: bind exact closed predecessor")
    continuation.add_argument("--manifest", type=Path, required=True)
    continuation.add_argument("--config", type=Path, required=True)
    continuation.add_argument("--program-ledger", type=Path, required=True)
    continuation.add_argument("--private-key", type=Path, required=True)
    _add_predecessor_arguments(continuation)
    continuation.set_defaults(func=_continuation_plan)
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
    blind_init = commands.add_parser("blind-init", help="create private balanced allocation before authorization")
    blind_init.add_argument("--manifest", type=Path, required=True)
    blind_init.add_argument("--private-key", type=Path, required=True)
    blind_init.set_defaults(func=_blind_init)
    blind = commands.add_parser("blind-packet", help="export packet using the precommitted private key")
    blind.add_argument("--manifest", type=Path, required=True)
    blind.add_argument("--config", type=Path, required=True)
    blind.add_argument("--authorization", type=Path, required=True)
    blind.add_argument("--private-records", type=Path, required=True)
    blind.add_argument("--private-packet", type=Path, required=True)
    blind.add_argument("--private-key", type=Path, required=True)
    blind.add_argument("--allow-partial", action="store_true", help="derived view of terminal incomplete campaign")
    blind.set_defaults(func=_blind_packet)
    freeze = commands.add_parser("freeze-scores", help="freeze blind scores before key disclosure")
    freeze.add_argument("--private-packet", type=Path, required=True)
    freeze.add_argument("--scores", type=Path, required=True)
    freeze.add_argument("--frozen-scores", type=Path, required=True)
    freeze.set_defaults(func=_freeze_scores)
    report = commands.add_parser("public-report", help="build deterministic text-free aggregate report")
    report.add_argument("--private-records", type=Path, required=True)
    report.add_argument("--scores", type=Path)
    report.add_argument("--private-key", type=Path)
    report.add_argument("--manifest", type=Path)
    report.add_argument("--config", type=Path)
    report.add_argument("--authorization", type=Path)
    report.add_argument("--frozen-scores-sha256")
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--allow-partial", action="store_true", help="derived view of terminal incomplete campaign")
    report.set_defaults(func=_public_report)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (AuthorizationError, CampaignLocked, CheckpointError, ManifestError, ValueError,
            programs.ProgramLedgerError, OSError) as error:
        raise SystemExit(f"refused: {error}") from None


if __name__ == "__main__":
    raise SystemExit(main())
