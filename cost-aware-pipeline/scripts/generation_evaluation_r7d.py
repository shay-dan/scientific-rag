"""R7D fixed-evidence paired generation evaluation.

``dry-run`` is offline and always makes zero provider calls. ``paid-run`` exists for a
future separately authorized execution and refuses before provider construction unless
the authorization exactly matches the frozen manifest, accepted code and full config.
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

from app.benchmark.generation_evaluation import (  # noqa: E402
    AuthorizationError,
    ManifestError,
    build_blind_scoring_packet,
    build_public_report,
    canonical_sha256,
    dry_run,
    run_paid_evaluation,
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
    _write(args.out, plan)
    print(json.dumps({key: plan[key] for key in (
        "provider_calls_made", "scenarios", "max_physical_calls", "planning_cost_bound_usd",
        "eligible_after_exact_owner_authorization", "blockers",
    )}, sort_keys=True))
    return 0


def _paid_run(args) -> int:
    if args.authorization is None:
        raise AuthorizationError("paid-run requires --authorization from a future explicit owner approval")
    manifest = _load(_private(args.manifest, "manifest"))
    config = _load(args.config)
    authorization = _load(_private(args.authorization, "authorization"))
    records, execution = asyncio.run(run_paid_evaluation(
        manifest, config, authorization=authorization,
    ))
    _write(_private(args.private_records, "private records"), {"manifest_sha256": manifest["manifest_sha256"],
                                  "config_sha256": canonical_sha256(config),
                                  "records": records, "execution": execution})
    return 0


def _incremental_run(args) -> int:
    if args.authorization is None:
        raise AuthorizationError("incremental-run requires --authorization from an exact owner approval")
    manifest = _load(_private(args.manifest, "manifest"))
    config = _load(_private(args.config, "config"))
    authorization = _load(_private(args.authorization, "authorization"))
    store = AtomicCheckpoint(_private(args.private_checkpoint, "private checkpoint"))
    with store.locked():
        checkpoint = asyncio.run(run_incremental_evaluation(
            manifest, config, authorization=authorization, checkpoint_store=store,
            pairs=args.pairs, resume_after_rate_limit=args.resume_after_rate_limit,
            wait_for_token_reset=args.wait_for_token_reset,
        ))
    records = checkpoint["records"]
    print(json.dumps({
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "completed_pairs": sum(record.get("status") == "completed" for record in records),
        "physical_requests": checkpoint["execution"]["budget"]["provider_calls_charged"],
        "stopped": checkpoint["execution"].get("stopped"),
        "stop_code": checkpoint["execution"].get("stop_code"),
        "pending_arm": ({key: checkpoint["active_arm"].get(key) for key in ("scenario_id", "prompt_version", "phase")}
                        if checkpoint.get("active_arm") else None),
    }, sort_keys=True))
    return 0


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
    dry.set_defaults(func=_dry_run)
    paid = commands.add_parser("paid-run", help="future authorized execution; not authorized by R7D-A")
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
    incremental.add_argument(
        "--resume-after-rate-limit", action="store_true",
        help="explicitly continue unexecuted work after a recorded capacity reset has elapsed")
    incremental.add_argument(
        "--wait-for-token-reset", action="store_true",
        help="opt-in: wait in-process for a known per-minute token reset of at most 65 seconds; "
             "never for request-per-day or unknown limits")
    incremental.set_defaults(func=_incremental_run)
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
    except (AuthorizationError, CampaignLocked, CheckpointError, ManifestError, ValueError) as error:
        raise SystemExit(f"refused: {error}") from None


if __name__ == "__main__":
    raise SystemExit(main())
