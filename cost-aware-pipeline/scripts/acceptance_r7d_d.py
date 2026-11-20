"""R7D-D configuration acceptance (synthetic): dry-run, authorized run, read-only report.

``dry-run`` is offline and makes zero provider calls. ``run`` requires an exact acceptance
authorization, the separate acceptance program ledger (create it with
``generation_evaluation_r7d.py ledger-init``) and a private checkpoint; it never retries.
``report`` reads a checkpoint (read-only) after verifying it against the dataset, config,
schedule and the authorized reference hash, and writes metrics only (no prompts, evidence or
answer text). ``--answers-out`` additionally writes the generated answers to a separate private
file for the owner's scientific review, which is required before a configuration is frozen.
Private files (config, authorization, checkpoint, ledger, outputs) stay outside the repository.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.benchmark import acceptance_evaluation as acceptance  # noqa: E402
from app.benchmark import program_ledger as programs  # noqa: E402
from app.benchmark.generation_evaluation import AuthorizationError  # noqa: E402
from app.benchmark.incremental_generation_evaluation import (  # noqa: E402
    AtomicCheckpoint,
    CampaignLocked,
    CheckpointError,
)

DEFAULT_DATASET = ROOT / "evaluation" / "r7d-d" / "synthetic-acceptance-v1.json"
DEFAULT_REFERENCES = ROOT / "evaluation" / "r7d-d" / "synthetic-acceptance-references-v1.json"


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON literal {value} is not permitted")


def _load(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _private(path: Path, label: str) -> Path:
    try:
        Path(path).resolve().relative_to(ROOT.resolve())
    except ValueError:
        return Path(path)
    raise ValueError(f"{label} must stay outside the public repository")


def _write_private(path: Path, value: object) -> None:
    path = _private(path, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    path.chmod(0o600)


def _dry_run(args) -> int:
    plan = acceptance.acceptance_dry_run(_load(args.dataset), _load(_private(args.config, "config")),
                                         _load(args.references))
    _write_private(args.out, plan)
    print(json.dumps({key: plan[key] for key in (
        "provider_calls_made", "max_physical_requests", "deterministic_reservation_usd",
        "typical_conditional_repair_estimate_usd", "planning_cost_bound_usd", "cost_ceiling_usd",
        "dataset_sha256", "config_sha256", "schedule_sha256", "plan_sha256", "selection_rules_sha256",
        "eligible_after_exact_owner_authorization", "blockers", "warnings")}, sort_keys=True))
    return 0


def _run(args) -> int:
    store = AtomicCheckpoint(_private(args.private_checkpoint, "private checkpoint"))
    ledger = programs.ProgramLedgerStore(_private(args.program_ledger, "acceptance program ledger"))
    with store.locked():
        checkpoint = asyncio.run(acceptance.run_acceptance(
            _load(args.dataset), _load(_private(args.config, "config")),
            authorization=_load(_private(args.authorization, "authorization")), checkpoint_store=store,
            program_ledger=ledger, operations=args.operations, resume_after_rate_limit=args.resume_after_rate_limit,
            wait_for_token_reset=args.wait_for_token_reset))
    execution = checkpoint["execution"]
    print(json.dumps({
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "operations_recorded": len(checkpoint["records"]),
        "physical_requests_cumulative": execution["budget"]["provider_calls_charged"],
        "budget": execution["budget"], "stop_code": execution.get("stop_code"), "stopped": execution.get("stopped"),
        "pending_operation": (checkpoint["active_arm"] or {}).get("operation"),
    }, sort_keys=True))
    return 0


def _report(args) -> int:
    checkpoint = AtomicCheckpoint(_private(args.checkpoint, "checkpoint")).load()   # read-only
    report = acceptance.build_acceptance_report(checkpoint, _load(args.dataset), _load(args.references),
                                                _load(_private(args.config, "config")))
    _write_private(args.out, report)
    if args.answers_out is not None:
        _write_private(args.answers_out, acceptance.review_answers(checkpoint))
    print(json.dumps({"experiment_status": report["experiment_status"], "selection": report["selection"],
                      "missing_operations": report["missing_operations"],
                      "interrupted_operations": report["interrupted_operations"],
                      "stop_code": report["stop_code"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)
    dry = commands.add_parser("dry-run", help="offline; zero provider calls")
    dry.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    dry.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    dry.add_argument("--config", type=Path, required=True)
    dry.add_argument("--out", type=Path, required=True)
    dry.set_defaults(func=_dry_run)
    run = commands.add_parser("run", help="authorized live acceptance run; no automatic retries")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--private-checkpoint", type=Path, required=True)
    run.add_argument("--program-ledger", type=Path, required=True)
    run.add_argument("--operations", type=int, help="run at most this many remaining operations")
    run.add_argument("--resume-after-rate-limit", action="store_true",
                     help="continue only after a recorded pre-transmission capacity stop")
    run.add_argument("--wait-for-token-reset", action="store_true",
                     help="opt-in bounded wait (<= 65 s) for a known short token-per-minute reset")
    run.set_defaults(func=_run)
    report = commands.add_parser("report", help="read-only metrics export")
    report.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    report.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--checkpoint", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--answers-out", type=Path,
                        help="private file with the generated answers for the owner's scientific review")
    report.set_defaults(func=_report)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (AuthorizationError, CampaignLocked, CheckpointError, ValueError, programs.ProgramLedgerError) as error:
        raise SystemExit(f"refused: {error}") from None


if __name__ == "__main__":
    raise SystemExit(main())
