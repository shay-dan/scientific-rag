"""R7D-F: explicit successor of a closed terminal generation rejection.

Inherit finalized records, not their calls or charges. Recheck the original immutable
checkpoint, plan and blind commitment. Never clear a terminal stop or replay an old arm.
"""
from __future__ import annotations

import copy

from . import program_ledger as programs
from .generation_evaluation import (
    CONFIG_SCHEMA_V2, SCORABLE_ARM_STATUSES, AuthorizationError,
    canonical_sha256, check_paid_authorization, dry_run, validate_blind_key,
    _not_executed_pair, _execution_order,
)

POLICY = "r7d-f-untouched-arms/v1"
STOP = "provider_structured_generation_failure"


def _require(condition, message):
    if not condition:
        raise AuthorizationError(message)


def _ordered_arms(records):
    return [{"scenario_id": row["scenario_id"], "arm": arm}
            for row in records for arm in row["arms"]]


def build_continuation_contract(*, manifest, config, key, program_ledger,
                                predecessor_store, predecessor_config,
                                predecessor_authorization, predecessor_plan):
    """Read only: require a closed reconciled predecessor with one terminal rejection."""
    from .incremental_generation_evaluation import verify_checkpoint, _pair_status

    old = predecessor_store.load()
    verified, file_hash = programs._read_checkpoint(predecessor_store.path)
    _require(verified == old, "predecessor changed during read")
    verify_checkpoint(old, manifest, predecessor_config, predecessor_authorization)
    _require(predecessor_config.get("schema") == config.get("schema") == CONFIG_SCHEMA_V2,
             "continuation requires v2 configurations")
    _require("continuation" not in predecessor_authorization and "continuation" not in old,
             "chained successors are not supported")
    _require(config == dict(predecessor_config, code_sha=config.get("code_sha"))
             and config["code_sha"] != predecessor_config["code_sha"],
             "only accepted runtime code SHA may change in successor config")
    validate_blind_key(manifest, key, predecessor_authorization.get("blind_key_sha256"))

    original_plan = dry_run(manifest, predecessor_config)
    next_plan = dry_run(manifest, config)
    _require({k: v for k, v in predecessor_plan.items() if k != "program"} == original_plan,
             "predecessor plan does not reproduce exactly")
    _require({k: v for k, v in original_plan.items() if k != "config_sha256"}
             == {k: v for k, v in next_plan.items() if k != "config_sha256"},
             "successor prompt, schema, schedule or planning contract drift")
    _require(not next_plan["blockers"], "successor dry run has blockers")
    if "program" in predecessor_plan:
        _require(predecessor_plan["program"]["baseline_sha256"]
                 == predecessor_authorization["program_baseline_sha256"],
                 "predecessor plan baseline mismatch")

    ex = old["execution"]
    calls = ex["calls"]
    records = old["records"]
    flat = _ordered_arms(records)
    _require(old.get("active_arm") is None and old.get("active_attempt") is None
             and ex["budget"]["unsettled_reserved_usd"] == 0,
             "predecessor has pending or unsettled work")
    context = ex.get("stop_context") or {}
    _require(ex.get("stop_code") == STOP and bool(ex.get("stopped"))
             and context.get("kind") == "provider_failure"
             and context.get("failure_class") == STOP
             and context.get("transmission") == "confirmed"
             and context.get("call_index") == len(calls) - 1 and bool(calls),
             "predecessor is not the supported terminal provider rejection")
    _require(bool(flat) and all(item["arm"]["status"] in SCORABLE_ARM_STATUSES
                               for item in flat[:-1])
             and flat[-1]["arm"]["status"] == "interrupted",
             "predecessor must have exactly one final interrupted arm")
    last = calls[-1]
    failure = last.get("provider_failure") or {}
    failed = flat[-1]["arm"]
    _require(last["phase"] == "initial" and failed.get("failure_phase") == "initial"
             and failed.get("provider_failure") == failure
             and failure.get("failure_class") == STOP and failure.get("transmission") == "confirmed"
             and failure.get("http_status") == 400 and failure.get("provider_error_code") == "json_validate_failed"
             and last["cost_status"] == "incomplete" and last["reserved_cost_usd"] > 0
             and all(c["cost_status"] == "complete" for c in calls[:-1]),
             "unsupported failed request or earlier uncertain consumption")
    schedule = [(row["scenario_id"], version)
                for i, row in enumerate(manifest["scenarios"])
                for version in _execution_order(predecessor_config, i)]
    identities = [(item["scenario_id"], item["arm"]["prompt_version"]) for item in flat]
    _require(identities == schedule[:len(identities)] and len(identities) < len(schedule),
             "predecessor arms are not a strict prefix of the frozen schedule")
    audits = []
    for row in records:
        _require(row["status"] == _pair_status(row["arms"]) and bool(row["arms"])
                 and row["physical_calls"] == sum(a["physical_calls"] for a in row["arms"]),
                 "predecessor record accounting mismatch")
        for arm in row["arms"]:
            attempts = arm.get("attempts") or []
            _require(len(attempts) in (1, 2) and arm["physical_calls"] == len(attempts)
                     and [a["phase"] for a in attempts] == ["initial", "repair"][:len(attempts)],
                     "predecessor arm attempt history invalid")
            for attempt in attempts:
                _require(attempt["scenario_id"] == row["scenario_id"]
                         and attempt["prompt_version"] == arm["prompt_version"]
                         and attempt["provider_calls"] == 1,
                         "predecessor attempt identity invalid")
                audits.append({k: v for k, v in attempt.items() if k not in ("result", "error_type")})
    _require(audits == calls, "predecessor arm attempts do not reproduce the call audit")

    ledger = program_ledger.load()
    closed = [c for c in ledger["campaigns"] if c["state"] == "closed"]
    _require(bool(closed), "predecessor must be closed in the program ledger")
    entry = closed[-1]
    _require(entry["campaign_id"] == old["authorization_id"]
             and entry["checkpoint_sha256"] == old["checkpoint_sha256"]
             and entry["checkpoint_file_sha256"] == file_hash
             and entry["checkpoint_path"] == str(predecessor_store.path.resolve())
             and entry["authorization_sha256"] == old["authorization_sha256"]
             and entry["config_sha256"] == old["config_sha256"]
             and entry["manifest_sha256"] == old["manifest_sha256"]
             and entry["stop_code"] == STOP and entry["unresolved_attempt"] is False,
             "predecessor is not the last exact closed campaign")
    baseline = programs.program_baseline(ledger)
    settled, unresolved = programs._budget_from_checkpoint(old)
    _require(not unresolved and baseline["budget"] == settled
             and baseline["program_id"] == predecessor_authorization["program_id"],
             "closed baseline does not match predecessor cumulative consumption")
    remaining = len(schedule) - len(identities)
    _require(baseline["remaining"]["calls"] >= 2 * remaining,
             "remaining request allowance cannot cover all untouched arms and one repair each")
    return {
        "policy": POLICY,
        "predecessor_id": old["authorization_id"],
        "predecessor_code_sha": old["code_sha"],
        "predecessor_config_sha256": old["config_sha256"],
        "predecessor_authorization_sha256": old["authorization_sha256"],
        "predecessor_checkpoint_sha256": old["checkpoint_sha256"],
        "predecessor_checkpoint_file_sha256": file_hash,
        "predecessor_plan_sha256": canonical_sha256(predecessor_plan),
        "inherited_arms_sha256": canonical_sha256(flat),
        "inherited_arm_count": len(flat),
        "inherited_call_count": len(calls),
        "remaining_arm_count": remaining,
        "program_baseline_sha256": baseline["sha256"],
    }


def authorized_seed(*, manifest, config, authorization, key, program_ledger,
                    checkpoint_store, predecessor):
    """Revalidate the exact predecessor every run, including capacity continuations."""
    contract = (authorization or {}).get("continuation")
    if predecessor is None:
        _require("continuation" not in (authorization or {}), "successor requires predecessor inputs")
        return None
    _require(isinstance(contract, dict) and program_ledger is not None,
             "predecessor inputs require explicit successor authorization and ledger")
    old_store, old_config, old_auth, old_plan = predecessor
    _require(old_store.path != checkpoint_store.path, "successor checkpoint must use a new path")
    check_paid_authorization(authorization, manifest=manifest, config=config)
    _require(authorization["authorization_id"] != old_auth["authorization_id"]
             and authorization.get("blind_key_sha256") == old_auth.get("blind_key_sha256")
             and authorization.get("program_id") == old_auth.get("program_id"),
             "successor identity or original blind commitment mismatch")
    actual = build_continuation_contract(
        manifest=manifest, config=config, key=key, program_ledger=program_ledger,
        predecessor_store=old_store, predecessor_config=old_config,
        predecessor_authorization=old_auth, predecessor_plan=old_plan)
    _require(contract == actual and authorization["program_baseline_sha256"] == actual["program_baseline_sha256"],
             "successor authorization does not bind the exact continuation contract")
    old, file_hash = programs._read_checkpoint(old_store.path)
    _require(file_hash == actual["predecessor_checkpoint_file_sha256"]
             and old["checkpoint_sha256"] == actual["predecessor_checkpoint_sha256"],
             "predecessor changed before seeding")
    return copy.deepcopy(old["records"])


def verify_inherited_records(checkpoint, authorization):
    """Reject edited inherited outcomes and replayed call or active-arm identities."""
    contract = authorization.get("continuation")
    if contract is None:
        _require("continuation" not in authorization and "continuation" not in checkpoint,
                 "unexpected continuation metadata")
        return
    _require(isinstance(contract, dict) and contract.get("policy") == POLICY
             and checkpoint.get("continuation") == contract,
             "checkpoint continuation binding mismatch")
    n = contract.get("inherited_arm_count")
    flat = _ordered_arms(checkpoint["records"])
    _require(type(n) is int and 0 < n < 40 and len(flat) >= n
             and canonical_sha256(flat[:n]) == contract.get("inherited_arms_sha256"),
             "inherited arm records changed or disappeared")
    inherited = {(x["scenario_id"], x["arm"]["prompt_version"]) for x in flat[:n]}
    attempts = list(checkpoint["execution"]["calls"])
    if checkpoint.get("active_arm") is not None:
        attempts.append(checkpoint["active_arm"])
    _require(not any((a["scenario_id"], a["prompt_version"]) in inherited for a in attempts),
             "successor attempted to replay an inherited arm")


def reporting_records(manifest, config, checkpoint):
    """Derived non-executable complete-denominator view; never writes a checkpoint."""
    if checkpoint.get("active_arm") is not None or checkpoint.get("active_attempt") is not None:
        raise ValueError("cannot export with pending work")
    records = checkpoint["records"]
    by_id = {r["scenario_id"]: r for r in records}
    ids = {s["scenario_id"] for s in manifest["scenarios"]}
    if len(by_id) != len(records) or not set(by_id) <= ids:
        raise ValueError("invalid report scenario identities")
    incomplete = len(records) != len(ids) or any(len(r["arms"]) < 2 for r in records)
    ex = checkpoint["execution"]
    if incomplete and (not ex.get("stopped") or ex.get("stop_code") == "rate_limit"):
        raise ValueError("partial export requires a terminal stop, not an unfinished or resumable run")
    return [copy.deepcopy(by_id[s["scenario_id"]]) if s["scenario_id"] in by_id
            else _not_executed_pair(s, config, i, "not attempted before terminal execution stop")
            for i, s in enumerate(manifest["scenarios"])]
