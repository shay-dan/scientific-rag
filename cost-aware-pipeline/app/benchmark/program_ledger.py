"""R7D-C cumulative benchmark-program accounting (private, file-based).

One private ledger per benchmark program holds the fixed cumulative ceilings (at most the
R7D 80 physical requests and $1.50), every campaign that consumed them, and a separate
diagnostic category. It reuses the campaign checkpoint's guarantees: mode-600 JSON, a
checksum, same-directory fsync+replace and a non-blocking advisory lock. No database.

Accounting categories kept apart:
1. provider attempts: each campaign checkpoint's own call audit (unchanged);
2. per-experiment: a campaign's consumption beyond the baseline it was opened from;
3. cumulative program: closed campaigns plus the open campaign, against the fixed ceilings;
4. diagnostics: separately recorded requests that are never benchmark pairs;
5. actual provider billing: not available here; internal values are estimates/reservations.

Rules: historical checkpoints are only read (never modified); an import is idempotent by
checkpoint checksum and refuses a different state of a recorded campaign; unknown usage is
never zero (unresolved and unsettled reservations become uncertain charges); at most one
campaign is open; a campaign id is never reused; a closed campaign never runs again; a new
campaign starts from the exact baseline its owner authorization names.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterator

from .generation_evaluation import MAX_COST_USD, MAX_PHYSICAL_CALLS, canonical_sha256

PROGRAM_LEDGER_SCHEMA = "scientific-rag-generation-eval-program-ledger/v1"
# The campaign checkpoint schema (incremental runner); read-only here.
READABLE_CHECKPOINT_SCHEMAS = ("scientific-rag-generation-eval-checkpoint/v2",)
BUDGET_FIELDS = ("provider_calls_charged", "cost_complete_usd", "known_cost_lower_bound_usd",
                 "uncertain_reserved_usd", "unsettled_reserved_usd", "budget_charged_usd")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProgramLedgerError(ValueError):
    """The program ledger, or an operation on it, is unsafe, inconsistent or refused."""


class ProgramLedgerLocked(ProgramLedgerError):
    """Another process holds the program ledger lock."""


def _ledger_hash(value: dict) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "ledger_sha256"})


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON literal {value} is not permitted")


def _round(value: float) -> float:
    return round(max(float(value), 0.0), 8)


def _zero_budget() -> dict:
    return {field: (0 if field == "provider_calls_charged" else 0.0) for field in BUDGET_FIELDS}


def _private_parent(path: Path) -> None:
    if path.parent.exists():
        if stat.S_IMODE(path.parent.stat().st_mode) & 0o077:
            raise ProgramLedgerError("ledger directory must not grant group or other access")
    else:
        path.parent.mkdir(parents=True, mode=0o700)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ProgramLedgerStore:
    """Mode-600 ledger file with a checksum and a non-blocking advisory lock."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def locked(self) -> Iterator[None]:
        _private_parent(self.path)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ProgramLedgerLocked("another process is using this program ledger") from error
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict:
        if not self.path.exists():
            raise ProgramLedgerError("program ledger does not exist; initialize it explicitly")
        if stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise ProgramLedgerError("program ledger permissions must not grant group or other access")
        value = json.loads(self.path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        if not isinstance(value, dict) or value.get("schema") != PROGRAM_LEDGER_SCHEMA:
            raise ProgramLedgerError(f"program ledger must use {PROGRAM_LEDGER_SCHEMA}")
        if value.get("ledger_sha256") != _ledger_hash(value):
            raise ProgramLedgerError("program ledger checksum mismatch")
        _check_ceilings(value.get("ceilings"))
        return value

    def write(self, value: dict) -> dict:
        value["revision"] = int(value.get("revision") or 0) + 1
        value["ledger_sha256"] = _ledger_hash(value)
        _atomic_write(self.path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False) + "\n").encode("utf-8"))
        return value


def _check_ceilings(ceilings: object) -> dict:
    if not isinstance(ceilings, dict) or set(ceilings) != {"max_calls", "max_cost_usd"}:
        raise ProgramLedgerError("program ceilings must be max_calls and max_cost_usd")
    calls, cost = ceilings["max_calls"], ceilings["max_cost_usd"]
    if type(calls) is not int or not 0 < calls <= MAX_PHYSICAL_CALLS:
        raise ProgramLedgerError(f"program max_calls must be 1..{MAX_PHYSICAL_CALLS}")
    if (not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost)
            or not 0 < cost <= MAX_COST_USD):
        raise ProgramLedgerError(f"program max_cost_usd must be in (0, {MAX_COST_USD}]")
    return ceilings


def initialize_program_ledger(store: ProgramLedgerStore, *, program_id: str, max_calls: int,
                              max_cost_usd: float, now: float) -> dict:
    """Create a new program ledger. The ceilings are fixed for its lifetime."""
    if not isinstance(program_id, str) or not _ID.fullmatch(program_id):
        raise ProgramLedgerError("program_id must be a short identifier")
    ceilings = _check_ceilings({"max_calls": max_calls, "max_cost_usd": max_cost_usd})
    with store.locked():
        if store.exists():
            raise ProgramLedgerError("program ledger already exists; it is never re-initialized")
        return store.write({"schema": PROGRAM_LEDGER_SCHEMA, "program_id": program_id, "ceilings": ceilings,
                            "created_at_unix": float(now), "revision": 0, "campaigns": [], "diagnostics": []})


def _budget_from_checkpoint(checkpoint: dict) -> tuple[dict, bool]:
    """Cumulative charge recorded by a checkpoint, with unsettled reservations made uncertain."""
    execution = checkpoint.get("execution")
    budget = execution.get("budget") if isinstance(execution, dict) else None
    if not isinstance(budget, dict) or set(budget) != set(BUDGET_FIELDS):
        raise ProgramLedgerError("checkpoint budget does not match the ledger contract")
    calls = budget["provider_calls_charged"]
    values = {key: budget[key] for key in BUDGET_FIELDS if key != "provider_calls_charged"}
    if type(calls) is not int or calls < 0 or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0
            for value in values.values()):
        raise ProgramLedgerError("checkpoint budget values must be finite and non-negative")
    unresolved = checkpoint.get("active_attempt") is not None
    return {
        "provider_calls_charged": calls,
        "cost_complete_usd": _round(values["cost_complete_usd"]),
        "known_cost_lower_bound_usd": _round(values["known_cost_lower_bound_usd"]),
        # Unknown usage is never zero: an unsettled (possibly transmitted) reservation stays charged.
        "uncertain_reserved_usd": _round(values["uncertain_reserved_usd"] + values["unsettled_reserved_usd"]),
        "unsettled_reserved_usd": 0.0,
        "budget_charged_usd": _round(values["budget_charged_usd"]),
    }, unresolved


def _reconcile(checkpoint: dict, budget: dict, baseline: dict | None) -> None:
    """The checkpoint's call audit must account for its whole charge (minus any baseline)."""
    calls = checkpoint["execution"].get("calls")
    if not isinstance(calls, list):
        raise ProgramLedgerError("checkpoint call audit must be a list")
    base = baseline or _zero_budget()
    charged = complete = uncertain = 0.0
    charged_calls = 0
    for call in calls:
        provider_calls = call.get("provider_calls") if isinstance(call, dict) else None
        charged_calls += max(1, provider_calls) if type(provider_calls) is int and provider_calls >= 0 else 1
        if call.get("cost_status") == "complete":
            complete += float(call.get("cost_usd") or 0.0)
        else:
            uncertain += float(call.get("reserved_cost_usd") or 0.0)
    active = checkpoint.get("active_attempt")
    if active is not None:
        charged_calls += 1
        uncertain += float(active.get("reserved_cost_usd") or 0.0) if isinstance(active, dict) else 0.0
    charged = complete + uncertain
    expected = {
        "provider_calls_charged": base["provider_calls_charged"] + charged_calls,
        "cost_complete_usd": base["cost_complete_usd"] + complete,
        "uncertain_reserved_usd": base["uncertain_reserved_usd"] + uncertain,
        "budget_charged_usd": base["budget_charged_usd"] + charged,
    }
    # An unresolved attempt's reservation is still unsettled in the snapshot; it counts as uncertain.
    actual = dict(budget, uncertain_reserved_usd=budget["uncertain_reserved_usd"] + budget["unsettled_reserved_usd"])
    if actual["provider_calls_charged"] != expected["provider_calls_charged"] or not all(
            math.isclose(actual[key], expected[key], abs_tol=1e-8) for key in expected if key != "provider_calls_charged"):
        raise ProgramLedgerError("checkpoint call audit does not reconcile with its budget")


def _read_checkpoint(path: Path) -> tuple[dict, str]:
    path = Path(path)
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ProgramLedgerError("checkpoint permissions must not grant group or other access")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict) or value.get("schema") not in READABLE_CHECKPOINT_SCHEMAS:
        raise ProgramLedgerError("unsupported checkpoint schema")
    if value.get("checkpoint_sha256") != canonical_sha256(
            {key: item for key, item in value.items() if key != "checkpoint_sha256"}):
        raise ProgramLedgerError("checkpoint checksum mismatch")
    return value, hashlib.sha256(raw).hexdigest()


def _campaign(ledger: dict, campaign_id: str) -> dict | None:
    return next((item for item in ledger["campaigns"] if item["campaign_id"] == campaign_id), None)


def _closed_entry(checkpoint: dict, file_sha256: str, budget: dict, unresolved: bool, source: str) -> dict:
    return {
        "campaign_id": checkpoint["authorization_id"], "state": "closed", "source": source,
        "checkpoint_schema": checkpoint["schema"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_revision": checkpoint.get("checkpoint_revision"), "checkpoint_file_sha256": file_sha256,
        "config_sha256": checkpoint.get("config_sha256"), "manifest_sha256": checkpoint.get("manifest_sha256"),
        "code_sha": checkpoint.get("code_sha"), "stop_code": (checkpoint.get("execution") or {}).get("stop_code"),
        "unresolved_attempt": unresolved, "consumed": budget,
    }


def import_campaign(store: ProgramLedgerStore, checkpoint_path: Path) -> dict:
    """Record a pre-ledger campaign (read-only) as closed. Idempotent for identical checkpoints."""
    with store.locked():
        ledger = store.load()
        checkpoint, file_sha256 = _read_checkpoint(checkpoint_path)
        if checkpoint.get("program") is not None:
            raise ProgramLedgerError("this checkpoint belongs to a ledger campaign; close it instead of importing")
        campaign_id = checkpoint.get("authorization_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise ProgramLedgerError("checkpoint has no campaign identity")
        existing = _campaign(ledger, campaign_id)
        if existing is not None:
            if existing.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"]:
                return existing   # same campaign state: already imported
            raise ProgramLedgerError("campaign is already recorded with a different checkpoint state")
        budget, unresolved = _budget_from_checkpoint(checkpoint)
        _reconcile(checkpoint, _checkpoint_budget(checkpoint), None)
        entry = _closed_entry(checkpoint, file_sha256, budget, unresolved, "imported")
        _require_capacity(ledger, budget)
        ledger["campaigns"].append(entry)
        store.write(ledger)
        return entry


def _checkpoint_budget(checkpoint: dict) -> dict:
    return dict(checkpoint["execution"]["budget"])


def _sum_budgets(budgets: list[dict]) -> dict:
    total = _zero_budget()
    for budget in budgets:
        total["provider_calls_charged"] += int(budget["provider_calls_charged"])
        for key in BUDGET_FIELDS[1:]:
            total[key] += float(budget[key])
    return {key: (value if key == "provider_calls_charged" else _round(value)) for key, value in total.items()}


def _require_capacity(ledger: dict, added: dict) -> None:
    closed = [item["consumed"] for item in ledger["campaigns"] if item["state"] == "closed"]
    total = _sum_budgets(closed + [added])
    ceilings = ledger["ceilings"]
    if total["provider_calls_charged"] > ceilings["max_calls"] or total["budget_charged_usd"] > ceilings["max_cost_usd"] + 1e-9:
        # Recorded history can exceed a ceiling only if it was already spent elsewhere; refuse
        # silently absorbing it. The owner must decide (a new program, or an approved ceiling).
        raise ProgramLedgerError("recorded consumption would exceed the program ceilings")


def program_baseline(ledger: dict) -> dict:
    """Cumulative consumption of every closed campaign; a new campaign starts here."""
    closed = [item for item in ledger["campaigns"] if item["state"] == "closed"]
    budget = _sum_budgets([item["consumed"] for item in closed])
    commitment = {
        "program_id": ledger["program_id"], "ceilings": ledger["ceilings"],
        "closed_campaigns": [{"campaign_id": item["campaign_id"], "checkpoint_sha256": item["checkpoint_sha256"],
                              "consumed": item["consumed"]} for item in closed],
        "budget": budget,
    }
    ceilings = ledger["ceilings"]
    return {
        "sha256": canonical_sha256(commitment), "program_id": ledger["program_id"], "budget": budget,
        "ceilings": dict(ceilings), "closed_campaigns": len(closed),
        "remaining": {"calls": ceilings["max_calls"] - budget["provider_calls_charged"],
                      "cost_usd": _round(ceilings["max_cost_usd"] - budget["budget_charged_usd"])},
    }


def open_campaign(store: ProgramLedgerStore, *, authorization: dict, config: dict, manifest_sha256: str) -> dict:
    """Register a new campaign from the exact authorized baseline; returns that baseline."""
    with store.locked():
        ledger = store.load()
        baseline = program_baseline(ledger)
        if authorization.get("program_id") != ledger["program_id"]:
            raise ProgramLedgerError("authorization names a different benchmark program")
        if authorization.get("program_baseline_sha256") != baseline["sha256"]:
            raise ProgramLedgerError("authorization baseline does not match the current program ledger")
        if {"max_calls": config["max_calls"], "max_cost_usd": config["max_cost_usd"]} != ledger["ceilings"]:
            raise ProgramLedgerError("config ceilings must equal the program's cumulative ceilings")
        campaign_id = authorization["authorization_id"]
        existing = _campaign(ledger, campaign_id)
        identity = {"config_sha256": canonical_sha256(config), "manifest_sha256": manifest_sha256,
                    "authorization_sha256": canonical_sha256(authorization), "baseline_sha256": baseline["sha256"]}
        if existing is not None:
            if existing["state"] == "open" and all(existing.get(key) == value for key, value in identity.items()):
                return baseline   # opened, then interrupted before its checkpoint existed
            raise ProgramLedgerError("campaign identity is already recorded; an authorization is never reused")
        if any(item["state"] == "open" for item in ledger["campaigns"]):
            raise ProgramLedgerError("another campaign is already open; close it before opening a new one")
        ledger["campaigns"].append({"campaign_id": campaign_id, "state": "open", "source": "opened",
                                    "code_sha": config["code_sha"], "baseline": baseline["budget"], **identity})
        store.write(ledger)
        return baseline


def require_runnable_campaign(store: ProgramLedgerStore, campaign_id: str) -> dict:
    """The ledger entry of an existing campaign checkpoint; refuses closed or unknown campaigns."""
    ledger = store.load()
    entry = _campaign(ledger, campaign_id)
    if entry is None:
        raise ProgramLedgerError("campaign is not recorded in this program ledger")
    if entry["state"] != "open":
        raise ProgramLedgerError("campaign is closed; a closed campaign never runs again")
    return entry


def close_campaign(store: ProgramLedgerStore, checkpoint_store) -> dict:
    """Close the open campaign, recording its consumption. Its checkpoint is not modified."""
    with store.locked(), checkpoint_store.locked():
        ledger = store.load()
        checkpoint, file_sha256 = _read_checkpoint(checkpoint_store.path)
        program = checkpoint.get("program")
        if not isinstance(program, dict) or program.get("program_id") != ledger["program_id"]:
            raise ProgramLedgerError("checkpoint does not belong to this program ledger")
        entry = _campaign(ledger, checkpoint.get("authorization_id"))
        if entry is None:
            raise ProgramLedgerError("campaign is not recorded in this program ledger")
        if entry["state"] == "closed":
            if entry.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"]:
                return entry
            raise ProgramLedgerError("campaign is already closed with a different checkpoint state")
        if program.get("baseline_sha256") != entry["baseline_sha256"] or program.get("baseline") != entry["baseline"]:
            raise ProgramLedgerError("checkpoint baseline does not match the ledger's open campaign")
        cumulative, unresolved = _budget_from_checkpoint(checkpoint)
        _reconcile(checkpoint, _checkpoint_budget(checkpoint), entry["baseline"])
        base = entry["baseline"]
        consumed = {"provider_calls_charged": cumulative["provider_calls_charged"] - base["provider_calls_charged"]}
        for key in BUDGET_FIELDS[1:]:
            consumed[key] = _round(cumulative[key] - base[key])
        closed = _closed_entry(checkpoint, file_sha256, consumed, unresolved, "opened")
        closed.update({key: entry[key] for key in ("authorization_sha256", "baseline_sha256", "baseline")})
        ledger["campaigns"][ledger["campaigns"].index(entry)] = closed
        store.write(ledger)
        return closed


def record_diagnostic(store: ProgramLedgerStore, source_path: Path, *, label: str, requests: int,
                      estimated_cost_usd: float, uncertain_reserved_usd: float) -> dict:
    """Record separately-run diagnostic requests (never benchmark pairs). Idempotent per source file."""
    if not isinstance(label, str) or not _ID.fullmatch(label):
        raise ProgramLedgerError("diagnostic label must be a short identifier")
    if type(requests) is not int or requests < 0:
        raise ProgramLedgerError("diagnostic request count must be a non-negative integer")
    for value in (estimated_cost_usd, uncertain_reserved_usd):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ProgramLedgerError("diagnostic amounts must be finite and non-negative")
    source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    entry = {"label": label, "source_sha256": source_sha256, "requests": requests,
             "estimated_cost_usd": _round(estimated_cost_usd), "uncertain_reserved_usd": _round(uncertain_reserved_usd),
             "basis": "operator-declared from the hashed source; internal estimates, not provider billing"}
    with store.locked():
        ledger = store.load()
        existing = next((item for item in ledger["diagnostics"] if item["source_sha256"] == source_sha256), None)
        if existing is not None:
            if existing == entry:
                return existing
            raise ProgramLedgerError("diagnostic source is already recorded with different values")
        ledger["diagnostics"].append(entry)
        store.write(ledger)
        return entry


def accounting_summary(ledger: dict, checkpoint: dict | None = None) -> dict:
    """The five accounting categories, kept separate."""
    baseline = program_baseline(ledger)
    program = dict(baseline["budget"])
    experiment = None
    attempts = None
    if checkpoint is not None and isinstance(checkpoint.get("program"), dict):
        cumulative = checkpoint["execution"]["budget"]
        base = checkpoint["program"]["baseline"]
        experiment = {"provider_calls_charged": cumulative["provider_calls_charged"] - base["provider_calls_charged"]}
        for key in BUDGET_FIELDS[1:]:
            experiment[key] = _round(cumulative[key] - base[key])
        program = dict(cumulative)
        attempts = len(checkpoint["execution"].get("calls") or [])
    diagnostics = ledger["diagnostics"]
    return {
        "provider_attempts": {"campaign_call_records": attempts,
                              "campaigns": [{"campaign_id": item["campaign_id"], "state": item["state"]}
                                            for item in ledger["campaigns"]]},
        "experiment": experiment,
        "program": {**program, "ceilings": baseline["ceilings"],
                    "remaining_calls": baseline["ceilings"]["max_calls"] - program["provider_calls_charged"],
                    "remaining_cost_usd": _round(baseline["ceilings"]["max_cost_usd"] - program["budget_charged_usd"])},
        "diagnostics": {"entries": len(diagnostics), "requests": sum(item["requests"] for item in diagnostics),
                        "estimated_cost_usd": _round(sum(item["estimated_cost_usd"] for item in diagnostics)),
                        "uncertain_reserved_usd": _round(sum(item["uncertain_reserved_usd"] for item in diagnostics)),
                        "counted_against_benchmark_ceilings": False},
        "actual_provider_billing": {"status": "not_available",
                                    "note": "values are token-based estimates and conservative reservations, "
                                            "not provider charges"},
    }


__all__ = [
    "PROGRAM_LEDGER_SCHEMA", "ProgramLedgerError", "ProgramLedgerLocked", "ProgramLedgerStore",
    "accounting_summary", "close_campaign", "import_campaign", "initialize_program_ledger", "open_campaign",
    "program_baseline", "record_diagnostic", "require_runnable_campaign",
]
