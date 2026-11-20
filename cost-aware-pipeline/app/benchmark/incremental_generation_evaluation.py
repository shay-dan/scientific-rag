"""Opt-in, resumable R7D execution with private atomic checkpoints.

A deliberately small stage runner executes whole pairs in frozen-manifest order, writes a
durable write-ahead record before every provider attempt, and refuses to resume an
unresolved attempt: without provider idempotency, repeating an attempt after process death
could duplicate a billable request.

Checkpoint v2 (R7D execution-safety correction) makes the arm itself durable. Before a
repair is sent, the arm is saved as ``repair_pending`` with its initial output, validation
errors and repair-prompt fingerprint. A pre-transmission capacity pause therefore leaves the
arm pending instead of failed, and a verified continuation sends only that repair. A crash
between writes resumes from saved results; nothing that reached the provider is repeated.

Per-minute capacity admission is separate from cost accounting. The cost ledger keeps its
conservative byte-per-token reservation. Admission uses an *estimate* calibrated from
provider-reported input tokens for each response schema, with a safety margin; it is never
treated as an upper bound. An under-estimate surfaces as a provider 429, which is terminal.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Awaitable, Callable, Iterator

from scientific_rag import provider_failures as pf

from . import program_ledger as programs
from .generation_evaluation import (
    ADMISSION_MARGIN,
    CONFIG_SCHEMA_V2,
    EXECUTION_STOP,
    MAX_TOKEN_RESET_WAIT_SECONDS,
    SCORABLE_ARM_STATUSES,
    AuthorizationError,
    EvaluationStopped,
    Gateway,
    ResumeIntegrityError,
    _dump,
    _evaluate_arm,
    _execution_order,
    _is_model_output_error,
    _planning_prompts,
    _scores_completion_budget_failures,
    _usage_from_trace,
    canonical_sha256,
    check_paid_authorization,
    outcome_dimensions,
    reservation_input_tokens,
    schema_bytes,
    validate_config,
    validate_manifest,
    validate_blind_key,
)
from .production_verifier_replay import BudgetLedger


CHECKPOINT_SCHEMA = "scientific-rag-generation-eval-checkpoint/v2"
LEGACY_CHECKPOINT_SCHEMAS = {"scientific-rag-generation-eval-checkpoint/v1"}
# Capacity dimensions: Groq documents ``requests`` (per day) and ``tokens`` (per minute).
# Separate input/output token limits are enforced only when the provider reports them.
TOKEN_DIMENSIONS = ("tokens", "input-tokens", "output-tokens")
CAPACITY_DIMENSIONS = ("requests",) + TOKEN_DIMENSIONS
SAFE_RATE_LIMIT_HEADERS = {"retry-after"} | {
    f"x-ratelimit-{kind}-{dimension}" for kind in ("limit", "remaining", "reset") for dimension in CAPACITY_DIMENSIONS
}
FREE_REFERENCE_TOKENS_PER_DAY = 200_000
RESET_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+(?:\.\d+)?)d)?"
    r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?$",
    re.IGNORECASE,
)
MAX_RESET_SECONDS = 7 * 24 * 60 * 60


class CheckpointError(ValueError):
    """A private checkpoint is unsafe, inconsistent, concurrent, or tampered with."""


class CampaignLocked(CheckpointError):
    """Another process already holds the campaign checkpoint lock."""


def _checkpoint_hash(value: dict) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "checkpoint_sha256"})


def _serialized(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


class AtomicCheckpoint:
    """Mode-600 JSON checkpoint with same-directory fsync+replace and an advisory lock."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def locked(self) -> Iterator[None]:
        if self.path.parent.exists():
            parent_mode = stat.S_IMODE(self.path.parent.stat().st_mode)
            if parent_mode & 0o077:
                raise CheckpointError("checkpoint directory must not grant group or other access")
        else:
            self.path.parent.mkdir(parents=True, mode=0o700)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CampaignLocked("another process is executing this campaign checkpoint") from error
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict:
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode & 0o077:
            raise CheckpointError("checkpoint permissions must not grant group or other access")
        value = json.loads(self.path.read_text(encoding="utf-8"), parse_constant=self._reject_constant)
        if isinstance(value, dict) and value.get("schema") in LEGACY_CHECKPOINT_SCHEMAS:
            raise CheckpointError(
                f"checkpoint uses {value['schema']}, written by an earlier accepted code version; this code "
                f"requires {CHECKPOINT_SCHEMA}. The file was not modified; it stays bound to its original "
                "authorization and code SHA.")
        if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointError(f"checkpoint must use {CHECKPOINT_SCHEMA}")
        if value.get("checkpoint_sha256") != _checkpoint_hash(value):
            raise CheckpointError("checkpoint checksum mismatch")
        return value

    def write(self, value: dict) -> dict:
        value["checkpoint_revision"] = int(value.get("checkpoint_revision") or 0) + 1
        value["checkpoint_sha256"] = _checkpoint_hash(value)
        payload = _serialized(value)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return value

    @staticmethod
    def _reject_constant(value: str):
        raise ValueError(f"non-finite JSON literal {value} is not permitted")


def _prompt_contracts(manifest: dict, config: dict) -> list[dict]:
    return [{key: item[key] for key in ("scenario_id", "prompt_version", "prompt_sha256")}
            for item in _planning_prompts(manifest, config)]


def _program_binding(authorization: dict, baseline: dict) -> dict:
    return {"program_id": authorization["program_id"], "baseline_sha256": baseline["sha256"],
            "baseline": dict(baseline["budget"])}


def new_checkpoint(manifest: dict, config: dict, authorization: dict, provenance: dict,
                   program_baseline: dict | None = None) -> dict:
    """A fresh campaign checkpoint. A v2 campaign starts from its program baseline (R7D-C)."""
    auth = check_paid_authorization(authorization, manifest=manifest, config=config)
    v2 = config["schema"] == CONFIG_SCHEMA_V2
    if v2 != (program_baseline is not None):
        raise CheckpointError("a v2 campaign requires its program baseline; a v1 campaign has none")
    budget = (dict(program_baseline["budget"]) if v2
              else BudgetLedger(config["max_calls"], config["max_cost_usd"]).snapshot())
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_revision": 0,
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": canonical_sha256(config),
        "authorization_sha256": canonical_sha256(authorization),
        "authorization_id": auth["authorization_id"],
        "code_sha": config["code_sha"],
        "provider": config["provider"],
        "model": config["model"],
        "prompt_contracts": _prompt_contracts(manifest, config),
        "records": [],
        "active_arm": None,
        "active_attempt": None,
        "execution": {
            "authorization_id": auth["authorization_id"],
            "budget": budget,
            "code": provenance,
            "stopped": None,
            "stop_code": None,
            "stop_context": None,
            "calls": [],
            "rate_limit_state": {},
            "rate_limit_continuations": [],
            "stages": [],
        },
    }
    if v2:
        value["program"] = _program_binding(authorization, program_baseline)
    return value


UNRESOLVED_MESSAGE = "checkpoint contains an unresolved provider attempt or arm; refusing a duplicate request"
REPAIR_SCHEMA = "V2RepairBatch"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _record_map(checkpoint: dict) -> dict[str, dict]:
    return {record["scenario_id"]: record for record in checkpoint["records"]}


def _next_arm(checkpoint: dict, config: dict, scenarios: list[dict]) -> tuple[int, str] | None:
    """The next unexecuted arm in frozen manifest and execution order."""
    records = _record_map(checkpoint)
    for index, scenario in enumerate(scenarios):
        done = {arm["prompt_version"] for arm in records.get(scenario["scenario_id"], {}).get("arms", [])}
        for version in _execution_order(config, index):
            if version not in done:
                return index, version
    return None


def _resolved_attempt(attempt: object) -> bool:
    """A response was received and fully accounted: it may be reused, never re-sent."""
    return (isinstance(attempt, dict) and attempt.get("result") is not None and attempt.get("error_type") is None
            and attempt.get("cost_status") == "complete" and attempt.get("provider_calls") == 1
            and isinstance(attempt.get("prompt_sha256"), str) and bool(_HEX64.match(attempt["prompt_sha256"])))


def _verify_active_arm(checkpoint: dict, config: dict, scenarios: list[dict], prompt_hashes: dict) -> None:
    """Accept a saved in-progress arm only if every recorded attempt is resolved.

    Anything that may have reached the provider without a recorded, fully accounted response
    is refused (fail closed). A resumable arm is the next arm in frozen order, in phase
    ``initial_pending`` (no response, or one response not yet validated) or
    ``repair_pending`` (initial response saved with its errors and repair fingerprint).
    """
    active = checkpoint.get("active_arm")
    if active is None:
        return
    unsafe = CheckpointError(UNRESOLVED_MESSAGE)
    if not isinstance(active, dict) or not isinstance(active.get("attempts"), list):
        raise unsafe
    expected = _next_arm(checkpoint, config, scenarios)
    if (expected is None or (active.get("scenario_index"), active.get("prompt_version")) != expected
            or active.get("scenario_id") != scenarios[expected[0]]["scenario_id"]):
        raise unsafe
    attempts = active["attempts"]
    if [attempt.get("phase") if isinstance(attempt, dict) else None for attempt in attempts] not in (
            [], ["initial"], ["initial", "repair"]):
        raise unsafe
    if not all(_resolved_attempt(attempt) for attempt in attempts):
        raise unsafe
    key = (active["scenario_id"], active["prompt_version"])
    if attempts and attempts[0]["prompt_sha256"] != prompt_hashes[key]:
        raise CheckpointError(f"checkpoint prompt fingerprint drifted for {key[0]}/{key[1]}")
    phase = active.get("phase")
    if phase == "initial_pending":
        if len(attempts) > 1:
            raise unsafe
    elif phase == "repair_pending":
        initial, pending = active.get("initial"), active.get("pending_operation")
        if (not attempts or not isinstance(initial, dict) or initial.get("output") != attempts[0]["result"]
                or not isinstance(initial.get("validation_errors"), dict) or not initial["validation_errors"]
                or not isinstance(pending, dict) or pending.get("phase") != "repair"
                or pending.get("schema") != REPAIR_SCHEMA
                or not isinstance(pending.get("prompt_sha256"), str) or not _HEX64.match(pending["prompt_sha256"])):
            raise unsafe
        if len(attempts) == 2 and attempts[1]["prompt_sha256"] != pending["prompt_sha256"]:
            raise unsafe
    else:
        raise unsafe
    audited = {(call.get("scenario_id"), call.get("prompt_version"), call.get("phase"), call.get("prompt_sha256"))
               for call in checkpoint["execution"].get("calls") or [] if isinstance(call, dict)}
    for attempt in attempts:
        if (key[0], key[1], attempt["phase"], attempt["prompt_sha256"]) not in audited:
            raise unsafe


def _resume_state(active: dict | None) -> dict | None:
    """Saved results for ``_evaluate_arm``; None when nothing reached the provider."""
    if not active or not active.get("attempts"):
        return None
    attempts = active["attempts"]
    resume = {"prompt_sha256": attempts[0]["prompt_sha256"], "initial_output": attempts[0]["result"]}
    if active.get("phase") == "repair_pending":
        resume["validation_errors"] = active["initial"]["validation_errors"]
        resume["repair_prompt_sha256"] = active["pending_operation"]["prompt_sha256"]
    if len(attempts) == 2:
        resume["repair_output"] = attempts[1]["result"]
        resume["repair_prompt_sha256"] = attempts[1]["prompt_sha256"]
    return resume


def verify_checkpoint(checkpoint: dict, manifest: dict, config: dict, authorization: dict) -> BudgetLedger:
    validate_manifest(manifest)
    validate_config(config)
    auth = check_paid_authorization(authorization, manifest=manifest, config=config)
    expected = {
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": canonical_sha256(config),
        "authorization_sha256": canonical_sha256(authorization),
        "authorization_id": auth["authorization_id"],
        "code_sha": config["code_sha"],
        "provider": config["provider"],
        "model": config["model"],
        "prompt_contracts": _prompt_contracts(manifest, config),
    }
    v2 = config["schema"] == CONFIG_SCHEMA_V2
    if v2:
        program = checkpoint.get("program")
        if (not isinstance(program, dict) or program.get("program_id") != authorization.get("program_id")
                or program.get("baseline_sha256") != authorization.get("program_baseline_sha256")):
            expected["program"] = "authorized program baseline"
    elif "program" in checkpoint:
        expected["program"] = None
    mismatched = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatched:
        raise CheckpointError("checkpoint binding mismatch: " + ", ".join(mismatched))
    baseline = checkpoint["program"]["baseline"] if v2 else BudgetLedger(1, 1.0).snapshot()
    if checkpoint.get("active_attempt") is not None:
        raise CheckpointError(UNRESOLVED_MESSAGE)
    execution = checkpoint.get("execution")
    if not isinstance(execution, dict) or execution.get("authorization_id") != auth["authorization_id"]:
        raise CheckpointError("checkpoint execution identity does not match the authorization")
    ledger = BudgetLedger.from_snapshot(config["max_calls"], config["max_cost_usd"], execution.get("budget"))
    scenarios = manifest["scenarios"]
    positions = {scenario["scenario_id"]: index for index, scenario in enumerate(scenarios)}
    seen: set[str] = set()
    last_position = -1
    prompt_hashes = {(item["scenario_id"], item["prompt_version"]): item["prompt_sha256"]
                     for item in expected["prompt_contracts"]}
    charged_calls = 0
    complete_cost = uncertain_cost = 0.0
    for record in checkpoint.get("records", []):
        scenario_id = record.get("scenario_id")
        if scenario_id not in positions or scenario_id in seen or positions[scenario_id] < last_position:
            raise CheckpointError("checkpoint records are duplicated or outside frozen manifest order")
        seen.add(scenario_id)
        last_position = positions[scenario_id]
        index = positions[scenario_id]
        if record.get("scenario_index") != index or record.get("execution_order") != _execution_order(config, index):
            raise CheckpointError(f"checkpoint execution order drifted for {scenario_id}")
        versions: set[str] = set()
        recorded_versions: list[str] = []
        for arm in record.get("arms", []):
            version = arm.get("prompt_version")
            if version in versions or version not in _execution_order(config, index):
                raise CheckpointError(f"checkpoint has invalid arms for {scenario_id}")
            versions.add(version)
            recorded_versions.append(version)
            if arm.get("prompt_sha256") != prompt_hashes[(scenario_id, version)]:
                raise CheckpointError(f"checkpoint prompt fingerprint drifted for {scenario_id}/{version}")
        if recorded_versions != _execution_order(config, index)[:len(recorded_versions)]:
            raise CheckpointError(f"checkpoint arm order drifted for {scenario_id}")
    _verify_active_arm(checkpoint, config, scenarios, prompt_hashes)
    from .generation_continuation import verify_inherited_records
    verify_inherited_records(checkpoint, authorization)
    calls = execution.get("calls")
    if not isinstance(calls, list):
        raise CheckpointError("checkpoint call audit must be a list")
    for call in calls:
        provider_calls = call.get("provider_calls")
        charged_calls += max(1, provider_calls) if type(provider_calls) is int and provider_calls >= 0 else 1
        if call.get("cost_status") == "complete":
            complete_cost += float(call.get("cost_usd") or 0.0)
        else:
            uncertain_cost += float(call.get("reserved_cost_usd") or 0.0)
    snapshot = ledger.snapshot()
    if charged_calls + baseline["provider_calls_charged"] != snapshot["provider_calls_charged"]:
        raise CheckpointError("checkpoint call audit does not reconcile with the cumulative call ledger")
    if not math.isclose(complete_cost + baseline["cost_complete_usd"], snapshot["cost_complete_usd"], abs_tol=1e-8):
        raise CheckpointError("checkpoint complete usage does not reconcile with the cumulative cost ledger")
    if not math.isclose(uncertain_cost + baseline["uncertain_reserved_usd"], snapshot["uncertain_reserved_usd"],
                        abs_tol=1e-8):
        raise CheckpointError("checkpoint uncertain usage does not reconcile with the cumulative cost ledger")
    return ledger


def _number(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _rate_headers(summary: dict) -> dict[str, str]:
    captured: dict[str, str] = {}
    for attempt in summary.get("attempts") or []:
        for key, value in (attempt.get("rate_limit_headers") or {}).items():
            lowered = str(key).lower()
            if lowered in SAFE_RATE_LIMIT_HEADERS:
                captured[lowered] = str(value)
    return captured


def _reset_seconds(value: object) -> float | None:
    """Parse Groq duration metadata such as ``1m0.5s`` without guessing units."""
    if not isinstance(value, str):
        return None
    match = RESET_DURATION_RE.fullmatch(value.strip())
    if match is None or not any(match.groupdict().values()):
        return None
    multipliers = {"days": 86_400, "hours": 3_600, "minutes": 60, "seconds": 1}
    seconds = sum(float(match.group(name) or 0) * multiplier
                  for name, multiplier in multipliers.items())
    if not math.isfinite(seconds) or seconds < 0 or seconds > MAX_RESET_SECONDS:
        return None
    return seconds


def _clock_now(clock: Callable[[], float]) -> float:
    now = float(clock())
    if not math.isfinite(now) or now < 0:
        raise CheckpointError("clock must return a finite non-negative Unix timestamp")
    return now


def _rate_limit_state(headers: dict[str, str], observed_at: float) -> dict:
    reset_at: dict[str, float] = {}
    for dimension in CAPACITY_DIMENSIONS:
        seconds = _reset_seconds(headers.get(f"x-ratelimit-reset-{dimension}"))
        if seconds is not None:
            reset_at[dimension] = observed_at + seconds
    return {
        "headers": dict(headers),
        "observed_at_unix": observed_at,
        "reset_at_unix": reset_at,
    }


def _token_calibration(calls: list[dict]) -> dict[str, float]:
    """Highest observed provider input tokens per request byte, per response schema."""
    ratios: dict[str, float] = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        tokens, request = call.get("input_tokens"), call.get("request_bytes")
        if (call.get("cost_status") == "complete" and call.get("provider_calls") == 1
                and type(tokens) is int and tokens > 0 and type(request) is int and request > 0
                and isinstance(call.get("schema"), str)):
            ratios[call["schema"]] = max(ratios.get(call["schema"], 0.0), tokens / request)
    return ratios


def admission_estimate(calls: list[dict], *, prompt_bytes: int, schema_name: str, schema_size: int,
                       max_output_tokens: int) -> dict:
    """Per-minute admission requirement. An estimate, never an upper bound.

    Same schema observed: calibrated ratio x request bytes x (1 + margin).
    Other schema observed only: calibrated text, plus this schema at one token per byte.
    Nothing observed: one token per request byte.
    """
    ratios = _token_calibration(calls)
    request = prompt_bytes + schema_size
    if schema_name in ratios:
        method = "calibrated_same_schema"
        input_tokens = math.ceil(ratios[schema_name] * request * (1 + ADMISSION_MARGIN))
    elif ratios:
        method = "calibrated_text_unobserved_schema_one_token_per_byte"
        input_tokens = math.ceil(max(ratios.values()) * prompt_bytes * (1 + ADMISSION_MARGIN)) + schema_size
    else:
        method = "uncalibrated_one_token_per_byte"
        input_tokens = request
    return {
        "method": method, "upper_bound": False, "margin": ADMISSION_MARGIN,
        "request_bytes": request, "estimated_input_tokens": input_tokens,
        "required": {"requests": 1, "tokens": input_tokens + max_output_tokens,
                     "input-tokens": input_tokens, "output-tokens": max_output_tokens},
    }


def _capacity_check(checkpoint: dict, estimate: dict, *, now: float, allow_elapsed_reset: bool) -> dict | None:
    """None when the next request may be sent; otherwise a pre-transmission refusal."""
    if not checkpoint["execution"]["calls"]:   # the first authorized inference establishes provider headers
        return None
    state = checkpoint["execution"].get("rate_limit_state") or {}
    headers = state.get("headers") if isinstance(state.get("headers"), dict) else {}
    remaining = {dimension: _number(headers.get(f"x-ratelimit-remaining-{dimension}"))
                 for dimension in CAPACITY_DIMENSIONS}
    token_capacity_known = remaining["tokens"] is not None or (
        remaining["input-tokens"] is not None and remaining["output-tokens"] is not None)
    if remaining["requests"] is None or not token_capacity_known:
        return {"code": "rate_limit_capacity_unknown", "blocked": [],
                "reason": "provider remaining request/token capacity is unavailable"}
    required = estimate["required"]
    blocked = [dimension for dimension in CAPACITY_DIMENSIONS
               if remaining[dimension] is not None and remaining[dimension] < required[dimension]]
    if not blocked:
        return None
    if not allow_elapsed_reset:
        return {"code": "rate_limit", "blocked": blocked,
                "reason": "known provider allowance cannot admit the next bounded attempt (estimated requirement)"}
    reset_at = state.get("reset_at_unix") if isinstance(state.get("reset_at_unix"), dict) else {}
    for dimension in blocked:
        deadline = reset_at.get(dimension)
        limit = _number(headers.get(f"x-ratelimit-limit-{dimension}"))
        if not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or limit is None:
            return {"code": "rate_limit_capacity_unknown", "blocked": blocked,
                    "reason": f"provider {dimension} reset metadata or account-specific limit is unavailable"}
        if now < float(deadline):
            return {"code": "rate_limit", "blocked": blocked, "reason": f"provider {dimension} reset has not elapsed"}
        if limit < required[dimension]:
            return {"code": "rate_limit_capacity_unknown", "blocked": blocked,
                    "reason": f"provider {dimension} limit cannot admit the next bounded attempt (estimated requirement)"}
    return None


def _short_token_wait(checkpoint: dict, refusal: dict, estimate: dict, now: float) -> float | None:
    """Seconds to wait for a known, short per-minute token reset; None when waiting is not allowed."""
    blocked = refusal.get("blocked") or []
    if refusal.get("code") != "rate_limit" or not blocked or any(item not in TOKEN_DIMENSIONS for item in blocked):
        return None   # never wait for request-per-day or unknown limits
    state = checkpoint["execution"].get("rate_limit_state") or {}
    headers = state.get("headers") if isinstance(state.get("headers"), dict) else {}
    reset_at = state.get("reset_at_unix") if isinstance(state.get("reset_at_unix"), dict) else {}
    delays = []
    for dimension in blocked:
        deadline = reset_at.get(dimension)
        limit = _number(headers.get(f"x-ratelimit-limit-{dimension}"))
        if (not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or limit is None
                or limit < estimate["required"][dimension]):
            return None
        delay = float(deadline) - now
        if delay > MAX_TOKEN_RESET_WAIT_SECONDS:
            return None
        delays.append(max(0.0, delay))
    return max(delays) + 1.0


def _reported_reasoning(summary: dict) -> dict:
    """Provider-reported reasoning tokens and finish reason (capture mode); never estimated."""
    diagnostics = [attempt.get("diagnostics") or {} for attempt in summary.get("attempts") or []]
    tokens = [item.get("reasoning_tokens") for item in diagnostics]
    reported = bool(tokens) and all(type(value) is int and value >= 0 for value in tokens)
    finish = [item.get("finish_reason") for item in diagnostics if isinstance(item.get("finish_reason"), str)]
    return {"reasoning_tokens": sum(tokens) if reported else None, "finish_reason": finish[-1] if finish else None}


def _quota_error(error: BaseException | None, summary: dict, failure: dict | None = None) -> bool:
    statuses = {attempt.get("status_code") for attempt in summary.get("attempts") or []}
    if 429 in statuses or (failure is not None and failure["failure_class"] == pf.PROVIDER_RATE_LIMIT):
        return True
    if failure is not None and failure["failure_class"] not in (pf.UNKNOWN_PROVIDER_FAILURE,
                                                                pf.TRANSIENT_INFRASTRUCTURE_FAILURE):
        # A typed provider error is not a quota error: its body (for example a failed
        # generation) can contain arbitrary scientific text that must not be pattern-matched.
        return False
    message = str(error or "").lower()
    return any(marker in message for marker in ("quota", "rate limit", "resource exhausted", "too many requests"))


def _pair_status(arms: list[dict]) -> str:
    """``completed`` only for two valid terminal model outcomes; never for an interruption."""
    if any(arm.get("status") not in SCORABLE_ARM_STATUSES for arm in arms):
        return "interrupted"
    return "completed" if len(arms) == 2 else "partial"


def _arm_usage(attempts: list[dict]) -> dict:
    return {
        "input_tokens": (sum(item["input_tokens"] for item in attempts)
                         if attempts and all(type(item.get("input_tokens")) is int for item in attempts) else None),
        "output_tokens": (sum(item["output_tokens"] for item in attempts)
                          if attempts and all(type(item.get("output_tokens")) is int for item in attempts) else None),
        "latency_ms": round(sum(float(item.get("latency_ms") or 0.0) for item in attempts), 1) if attempts else None,
        "cost_usd": (round(sum(float(item["cost_usd"]) for item in attempts), 8)
                     if attempts and all(item.get("cost_usd") is not None for item in attempts) else None),
        "cost_status": ("complete" if attempts and all(item.get("cost_status") == "complete" for item in attempts)
                        else "incomplete"),
    }


class GuardedExecution:
    """Pre-transmission safety, one bounded transmission and settlement (R7D-B/C, shared).

    Used by the benchmark runner and by the R7D-D acceptance runner, so both apply exactly the
    same admission, reservation, ledger-mark, stop-precedence and settlement rules. ``state``
    must be checkpoint-shaped (``execution`` calls/budget/rate-limit/stop fields,
    ``active_arm`` and ``active_attempt``) and is mutated in place.
    """

    def __init__(self, *, checkpoint: dict, checkpoint_store: AtomicCheckpoint, ledger: BudgetLedger,
                 config: dict, v2: bool, mark: Callable[[dict], None], clock: Callable[[], float],
                 sleep: Callable[[float], Awaitable[None]], wait_for_token_reset: bool,
                 reset_continuation_authorized: bool, stage: dict, gateway: Gateway) -> None:
        self.checkpoint, self.checkpoint_store, self.ledger = checkpoint, checkpoint_store, ledger
        self.config, self.v2, self.mark, self.clock, self.sleep = config, v2, mark, clock, sleep
        self.wait_for_token_reset = wait_for_token_reset
        self.reset_continuation_authorized = reset_continuation_authorized
        self.stage, self.gateway = stage, gateway

    def record_stop(self, code: str | None, reason: str | None, context: dict | None = None) -> None:
        self.checkpoint["execution"]["stop_code"] = code
        self.checkpoint["execution"]["stopped"] = reason
        self.checkpoint["execution"]["stop_context"] = context

    async def metered(self, role, temperature, prompt, schema, *, max_output_tokens, **settings):
        from scientific_rag import llm_gateway   # resolved at call time (tests patch its trace hooks)
        from scientific_rag.costing import calculate_cost
        schema_name = getattr(schema, "__name__", str(schema))
        phase = "initial" if schema_name == "StructuredEvidenceAnswerV2" else "repair"
        prompt_bytes = len(prompt.encode("utf-8"))
        estimate = admission_estimate(self.checkpoint["execution"]["calls"], prompt_bytes=prompt_bytes,
                                      schema_name=schema_name, schema_size=schema_bytes(schema),
                                      max_output_tokens=max_output_tokens)
        now = _clock_now(self.clock)
        refusal = _capacity_check(self.checkpoint, estimate, now=now, allow_elapsed_reset=self.reset_continuation_authorized)
        continuation_kind = "operator_resume_after_reset" if self.reset_continuation_authorized else None
        if refusal and self.wait_for_token_reset:
            delay = _short_token_wait(self.checkpoint, refusal, estimate, now)
            if delay is not None:
                # Durable before waiting: a restart here needs the normal verified continuation.
                self.record_stop("rate_limit", refusal["reason"], {
                    "kind": "preflight_capacity", "blocked_dimensions": refusal["blocked"],
                    "observed_at_unix": now, "phase": phase, "waiting_until_unix": now + delay,
                    "admission": estimate})
                self.checkpoint = self.checkpoint_store.write(self.checkpoint)
                await self.sleep(delay)
                now = _clock_now(self.clock)
                self.record_stop(None, None, None)
                self.checkpoint = self.checkpoint_store.write(self.checkpoint)
                refusal = _capacity_check(self.checkpoint, estimate, now=now, allow_elapsed_reset=True)
                continuation_kind = "in_process_token_reset_wait"
        if refusal:
            resumable = refusal["code"] == "rate_limit"
            self.record_stop(refusal["code"], refusal["reason"], {
                "kind": "preflight_capacity", "blocked_dimensions": refusal["blocked"],
                "observed_at_unix": now, "phase": phase, "admission": estimate})
            self.checkpoint_store.write(self.checkpoint)
            raise EvaluationStopped(refusal["reason"], code=refusal["code"], resumable=resumable)
        if continuation_kind is not None:
            self.checkpoint["execution"]["rate_limit_continuations"].append({
                "kind": continuation_kind,
                "authorized_at_unix": now,
                "stage_number": self.stage["stage_number"],
                "next_scenario_index": self.checkpoint["active_arm"]["scenario_index"],
                "next_prompt_version": self.checkpoint["active_arm"]["prompt_version"],
                "next_phase": phase,
            })
            self.reset_continuation_authorized = False
            self.checkpoint = self.checkpoint_store.write(self.checkpoint)
        reservation_input = reservation_input_tokens(self.config, prompt_bytes, schema)
        if reservation_input > self.config["max_input_tokens"]:
            reason = "conservative input-token ceiling reached before the call"
            self.record_stop("input_ceiling", reason)
            self.checkpoint_store.write(self.checkpoint)
            raise EvaluationStopped(reason, code="input_ceiling")
        reserved, _ = calculate_cost(self.config["provider"], self.config["model"], reservation_input, max_output_tokens)
        if reserved is None or not self.ledger.reserve(reserved):
            reason = "authorized call or cost ceiling reached before the call"
            self.record_stop("budget", reason)
            self.checkpoint_store.write(self.checkpoint)
            raise EvaluationStopped(reason, code="budget")
        prompt_sha256 = canonical_sha256(prompt)
        self.checkpoint["execution"]["budget"] = self.ledger.snapshot()
        self.checkpoint["active_attempt"] = {
            "scenario_id": self.checkpoint["active_arm"]["scenario_id"],
            "scenario_index": self.checkpoint["active_arm"]["scenario_index"],
            "prompt_version": self.checkpoint["active_arm"]["prompt_version"],
            "phase": phase,
            "prompt_sha256": prompt_sha256,
            "reserved_cost_usd": reserved,
            "state": "reserved",
        }
        self.checkpoint = self.checkpoint_store.write(self.checkpoint)
        try:
            self.mark(self.checkpoint)   # the ledger must cover this in-flight reservation before anything is sent
        except Exception as error:
            # Nothing was transmitted: release the reservation and stop (terminal, no retry). If
            # this state cannot be persisted, the durable reserved attempt stays unresolved.
            self.ledger.release(reserved)
            self.checkpoint["execution"]["budget"] = self.ledger.snapshot()
            self.checkpoint["active_attempt"] = None
            reason = "the pre-transmission ledger mark could not be recorded; nothing was sent"
            self.record_stop("ledger_mark_failed", reason, {"kind": "ledger_persistence", "phase": "pre_transmission",
                                                       "error_type": type(error).__name__})
            self.checkpoint = self.checkpoint_store.write(self.checkpoint)
            raise EvaluationStopped(reason, code="ledger_mark_failed") from error
        token = llm_gateway.begin_trace(capture_rate_limits=True)
        started = time.perf_counter()
        result = None
        error: BaseException | None = None
        try:
            result = await self.gateway(role, temperature, prompt, schema, max_output_tokens=max_output_tokens, **settings)
        except BaseException as caught:
            error = caught
        finally:
            summary = llm_gateway.summarize_trace(llm_gateway.end_trace(token))
            usage = _usage_from_trace(summary, reserved)
            failure = pf.describe_provider_failure(error, schema) if isinstance(error, Exception) else None
            local_failure = failure is not None and failure["failure_class"] == pf.LOCAL_STRUCTURED_OUTPUT_FAILURE
            # Predeclared policy (v2): explicit completion-budget exhaustion is a scored model
            # outcome. Its reservation stays charged when usage is unknown; it is never re-sent.
            scored_budget_failure = (failure is not None
                                     and failure["failure_class"] == pf.COMPLETION_BUDGET_EXHAUSTED
                                     and _scores_completion_budget_failures(self.config))
            if failure is not None:
                usage["provider_failure"] = failure
            usage["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            usage["rate_limit_headers"] = _rate_headers(summary)
            usage["rate_limit_observed_at_unix"] = _clock_now(self.clock)
            usage["phase"] = phase
            usage["schema"] = schema_name
            usage["prompt_sha256"] = prompt_sha256
            usage["prompt_bytes"] = prompt_bytes
            usage["request_bytes"] = estimate["request_bytes"]
            usage["reservation_input_tokens"] = reservation_input
            usage["admission"] = estimate
            usage.update(_reported_reasoning(summary))
            usage["scenario_id"] = self.checkpoint["active_arm"]["scenario_id"]
            usage["prompt_version"] = self.checkpoint["active_arm"]["prompt_version"]
            self.ledger.settle(reserved, usage)
            self.checkpoint["execution"]["budget"] = self.ledger.snapshot()
            self.checkpoint["execution"]["calls"].append(usage)
            self.checkpoint["execution"]["rate_limit_state"] = _rate_limit_state(
                usage["rate_limit_headers"], usage["rate_limit_observed_at_unix"])
            self.checkpoint["active_arm"]["attempts"].append({
                **usage,
                "result": _dump(result) if result is not None else None,
                "error_type": type(error).__name__ if error is not None else None,
            })
            self.checkpoint["active_attempt"] = None
            allowed = f"{self.config['provider']}:{self.config['model']}"
            if _quota_error(error, summary, failure):
                self.record_stop("rate_limit", "provider returned a rate-limit or quota-exhaustion response", {
                    "kind": "provider_rate_limit", "call_index": len(self.checkpoint["execution"]["calls"]) - 1})
            elif any(model != allowed for model in usage["provider_models"]):
                self.record_stop("model_drift", "a call used a provider/model outside the authorization")
            elif usage["cost_status"] == "complete" and usage["cost_usd"] > reserved + 1e-12:
                # Fail closed: the measured cost is kept (never replaced by the reservation).
                self.record_stop("reservation_exceeded", "measured cost exceeded the call's reservation")
            elif self.ledger.charged > self.config["max_cost_usd"] + 1e-9:
                self.record_stop("cost_ceiling_exceeded", "settled cost exceeded the cumulative cost ceiling")
            elif self.v2 and failure is not None and not local_failure and not scored_budget_failure:
                # Typed, terminal provider failure: never scored, never re-sent automatically.
                self.record_stop(failure["failure_class"], "the provider request failed; execution stopped", {
                    "kind": "provider_failure", "failure_class": failure["failure_class"],
                    "transmission": failure["transmission"], "error_type": type(error).__name__,
                    "call_index": len(self.checkpoint["execution"]["calls"]) - 1})
            elif usage["cost_status"] != "complete" and not scored_budget_failure:
                self.record_stop("billing_unknown", "provider usage was incomplete; spend is uncertain")
            elif usage["provider_calls"] != 1:
                self.record_stop("unexpected_retry", "an unexpected retry or fallback was observed")
            elif isinstance(usage["output_tokens"], int) and usage["output_tokens"] > max_output_tokens:
                self.record_stop("output_limit", "provider reported output beyond the output-token limit")
            elif isinstance(error, Exception) and not _is_model_output_error(error) and not scored_budget_failure:
                # A provider/network failure is not evidence about the prompt, and the next
                # request would meet the same provider state: stop the campaign (terminal).
                self.record_stop("provider_failure", "the provider request failed; execution stopped", {
                    "kind": "provider_failure", "error_type": type(error).__name__,
                    "call_index": len(self.checkpoint["execution"]["calls"]) - 1})
            self.checkpoint = self.checkpoint_store.write(self.checkpoint)
            if self.v2:
                # Refresh the ledger mark immediately after durable settlement, so a lost
                # checkpoint never hides settled consumption (cost above a reservation, retries).
                try:
                    self.mark(self.checkpoint)
                except Exception as mark_error:   # the mark stays "in_flight": lost-checkpoint close refuses
                    if not self.checkpoint["execution"].get("stopped"):
                        self.record_stop("ledger_mark_failed",
                                    "settled consumption could not be recorded in the program ledger",
                                    {"kind": "ledger_persistence", "phase": "post_settlement",
                                     "error_type": type(mark_error).__name__})
                        self.checkpoint = self.checkpoint_store.write(self.checkpoint)
        stop_code = self.checkpoint["execution"].get("stop_code")
        if stop_code is not None and scored_budget_failure:
            # P0.3: an execution-integrity stop outranks the scored completion-budget outcome.
            raise EvaluationStopped(self.checkpoint["execution"]["stopped"], code=stop_code, transmitted=True) from error
        if error is not None and (not isinstance(error, Exception) or not _is_model_output_error(error)
                                  or stop_code is None):
            # Cancellation and provider/network failures keep their own classification.
            raise error
        if stop_code is not None:
            # Safety stops (uncertain accounting, quota, model drift, retries, output limit)
            # take precedence over model-quality classification: never a scorable failure.
            raise EvaluationStopped(self.checkpoint["execution"]["stopped"], code=stop_code,
                                    transmitted=True) from error
        return result


async def run_incremental_evaluation(
    manifest: dict,
    config: dict,
    *,
    authorization: dict | None,
    checkpoint_store: AtomicCheckpoint,
    pairs: int,
    gateway: Gateway | None = None,
    resume_after_rate_limit: bool = False,
    provenance_resolver: Callable[[], dict] | None = None,
    clock: Callable[[], float] = time.time,
    wait_for_token_reset: bool = False,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    program_ledger: "programs.ProgramLedgerStore | None" = None,
    blind_key: dict | None = None,
    predecessor: tuple | None = None,
) -> dict:
    """Run up to ``pairs`` unexecuted pairs (see ``_run_incremental_evaluation``).

    R7D-C: the real provider gateway runs only for a v2 config with the program ledger, so no
    live route bypasses cumulative accounting; injected gateways keep historical v1 behavior
    for offline tests. A v2 campaign runs under its per-campaign lock, independent of the
    checkpoint path, so two processes cannot spend the same allowance concurrently.
    """
    from scientific_rag import llm_gateway
    v2 = isinstance(config, dict) and config.get("schema") == CONFIG_SCHEMA_V2
    if (gateway is None or gateway is llm_gateway.ainvoke_structured) and not (v2 and program_ledger is not None):
        raise AuthorizationError("real-provider execution requires a v2 config and the program ledger "
                                 "(cumulative accounting)")
    if gateway is None or gateway is llm_gateway.ainvoke_structured:
        try:
            validate_blind_key(manifest, blind_key, (authorization or {}).get("blind_key_sha256"))
        except ValueError as error:
            raise AuthorizationError("real-provider execution requires the precommitted private blind allocation") from error
    arguments = dict(authorization=authorization, checkpoint_store=checkpoint_store, pairs=pairs, gateway=gateway,
                     resume_after_rate_limit=resume_after_rate_limit, provenance_resolver=provenance_resolver,
                     clock=clock, wait_for_token_reset=wait_for_token_reset, sleep=sleep,
                     program_ledger=program_ledger)
    if not (v2 and program_ledger is not None):
        if predecessor is not None or "continuation" in (authorization or {}):
            raise AuthorizationError("successor requires v2 configuration and cumulative ledger")
        return await _run_incremental_evaluation(manifest, config, **arguments)
    campaign_id = str((authorization or {}).get("authorization_id") or "")
    if not campaign_id:
        raise AuthorizationError("authorization_id is required")
    with program_ledger.campaign_lock(campaign_id):
        # Marks are recorded before every transmission and immediately after its settlement.
        from .generation_continuation import authorized_seed
        seed = authorized_seed(manifest=manifest, config=config, authorization=authorization,
                               key=blind_key, program_ledger=program_ledger,
                               checkpoint_store=checkpoint_store, predecessor=predecessor)
        return await _run_incremental_evaluation(manifest, config, inherited_records=seed, **arguments)


async def _run_incremental_evaluation(
    manifest: dict,
    config: dict,
    *,
    authorization: dict,
    checkpoint_store: AtomicCheckpoint,
    pairs: int,
    gateway: Gateway | None = None,
    resume_after_rate_limit: bool = False,
    provenance_resolver: Callable[[], dict] | None = None,
    clock: Callable[[], float] = time.time,
    wait_for_token_reset: bool = False,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    program_ledger: "programs.ProgramLedgerStore | None" = None,
    inherited_records: list[dict] | None = None,
) -> dict:
    """Execute at most ``pairs`` frozen pairs and persist every provider attempt.

    ``wait_for_token_reset`` (off by default) lets a pre-transmission per-minute token pause
    wait in-process for a known reset of at most ``MAX_TOKEN_RESET_WAIT_SECONDS``. The pause
    is recorded durably before waiting, so a restart during the wait requires the normal
    ``resume_after_rate_limit`` continuation. Request-per-day and unknown limits never wait.

    A v2 config (R7D-C) requires ``program_ledger``: a new campaign opens there from the exact
    authorized cumulative baseline, and an existing campaign must still be open in it.
    """
    if type(pairs) is not int or pairs <= 0:
        raise ValueError("pairs must be a positive integer")
    if type(wait_for_token_reset) is not bool:
        raise ValueError("wait_for_token_reset must be a boolean")
    auth = check_paid_authorization(authorization, manifest=manifest, config=config)
    from ..evaluation_report import resolve_git_provenance
    from scientific_rag import llm_gateway
    from scientific_rag.costing import calculate_cost

    provenance = (provenance_resolver or resolve_git_provenance)()
    if provenance.get("sha") != config["code_sha"]:
        raise AuthorizationError("runtime code SHA does not match the owner-accepted code SHA")
    route = llm_gateway.provider_order("expert")
    if route != (config["provider"],) or llm_gateway.model_for(config["provider"], "expert") != config["model"]:
        raise AuthorizationError("expert gateway route is not pinned to the authorized provider/model")
    gateway = gateway or llm_gateway.ainvoke_structured
    v2 = config["schema"] == CONFIG_SCHEMA_V2
    if v2 and program_ledger is None:
        raise AuthorizationError("v2 configurations require the program ledger (cumulative accounting)")
    if not v2 and program_ledger is not None:
        raise AuthorizationError("the program ledger requires a v2 configuration")

    campaign_id = str(authorization["authorization_id"])

    def mark(value: dict) -> None:
        # Durable high-water mark of the authoritative checkpoint (P0.1): before transmission.
        if v2:
            programs.record_checkpoint_mark(program_ledger, campaign_id, checkpoint_store.path, value)

    if checkpoint_store.exists():
        checkpoint = checkpoint_store.load()
        if v2:
            if checkpoint.get("authorization_id") != campaign_id:
                raise CheckpointError("checkpoint binding mismatch: authorization_id")
            programs.require_runnable_campaign(program_ledger, campaign_id, checkpoint_store.path, checkpoint)
        ledger = verify_checkpoint(checkpoint, manifest, config, authorization)
    elif v2:
        entry = next((item for item in program_ledger.load()["campaigns"] if item["campaign_id"] == campaign_id), None)
        if entry is not None:
            # Registered earlier: only its own path, and only if nothing was ever marked.
            programs.require_runnable_campaign(program_ledger, campaign_id, checkpoint_store.path, None)
        baseline = programs.program_baseline(program_ledger.load())
        if entry is None and (authorization.get("program_id") != baseline["program_id"]
                              or authorization.get("program_baseline_sha256") != baseline["sha256"]):
            raise AuthorizationError("authorization program baseline does not match the program ledger")
        baseline = programs.open_campaign(program_ledger, authorization=authorization, config=config,
                                          manifest_sha256=manifest["manifest_sha256"],
                                          checkpoint_path=checkpoint_store.path)
        initial = new_checkpoint(manifest, config, authorization, provenance, baseline)
        if inherited_records is not None:
            initial["records"] = inherited_records
            initial["continuation"] = authorization["continuation"]
            verify_checkpoint(initial, manifest, config, authorization)
        checkpoint = checkpoint_store.write(initial)
        mark(checkpoint)
        ledger = BudgetLedger.from_snapshot(config["max_calls"], config["max_cost_usd"], checkpoint["execution"]["budget"])
    else:
        checkpoint = checkpoint_store.write(new_checkpoint(manifest, config, authorization, provenance))
        ledger = BudgetLedger(config["max_calls"], config["max_cost_usd"])
    stopped = checkpoint["execution"].get("stopped")
    stop_code = checkpoint["execution"].get("stop_code")
    reset_continuation_authorized = False
    if stopped:
        if not (resume_after_rate_limit and stop_code == "rate_limit"):
            raise CheckpointError(f"campaign checkpoint is stopped ({stop_code}): {stopped}")
        stop_context = checkpoint["execution"].get("stop_context") or {}
        if stop_context.get("kind") != "preflight_capacity":
            raise CheckpointError(
                "rate-limit continuation is allowed only after a pre-transmission capacity stop")
        reset_continuation_authorized = True
        checkpoint["execution"]["stopped"] = None
        checkpoint["execution"]["stop_code"] = None
        checkpoint["execution"]["stop_context"] = None
        checkpoint = checkpoint_store.write(checkpoint)
    elif resume_after_rate_limit:
        raise CheckpointError("--resume-after-rate-limit requires a recorded rate-limit stop")

    scenarios = manifest["scenarios"]
    records = _record_map(checkpoint)
    candidates = [index for index, scenario in enumerate(scenarios)
                  if len(records.get(scenario["scenario_id"], {}).get("arms", [])) < 2]
    if not candidates:
        raise CheckpointError("campaign has no unexecuted pairs; refusing repeated execution")
    selected = candidates[:pairs]
    stage = {
        "stage_number": len(checkpoint["execution"]["stages"]) + 1,
        "requested_pairs": pairs,
        "selected_scenario_indices": selected,
        "completed_pairs": 0,
        "rate_limit_continuation": reset_continuation_authorized,
        "resumed_active_arm": _resume_state(checkpoint.get("active_arm")) is not None,
        "wait_for_token_reset": wait_for_token_reset,
        "stopped": None,
    }
    checkpoint["execution"]["stages"].append(stage)
    checkpoint = checkpoint_store.write(checkpoint)

    guard = GuardedExecution(
        checkpoint=checkpoint, checkpoint_store=checkpoint_store, ledger=ledger, config=config, v2=v2,
        mark=mark, clock=clock, sleep=sleep, wait_for_token_reset=wait_for_token_reset,
        reset_continuation_authorized=reset_continuation_authorized, stage=stage, gateway=gateway)
    record_stop = guard.record_stop
    metered = guard.metered

    async def persist_repair_pending(state: dict) -> None:
        nonlocal checkpoint
        checkpoint["active_arm"]["phase"] = "repair_pending"
        checkpoint["active_arm"]["initial"] = {"output": state["initial_output"],
                                               "validation_errors": state["validation_errors"]}
        checkpoint["active_arm"]["pending_operation"] = {"phase": "repair", "schema": REPAIR_SCHEMA,
                                                         "prompt_sha256": state["repair_prompt_sha256"]}
        checkpoint = checkpoint_store.write(checkpoint)

    try:
        for scenario_index in selected:
            scenario = scenarios[scenario_index]
            scenario_id = scenario["scenario_id"]
            record = records.get(scenario_id)
            if record is None:
                record = {
                    "scenario_id": scenario_id,
                    "scenario_index": scenario_index,
                    "category": scenario["category"],
                    "execution_order": _execution_order(config, scenario_index),
                    "status": "partial",
                    "arms": [],
                    "physical_calls": 0,
                }
                checkpoint["records"].append(record)
                records[scenario_id] = record
                checkpoint = checkpoint_store.write(checkpoint)
            completed_versions = {arm["prompt_version"] for arm in record["arms"]}
            for version in record["execution_order"]:
                if version in completed_versions:
                    continue
                active = checkpoint.get("active_arm")
                if active and (active["scenario_index"], active["prompt_version"]) == (scenario_index, version):
                    resume = _resume_state(active)   # verified on load; never repeats a transmitted request
                else:
                    resume = None
                    checkpoint["active_arm"] = {
                        "scenario_id": scenario_id,
                        "scenario_index": scenario_index,
                        "prompt_version": version,
                        "phase": "initial_pending",
                        "attempts": [],
                    }
                    checkpoint = checkpoint_store.write(checkpoint)
                try:
                    arm = await _evaluate_arm(scenario, version, config, metered, resume=resume,
                                              before_repair=persist_repair_pending)
                except ResumeIntegrityError as error:
                    raise CheckpointError(f"saved arm state for {scenario_id}/{version} failed verification: "
                                          f"{error}") from error
                attempts = checkpoint["active_arm"]["attempts"]
                interruption = arm.get("interruption") or {}
                if not attempts and arm.get("status") == "interrupted" and arm.get("failure_class") != EXECUTION_STOP:
                    # An internal error before any settled transmission: fail closed with a terminal
                    # stop. Any reserved active attempt is kept as-is (unresolved, charged as
                    # uncertain), never cleared: whether it was sent cannot be established here.
                    record_stop("internal_error", "an internal error occurred before a settled transmission",
                                {"kind": "internal_error", "error_type": arm.get("error_type"),
                                 "reserved_attempt_kept": checkpoint.get("active_attempt") is not None})
                    checkpoint = checkpoint_store.write(checkpoint)
                    break
                if not attempts or (arm.get("failure_class") == EXECUTION_STOP and not interruption.get("transmitted")):
                    # Stopped before transmission: nothing was sent for this arm yet, or its
                    # untransmitted repair stays durably pending. Never a failed answer; the
                    # campaign stop code alone decides whether a continuation is allowed.
                    if not attempts:
                        checkpoint["active_arm"] = None
                    checkpoint["active_attempt"] = None
                    checkpoint = checkpoint_store.write(checkpoint)
                    break
                arm["gateway_invocations"] = len(attempts)
                arm["physical_calls"] = sum(int(item.get("provider_calls") or 0) for item in attempts)
                arm["usage"] = _arm_usage(attempts)
                arm["attempts"] = attempts
                arm["outcome_dimensions"] = outcome_dimensions(arm, attempts)
                record["arms"].append(arm)
                record["physical_calls"] = sum(item["physical_calls"] for item in record["arms"])
                record["status"] = _pair_status(record["arms"])
                checkpoint["active_arm"] = None
                checkpoint["active_attempt"] = None
                checkpoint = checkpoint_store.write(checkpoint)
                if checkpoint["execution"].get("stopped"):
                    break
            if record["status"] == "completed":
                stage["completed_pairs"] += 1
            if checkpoint["execution"].get("stopped"):
                break
    except asyncio.CancelledError:
        context = checkpoint["execution"].get("stop_context") or {}
        if not (checkpoint.get("active_attempt") is None and context.get("kind") == "preflight_capacity"):
            # A pre-transmission pause (for example during the opt-in wait) keeps its resumable
            # stop; anything else is terminal and refused on resume.
            record_stop("cancelled", "execution was cancelled with an unresolved attempted arm")
        stage["stopped"] = checkpoint["execution"]["stopped"]
        checkpoint_store.write(checkpoint)
        raise
    stage["stopped"] = checkpoint["execution"].get("stopped")
    checkpoint = checkpoint_store.write(checkpoint)
    measured_tokens = sum((call.get("input_tokens") or 0) + (call.get("output_tokens") or 0)
                          for call in checkpoint["execution"]["calls"]
                          if type(call.get("input_tokens")) is int and type(call.get("output_tokens")) is int)
    checkpoint["execution"]["rate_limit_state"]["campaign_estimated_remaining_free_tpd"] = str(
        max(0, FREE_REFERENCE_TOKENS_PER_DAY - measured_tokens))
    return checkpoint_store.write(checkpoint)


__all__ = [
    "AtomicCheckpoint", "CampaignLocked", "CheckpointError", "CHECKPOINT_SCHEMA",
    "admission_estimate", "new_checkpoint", "run_incremental_evaluation", "verify_checkpoint",
]
