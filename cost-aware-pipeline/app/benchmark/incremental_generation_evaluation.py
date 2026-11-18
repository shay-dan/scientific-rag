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

from .generation_evaluation import (
    ADMISSION_MARGIN,
    EXECUTION_STOP,
    MAX_TOKEN_RESET_WAIT_SECONDS,
    AuthorizationError,
    EvaluationStopped,
    Gateway,
    ResumeIntegrityError,
    _dump,
    _evaluate_arm,
    _execution_order,
    _planning_prompts,
    _usage_from_trace,
    canonical_sha256,
    check_paid_authorization,
    schema_bytes,
    validate_config,
    validate_manifest,
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


def new_checkpoint(manifest: dict, config: dict, authorization: dict, provenance: dict) -> dict:
    auth = check_paid_authorization(authorization, manifest=manifest, config=config)
    return {
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
            "budget": BudgetLedger(config["max_calls"], config["max_cost_usd"]).snapshot(),
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
    mismatched = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatched:
        raise CheckpointError("checkpoint binding mismatch: " + ", ".join(mismatched))
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
    if charged_calls != snapshot["provider_calls_charged"]:
        raise CheckpointError("checkpoint call audit does not reconcile with the cumulative call ledger")
    if not math.isclose(complete_cost, snapshot["cost_complete_usd"], abs_tol=1e-8):
        raise CheckpointError("checkpoint complete usage does not reconcile with the cumulative cost ledger")
    if not math.isclose(uncertain_cost, snapshot["uncertain_reserved_usd"], abs_tol=1e-8):
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


def _quota_error(error: BaseException | None, summary: dict) -> bool:
    statuses = {attempt.get("status_code") for attempt in summary.get("attempts") or []}
    if 429 in statuses:
        return True
    message = str(error or "").lower()
    return any(marker in message for marker in ("quota", "rate limit", "resource exhausted", "too many requests"))


def _pair_status(arms: list[dict]) -> str:
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


async def run_incremental_evaluation(
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
) -> dict:
    """Execute at most ``pairs`` frozen pairs and persist every provider attempt.

    ``wait_for_token_reset`` (off by default) lets a pre-transmission per-minute token pause
    wait in-process for a known reset of at most ``MAX_TOKEN_RESET_WAIT_SECONDS``. The pause
    is recorded durably before waiting, so a restart during the wait requires the normal
    ``resume_after_rate_limit`` continuation. Request-per-day and unknown limits never wait.
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

    if checkpoint_store.exists():
        checkpoint = checkpoint_store.load()
        ledger = verify_checkpoint(checkpoint, manifest, config, authorization)
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

    def record_stop(code: str, reason: str, context: dict | None = None) -> None:
        checkpoint["execution"]["stop_code"] = code
        checkpoint["execution"]["stopped"] = reason
        checkpoint["execution"]["stop_context"] = context

    async def metered(role, temperature, prompt, schema, *, max_output_tokens):
        nonlocal checkpoint, reset_continuation_authorized
        schema_name = getattr(schema, "__name__", str(schema))
        phase = "initial" if schema_name == "StructuredEvidenceAnswerV2" else "repair"
        prompt_bytes = len(prompt.encode("utf-8"))
        estimate = admission_estimate(checkpoint["execution"]["calls"], prompt_bytes=prompt_bytes,
                                      schema_name=schema_name, schema_size=schema_bytes(schema),
                                      max_output_tokens=max_output_tokens)
        now = _clock_now(clock)
        refusal = _capacity_check(checkpoint, estimate, now=now, allow_elapsed_reset=reset_continuation_authorized)
        continuation_kind = "operator_resume_after_reset" if reset_continuation_authorized else None
        if refusal and wait_for_token_reset:
            delay = _short_token_wait(checkpoint, refusal, estimate, now)
            if delay is not None:
                # Durable before waiting: a restart here needs the normal verified continuation.
                record_stop("rate_limit", refusal["reason"], {
                    "kind": "preflight_capacity", "blocked_dimensions": refusal["blocked"],
                    "observed_at_unix": now, "phase": phase, "waiting_until_unix": now + delay,
                    "admission": estimate})
                checkpoint = checkpoint_store.write(checkpoint)
                await sleep(delay)
                now = _clock_now(clock)
                record_stop(None, None, None)
                checkpoint = checkpoint_store.write(checkpoint)
                refusal = _capacity_check(checkpoint, estimate, now=now, allow_elapsed_reset=True)
                continuation_kind = "in_process_token_reset_wait"
        if refusal:
            resumable = refusal["code"] == "rate_limit"
            record_stop(refusal["code"], refusal["reason"], {
                "kind": "preflight_capacity", "blocked_dimensions": refusal["blocked"],
                "observed_at_unix": now, "phase": phase, "admission": estimate})
            checkpoint_store.write(checkpoint)
            raise EvaluationStopped(refusal["reason"], code=refusal["code"], resumable=resumable)
        if continuation_kind is not None:
            checkpoint["execution"]["rate_limit_continuations"].append({
                "kind": continuation_kind,
                "authorized_at_unix": now,
                "stage_number": stage["stage_number"],
                "next_scenario_index": checkpoint["active_arm"]["scenario_index"],
                "next_prompt_version": checkpoint["active_arm"]["prompt_version"],
                "next_phase": phase,
            })
            reset_continuation_authorized = False
            checkpoint = checkpoint_store.write(checkpoint)
        if prompt_bytes > config["max_input_tokens"]:
            reason = "conservative input-token ceiling reached before the call"
            record_stop("input_ceiling", reason)
            checkpoint_store.write(checkpoint)
            raise EvaluationStopped(reason, code="input_ceiling")
        reserved, _ = calculate_cost(config["provider"], config["model"], prompt_bytes, max_output_tokens)
        if reserved is None or not ledger.reserve(reserved):
            reason = "authorized call or cost ceiling reached before the call"
            record_stop("budget", reason)
            checkpoint_store.write(checkpoint)
            raise EvaluationStopped(reason, code="budget")
        prompt_sha256 = canonical_sha256(prompt)
        checkpoint["execution"]["budget"] = ledger.snapshot()
        checkpoint["active_attempt"] = {
            "scenario_id": checkpoint["active_arm"]["scenario_id"],
            "scenario_index": checkpoint["active_arm"]["scenario_index"],
            "prompt_version": checkpoint["active_arm"]["prompt_version"],
            "phase": phase,
            "prompt_sha256": prompt_sha256,
            "reserved_cost_usd": reserved,
            "state": "reserved",
        }
        checkpoint_store.write(checkpoint)
        token = llm_gateway.begin_trace(capture_rate_limits=True)
        started = time.perf_counter()
        result = None
        error: BaseException | None = None
        try:
            result = await gateway(role, temperature, prompt, schema, max_output_tokens=max_output_tokens)
        except BaseException as caught:
            error = caught
        finally:
            summary = llm_gateway.summarize_trace(llm_gateway.end_trace(token))
            usage = _usage_from_trace(summary, reserved)
            usage["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            usage["rate_limit_headers"] = _rate_headers(summary)
            usage["rate_limit_observed_at_unix"] = _clock_now(clock)
            usage["phase"] = phase
            usage["schema"] = schema_name
            usage["prompt_sha256"] = prompt_sha256
            usage["prompt_bytes"] = prompt_bytes
            usage["request_bytes"] = estimate["request_bytes"]
            usage["admission"] = estimate
            usage["scenario_id"] = checkpoint["active_arm"]["scenario_id"]
            usage["prompt_version"] = checkpoint["active_arm"]["prompt_version"]
            ledger.settle(reserved, usage)
            checkpoint["execution"]["budget"] = ledger.snapshot()
            checkpoint["execution"]["calls"].append(usage)
            checkpoint["execution"]["rate_limit_state"] = _rate_limit_state(
                usage["rate_limit_headers"], usage["rate_limit_observed_at_unix"])
            checkpoint["active_arm"]["attempts"].append({
                **usage,
                "result": _dump(result) if result is not None else None,
                "error_type": type(error).__name__ if error is not None else None,
            })
            checkpoint["active_attempt"] = None
            allowed = f"{config['provider']}:{config['model']}"
            if _quota_error(error, summary):
                record_stop("rate_limit", "provider returned a rate-limit or quota-exhaustion response", {
                    "kind": "provider_rate_limit", "call_index": len(checkpoint["execution"]["calls"]) - 1})
            elif any(model != allowed for model in usage["provider_models"]):
                record_stop("model_drift", "a call used a provider/model outside the authorization")
            elif usage["cost_status"] != "complete":
                record_stop("billing_unknown", "provider usage was incomplete; spend is uncertain")
            elif usage["provider_calls"] != 1:
                record_stop("unexpected_retry", "an unexpected retry or fallback was observed")
            elif isinstance(usage["output_tokens"], int) and usage["output_tokens"] > max_output_tokens:
                record_stop("output_limit", "provider reported output beyond the output-token limit")
            checkpoint_store.write(checkpoint)
        if error is not None:
            raise error
        if checkpoint["execution"].get("stopped"):
            raise EvaluationStopped(checkpoint["execution"]["stopped"],
                                    code=checkpoint["execution"]["stop_code"], transmitted=True)
        return result

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
