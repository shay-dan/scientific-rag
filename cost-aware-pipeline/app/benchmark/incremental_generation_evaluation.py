"""Opt-in, resumable R7D execution with private atomic checkpoints.

The existing one-shot runner remains unchanged.  This module adds a deliberately small
stage runner that executes whole pairs in frozen-manifest order, writes a durable
write-ahead record before every provider attempt, and refuses to resume an unresolved
attempt.  That refusal is intentional: without provider idempotency, repeating an
attempt after process death could duplicate a billable request.
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
from typing import Callable, Iterator

from .generation_evaluation import (
    AuthorizationError,
    EvaluationStopped,
    Gateway,
    _dump,
    _evaluate_arm,
    _execution_order,
    _planning_prompts,
    _usage_from_trace,
    canonical_sha256,
    check_paid_authorization,
    validate_config,
    validate_manifest,
)
from .production_verifier_replay import BudgetLedger


CHECKPOINT_SCHEMA = "scientific-rag-generation-eval-checkpoint/v1"
SAFE_RATE_LIMIT_HEADERS = {
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
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


def _record_map(checkpoint: dict) -> dict[str, dict]:
    return {record["scenario_id"]: record for record in checkpoint["records"]}


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
    if checkpoint.get("active_attempt") is not None or checkpoint.get("active_arm") is not None:
        raise CheckpointError(
            "checkpoint contains an unresolved provider attempt or arm; refusing a duplicate request")
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
    for dimension in ("requests", "tokens"):
        seconds = _reset_seconds(headers.get(f"x-ratelimit-reset-{dimension}"))
        if seconds is not None:
            reset_at[dimension] = observed_at + seconds
    return {
        "headers": dict(headers),
        "observed_at_unix": observed_at,
        "reset_at_unix": reset_at,
    }


def _capacity_refusal(
    checkpoint: dict,
    prompt: str,
    max_output_tokens: int,
    *,
    now: float,
    allow_elapsed_reset: bool,
) -> tuple[str, str, list[str]] | None:
    calls = checkpoint["execution"]["calls"]
    if not calls:  # the first authorized benchmark inference establishes provider headers
        return None
    state = checkpoint["execution"].get("rate_limit_state") or {}
    headers = state.get("headers") if isinstance(state.get("headers"), dict) else {}
    remaining_requests = _number(headers.get("x-ratelimit-remaining-requests"))
    remaining_tokens = _number(headers.get("x-ratelimit-remaining-tokens"))
    if remaining_requests is None or remaining_tokens is None:
        return ("rate_limit_capacity_unknown",
                "provider remaining request/token capacity is unavailable", [])
    conservative_tokens = len(prompt.encode("utf-8")) + max_output_tokens
    blocked = []
    if remaining_requests < 1:
        blocked.append("requests")
    if remaining_tokens < conservative_tokens:
        blocked.append("tokens")
    if not blocked:
        return None
    if not allow_elapsed_reset:
        return ("rate_limit", "known provider request/token allowance cannot admit the next bounded attempt",
                blocked)
    reset_at = state.get("reset_at_unix") if isinstance(state.get("reset_at_unix"), dict) else {}
    required = {"requests": 1, "tokens": conservative_tokens}
    for dimension in blocked:
        deadline = reset_at.get(dimension)
        limit = _number(headers.get(f"x-ratelimit-limit-{dimension}"))
        if not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or limit is None:
            return ("rate_limit_capacity_unknown",
                    f"provider {dimension} reset metadata or account-specific limit is unavailable", blocked)
        if now < float(deadline):
            return ("rate_limit", f"provider {dimension} reset has not elapsed", blocked)
        if limit < required[dimension]:
            return ("rate_limit_capacity_unknown",
                    f"provider {dimension} limit cannot admit the next bounded attempt", blocked)
    return None


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
) -> dict:
    """Execute at most ``pairs`` frozen pairs and persist every provider attempt."""
    if type(pairs) is not int or pairs <= 0:
        raise ValueError("pairs must be a positive integer")
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
        "stopped": None,
    }
    checkpoint["execution"]["stages"].append(stage)
    checkpoint = checkpoint_store.write(checkpoint)

    async def metered(role, temperature, prompt, schema, *, max_output_tokens):
        nonlocal checkpoint, reset_continuation_authorized
        now = _clock_now(clock)
        refusal = _capacity_refusal(
            checkpoint, prompt, max_output_tokens, now=now,
            allow_elapsed_reset=reset_continuation_authorized,
        )
        if refusal:
            code, reason, blocked = refusal
            checkpoint["execution"]["stop_code"] = code
            checkpoint["execution"]["stopped"] = reason
            checkpoint["execution"]["stop_context"] = {
                "kind": "preflight_capacity",
                "blocked_dimensions": blocked,
                "observed_at_unix": now,
            }
            checkpoint_store.write(checkpoint)
            raise EvaluationStopped(reason)
        if reset_continuation_authorized:
            checkpoint["execution"]["rate_limit_continuations"].append({
                "authorized_at_unix": now,
                "stage_number": stage["stage_number"],
                "next_scenario_index": checkpoint["active_arm"]["scenario_index"],
                "next_prompt_version": checkpoint["active_arm"]["prompt_version"],
            })
            reset_continuation_authorized = False
            checkpoint = checkpoint_store.write(checkpoint)
        input_ceiling = len(prompt.encode("utf-8"))
        if input_ceiling > config["max_input_tokens"]:
            reason = "conservative input-token ceiling reached before the call"
            checkpoint["execution"]["stop_code"] = "input_ceiling"
            checkpoint["execution"]["stopped"] = reason
            checkpoint_store.write(checkpoint)
            raise EvaluationStopped(reason)
        reserved, _ = calculate_cost(config["provider"], config["model"], input_ceiling, max_output_tokens)
        if reserved is None or not ledger.reserve(reserved):
            reason = "authorized call or cost ceiling reached before the call"
            checkpoint["execution"]["stop_code"] = "budget"
            checkpoint["execution"]["stopped"] = reason
            checkpoint_store.write(checkpoint)
            raise EvaluationStopped(reason)
        phase = "initial" if getattr(schema, "__name__", "") == "StructuredEvidenceAnswerV2" else "repair"
        checkpoint["execution"]["budget"] = ledger.snapshot()
        checkpoint["active_attempt"] = {
            "scenario_id": checkpoint["active_arm"]["scenario_id"],
            "scenario_index": checkpoint["active_arm"]["scenario_index"],
            "prompt_version": checkpoint["active_arm"]["prompt_version"],
            "phase": phase,
            "prompt_sha256": canonical_sha256(prompt),
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
                checkpoint["execution"]["stop_code"] = "rate_limit"
                checkpoint["execution"]["stopped"] = "provider returned a rate-limit or quota-exhaustion response"
                checkpoint["execution"]["stop_context"] = {
                    "kind": "provider_rate_limit",
                    "call_index": len(checkpoint["execution"]["calls"]) - 1,
                }
            elif any(model != allowed for model in usage["provider_models"]):
                checkpoint["execution"]["stop_code"] = "model_drift"
                checkpoint["execution"]["stopped"] = "a call used a provider/model outside the authorization"
            elif usage["cost_status"] != "complete":
                checkpoint["execution"]["stop_code"] = "billing_unknown"
                checkpoint["execution"]["stopped"] = "provider usage was incomplete; spend is uncertain"
            elif usage["provider_calls"] != 1:
                checkpoint["execution"]["stop_code"] = "unexpected_retry"
                checkpoint["execution"]["stopped"] = "an unexpected retry or fallback was observed"
            elif isinstance(usage["output_tokens"], int) and usage["output_tokens"] > max_output_tokens:
                checkpoint["execution"]["stop_code"] = "output_limit"
                checkpoint["execution"]["stopped"] = "provider reported output beyond the output-token limit"
            checkpoint_store.write(checkpoint)
        if error is not None:
            raise error
        if checkpoint["execution"].get("stopped"):
            raise EvaluationStopped(checkpoint["execution"]["stopped"])
        return result

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
                checkpoint["active_arm"] = {
                    "scenario_id": scenario_id,
                    "scenario_index": scenario_index,
                    "prompt_version": version,
                    "attempts": [],
                }
                checkpoint = checkpoint_store.write(checkpoint)
                arm = await _evaluate_arm(scenario, version, config, metered)
                attempts = checkpoint["active_arm"]["attempts"]
                if not attempts:
                    # A pre-transmission capacity/budget refusal leaves this arm genuinely
                    # unexecuted so a later explicitly approved stage cannot mistake it for
                    # a failed provider response or duplicate a completed request.
                    checkpoint["active_arm"] = None
                    checkpoint["active_attempt"] = None
                    checkpoint = checkpoint_store.write(checkpoint)
                    break
                arm["gateway_invocations"] = arm["physical_calls"]
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
        checkpoint["execution"]["stop_code"] = "cancelled"
        checkpoint["execution"]["stopped"] = "execution was cancelled with an unresolved attempted arm"
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
    "new_checkpoint", "run_incremental_evaluation", "verify_checkpoint",
]
