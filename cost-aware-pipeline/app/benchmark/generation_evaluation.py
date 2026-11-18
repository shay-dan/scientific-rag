"""R7D offline paired evaluation harness for the v15/v16 generation prompts.

This module is deliberately isolated from production orchestration.  It accepts only a
frozen, label-free development manifest and an injected structured-output gateway.  It
does not retrieve, mine, embed, plan, score scientific correctness, or select evidence.
Paid execution is fail-closed behind a future, versioned owner authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from typing import Awaitable, Callable

from langchain_core.documents import Document

from ..claim_checks import audit_answer_claims
from ..weak_model_synthesis import (
    EvidenceBlock,
    StructuredEvidenceAnswerV2,
    V2RepairBatch,
    generation_prompt_v2,
    merge_v2_repairs,
    render_evidence_block,
    repair_prompt_v2,
    validate_answer_v2,
)
from .production_verifier_replay import BudgetLedger, output_limit_status

MANIFEST_SCHEMA = "scientific-rag-generation-eval-manifest/v1"
CONFIG_SCHEMA = "scientific-rag-generation-eval-config/v1"
AUTHORIZATION_SCHEMA = "scientific-rag-generation-eval-authorization/v1"
ADAPTER = "paired_structured_generation_v15_v16"
ALLOWED_CATEGORIES = {"numeric_fidelity", "attribution", "study_design", "coverage"}
FORBIDDEN_LABEL_FIELDS = {
    "label", "labels", "gold", "gold_claim", "gold_claims", "reference", "reference_answer",
    "score", "scores", "winner", "correctness", "adjudication",
}
MAX_SCENARIOS = 20
MAX_PHYSICAL_CALLS = 80
MAX_COST_USD = 1.50
# A model-independent preflight upper bound: a tokenizer cannot emit more tokens than
# there are UTF-8 bytes available to encode. This is intentionally much more conservative
# than the ordinary four-character planning estimate.
INPUT_BYTES_PER_TOKEN_CEILING = 1
# Pydantic field maxima permit at most 20 cited answer fields.  This allowance
# conservatively covers their serialized text/quotes, targets and validation messages
# in the single repair prompt; actual calls are always reserved from actual prompt size.
REPAIR_PROMPT_ADDITIONAL_CHARS_CEILING = 70_000
# Per-minute capacity admission (incremental runner). This is an estimate, never a bound:
# input tokens are calibrated from provider-reported usage per response schema, then
# multiplied by (1 + ADMISSION_MARGIN). The cost ledger keeps its own byte-per-token bound.
ADMISSION_MARGIN = 0.25
# Opt-in in-process wait: only a known per-minute token reset this close is waited for.
MAX_TOKEN_RESET_WAIT_SECONDS = 65
REFERENCE_FREE_TOKENS_PER_MINUTE = 8_000   # published Groq Free sample; display only
Gateway = Callable[..., Awaitable[object]]
SCORE_METRICS = {
    "critical_error", "numeric_fidelity", "attribution", "study_design_causality",
    "coverage", "structural_validity", "repair_quality", "failure", "abstention",
}


class ManifestError(ValueError):
    """A fixed-evidence manifest is incomplete, mutable, labelled, or tampered with."""


class AuthorizationError(PermissionError):
    """A paid evaluation was requested without an exact future owner authorization."""


class EvaluationStopped(RuntimeError):
    """The runner, not the model, stopped an arm: capacity, budget, ceilings or accounting.

    ``transmitted`` tells whether the stopped operation reached the provider. A resumable
    stop is a pre-transmission capacity pause whose operation may run after a verified reset.
    """

    def __init__(self, reason: str = "evaluation stopped", *, code: str = "stopped",
                 transmitted: bool = False, resumable: bool = False) -> None:
        super().__init__(reason)
        self.code, self.transmitted, self.resumable = code, bool(transmitted), bool(resumable)


class ResumeIntegrityError(RuntimeError):
    """Saved arm state does not reproduce the frozen prompt, output or validation exactly."""


class _ModelOutputFailure(Exception):
    """A provider answered, and the answer itself failed the structured contract."""

    def __init__(self, kind: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.kind, self.cause = kind, cause


# Arm outcomes. Only ``completed`` and ``failed`` (a model-output failure) are valid terminal
# model outcomes and may enter paired evaluation. ``interrupted`` (runner stop, quota pause,
# provider/network/billing failure) and ``not_executed`` never count as prompt quality.
SCORABLE_ARM_STATUSES = frozenset({"completed", "failed"})
MODEL_OUTPUT = "model_output"
INFRASTRUCTURE = "infrastructure"
EXECUTION_STOP = "execution_stop"


def pair_is_scorable(arms: dict[str, dict]) -> bool:
    """A pair is comparable only when both arms reached a valid terminal model outcome."""
    return all(arm.get("status") in SCORABLE_ARM_STATUSES for arm in arms.values())


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def schema_bytes(schema: type) -> int:
    """UTF-8 size of a response schema's canonical JSON (its request overhead proxy)."""
    return len(canonical_json(schema.model_json_schema()).encode("utf-8"))


def canonical_sha256(value: object) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_hash(manifest: dict) -> str:
    return canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def _context_parts(scenario: dict) -> list[dict]:
    return [
        {
            "source_id": item.get("source_id"),
            "title": item.get("title"),
            "passage": item.get("passage"),
            "section": item.get("section"),
            "production_notes": item.get("production_notes", ""),
        }
        for item in scenario.get("evidence", [])
    ]


def question_context_hash(scenario: dict) -> str:
    return canonical_sha256({"question": scenario.get("question"), "evidence": _context_parts(scenario)})


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be non-empty text")
    return value


def _forbidden_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return (FORBIDDEN_LABEL_FIELDS.intersection(value)
                | set().union(*(_forbidden_keys(item) for item in value.values()), set()))
    if isinstance(value, list):
        return set().union(*(_forbidden_keys(item) for item in value), set())
    return set()


def validate_manifest(manifest: dict) -> dict:
    """Validate the immutable, label-free 20-scenario development contract."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"manifest must use {MANIFEST_SCHEMA}")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("split") != "development":
        raise ManifestError("only the reviewed development split is permitted")
    if selection.get("frozen") is not True or not str(selection.get("version") or "").strip():
        raise ManifestError("selection must have a version and be frozen before evaluation")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ManifestError("manifest provenance is required")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != MAX_SCENARIOS:
        raise ManifestError(f"manifest must contain exactly {MAX_SCENARIOS} scenarios")
    ids: set[str] = set()
    categories: set[str] = set()
    for position, scenario in enumerate(scenarios, 1):
        if not isinstance(scenario, dict):
            raise ManifestError(f"scenario {position} must be an object")
        forbidden = _forbidden_keys(scenario)
        if forbidden:
            raise ManifestError(f"scenario {position} contains prohibited scoring fields: {sorted(forbidden)}")
        scenario_id = _require_text(scenario.get("scenario_id"), f"scenario {position} scenario_id")
        if not re.fullmatch(r"D\d{3}", scenario_id):
            raise ManifestError(f"scenario {position} must use an opaque D### identifier")
        if scenario_id in ids:
            raise ManifestError(f"duplicate scenario_id {scenario_id}")
        ids.add(scenario_id)
        category = scenario.get("category")
        if category not in ALLOWED_CATEGORIES:
            raise ManifestError(f"scenario {scenario_id} has unsupported category")
        categories.add(category)
        _require_text(scenario.get("question"), f"scenario {scenario_id} question")
        evidence = scenario.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ManifestError(f"scenario {scenario_id} requires fixed evidence")
        source_ids: set[str] = set()
        for evidence_index, item in enumerate(evidence, 1):
            if not isinstance(item, dict):
                raise ManifestError(f"scenario {scenario_id} evidence {evidence_index} must be an object")
            source_id = _require_text(item.get("source_id"), "source_id")
            if not re.fullmatch(r"S\d+", source_id) or source_id in source_ids:
                raise ManifestError(f"scenario {scenario_id} has invalid or duplicate source_id {source_id}")
            source_ids.add(source_id)
            _require_text(item.get("title"), "title")
            passage = _require_text(item.get("passage"), "passage")
            _require_text(item.get("production_notes"), "production_notes")
            if item.get("evidence_sha256") != canonical_sha256(passage):
                raise ManifestError(f"scenario {scenario_id} evidence hash mismatch")
        if scenario.get("question_context_sha256") != question_context_hash(scenario):
            raise ManifestError(f"scenario {scenario_id} question/context hash mismatch")
    if categories != ALLOWED_CATEGORIES:
        raise ManifestError("the frozen selection must cover every required category")
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        raise ManifestError("manifest hash mismatch")
    return manifest


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config must use {CONFIG_SCHEMA}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(config.get("code_sha") or "")):
        raise ValueError("config code_sha must be an exact commit SHA")
    for field in ("provider", "model"):
        if not str(config.get(field) or "").strip():
            raise ValueError(f"config {field} is required")
    if config.get("temperature") != 0:
        raise ValueError("paired generation temperature must be zero")
    if config.get("prompt_versions") != ["v15", "v16"]:
        raise ValueError("the paired prompt versions must be exactly v15 and v16")
    if config.get("semantic_verification_mode") != "off":
        raise ValueError("semantic verification must be off for the paired evaluation")
    if config.get("claim_recovery_mode") != "strict":
        raise ValueError("claim recovery must be strict")
    if config.get("claim_audit_mode") not in {"off", "audit"}:
        raise ValueError("claim audit mode must be off or audit")
    for field in ("model_context_tokens", "max_input_tokens", "max_output_tokens", "max_calls"):
        if not isinstance(config.get(field), int) or isinstance(config[field], bool) or config[field] <= 0:
            raise ValueError(f"config {field} must be a positive integer")
    cost = config.get("max_cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("config max_cost_usd must be finite and positive")
    if config["max_calls"] > MAX_PHYSICAL_CALLS or cost > MAX_COST_USD:
        raise ValueError("config exceeds the proposed R7D pilot ceiling")
    if config["max_input_tokens"] + config["max_output_tokens"] > config["model_context_tokens"]:
        raise ValueError("input and output ceilings exceed the pinned model context window")
    return config


def _blocks_and_context(scenario: dict) -> tuple[list[EvidenceBlock], str]:
    blocks, rendered = [], []
    for item in scenario["evidence"]:
        document = Document(page_content=item["passage"], metadata={
            "citation_id": item["source_id"], "title": item["title"],
        })
        blocks.append(EvidenceBlock(item["source_id"], item["title"], item["passage"], document))
        rendered.append(render_evidence_block(
            item["source_id"], item["title"], item["passage"],
            notes=item.get("production_notes", ""), section=item.get("section"),
        ))
    return blocks, "\n\n".join(rendered)


def _dump(value: object) -> object:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _is_model_output_error(error: BaseException) -> bool:
    name = type(error).__name__
    return "InvalidStructured" in name or "ValidationError" in name


def _not_executed_arm(scenario: dict, prompt_version: str, reason: str) -> dict:
    """A complete arm-shaped record for a call that was never started."""
    _, context = _blocks_and_context(scenario)
    prompt = generation_prompt_v2(scenario["question"], context, prompt_version=prompt_version)
    return {
        "prompt_version": prompt_version,
        "question_context_sha256": scenario["question_context_sha256"],
        "evidence_sha256": [item["evidence_sha256"] for item in scenario["evidence"]],
        "prompt_sha256": canonical_sha256(prompt),
        "status": "not_executed", "reason": reason,
        "repair_attempted": False, "physical_calls": 0,
        "initial_output": None, "post_repair_output": None, "validation_errors": {},
    }


def _execution_order(config: dict, execution_index: int) -> list[str]:
    """Alternate the first arm by frozen manifest position."""
    versions = list(config["prompt_versions"])
    return versions if execution_index % 2 == 0 else list(reversed(versions))


async def _evaluate_arm(scenario: dict, prompt_version: str, config: dict, gateway: Gateway, *,
                        resume: dict | None = None,
                        before_repair: Callable[[dict], Awaitable[None]] | None = None) -> dict:
    """Run one arm: one generation, then at most one repair.

    ``before_repair`` (incremental runner) durably records the pending repair before it is
    sent. ``resume`` continues an arm from saved results without repeating a transmitted
    request: ``initial_output`` (and optionally ``repair_output``) are restored, and the
    saved prompt fingerprints and validation errors must reproduce exactly.
    """
    blocks, context = _blocks_and_context(scenario)
    prompt = generation_prompt_v2(scenario["question"], context, prompt_version=prompt_version)
    prompt_sha256 = canonical_sha256(prompt)
    result = {
        "prompt_version": prompt_version,
        "question_context_sha256": scenario["question_context_sha256"],
        "evidence_sha256": [item["evidence_sha256"] for item in scenario["evidence"]],
        "prompt_sha256": prompt_sha256,
        "status": "failed", "repair_attempted": False, "physical_calls": 0,
        "initial_output": None, "post_repair_output": None, "validation_errors": {},
    }
    phase = "initial"

    async def call(call_prompt: str, schema: type):
        try:
            raw = await gateway("expert", 0, call_prompt, schema, max_output_tokens=config["max_output_tokens"])
        except (EvaluationStopped, ResumeIntegrityError):
            raise
        except Exception as error:
            if _is_model_output_error(error):
                raise _ModelOutputFailure("invalid_structured_output", error) from error
            raise
        return _parsed(raw, schema)

    try:
        if resume is None:
            result["physical_calls"] += 1
            answer = await call(prompt, StructuredEvidenceAnswerV2)
        else:
            if resume.get("prompt_sha256") != prompt_sha256:
                raise ResumeIntegrityError("saved generation prompt fingerprint does not match the frozen prompt")
            answer = _restored(StructuredEvidenceAnswerV2, resume.get("initial_output"), "initial output")
            result["physical_calls"] = 1
        result["initial_output"] = _dump(answer)
        errors = validate_answer_v2(answer, blocks)
        result["initial_structurally_valid"] = not errors
        result["validation_errors"] = {key: list(value) for key, value in errors.items()}
        if resume is not None and resume.get("validation_errors") is not None \
                and resume["validation_errors"] != result["validation_errors"]:
            raise ResumeIntegrityError("saved validation errors do not reproduce from the saved initial output")
        audit = {"preserved": [], "repaired": [], "dropped": []}
        if errors:
            phase = "repair"
            result["repair_attempted"] = True
            repair_prompt = repair_prompt_v2(scenario["question"], context, answer, errors)
            result["repair_prompt_sha256"] = canonical_sha256(repair_prompt)
            expected = (resume or {}).get("repair_prompt_sha256")
            if expected is not None and expected != result["repair_prompt_sha256"]:
                raise ResumeIntegrityError("saved repair prompt fingerprint does not match the rebuilt repair prompt")
            if resume is not None and resume.get("repair_output") is not None:
                batch = _restored(V2RepairBatch, resume["repair_output"], "repair output")
                result["physical_calls"] = 2
            else:
                if before_repair is not None:
                    await before_repair({"initial_output": result["initial_output"],
                                         "validation_errors": result["validation_errors"],
                                         "repair_prompt_sha256": result["repair_prompt_sha256"]})
                result["physical_calls"] += 1
                batch = await call(repair_prompt, V2RepairBatch)
            result["repair_output"] = _dump(batch)
            try:
                answer, audit, remaining = merge_v2_repairs(answer, batch, blocks)
            except ValueError as error:
                raise _ModelOutputFailure("invalid_repair_response", error) from error
            if remaining:
                raise _ModelOutputFailure("validation_failed_after_repair", ValueError(
                    f"structured validation failed after bounded repair: {sorted(remaining)}"))
        result["post_repair_output"] = _dump(answer)
        result["unanswered_parts"] = len(answer.unanswered_parts)
        result["repair_audit"] = audit
        if config["claim_audit_mode"] == "audit":
            result["claim_audit"] = audit_answer_claims(answer, blocks)
        result["status"] = "completed"
    except ResumeIntegrityError:
        raise
    except _ModelOutputFailure as failure:
        result.update(status="failed", failure_class=MODEL_OUTPUT, failure_kind=failure.kind,
                      failure_phase=phase, error_type=type(failure.cause).__name__, error=str(failure.cause))
    except EvaluationStopped as stop:
        result.update(status="interrupted", failure_class=EXECUTION_STOP,
                      failure_kind="capacity_pause" if stop.resumable else "budget_or_accounting_stop",
                      failure_phase=phase, error_type=type(stop).__name__, error=str(stop),
                      interruption={"phase": phase, "code": stop.code, "transmitted": stop.transmitted,
                                    "resumable": stop.resumable})
    except Exception as error:
        # Provider, network or gateway failure: not evidence about the prompt.
        result.update(status="interrupted", failure_class=INFRASTRUCTURE, failure_kind="gateway_failure",
                      failure_phase=phase, error_type=type(error).__name__, error=str(error),
                      interruption={"phase": phase, "code": "infrastructure", "transmitted": True,
                                    "resumable": False})
    return result


def _parsed(raw: object, schema: type):
    try:
        return raw if isinstance(raw, schema) else schema.model_validate(raw)
    except Exception as error:
        raise _ModelOutputFailure("invalid_structured_output", error) from error


def _restored(schema: type, value: object, label: str):
    try:
        return schema.model_validate(value)
    except Exception as error:
        raise ResumeIntegrityError(f"saved {label} does not match its schema") from error


async def evaluate_pair(scenario: dict, *, config: dict, gateway: Gateway,
                        execution_index: int = 0) -> dict:
    """Evaluate both arms sequentially on one fixed scenario; recovery makes no extra calls."""
    validate_config(config)
    if not isinstance(execution_index, int) or isinstance(execution_index, bool) or execution_index < 0:
        raise ValueError("execution_index must be a non-negative integer")
    order = _execution_order(config, execution_index)
    arms = []
    for position, version in enumerate(order):
        arm = await _evaluate_arm(scenario, version, config, gateway)
        arms.append(arm)
        if arm.get("failure_class") == EXECUTION_STOP:
            arms.extend(_not_executed_arm(scenario, pending, arm.get("error") or "evaluation stopped")
                        for pending in order[position + 1:])
            break
    interrupted = any(arm["status"] in {"not_executed", "interrupted"} for arm in arms)
    return {"scenario_id": scenario["scenario_id"], "category": scenario["category"],
            "execution_order": order, "status": "interrupted" if interrupted else "completed", "arms": arms,
            "physical_calls": sum(arm["physical_calls"] for arm in arms)}


def _not_executed_pair(scenario: dict, config: dict, execution_index: int, reason: str) -> dict:
    order = _execution_order(config, execution_index)
    return {"scenario_id": scenario["scenario_id"], "category": scenario["category"],
            "execution_order": order, "status": "not_executed", "reason": reason,
            "arms": [_not_executed_arm(scenario, version, reason) for version in order],
            "physical_calls": 0}


def _normalized_arms(record: dict) -> dict[str, dict]:
    """Return both arm identities, safely filling an interrupted legacy omission."""
    arms: dict[str, dict] = {}
    for arm in record.get("arms", []):
        version = arm.get("prompt_version")
        if version not in {"v15", "v16"} or version in arms:
            raise ValueError(f"record {record.get('scenario_id')} has invalid or duplicate prompt-version arms")
        arms[version] = arm
    reason = str(record.get("reason") or "arm was absent from the interrupted experiment record")
    for version in ("v15", "v16"):
        arms.setdefault(version, {"prompt_version": version, "status": "not_executed", "reason": reason,
                                  "repair_attempted": False, "physical_calls": 0,
                                  "initial_output": None, "post_repair_output": None})
    return arms


def _planning_prompts(manifest: dict, config: dict) -> list[dict]:
    prompts = []
    for scenario in manifest["scenarios"]:
        _, context = _blocks_and_context(scenario)
        for version in config["prompt_versions"]:
            prompt = generation_prompt_v2(scenario["question"], context, prompt_version=version)
            prompts.append({"scenario_id": scenario["scenario_id"], "prompt_version": version,
                            "prompt_sha256": canonical_sha256(prompt), "prompt_chars": len(prompt),
                            "prompt_bytes": len(prompt.encode("utf-8"))})
    return prompts


def dry_run(manifest: dict, config: dict) -> dict:
    """Validate and plan the complete paired run without constructing a provider client."""
    validate_manifest(manifest)
    validate_config(config)
    from scientific_rag.costing import calculate_cost, rate_for
    prompts = _planning_prompts(manifest, config)
    generation_calls = len(prompts)
    maximum_calls = generation_calls * 2
    initial_input = sum(item["prompt_bytes"] for item in prompts)
    repair_input = sum(item["prompt_bytes"] + REPAIR_PROMPT_ADDITIONAL_CHARS_CEILING for item in prompts)
    reserved_input = initial_input + repair_input
    reserved_output = maximum_calls * config["max_output_tokens"]
    cost, source = calculate_cost(config["provider"], config["model"], reserved_input, reserved_output)
    limit = output_limit_status(config["provider"], config["max_output_tokens"])
    blockers = []
    rate = rate_for(config["provider"], config["model"])
    if (rate is None or not all(math.isfinite(value) and value >= 0
                                for value in (rate.input_usd_per_million, rate.output_usd_per_million))):
        blockers.append("provider/model price is unknown")
    if not limit["enforced"]:
        blockers.append("the gateway cannot enforce the output-token limit")
    if maximum_calls > config["max_calls"]:
        blockers.append("worst-case calls exceed the configured call ceiling")
    if max((item["prompt_bytes"] + REPAIR_PROMPT_ADDITIONAL_CHARS_CEILING
            for item in prompts), default=0) > config["max_input_tokens"]:
        blockers.append("a conservative repair prompt exceeds the input-token ceiling")
    if cost is None or not math.isfinite(cost) or cost < 0 or cost > config["max_cost_usd"]:
        blockers.append("worst-case reservation exceeds the configured cost ceiling")
    schedule = [{"scenario_id": scenario["scenario_id"],
                 "execution_order": _execution_order(config, index)}
                for index, scenario in enumerate(manifest["scenarios"])]
    return {
        "adapter": ADAPTER, "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": canonical_sha256(config), "provider_calls_made": 0,
        "scenarios": len(manifest["scenarios"]), "generation_calls_planned": generation_calls,
        "max_repair_calls": generation_calls, "max_physical_calls": maximum_calls,
        "reserved_input_tokens": reserved_input, "reserved_output_tokens": reserved_output,
        "token_ceilings": {"model_context_tokens": config["model_context_tokens"],
                           "max_input_tokens_per_call": config["max_input_tokens"],
                           "max_output_tokens_per_call": config["max_output_tokens"]},
        "planning_cost_bound_usd": cost, "rate_source": source, "output_limit": limit,
        "eligible_after_exact_owner_authorization": not blockers,
        "blockers": blockers,
        "single_arm_difference": {"field": "prompt_version", "prompt_versions": config["prompt_versions"]},
        "execution_schedule": schedule,
        "first_arm_counts": {version: sum(row["execution_order"][0] == version for row in schedule)
                             for version in config["prompt_versions"]},
        "schemas": {
            "structured_answer_sha256": canonical_sha256(StructuredEvidenceAnswerV2.model_json_schema()),
            "repair_batch_sha256": canonical_sha256(V2RepairBatch.model_json_schema()),
        },
        "capacity_admission": _capacity_admission_plan(prompts, config),
        "prompt_contracts": prompts,
    }


def _capacity_admission_plan(prompts: list[dict], config: dict) -> dict:
    """Informational: per-minute admission cannot be checked offline (account limits are unknown)."""
    largest = max((item["prompt_bytes"] for item in prompts), default=0) + schema_bytes(StructuredEvidenceAnswerV2)
    uncalibrated = largest + config["max_output_tokens"]
    warnings = []
    if uncalibrated > REFERENCE_FREE_TOKENS_PER_MINUTE:
        warnings.append("before calibration, the largest request's one-token-per-byte requirement exceeds the "
                        "published Free tokens-per-minute sample; the first measured call calibrates admission")
    return {
        "method": "provider-reported input tokens per response schema x (1 + margin) + output limit; estimate, "
                  "not an upper bound",
        "margin": ADMISSION_MARGIN,
        "largest_initial_request_bytes": largest,
        "uncalibrated_requirement_tokens": uncalibrated,
        "reference_free_tokens_per_minute": REFERENCE_FREE_TOKENS_PER_MINUTE,
        "warnings": warnings,
    }


def check_paid_authorization(authorization: dict | None, *, manifest: dict, config: dict) -> dict:
    """Bind future authorization to content, accepted code, full config and hard ceilings."""
    validate_manifest(manifest)
    validate_config(config)
    if not isinstance(authorization, dict) or authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise AuthorizationError(f"paid execution requires future owner authorization ({AUTHORIZATION_SCHEMA})")
    expected = {
        "manifest_sha256": manifest["manifest_sha256"], "accepted_code_sha": config["code_sha"],
        "config_sha256": canonical_sha256(config), "provider": config["provider"], "model": config["model"],
        "max_calls": config["max_calls"], "max_cost_usd": config["max_cost_usd"],
        "model_context_tokens": config["model_context_tokens"], "max_input_tokens": config["max_input_tokens"],
        "max_output_tokens": config["max_output_tokens"],
    }
    missing = [field for field in ("authorization_id", "approved_by") if not str(authorization.get(field) or "").strip()]
    mismatched = [field for field, value in expected.items() if authorization.get(field) != value]
    if missing or mismatched:
        raise AuthorizationError("authorization missing or mismatched: " + ", ".join(missing + mismatched))
    from scientific_rag.costing import rate_for
    rate = rate_for(config["provider"], config["model"])
    if (rate is None or not all(math.isfinite(value) and value >= 0
                                for value in (rate.input_usd_per_million, rate.output_usd_per_million))):
        raise AuthorizationError("authorized provider/model has no known price")
    limit = output_limit_status(config["provider"], config["max_output_tokens"])
    if not limit["enforced"]:
        raise AuthorizationError(f"output-token limit is not enforceable: {limit['source']}")
    plan = dry_run(manifest, config)
    if plan["blockers"]:
        raise AuthorizationError("paid plan is not eligible: " + "; ".join(plan["blockers"]))
    return authorization


def _usage_from_trace(summary: dict, reserved: float) -> dict:
    complete = bool(summary.get("token_usage_complete")) and summary.get("estimated_cost_usd") is not None
    return {
        "cost_status": "complete" if complete else "incomplete",
        "provider_calls": summary.get("provider_calls", 0),
        "input_tokens": summary.get("input_tokens"), "output_tokens": summary.get("output_tokens"),
        "cost_usd": summary.get("estimated_cost_usd") if complete else None,
        "known_cost_lower_bound_usd": summary.get("estimated_cost_usd") or 0.0,
        "reserved_cost_usd": reserved, "provider_models": sorted(summary.get("by_model") or {}),
    }


async def run_paid_evaluation(manifest: dict, config: dict, *, authorization: dict | None,
                              gateway: Gateway | None = None,
                              ledger: BudgetLedger | None = None) -> tuple[list[dict], dict]:
    """Future live entry point. R7D-A callers receive a refusal without exact authorization."""
    auth = check_paid_authorization(authorization, manifest=manifest, config=config)
    from ..evaluation_report import resolve_git_provenance
    provenance = resolve_git_provenance()
    if provenance.get("sha") != config["code_sha"]:
        raise AuthorizationError("runtime code SHA does not match the owner-accepted code SHA")
    from scientific_rag import llm_gateway
    from scientific_rag.costing import calculate_cost
    route = llm_gateway.provider_order("expert")
    if route != (config["provider"],) or llm_gateway.model_for(config["provider"], "expert") != config["model"]:
        raise AuthorizationError("expert gateway route is not pinned to the authorized provider/model")
    gateway = gateway or llm_gateway.ainvoke_structured
    ledger = ledger if ledger is not None else BudgetLedger(config["max_calls"], config["max_cost_usd"])
    stopped: str | None = None
    call_usage: list[dict] = []

    async def metered(role, temperature, prompt, schema, *, max_output_tokens):
        nonlocal stopped
        if stopped:
            raise EvaluationStopped(stopped, code="stopped")
        input_ceiling = len(prompt.encode("utf-8"))
        if input_ceiling > config["max_input_tokens"]:
            stopped = "conservative input-token ceiling reached before the call"
            raise EvaluationStopped(stopped, code="input_ceiling")
        reserved, _ = calculate_cost(config["provider"], config["model"], input_ceiling, max_output_tokens)
        if reserved is None or not ledger.reserve(reserved):
            stopped = "authorized call or cost ceiling reached before the call"
            raise EvaluationStopped(stopped, code="budget")
        token = llm_gateway.begin_trace()
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
            ledger.settle(reserved, usage)
            call_usage.append(usage)
            allowed = f"{config['provider']}:{config['model']}"
            if any(model != allowed for model in usage["provider_models"]):
                stopped = "a call used a provider/model outside the authorization"
            elif usage["cost_status"] != "complete":
                stopped = "provider usage was incomplete; spend is uncertain"
            elif usage["provider_calls"] != 1:
                stopped = "an unexpected retry or fallback was observed"
            elif isinstance(usage["output_tokens"], int) and usage["output_tokens"] > max_output_tokens:
                stopped = "provider reported output beyond the enforced output-token limit"
        if error is not None:
            raise error
        if stopped is not None:   # a response arrived, but its accounting is not acceptable
            raise EvaluationStopped(stopped, code="accounting", transmitted=True)
        return result

    records = []
    for execution_index, scenario in enumerate(manifest["scenarios"]):
        if stopped:
            records.append(_not_executed_pair(scenario, config, execution_index, stopped))
            continue
        usage_start = len(call_usage)
        pair = await evaluate_pair(scenario, config=config, gateway=metered, execution_index=execution_index)
        pair_usage = call_usage[usage_start:]
        offset = 0
        for arm in pair["arms"]:
            invocations = arm["physical_calls"]
            used = pair_usage[offset:offset + invocations]
            offset += len(used)
            arm["gateway_invocations"] = invocations
            arm["physical_calls"] = sum(int(item.get("provider_calls") or 0) for item in used)
            arm["usage"] = {
                "input_tokens": (sum(item["input_tokens"] for item in used)
                                 if used and all(isinstance(item.get("input_tokens"), int) for item in used) else None),
                "output_tokens": (sum(item["output_tokens"] for item in used)
                                  if used and all(isinstance(item.get("output_tokens"), int) for item in used) else None),
                "latency_ms": round(sum(item["latency_ms"] for item in used), 1) if used else None,
                "cost_usd": (round(sum(item["cost_usd"] for item in used), 8)
                             if used and all(item.get("cost_usd") is not None for item in used) else None),
                "cost_status": ("complete" if used and all(item["cost_status"] == "complete" for item in used)
                                else "incomplete"),
            }
        pair["physical_calls"] = sum(arm["physical_calls"] for arm in pair["arms"])
        records.append(pair)
    return records, {"authorization_id": auth["authorization_id"], "budget": ledger.snapshot(),
                     "code": provenance, "stopped": stopped, "calls": call_usage}


def build_public_report(records: list[dict], *, scores: dict | None,
                        manifest_sha256: str, config_sha256: str) -> dict:
    """Build a deterministic, text-free report. Owner scores remain a separate input."""
    scores = scores or {}
    record_ids = [record.get("scenario_id") for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("public report records contain duplicate scenario IDs")
    if set(scores) - set(record_ids):
        raise ValueError("owner scores contain scenario IDs absent from the attempted records")
    outcomes = {"v15": 0, "tie": 0, "v16": 0, "unscored": 0}
    rows, arm_totals = [], {version: Counter() for version in ("v15", "v16")}
    initial_outcomes = {version: Counter() for version in ("v15", "v16")}
    post_repair_outcomes = {version: Counter() for version in ("v15", "v16")}
    excluded = Counter()
    for record in sorted(records, key=lambda item: item["scenario_id"]):
        normalized = _normalized_arms(record)
        scorable = pair_is_scorable(normalized)
        if scorable:
            for version, arm in normalized.items():
                initial_outcomes[version][_initial_outcome(arm)] += 1
                post_repair_outcomes[version][_post_repair_outcome(arm)] += 1
        else:
            excluded["pairs"] += 1
            for arm in normalized.values():
                excluded[f"arms_{arm['status']}"] += 1
        score = scores.get(record["scenario_id"])
        if score is not None and not scorable:
            raise ValueError(f"owner score supplied for incomplete scenario {record['scenario_id']}")
        if isinstance(score, dict):
            unexpected = set(score) - {"winner", "metrics"}
            metrics = score.get("metrics", {})
            if unexpected or not isinstance(metrics, dict) or set(metrics) - SCORE_METRICS:
                raise ValueError(f"owner score for {record['scenario_id']} violates the scoring contract")
            for metric, arm_scores in metrics.items():
                if (not isinstance(arm_scores, dict) or set(arm_scores) - {"v15", "v16"}
                        or any(not isinstance(value, (bool, int)) or isinstance(value, str)
                               or (isinstance(value, int) and not 0 <= value <= 2)
                               for value in arm_scores.values())):
                    raise ValueError(f"owner score {metric} for {record['scenario_id']} must be bounded arm scores")
        winner = score.get("winner") if isinstance(score, dict) else None
        outcome = winner if winner in {"v15", "v16", "tie"} else "unscored"
        outcomes[outcome] += 1
        arms = []
        for version in ("v15", "v16"):
            arm = normalized[version]
            usage = arm.get("usage") or {}
            row = {"prompt_version": version, "status": arm["status"],
                   "repair_attempted": bool(arm.get("repair_attempted")),
                   "physical_calls": int(arm.get("physical_calls", 0)),
                   "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                   "latency_ms": usage.get("latency_ms"), "cost_usd": usage.get("cost_usd")}
            for field in ("failure_kind", "failure_class", "failure_phase", "initial_structurally_valid"):
                if arm.get(field) is not None:
                    row[field] = arm[field]
            arms.append(row)
            arm_totals[version]["scheduled"] += 1
            arm_totals[version]["attempted"] += arm["status"] != "not_executed"
            arm_totals[version]["not_executed"] += arm["status"] == "not_executed"
            arm_totals[version]["completed"] += arm["status"] == "completed"
            arm_totals[version]["failures"] += arm["status"] == "failed"   # model-output failures only
            arm_totals[version]["interrupted"] += arm["status"] == "interrupted"
            arm_totals[version]["infrastructure_failures"] += arm.get("failure_class") == INFRASTRUCTURE
            arm_totals[version]["execution_stops"] += arm.get("failure_class") == EXECUTION_STOP
            arm_totals[version]["repairs"] += bool(arm.get("repair_attempted"))
            arm_totals[version]["physical_calls"] += int(arm.get("physical_calls", 0))
            arm_totals[version]["abstentions"] += int(arm.get("unanswered_parts", 0) > 0)
            for status, count in (arm.get("claim_audit") or {}).get("summary", {}).items():
                arm_totals[version][f"audit_{status}"] += int(count)
        rows.append({"scenario_id": record["scenario_id"], "outcome": outcome, "scorable": scorable,
                     "execution_order": record.get("execution_order"), "arms": arms,
                     "metric_scores": score.get("metrics", {}) if isinstance(score, dict) else {}})
    return {
        "schema": "scientific-rag-generation-eval-public-report/v1",
        "manifest_sha256": manifest_sha256, "config_sha256": config_sha256,
        "attempted_scenarios": len(records), "paired_outcomes": outcomes,
        "arms": {version: dict(sorted(total.items())) for version, total in arm_totals.items()},
        # Comparable pairs only: both arms reached a valid terminal model outcome.
        "initial_generation": {version: dict(sorted(total.items())) for version, total in initial_outcomes.items()},
        "post_repair": {version: dict(sorted(total.items())) for version, total in post_repair_outcomes.items()},
        "excluded_from_pairing": dict(sorted(excluded.items())),
        "scenarios": rows,
        "privacy": "contains opaque IDs and aggregate/operational results only; no source text, model text, or references",
    }


def _initial_outcome(arm: dict) -> str:
    if arm.get("initial_output") is None:
        return "invalid_structured_output"
    return "structurally_valid" if arm.get("initial_structurally_valid") else "structurally_invalid"


def _post_repair_outcome(arm: dict) -> str:
    if arm["status"] == "completed":
        return "completed_after_repair" if arm.get("repair_attempted") else "completed_without_repair"
    return str(arm.get("failure_kind") or "failed")


def build_blind_scoring_packet(manifest: dict, records: list[dict]) -> tuple[dict, dict]:
    """Return a private evidence/output packet and a separate arm-key document.

    Prompt versions never appear in the scoring packet.  The key must be withheld from
    the owner-side scorer until judgments are frozen.  Neither document is a public report.
    """
    validate_manifest(manifest)
    by_id = {record["scenario_id"]: record for record in records}
    cases, keys = [], []
    for scenario in sorted(manifest["scenarios"], key=lambda item: item["scenario_id"]):
        record = by_id.get(scenario["scenario_id"])
        if record is None:
            raise ValueError(f"missing attempted record for {scenario['scenario_id']}")
        arms = _normalized_arms(record)
        scorable = pair_is_scorable(arms)
        # Deterministic balancing without disclosing the version in the scoring packet.
        a_version = "v15" if int(canonical_sha256(scenario["scenario_id"]), 16) % 2 == 0 else "v16"
        b_version = "v16" if a_version == "v15" else "v15"
        cases.append({
            "scenario_id": scenario["scenario_id"], "question": scenario["question"],
            "evidence": _context_parts(scenario),
            "arms": {
                "A": {"status": arms[a_version]["status"], "output": arms[a_version].get("post_repair_output"),
                      "failure_kind": arms[a_version].get("failure_kind")},
                "B": {"status": arms[b_version]["status"], "output": arms[b_version].get("post_repair_output"),
                      "failure_kind": arms[b_version].get("failure_kind")},
            },
            "scorable": scorable,
            "score_fields": sorted(SCORE_METRICS),
        })
        keys.append({"scenario_id": scenario["scenario_id"], "A": a_version, "B": b_version})
    packet = {"schema": "scientific-rag-generation-eval-blind-scoring/v1",
              "manifest_sha256": manifest["manifest_sha256"], "cases": cases}
    key = {"schema": "scientific-rag-generation-eval-blind-key/v1",
           "manifest_sha256": manifest["manifest_sha256"], "arms": keys}
    return packet, key


def unblind_owner_scores(blind_scores: dict, key: dict) -> dict:
    """Translate frozen A/B owner scores using the separately held arm key."""
    if not isinstance(blind_scores, dict) or not isinstance(key, dict):
        raise ValueError("blind scores and arm key must be objects")
    if key.get("schema") != "scientific-rag-generation-eval-blind-key/v1":
        raise ValueError("unrecognized blind arm key schema")
    mappings = {row["scenario_id"]: row for row in key.get("arms", [])}
    if set(blind_scores) - set(mappings):
        raise ValueError("blind scores contain scenario IDs absent from the arm key")
    output = {}
    for scenario_id, score in blind_scores.items():
        if not isinstance(score, dict) or set(score) - {"winner", "metrics"}:
            raise ValueError(f"blind score for {scenario_id} violates the scoring contract")
        mapping = mappings[scenario_id]
        winner = score.get("winner")
        if winner not in {"A", "B", "tie"}:
            raise ValueError(f"blind winner for {scenario_id} must be A, B, or tie")
        metrics = score.get("metrics", {})
        if not isinstance(metrics, dict) or set(metrics) - SCORE_METRICS:
            raise ValueError(f"blind metrics for {scenario_id} violate the scoring contract")
        translated = {}
        for metric, values in metrics.items():
            if not isinstance(values, dict) or set(values) != {"A", "B"}:
                raise ValueError(f"blind metric {metric} for {scenario_id} requires A and B")
            translated[metric] = {mapping[arm]: value for arm, value in values.items()}
        output[scenario_id] = {
            "winner": winner if winner == "tie" else mapping[winner],
            "metrics": translated,
        }
    return output


__all__ = [
    "ADAPTER", "AUTHORIZATION_SCHEMA", "AuthorizationError", "EvaluationStopped", "ManifestError",
    "ResumeIntegrityError", "SCORABLE_ARM_STATUSES", "pair_is_scorable", "schema_bytes",
    "build_blind_scoring_packet", "build_public_report", "canonical_sha256", "check_paid_authorization", "dry_run", "evaluate_pair",
    "manifest_hash", "question_context_hash", "run_paid_evaluation", "unblind_owner_scores", "validate_config", "validate_manifest",
]
