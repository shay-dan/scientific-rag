"""R7D-D focused live configuration acceptance harness (synthetic data only).

Selects the reasoning/completion configuration for the forthcoming R7D benchmark from a
small, fixed matrix: Groq ``openai/gpt-oss-120b``, low reasoning, 2,000 versus 3,000
completion tokens, v15 and v16, three synthetic scenarios, at most one repair per generation
and two dedicated repair-schema tests: at most 26 physical requests.

Why a dedicated adapter: the benchmark manifest contract is deliberately exactly 20 frozen
development scenarios with one configuration per campaign. Accepting a 3-scenario,
two-configuration matrix there would weaken that contract. This adapter therefore has its
own dataset/config/authorization contracts but reuses, unchanged, every production and
safety primitive:
- ``_evaluate_arm``: production v15/v16 prompt builders, strict StructuredEvidenceAnswerV2,
  Pydantic and evidence validators, the single bounded repair, typed failure classification;
- ``GuardedExecution``: capacity admission and bounded token-reset waits, reservations,
  ledger marks before transmission and after settlement, stop precedence, settlement,
  reasoning-token telemetry;
- ``AtomicCheckpoint`` and the program ledger: private durable state, one authoritative
  checkpoint, a per-campaign lock, cumulative accounting (a separate acceptance program
  ledger, so acceptance spending never mixes with benchmark spending).

Dry run is the default and makes zero provider calls. Live execution requires an exact
acceptance authorization bound to the code, dataset, config, schedule and plan.
Reference predicates are never read by the runner or passed to any prompt.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Any, Awaitable, Callable

from scientific_rag.provider_capabilities import validate_reasoning_effort

from ..weak_model_synthesis import (
    StructuredEvidenceAnswerV2,
    V2RepairBatch,
    generation_prompt_v2,
    repair_prompt_v2,
    validate_answer_v2,
)
from .generation_evaluation import (
    CONFIG_SCHEMA_V2,
    EXECUTION_STOP,
    REFERENCE_FREE_TOKENS_PER_MINUTE,
    REPAIR_PROMPT_ADDITIONAL_CHARS_CEILING,
    AuthorizationError,
    _blocks_and_context,
    _evaluate_arm,
    canonical_sha256,
    outcome_dimensions,
    question_context_hash,
    reservation_input_tokens,
    schema_bytes,
    validate_config,
)
from .incremental_generation_evaluation import (
    CHECKPOINT_SCHEMA,
    REPAIR_SCHEMA,
    UNRESOLVED_MESSAGE,
    AtomicCheckpoint,
    CheckpointError,
    GuardedExecution,
    _arm_usage,
)
from .production_verifier_replay import BudgetLedger

DATASET_SCHEMA = "scientific-rag-generation-acceptance-dataset/v1"
REFERENCES_SCHEMA = "scientific-rag-generation-acceptance-references/v1"
CONFIG_SCHEMA = "scientific-rag-generation-acceptance-config/v1"
AUTHORIZATION_SCHEMA = "scientific-rag-generation-acceptance-authorization/v1"
CAMPAIGN_KIND = "r7d-d-acceptance"
ADAPTER = "r7d_d_acceptance_v1"
MAX_PHYSICAL_REQUESTS = 26
PROPOSED_DIAGNOSTIC_BUDGET_USD = 0.15
SCENARIO_IDS = ("A001", "A002", "A003")
FIXTURE_IDS = ("RF1", "RF2")
CONFIG_FIELDS = frozenset({
    "schema", "code_sha", "provider", "model", "temperature", "prompt_versions", "model_context_tokens",
    "max_input_tokens", "max_calls", "max_cost_usd", "semantic_verification_mode", "claim_recovery_mode",
    "claim_audit_mode", "completion_budget_failure_policy", "configurations",
})
PREDICATE_KINDS = ("expected_fact", "required_uncertainty", "required_abstention", "forbidden_claim", "conditional")
# Frozen, balanced execution order (configuration index into config["configurations"]). Within
# every scenario each configuration runs one v15 and one v16 arm; the first arm alternates
# A/B/A across scenarios, and the two dedicated repair tests sit between scenario blocks.
SCHEDULE_PATTERN = (
    ("arm", "A001", 0, "v15"), ("arm", "A001", 1, "v16"), ("arm", "A001", 1, "v15"), ("arm", "A001", 0, "v16"),
    ("dedicated_repair", "RF1", 1, "v16"),
    ("arm", "A002", 1, "v15"), ("arm", "A002", 0, "v16"), ("arm", "A002", 0, "v15"), ("arm", "A002", 1, "v16"),
    ("dedicated_repair", "RF2", 0, "v15"),
    ("arm", "A003", 0, "v16"), ("arm", "A003", 1, "v15"), ("arm", "A003", 1, "v16"), ("arm", "A003", 0, "v15"),
)
# Predeclared configuration-selection rules (frozen before live execution; hashed into the plan).
SELECTION_RULES = {
    "version": "r7d-d-selection/v1",
    "gates": [
        "initial-generation and dedicated repair schemas both parsed on the actual gateway",
        "no execution-integrity stop",
        "no unresolved accounting defect",
        "every scheduled operation completed, or each missing one listed",
        "final structural validity in at least 5 of 6 arms",
        "no material scientific regression against the other configuration",
    ],
    "adequate_final_valid_arms": 5,
    "material_regression": "at least 2 more critical violations, or at least 2 more failed reference predicates, "
                           "than the other configuration on matched completed arms",
    "tie_break": "prefer the 2,000-token configuration unless the 3,000-token one shows a meaningful advantage: "
                 "fewer completion-budget failures, at least one more final-valid arm, at least 3 more passed "
                 "reference predicates, or minimum completion headroom of at least 10% where 2,000 had less",
    "never": ["select by shorter answers", "claim statistical superiority", "expand the sample automatically"],
}
INTEGRITY_STOPS = frozenset({
    "model_drift", "unexpected_retry", "output_limit", "reservation_exceeded", "cost_ceiling_exceeded",
    "internal_error", "ledger_mark_failed",
})


ACCOUNTING_STOPS = frozenset({"billing_unknown"})


class AcceptanceError(ValueError):
    """The acceptance dataset, references, config or plan is invalid."""


def _hash_without(value: dict, key: str) -> str:
    return canonical_sha256({k: v for k, v in value.items() if k != key})


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceError(f"{label} must be non-empty text")
    return value


# --------------------------------------------------------------------------- contracts

def validate_dataset(dataset: dict) -> dict:
    if not isinstance(dataset, dict) or dataset.get("schema") != DATASET_SCHEMA:
        raise AcceptanceError(f"dataset must use {DATASET_SCHEMA}")
    if dataset.get("synthetic") is not True:
        raise AcceptanceError("the acceptance dataset must be explicitly synthetic")
    _text(dataset.get("dataset_id"), "dataset_id")
    scenarios = dataset.get("scenarios")
    if not isinstance(scenarios, list) or [item.get("scenario_id") for item in scenarios] != list(SCENARIO_IDS):
        raise AcceptanceError(f"dataset must contain exactly the synthetic scenarios {', '.join(SCENARIO_IDS)}")
    for scenario in scenarios:
        sid = scenario["scenario_id"]
        forbidden = set(scenario) & {"expected", "references", "predicates", "labels", "gold", "answers"}
        if forbidden:
            raise AcceptanceError(f"scenario {sid} must not contain reference fields: {sorted(forbidden)}")
        _text(scenario.get("question"), f"{sid} question")
        _text(scenario.get("category"), f"{sid} category")
        evidence = scenario.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise AcceptanceError(f"scenario {sid} requires fixed evidence")
        seen: set[str] = set()
        for item in evidence:
            source_id = _text(item.get("source_id"), "source_id")
            if not re.fullmatch(r"S\d+", source_id) or source_id in seen:
                raise AcceptanceError(f"scenario {sid} has invalid or duplicate source_id {source_id}")
            seen.add(source_id)
            _text(item.get("title"), "title")
            _text(item.get("production_notes"), "production_notes")
            if item.get("evidence_sha256") != canonical_sha256(_text(item.get("passage"), "passage")):
                raise AcceptanceError(f"scenario {sid} evidence hash mismatch")
        if scenario.get("question_context_sha256") != question_context_hash(scenario):
            raise AcceptanceError(f"scenario {sid} question/context hash mismatch")
    fixtures = dataset.get("repair_fixtures")
    if not isinstance(fixtures, list) or [item.get("fixture_id") for item in fixtures] != list(FIXTURE_IDS):
        raise AcceptanceError(f"dataset must contain exactly the repair fixtures {', '.join(FIXTURE_IDS)}")
    by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}
    for fixture in fixtures:
        scenario = by_id.get(fixture.get("scenario_id"))
        if scenario is None:
            raise AcceptanceError(f"fixture {fixture['fixture_id']} names an unknown scenario")
        answer = StructuredEvidenceAnswerV2.model_validate(fixture.get("answer"))   # strict schema-valid
        blocks, _ = _blocks_and_context(scenario)
        if not validate_answer_v2(answer, blocks):
            raise AcceptanceError(f"fixture {fixture['fixture_id']} must fail evidence validation to exercise repair")
    if dataset.get("dataset_sha256") != _hash_without(dataset, "dataset_sha256"):
        raise AcceptanceError("dataset hash mismatch")
    return dataset


def validate_references(references: dict, dataset: dict) -> dict:
    """Synthetic acceptance references: deterministic predicates, not human-reviewed gold labels."""
    if not isinstance(references, dict) or references.get("schema") != REFERENCES_SCHEMA:
        raise AcceptanceError(f"references must use {REFERENCES_SCHEMA}")
    if references.get("label") != "synthetic acceptance references; not independent human-reviewed gold labels":
        raise AcceptanceError("references must carry the synthetic-reference label")
    if references.get("dataset_sha256") != dataset["dataset_sha256"]:
        raise AcceptanceError("references are bound to a different dataset")
    predicates = references.get("predicates")
    if not isinstance(predicates, dict) or set(predicates) != set(SCENARIO_IDS):
        raise AcceptanceError("references must cover exactly the synthetic scenarios")
    for sid, items in predicates.items():
        for item in items:
            if item.get("kind") not in PREDICATE_KINDS:
                raise AcceptanceError(f"{sid} predicate {item.get('id')} has an unsupported kind")
            for key in ("any_of", "all_of", "unless_any_of", "if_any_of", "then_any_of"):
                for pattern in item.get(key) or []:
                    re.compile(pattern, re.IGNORECASE)
    if references.get("references_sha256") != _hash_without(references, "references_sha256"):
        raise AcceptanceError("references hash mismatch")
    return references


def validate_acceptance_config(config: dict) -> dict:
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA or set(config) != CONFIG_FIELDS:
        raise AcceptanceError(f"config must use {CONFIG_SCHEMA} with exactly: {', '.join(sorted(CONFIG_FIELDS))}")
    configurations = config["configurations"]
    if not isinstance(configurations, list) or len(configurations) != 2:
        raise AcceptanceError("the acceptance matrix compares exactly two configurations")
    labels = [item.get("label") for item in configurations]
    if labels != ["A", "B"]:
        raise AcceptanceError("configuration labels must be A and B")
    for item in configurations:
        if set(item) != {"label", "reasoning_effort", "max_output_tokens"}:
            raise AcceptanceError("each configuration binds exactly label, reasoning_effort and max_output_tokens")
        try:
            validate_reasoning_effort(config["provider"], config["model"], item["reasoning_effort"])
        except ValueError as error:
            raise AcceptanceError(str(error)) from None
        validate_config(arm_config(config, item["label"]))                 # the production v2 contract
    if len({item["reasoning_effort"] for item in configurations}) != 1:
        raise AcceptanceError("this matrix varies only the completion ceiling; reasoning effort must match")
    if config["max_calls"] != MAX_PHYSICAL_REQUESTS:
        raise AcceptanceError(f"max_calls must equal the matrix ceiling of {MAX_PHYSICAL_REQUESTS}")
    if config["prompt_versions"] != ["v15", "v16"]:
        raise AcceptanceError("prompt_versions must be v15 and v16")
    return config


def arm_config(config: dict, label: str) -> dict:
    """The production v2 generation config for one configuration (initial and repair)."""
    item = next(entry for entry in config["configurations"] if entry["label"] == label)
    return {
        "schema": CONFIG_SCHEMA_V2, "code_sha": config["code_sha"], "provider": config["provider"],
        "model": config["model"], "temperature": config["temperature"], "prompt_versions": config["prompt_versions"],
        "model_context_tokens": config["model_context_tokens"], "max_input_tokens": config["max_input_tokens"],
        "max_output_tokens": item["max_output_tokens"], "max_calls": config["max_calls"],
        "max_cost_usd": config["max_cost_usd"], "semantic_verification_mode": config["semantic_verification_mode"],
        "claim_recovery_mode": config["claim_recovery_mode"], "claim_audit_mode": config["claim_audit_mode"],
        "reasoning_effort": item["reasoning_effort"],
        "completion_budget_failure_policy": config["completion_budget_failure_policy"],
    }


def acceptance_schedule(config: dict) -> list[dict]:
    labels = [item["label"] for item in config["configurations"]]
    return [{"operation": index, "kind": kind, "target": target, "configuration": labels[position],
             "prompt_version": version}
            for index, (kind, target, position, version) in enumerate(SCHEDULE_PATTERN, 1)]


# --------------------------------------------------------------------------- planning

def _scenario(dataset: dict, scenario_id: str) -> dict:
    return next(item for item in dataset["scenarios"] if item["scenario_id"] == scenario_id)


def _fixture(dataset: dict, fixture_id: str) -> dict:
    return next(item for item in dataset["repair_fixtures"] if item["fixture_id"] == fixture_id)


def _operation_inputs(dataset: dict, operation: dict) -> tuple[dict, dict | None]:
    """(scenario, fixture resume state or None) for one scheduled operation."""
    if operation["kind"] == "arm":
        return _scenario(dataset, operation["target"]), None
    fixture = _fixture(dataset, operation["target"])
    scenario = _scenario(dataset, fixture["scenario_id"])
    _, context = _blocks_and_context(scenario)
    prompt = generation_prompt_v2(scenario["question"], context, prompt_version=operation["prompt_version"])
    return scenario, {"prompt_sha256": canonical_sha256(prompt), "initial_output": fixture["answer"]}


def _reserved(config: dict, prompt_bytes: int, schema: type, max_output_tokens: int) -> tuple[int, float]:
    from scientific_rag.costing import calculate_cost
    basis = reservation_input_tokens(config, prompt_bytes, schema)
    cost, _ = calculate_cost(config["provider"], config["model"], basis, max_output_tokens)
    if cost is None or not math.isfinite(cost):
        raise AcceptanceError("the configured provider/model has no builtin price; reservations cannot be planned")
    return basis, round(cost, 8)


def acceptance_dry_run(dataset: dict, config: dict, references: dict | None = None) -> dict:
    """Zero provider calls: frozen fingerprints, schedule, reservations and selection rules."""
    validate_dataset(dataset)
    validate_acceptance_config(config)
    if references is not None:
        validate_references(references, dataset)
    schedule = acceptance_schedule(config)
    operations, blockers, warnings = [], [], []
    deterministic = conditional = typical = 0.0
    for operation in schedule:
        settings = arm_config(config, operation["configuration"])
        limit = settings["max_output_tokens"]
        scenario, fixture_state = _operation_inputs(dataset, operation)
        blocks, context = _blocks_and_context(scenario)
        initial_prompt = generation_prompt_v2(scenario["question"], context, prompt_version=operation["prompt_version"])
        entry = {**operation, "scenario_id": scenario["scenario_id"], "max_output_tokens": limit,
                 "reasoning_effort": settings["reasoning_effort"],
                 "question_context_sha256": scenario["question_context_sha256"],
                 "initial_prompt_sha256": canonical_sha256(initial_prompt)}
        if fixture_state is None:
            initial_bytes = len(initial_prompt.encode("utf-8"))
            basis, cost = _reserved(settings, initial_bytes, StructuredEvidenceAnswerV2, limit)
            repair_bytes = initial_bytes + REPAIR_PROMPT_ADDITIONAL_CHARS_CEILING
            repair_basis, repair_cost = _reserved(settings, repair_bytes, V2RepairBatch, limit)
            # Typical repair size (an ESTIMATE): the initial prompt plus an answer at ~4 bytes per token.
            _, typical_cost = _reserved(settings, initial_bytes + 4 * limit, V2RepairBatch, limit)
            deterministic += cost
            conditional += repair_cost
            typical += typical_cost
            entry.update(initial_prompt_bytes=initial_bytes, initial_reservation_input_tokens=basis,
                         initial_reserved_cost_usd=cost, conditional_repair_planning_reserved_cost_usd=repair_cost,
                         uncalibrated_initial_requirement_tokens=initial_bytes + schema_bytes(StructuredEvidenceAnswerV2)
                         + limit)
        else:
            answer = StructuredEvidenceAnswerV2.model_validate(fixture_state["initial_output"])
            errors = validate_answer_v2(answer, blocks)
            repair_prompt = repair_prompt_v2(scenario["question"], context, answer, errors)
            repair_bytes = len(repair_prompt.encode("utf-8"))
            basis, cost = _reserved(settings, repair_bytes, V2RepairBatch, limit)
            deterministic += cost
            entry.update(fixture_answer_sha256=canonical_sha256(fixture_state["initial_output"]),
                         fixture_validation_targets=sorted(errors), repair_prompt_sha256=canonical_sha256(repair_prompt),
                         repair_prompt_bytes=repair_bytes, repair_reservation_input_tokens=basis,
                         repair_reserved_cost_usd=cost)
        operations.append(entry)
    planning_bound = round(deterministic + conditional, 8)
    if deterministic > config["max_cost_usd"]:
        blockers.append("the deterministic reservations alone exceed the diagnostic cost ceiling")
    if planning_bound > config["max_cost_usd"]:
        warnings.append("the conservative planning bound for all 12 conditional repairs (70,000 extra repair "
                        "characters each) exceeds the cost ceiling; each actual repair is reserved from its real "
                        "prompt and refused before transmission if it would not fit")
    # Only the first request of a schema is admitted uncalibrated (one token per byte); later ones
    # use provider-reported token ratios from earlier calls of the same campaign.
    first = next(item for item in operations if item["kind"] == "arm")
    if first["uncalibrated_initial_requirement_tokens"] > REFERENCE_FREE_TOKENS_PER_MINUTE:
        blockers.append("the first, uncalibrated request cannot fit the published Free tokens-per-minute sample")
    initial = sum(item["kind"] == "arm" for item in schedule)
    plan = {
        "adapter": ADAPTER, "provider_calls_made": 0,
        "dataset_id": dataset["dataset_id"], "dataset_sha256": dataset["dataset_sha256"],
        "references_sha256": references["references_sha256"] if references is not None else None,
        "reference_isolation": "references are validated separately and never passed to prompts or the runner",
        "config_sha256": canonical_sha256(config), "code_sha": config["code_sha"],
        "provider": config["provider"], "model": config["model"],
        "configurations": config["configurations"],
        "schedule": schedule, "schedule_sha256": canonical_sha256(schedule),
        "initial_generations": initial, "conditional_repairs_max": initial,
        "dedicated_repairs": len(schedule) - initial,
        "max_physical_requests": 2 * initial + (len(schedule) - initial),
        "schemas": {"initial": canonical_sha256(StructuredEvidenceAnswerV2.model_json_schema()),
                    "repair": canonical_sha256(V2RepairBatch.model_json_schema())},
        "operations": operations,
        "deterministic_reservation_usd": round(deterministic, 8),
        "conditional_repair_planning_bound_usd": round(conditional, 8),
        "typical_conditional_repair_estimate_usd": round(typical, 8),
        "planning_cost_bound_usd": planning_bound,
        "cost_ceiling_usd": config["max_cost_usd"],
        "cost_basis": "token-based estimates and conservative reservations; not provider charges and not a "
                      "guaranteed upper bound",
        "selection_rules": SELECTION_RULES, "selection_rules_sha256": canonical_sha256(SELECTION_RULES),
        "blockers": blockers, "warnings": warnings,
    }
    if plan["max_physical_requests"] != MAX_PHYSICAL_REQUESTS:
        plan["blockers"].append("the schedule does not match the 26-request matrix")
    plan["eligible_after_exact_owner_authorization"] = not plan["blockers"]
    plan["plan_sha256"] = canonical_sha256({key: value for key, value in plan.items()
                                            if key not in ("references_sha256", "warnings")})
    return plan


def check_acceptance_authorization(authorization: dict | None, *, dataset: dict, config: dict) -> tuple[dict, dict]:
    plan = acceptance_dry_run(dataset, config)
    if not isinstance(authorization, dict) or authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise AuthorizationError(f"live acceptance execution requires an exact owner authorization "
                                 f"({AUTHORIZATION_SCHEMA})")
    if not plan["eligible_after_exact_owner_authorization"]:
        raise AuthorizationError("the acceptance plan has blockers: " + "; ".join(plan["blockers"]))
    expected = {
        "accepted_code_sha": config["code_sha"], "dataset_sha256": dataset["dataset_sha256"],
        "config_sha256": plan["config_sha256"], "schedule_sha256": plan["schedule_sha256"],
        "plan_sha256": plan["plan_sha256"], "provider": config["provider"], "model": config["model"],
        "reasoning_effort": config["configurations"][0]["reasoning_effort"],
        "max_output_tokens": [item["max_output_tokens"] for item in config["configurations"]],
        "max_physical_requests": MAX_PHYSICAL_REQUESTS, "max_cost_usd": config["max_cost_usd"],
        "completion_budget_failure_policy": config["completion_budget_failure_policy"],
        "selection_rules_sha256": plan["selection_rules_sha256"],
    }
    missing = [field for field in ("authorization_id", "approved_by", "program_id")
               if not str(authorization.get(field) or "").strip()]
    if not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get("program_baseline_sha256") or "")):
        missing.append("program_baseline_sha256")
    mismatched = [field for field, value in expected.items() if authorization.get(field) != value]
    if missing or mismatched:
        raise AuthorizationError("acceptance authorization missing or mismatched: " + ", ".join(missing + mismatched))
    return authorization, plan


# --------------------------------------------------------------------------- execution

def _new_checkpoint(dataset: dict, config: dict, authorization: dict, plan: dict, provenance: dict,
                    baseline: dict) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA, "campaign_kind": CAMPAIGN_KIND, "checkpoint_revision": 0,
        "manifest_sha256": dataset["dataset_sha256"], "config_sha256": plan["config_sha256"],
        "authorization_sha256": canonical_sha256(authorization), "authorization_id": authorization["authorization_id"],
        "schedule_sha256": plan["schedule_sha256"], "plan_sha256": plan["plan_sha256"],
        "code_sha": config["code_sha"], "provider": config["provider"], "model": config["model"],
        "program": {"program_id": authorization["program_id"], "baseline_sha256": baseline["sha256"],
                    "baseline": dict(baseline["budget"])},
        "records": [], "active_arm": None, "active_attempt": None,
        "execution": {"authorization_id": authorization["authorization_id"], "budget": dict(baseline["budget"]),
                      "code": provenance, "stopped": None, "stop_code": None, "stop_context": None, "calls": [],
                      "rate_limit_state": {}, "rate_limit_continuations": [], "stages": []},
    }


def _verify_bindings(checkpoint: dict, dataset: dict, authorization: dict, plan: dict) -> None:
    expected = {"campaign_kind": CAMPAIGN_KIND, "manifest_sha256": dataset["dataset_sha256"],
                "config_sha256": plan["config_sha256"], "authorization_sha256": canonical_sha256(authorization),
                "authorization_id": authorization["authorization_id"], "schedule_sha256": plan["schedule_sha256"],
                "plan_sha256": plan["plan_sha256"]}
    mismatched = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatched:
        raise CheckpointError("acceptance checkpoint binding mismatch: " + ", ".join(mismatched))


async def run_acceptance(
    dataset: dict,
    config: dict,
    *,
    authorization: dict | None,
    checkpoint_store: AtomicCheckpoint,
    program_ledger,
    gateway: Callable[..., Awaitable[Any]] | None = None,
    operations: int | None = None,
    resume_after_rate_limit: bool = False,
    wait_for_token_reset: bool = False,
    provenance_resolver: Callable[[], dict] | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    """Run up to ``operations`` scheduled operations (all remaining by default). No retries."""
    from . import program_ledger as programs
    if program_ledger is None:
        raise AuthorizationError("acceptance execution requires its separate program ledger")
    if operations is not None and (type(operations) is not int or operations <= 0):
        raise ValueError("operations must be a positive integer")
    authorization, plan = check_acceptance_authorization(authorization, dataset=dataset, config=config)
    from ..evaluation_report import resolve_git_provenance
    from scientific_rag import llm_gateway
    provenance = (provenance_resolver or resolve_git_provenance)()
    if provenance.get("sha") != config["code_sha"]:
        raise AuthorizationError("runtime code SHA does not match the owner-accepted code SHA")
    route = llm_gateway.provider_order("expert")
    if route != (config["provider"],) or llm_gateway.model_for(config["provider"], "expert") != config["model"]:
        raise AuthorizationError("expert gateway route is not pinned to the authorized provider/model")
    gateway = gateway or llm_gateway.ainvoke_structured
    campaign_id = str(authorization["authorization_id"])
    with program_ledger.campaign_lock(campaign_id):
        return await _run_locked(dataset, config, authorization, plan, provenance, checkpoint_store, program_ledger,
                                 programs, gateway, operations, resume_after_rate_limit, wait_for_token_reset,
                                 clock, sleep, campaign_id)


async def _run_locked(dataset, config, authorization, plan, provenance, checkpoint_store, program_ledger, programs,
                      gateway, operations, resume_after_rate_limit, wait_for_token_reset, clock, sleep,
                      campaign_id) -> dict:
    base_config = arm_config(config, config["configurations"][0]["label"])

    def mark(value: dict) -> None:
        programs.record_checkpoint_mark(program_ledger, campaign_id, checkpoint_store.path, value)

    if checkpoint_store.exists():
        checkpoint = checkpoint_store.load()
        programs.require_runnable_campaign(program_ledger, campaign_id, checkpoint_store.path, checkpoint)
        _verify_bindings(checkpoint, dataset, authorization, plan)
        if checkpoint.get("active_attempt") is not None:
            raise CheckpointError(UNRESOLVED_MESSAGE)
        ledger = BudgetLedger.from_snapshot(config["max_calls"], config["max_cost_usd"], checkpoint["execution"]["budget"])
    else:
        entry = next((item for item in program_ledger.load()["campaigns"] if item["campaign_id"] == campaign_id), None)
        if entry is not None:
            programs.require_runnable_campaign(program_ledger, campaign_id, checkpoint_store.path, None)
        baseline = programs.program_baseline(program_ledger.load())
        if entry is None and (authorization["program_id"] != baseline["program_id"]
                              or authorization["program_baseline_sha256"] != baseline["sha256"]):
            raise AuthorizationError("authorization program baseline does not match the acceptance program ledger")
        baseline = programs.open_campaign(program_ledger, authorization=authorization, config=config,
                                          manifest_sha256=dataset["dataset_sha256"],
                                          checkpoint_path=checkpoint_store.path)
        checkpoint = checkpoint_store.write(_new_checkpoint(dataset, config, authorization, plan, provenance, baseline))
        mark(checkpoint)
        ledger = BudgetLedger.from_snapshot(config["max_calls"], config["max_cost_usd"], checkpoint["execution"]["budget"])

    stop_code = checkpoint["execution"].get("stop_code")
    reset_continuation = False
    if checkpoint["execution"].get("stopped"):
        context = checkpoint["execution"].get("stop_context") or {}
        if not (resume_after_rate_limit and stop_code == "rate_limit" and context.get("kind") == "preflight_capacity"):
            raise CheckpointError(f"acceptance checkpoint is stopped ({stop_code}): {checkpoint['execution']['stopped']}")
        reset_continuation = True
        checkpoint["execution"].update(stopped=None, stop_code=None, stop_context=None)
        checkpoint = checkpoint_store.write(checkpoint)
    elif resume_after_rate_limit:
        raise CheckpointError("--resume-after-rate-limit requires a recorded pre-transmission capacity stop")

    schedule = plan["schedule"]
    done = {record["operation"] for record in checkpoint["records"]}
    remaining = [item for item in schedule if item["operation"] not in done]
    if not remaining:
        raise CheckpointError("every scheduled acceptance operation has run; refusing repeated execution")
    selected = remaining[:operations] if operations else remaining
    stage = {"stage_number": len(checkpoint["execution"]["stages"]) + 1, "requested_operations": operations,
             "selected_operations": [item["operation"] for item in selected], "completed_operations": 0,
             "rate_limit_continuation": reset_continuation, "wait_for_token_reset": wait_for_token_reset}
    checkpoint["execution"]["stages"].append(stage)
    checkpoint = checkpoint_store.write(checkpoint)
    guard = GuardedExecution(checkpoint=checkpoint, checkpoint_store=checkpoint_store, ledger=ledger,
                             config=base_config, v2=True, mark=mark, clock=clock, sleep=sleep,
                             wait_for_token_reset=wait_for_token_reset,
                             reset_continuation_authorized=reset_continuation, stage=stage, gateway=gateway)
    scenario_positions = {item["scenario_id"]: index for index, item in enumerate(dataset["scenarios"])}

    async def persist_repair_pending(state: dict) -> None:
        checkpoint["active_arm"]["phase"] = "repair_pending"
        checkpoint["active_arm"]["initial"] = {"output": state["initial_output"],
                                               "validation_errors": state["validation_errors"]}
        checkpoint["active_arm"]["pending_operation"] = {"phase": "repair", "schema": REPAIR_SCHEMA,
                                                         "prompt_sha256": state["repair_prompt_sha256"]}
        checkpoint_store.write(checkpoint)

    for operation in selected:
        scenario, fixture_state = _operation_inputs(dataset, operation)
        active = checkpoint.get("active_arm")
        if active is not None and active.get("operation") != operation["operation"]:
            raise CheckpointError(UNRESOLVED_MESSAGE)
        resume = dict(fixture_state) if fixture_state is not None else None
        if active is None:
            checkpoint["active_arm"] = {"operation": operation["operation"], "kind": operation["kind"],
                                        "scenario_id": scenario["scenario_id"],
                                        "scenario_index": scenario_positions[scenario["scenario_id"]],
                                        "prompt_version": operation["prompt_version"],
                                        "configuration": operation["configuration"], "phase": "initial_pending",
                                        "attempts": []}
            checkpoint_store.write(checkpoint)
        elif active.get("phase") == "repair_pending":
            initial = active["initial"]
            resume = {**(resume or {}), "initial_output": initial["output"],
                      "validation_errors": initial["validation_errors"],
                      "repair_prompt_sha256": active["pending_operation"]["prompt_sha256"]}
            if resume.get("prompt_sha256") is None:
                _, context = _blocks_and_context(scenario)
                resume["prompt_sha256"] = canonical_sha256(generation_prompt_v2(
                    scenario["question"], context, prompt_version=operation["prompt_version"]))
        arm = await _evaluate_arm(scenario, operation["prompt_version"], arm_config(config, operation["configuration"]),
                                  guard.metered, resume=resume, before_repair=persist_repair_pending)
        attempts = checkpoint["active_arm"]["attempts"]
        interruption = arm.get("interruption") or {}
        if not attempts and arm.get("status") == "interrupted" and arm.get("failure_class") != EXECUTION_STOP:
            guard.record_stop("internal_error", "an internal error occurred before a settled transmission",
                              {"kind": "internal_error", "error_type": arm.get("error_type"),
                               "reserved_attempt_kept": checkpoint.get("active_attempt") is not None})
            checkpoint_store.write(checkpoint)
            break
        if arm.get("failure_class") == EXECUTION_STOP and not interruption.get("transmitted"):
            if not attempts and checkpoint["active_arm"].get("phase") != "repair_pending":
                checkpoint["active_arm"] = None
            checkpoint_store.write(checkpoint)
            break
        record = {**operation, "scenario_id": scenario["scenario_id"], **arm,
                  "gateway_invocations": len(attempts),
                  "physical_calls": sum(int(item.get("provider_calls") or 0) for item in attempts),
                  "usage": _arm_usage(attempts), "attempts": attempts,
                  "outcome_dimensions": outcome_dimensions(arm, attempts)}
        if operation["kind"] == "dedicated_repair":
            record["initial_output"] = None   # the fixture is dataset content, not a generation
            record["repair_schema_parsed"] = arm.get("repair_output") is not None
        checkpoint["records"].append(record)
        checkpoint["active_arm"] = None
        stage["completed_operations"] += 1
        checkpoint_store.write(checkpoint)
        if checkpoint["execution"].get("stopped"):
            break
    return checkpoint


# --------------------------------------------------------------------------- report and selection

def _claims_text(answer: dict | None) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(answer, dict):
        return [], [], []
    claims = [str(item.get("text") or "") for item in answer.get("claims") or []]
    limitations = [str(item.get("text") or "") for item in answer.get("limitations") or []]
    unanswered = [str(item) for item in answer.get("unanswered_parts") or []]
    return claims, limitations, unanswered


def _search(patterns: list[str], texts: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns for text in texts)


def evaluate_predicates(answer: dict | None, predicates: list[dict]) -> list[dict]:
    """Deterministic synthetic-reference checks of one final answer (structure is not science)."""
    claims, limitations, unanswered = _claims_text(answer)
    answer_texts = claims + limitations
    results = []
    for item in predicates:
        kind = item["kind"]
        if kind in ("expected_fact", "required_uncertainty"):
            passed = _search(item["any_of"], answer_texts)
        elif kind == "required_abstention":
            passed = _search(item["any_of"], unanswered + answer_texts)
        elif kind == "forbidden_claim":
            passed = not any(all(re.search(p, text, re.IGNORECASE) for p in item["all_of"])
                             and not _search(item.get("unless_any_of") or [], [text]) for text in claims)
        else:   # conditional, within one claim
            passed = not any(_search(item["if_any_of"], [text]) and not _search(item["then_any_of"], [text])
                             for text in claims)
        results.append({"id": item["id"], "kind": kind, "dimension": item["dimension"],
                        "critical": bool(item.get("critical")), "passed": passed})
    return results


def build_acceptance_report(checkpoint: dict, dataset: dict, references: dict, config: dict) -> dict:
    """Offline, read-only metrics per configuration and prompt version, with full denominators."""
    validate_dataset(dataset)
    validate_references(references, dataset)
    validate_acceptance_config(config)
    schedule = acceptance_schedule(config)
    records = {record["operation"]: record for record in checkpoint.get("records") or []}
    limits = {item["label"]: item["max_output_tokens"] for item in config["configurations"]}
    cells: dict[str, dict] = {}
    for label in limits:
        for version in config["prompt_versions"]:
            cells[f"{label}/{version}"] = {"planned_arms": 0, "executed": 0, "unexecuted": [], "completed": 0,
                                           "failed": 0, "interrupted": 0, "initial_valid": 0,
                                           "completion_budget_failures": 0,
                                           "provider_structured_generation_failures": 0,
                                           "pydantic_failures": 0, "structural_failures": 0,
                                           "repairs_attempted": 0, "repairs_succeeded": 0, "final_valid": 0,
                                           "predicates_passed": 0, "predicates_evaluated": 0,
                                           "critical_violations": 0, "dimensions": {},
                                           "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                                           "reasoning_reported_calls": 0, "calls": 0, "latency_ms": [],
                                           "min_headroom_tokens": None, "known_cost_usd": 0.0,
                                           "uncertain_reserved_usd": 0.0}
    dedicated = {}
    predicates = references["predicates"]
    for operation in schedule:
        record = records.get(operation["operation"])
        if operation["kind"] == "dedicated_repair":
            dedicated[operation["configuration"]] = (
                {"operation": operation["operation"], "executed": False} if record is None else
                {"operation": operation["operation"], "executed": True,
                 "repair_schema_parsed": bool(record.get("repair_schema_parsed")), "status": record.get("status"),
                 "failure_kind": record.get("failure_kind")})
            continue
        cell = cells[f"{operation['configuration']}/{operation['prompt_version']}"]
        cell["planned_arms"] += 1
        if record is None:
            cell["unexecuted"].append(operation["operation"])
            continue
        cell["executed"] += 1
        status = record.get("status")
        cell[status if status in ("completed", "failed", "interrupted") else "interrupted"] += 1
        cell["initial_valid"] += bool(record.get("initial_structurally_valid"))
        kind = record.get("failure_kind")
        cell["completion_budget_failures"] += kind == "completion_budget_exhausted"
        cell["provider_structured_generation_failures"] += (
            record.get("provider_failure_class") == "provider_structured_generation_failure")
        cell["pydantic_failures"] += kind == "invalid_structured_output"
        cell["structural_failures"] += record.get("initial_structurally_valid") is False
        cell["repairs_attempted"] += bool(record.get("repair_attempted"))
        cell["repairs_succeeded"] += bool(record.get("repair_attempted")) and status == "completed"
        cell["final_valid"] += status == "completed"
        results = evaluate_predicates(record.get("post_repair_output") if status == "completed" else None,
                                      predicates[record["scenario_id"]])
        cell["predicates_evaluated"] += len(results)
        cell["predicates_passed"] += sum(item["passed"] for item in results)
        cell["critical_violations"] += sum(item["critical"] and not item["passed"] for item in results)
        for item in results:
            dimension = cell["dimensions"].setdefault(item["dimension"], {"passed": 0, "evaluated": 0})
            dimension["evaluated"] += 1
            dimension["passed"] += item["passed"]
        for attempt in record.get("attempts") or []:
            cell["calls"] += int(attempt.get("provider_calls") or 0)
            cell["input_tokens"] += int(attempt.get("input_tokens") or 0)
            output = attempt.get("output_tokens")
            cell["output_tokens"] += int(output or 0)
            if type(attempt.get("reasoning_tokens")) is int:
                cell["reasoning_tokens"] += attempt["reasoning_tokens"]
                cell["reasoning_reported_calls"] += 1
            if attempt.get("latency_ms") is not None:
                cell["latency_ms"].append(attempt["latency_ms"])
            if type(output) is int:
                headroom = limits[operation["configuration"]] - output
                cell["min_headroom_tokens"] = headroom if cell["min_headroom_tokens"] is None else min(
                    cell["min_headroom_tokens"], headroom)
            if attempt.get("cost_status") == "complete":
                cell["known_cost_usd"] = round(cell["known_cost_usd"] + float(attempt.get("cost_usd") or 0.0), 8)
            else:
                cell["uncertain_reserved_usd"] = round(cell["uncertain_reserved_usd"]
                                                       + float(attempt.get("reserved_cost_usd") or 0.0), 8)
    for cell in cells.values():
        latencies = cell.pop("latency_ms")
        cell["mean_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else None
        cell["non_reasoning_completion_tokens"] = (cell["output_tokens"] - cell["reasoning_tokens"]
                                                   if cell["reasoning_reported_calls"] == cell["calls"] and cell["calls"]
                                                   else None)
    execution = checkpoint.get("execution") or {}
    continuations = execution.get("rate_limit_continuations") or []
    report = {
        "adapter": ADAPTER, "dataset_sha256": dataset["dataset_sha256"],
        "references_sha256": references["references_sha256"],
        "reference_label": references["label"],
        "cells": cells, "dedicated_repairs": dedicated,
        "missing_operations": [item["operation"] for item in schedule if item["operation"] not in records],
        "stop_code": execution.get("stop_code"),
        "unresolved_attempt": checkpoint.get("active_attempt") is not None,
        "physical_requests": (execution.get("budget") or {}).get("provider_calls_charged"),
        "budget": execution.get("budget"),
        "quota_continuations": len(continuations),
        "caveats": ["synthetic acceptance references, not human-reviewed gold labels",
                    "structural validity is not scientific correctness",
                    "a 3-scenario sample cannot establish statistical superiority",
                    "costs are estimates and reservations, not provider charges"],
    }
    report["selection"] = select_configuration(report, limits)
    return report


def select_configuration(report: dict, limits: dict[str, int]) -> dict:
    """Apply the predeclared SELECTION_RULES to a report. Never claims superiority."""
    reasons: dict[str, list[str]] = {label: [] for label in limits}
    totals = {}
    for label in limits:
        cell_items = [cell for key, cell in report["cells"].items() if key.startswith(f"{label}/")]
        totals[label] = {
            "final_valid": sum(cell["final_valid"] for cell in cell_items),
            "planned": sum(cell["planned_arms"] for cell in cell_items),
            "initial_parsed": sum(cell["executed"] - cell["pydantic_failures"] - cell["completion_budget_failures"]
                                  - cell["provider_structured_generation_failures"] - cell["interrupted"]
                                  for cell in cell_items),
            "budget_failures": sum(cell["completion_budget_failures"] for cell in cell_items),
            "critical": sum(cell["critical_violations"] for cell in cell_items),
            "failed_predicates": sum(cell["predicates_evaluated"] - cell["predicates_passed"] for cell in cell_items),
            "passed_predicates": sum(cell["predicates_passed"] for cell in cell_items),
            "min_headroom": min((cell["min_headroom_tokens"] for cell in cell_items
                                 if cell["min_headroom_tokens"] is not None), default=None),
        }
    global_blocks = []
    if report["stop_code"] in INTEGRITY_STOPS:
        global_blocks.append(f"execution-integrity stop: {report['stop_code']}")
    if report["stop_code"] in ACCOUNTING_STOPS:
        global_blocks.append(f"accounting defect: {report['stop_code']}")
    if report["unresolved_attempt"]:
        global_blocks.append("an unresolved, potentially transmitted attempt remains")
    if report["missing_operations"]:
        global_blocks.append(f"missing operations: {report['missing_operations']}")
    for label, total in totals.items():
        repair = report["dedicated_repairs"].get(label) or {}
        if total["initial_parsed"] < 1:
            reasons[label].append("no initial generation parsed on the actual gateway")
        if not repair.get("repair_schema_parsed"):
            reasons[label].append("the dedicated repair schema test did not parse")
        if total["final_valid"] < SELECTION_RULES["adequate_final_valid_arms"]:
            reasons[label].append(f"final structural validity {total['final_valid']}/{total['planned']} below "
                                  f"{SELECTION_RULES['adequate_final_valid_arms']}")
    a, b = list(limits)
    for this, other in ((a, b), (b, a)):
        if (totals[this]["critical"] >= totals[other]["critical"] + 2
                or totals[this]["failed_predicates"] >= totals[other]["failed_predicates"] + 2):
            reasons[this].append("material scientific regression on the synthetic reference predicates")
    candidates = [label for label in limits if not reasons[label]] if not global_blocks else []
    if not candidates:
        decision, why = "inconclusive", "no configuration met every predeclared gate"
    elif len(candidates) == 1:
        decision, why = candidates[0], "the only configuration meeting every gate"
    else:
        smaller, larger = sorted(candidates, key=lambda label: limits[label])
        advantage = []
        if totals[larger]["budget_failures"] < totals[smaller]["budget_failures"]:
            advantage.append("fewer completion-budget failures")
        if totals[larger]["final_valid"] >= totals[smaller]["final_valid"] + 1:
            advantage.append("more final-valid arms")
        if totals[larger]["passed_predicates"] >= totals[smaller]["passed_predicates"] + 3:
            advantage.append("at least 3 more passed reference predicates")
        small_headroom = totals[smaller]["min_headroom"]
        large_headroom = totals[larger]["min_headroom"]
        if (small_headroom is not None and small_headroom < 0.10 * limits[smaller]
                and large_headroom is not None and large_headroom >= 0.10 * limits[larger]):
            advantage.append("adequate completion headroom where the smaller ceiling had less than 10%")
        decision = larger if advantage else smaller
        why = ("larger ceiling shows a meaningful advantage: " + "; ".join(advantage)) if advantage else \
            "both met every gate; the smaller completion ceiling is preferred"
    return {"rules_version": SELECTION_RULES["version"], "decision": decision, "reason": why,
            "global_blocks": global_blocks, "gate_failures": reasons, "totals": totals,
            "statistical_claim": "none: a 3-scenario acceptance sample cannot establish superiority"}


__all__ = [
    "ADAPTER", "AUTHORIZATION_SCHEMA", "CONFIG_SCHEMA", "DATASET_SCHEMA", "MAX_PHYSICAL_REQUESTS",
    "PROPOSED_DIAGNOSTIC_BUDGET_USD", "REFERENCES_SCHEMA", "SELECTION_RULES", "AcceptanceError",
    "acceptance_dry_run", "acceptance_schedule", "arm_config", "build_acceptance_report",
    "check_acceptance_authorization", "evaluate_predicates", "run_acceptance", "select_configuration",
    "validate_acceptance_config", "validate_dataset", "validate_references",
]
