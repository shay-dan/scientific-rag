"""R6C: replay the production semantic verifier on label-free benchmark claim inputs.

Opt-in evaluation adapter. It reuses the production verifier unchanged:
``ClaimVerificationInput`` -> ``plan_verification_batches`` (validation, R3 isolation,
prompt bound) -> ``verify_claims`` (the gateway call, response validation, and the R3
instruction-like-support guard). It never calls ``apply_semantic_verdicts`` and never
reads or changes the off/shadow/enforce setting: it produces verdicts only, which are
exported by opaque input ID for owner-side scoring. Nothing is scored here.

Representation (documented, not tuned): each benchmark claim is one target
(``claims.1``) verified in its own gateway call, so no claim sees another claim. The
benchmark supplies evidence segments but no citation quote, so the claim's text segments,
in evidence-index order and separated by blank lines, form the matched passage and stand
in for the quote (the R5/R6B convention). The research-question slot, which the
production prompt treats as untrusted context and never as evidence, carries a fixed
neutral sentence (``REPLAY_QUESTION``). Segments whose text is not redistributed are never
fetched; a claim without text is ``not_executed``.

Paid calls: ``dry_run`` plans every call and estimates tokens and a *planning* cost bound
without calling any provider. ``run_replay`` refuses to call a provider unless:
- an explicit authorization (``scientific-rag-paid-evaluation-authorization/v1``) names
  this adapter, the handoff, a single pinned priced provider and model, a positive integer
  call limit and a finite positive cost limit (parsed strictly: no NaN or Infinity); and
- the production chat model for that route enforces a provider-side output-token limit no
  larger than the reserved ``OUTPUT_TOKEN_CEILING_PER_CALL``. The production gateway sets
  no such limit today, so live verifier execution is refused: without it the per-call
  reservation is a planning assumption, not a spending ceiling.

Accounting follows R4: a call with complete provider usage is charged its measured cost
(``complete``); a call without complete usage is charged its full reservation and recorded
``incomplete`` with any known lower bound, and execution stops. Missing usage is never zero.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Awaitable, Callable

from ..claim_verification import (
    INSTRUCTION_LIKE_EVIDENCE_REASON, MAX_PASSAGE_CONTEXT_CHARS, SEMANTIC_VERIFICATION_PROMPT_MAX_CHARS,
    ClaimVerificationError, ClaimVerificationInput, build_semantic_verification_prompt, isolate_verification_evidence,
    plan_verification_batches, verify_claims,
)
from ..evaluation_report import sha256_hex
from .prediction_export import TEXT_REASON, _record

ADAPTER = "production_semantic_verifier"
AUTHORIZATION_SCHEMA = "scientific-rag-paid-evaluation-authorization/v1"
REPLAY_QUESTION = "Does the supplied evidence support the claim?"
REPLAY_TARGET = "claims.1"
UNTITLED = "[source title not supplied by the benchmark]"
GATEWAY_ROLE = "gatekeeper"
# Conservative output ceiling per call for the cost estimate: one assessment (reason <= 500
# characters) plus room for reasoning-model tokens that some providers bill as output.
OUTPUT_TOKEN_CEILING_PER_CALL = 2_000
# Pre-call ceiling: input tokens bounded at one token per 3 prompt characters (more
# conservative than the production 4-character estimate) plus the output ceiling.
INPUT_CHARS_PER_TOKEN_CEILING = 3
PROMPT_BOUND_REASON = (f"evidence does not fit the production verifier prompt bound "
                       f"({SEMANTIC_VERIFICATION_PROMPT_MAX_CHARS} characters); not truncated")
CEILING_REASON = "authorized cost or call ceiling reached before this input"
USAGE_REASON = "stopped: provider usage was not reported, so the cost ceiling could not be enforced"


class AuthorizationError(PermissionError):
    """A provider call was requested without a matching, explicit authorization."""


def _reject_constant(literal: str):
    raise AuthorizationError(f"authorization contains the non-finite JSON literal {literal}")


def parse_authorization(text: str) -> dict:
    """Strict JSON: NaN, Infinity and -Infinity are refused (Python's parser accepts them)."""
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise AuthorizationError(f"authorization is not valid JSON ({error.msg})") from None
    if not isinstance(value, dict):
        raise AuthorizationError("authorization must be a JSON object")
    return value


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value > 0)


def claim_inputs(fixed_items: list[dict]) -> list[dict]:
    """Claim-verification inputs: claim records, and pilot items typed as claim verification."""
    return sorted((item for item in fixed_items if item["kind"] == "claim_verification_input"
                   or item.get("item_type") == "claim_verification"), key=lambda item: item["input_id"])


def verification_input(item: dict) -> tuple[ClaimVerificationInput | None, str | None]:
    """Map one benchmark claim to a production ``ClaimVerificationInput`` (or a reason it cannot be)."""
    segments = sorted((s for s in item.get("segments", []) if s["availability"] == "text"),
                      key=lambda s: s["evidence_index"])
    if not segments:
        return None, TEXT_REASON
    passage = "\n\n".join(segment["text"] for segment in segments)
    titles = sorted({segment["title"] for segment in segments if segment.get("title")})
    claim = item.get("claim") or item.get("question")
    return ClaimVerificationInput(
        target=REPLAY_TARGET, claim_text=claim, source_id="S1", evidence_quote=passage,
        matched_source_title=" | ".join(titles) if titles else UNTITLED,
        # Same windowing as production: the quote is the whole passage, so it is kept whole.
        matched_passage_context=passage,
    ), None


def plan(item: dict) -> dict:
    """Validate and plan one replay call with the production planner; no provider call."""
    replay_input, reason = verification_input(item)
    if replay_input is None:
        return {"item": item, "status": "not_executed", "reason": reason}
    try:
        batches = plan_verification_batches([replay_input], question=REPLAY_QUESTION)
    except ClaimVerificationError as error:
        bound = "prompt limit" in str(error)
        return {"item": item, "status": "not_executed" if bound else "invalid",
                "reason": PROMPT_BOUND_REASON if bound else f"production verifier refused the input ({type(error).__name__})"}
    prompt = batches[0][1]
    return {"item": item, "status": "planned", "input": replay_input, "prompt_chars": len(prompt),
            "passage_chars": len(replay_input.matched_passage_context),
            "isolation_applied": isolate_verification_evidence(replay_input).instruction_like_content_removed}


def prompt_contract_hash() -> str:
    """Fingerprint of the production prompt for a fixed synthetic input (changes if the prompt changes)."""
    probe = ClaimVerificationInput(target=REPLAY_TARGET, claim_text="probe claim", source_id="S1",
                                   evidence_quote="probe evidence", matched_source_title="probe",
                                   matched_passage_context="probe evidence")
    return sha256_hex(build_semantic_verification_prompt([probe], REPLAY_QUESTION))


def call_token_ceiling(prompt_chars: int) -> tuple[int, int]:
    return math.ceil(prompt_chars / INPUT_CHARS_PER_TOKEN_CEILING), OUTPUT_TOKEN_CEILING_PER_CALL


def call_cost_ceiling(prompt_chars: int, provider: str, model: str) -> float | None:
    from scientific_rag.costing import calculate_cost
    return calculate_cost(provider, model, *call_token_ceiling(prompt_chars))[0]


_KEY_VARIABLES = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GOOGLE_API_KEY"}


def output_limit_status(provider: str, model: str, role: str = GATEWAY_ROLE) -> dict:
    """Inspect the production chat model for this route for a provider-side output-token limit.

    Builds the model exactly as the gateway does (``create_chat_model``); no request is sent.
    A placeholder key is set only for construction when none is configured, then removed.
    """
    from scientific_rag import llm_gateway
    variable = _KEY_VARIABLES[provider]
    placeholder = not llm_gateway._key_for(provider)
    try:
        if placeholder:
            os.environ[variable] = "placeholder-for-inspection-only"
        chat = llm_gateway.create_chat_model(provider, role, 0)
    except Exception as error:
        return {"enforced": False, "limit": None, "source": f"could not inspect the model ({type(error).__name__})"}
    finally:
        if placeholder:
            os.environ.pop(variable, None)
    built = getattr(chat, "model_name", None) or getattr(chat, "model", None)
    limits = [getattr(chat, name, None) for name in ("max_tokens", "max_completion_tokens", "max_output_tokens")]
    limits += [(getattr(chat, "model_kwargs", None) or {}).get(name) for name in ("max_tokens", "max_completion_tokens")]
    limit = next((value for value in limits if isinstance(value, int) and not isinstance(value, bool)), None)
    enforced = limit is not None and limit <= OUTPUT_TOKEN_CEILING_PER_CALL and str(built).endswith(model)
    return {"enforced": enforced, "limit": limit, "model": str(built),
            "source": f"production create_chat_model({provider!r}, {role!r}) attributes"}


def gateway_route() -> dict:
    """The configured gateway route for the verifier role (configuration only; no call)."""
    from scientific_rag.llm_gateway import model_for, provider_order
    order = provider_order(GATEWAY_ROLE)
    return {"role": GATEWAY_ROLE, "provider_order": list(order),
            "models": {provider: model_for(provider, GATEWAY_ROLE) for provider in order}}


def configuration() -> dict:
    return {"adapter": ADAPTER, "replay_question": REPLAY_QUESTION, "target": REPLAY_TARGET,
            "claims_per_call": 1, "temperature": 0, "gateway_role": GATEWAY_ROLE,
            "max_passage_context_chars": MAX_PASSAGE_CONTEXT_CHARS,
            "prompt_max_chars": SEMANTIC_VERIFICATION_PROMPT_MAX_CHARS,
            "prompt_contract_sha256": prompt_contract_hash(),
            "enforcement": "not applied (verdicts only; apply_semantic_verdicts is never called)"}


def dry_run(items: list[dict], *, route: dict | None = None) -> dict:
    """Plan every call; estimate tokens and a planning cost bound. Makes no provider call."""
    from scientific_rag.costing import calculate_cost
    planned = [plan(item) for item in claim_inputs(items)]
    calls = [p for p in planned if p["status"] == "planned"]
    input_tokens = sum(max(1, math.ceil(p["prompt_chars"] / 4)) for p in calls)
    input_reserve = sum(call_token_ceiling(p["prompt_chars"])[0] for p in calls)
    output_reserve = OUTPUT_TOKEN_CEILING_PER_CALL * len(calls)
    route = route or gateway_route()
    by_route, limits = {}, {}
    for provider in route["provider_order"]:
        model = route["models"][provider]
        estimate, source = calculate_cost(provider, model, input_tokens, output_reserve // 10)
        bound, _ = calculate_cost(provider, model, input_reserve, output_reserve)
        limits[provider] = output_limit_status(provider, model)
        by_route[f"{provider}:{model}"] = {"cost_estimate_usd": estimate, "planning_cost_bound_usd": bound,
                                           "output_limit_enforced": limits[provider]["enforced"], "rate_source": source}
    bounds = [entry["planning_cost_bound_usd"] for entry in by_route.values()]
    enforced = all(status["enforced"] for status in limits.values())
    return {
        "adapter": ADAPTER, "provider_calls_made": 0, "inputs": len(planned),
        "status": dict(sorted(Counter(p["status"] for p in planned).items())),
        "not_executed_reasons": dict(sorted(Counter(p["reason"] for p in planned if p["status"] != "planned").items())),
        "planned_calls": len(calls),
        "prompt_chars": {"total": sum(p["prompt_chars"] for p in calls),
                         "max": max((p["prompt_chars"] for p in calls), default=0)},
        "isolation_applied": sum(p["isolation_applied"] for p in calls),
        "estimated_input_tokens": input_tokens, "reserved_input_tokens": input_reserve,
        "reserved_output_tokens": output_reserve,
        "token_estimate_method": "estimate: ceil(prompt characters / 4) input (production estimate_text_tokens rule) "
                                 f"and {OUTPUT_TOKEN_CEILING_PER_CALL // 10} output tokens per call; reservation: "
                                 f"ceil(prompt characters / {INPUT_CHARS_PER_TOKEN_CEILING}) input and "
                                 f"{OUTPUT_TOKEN_CEILING_PER_CALL} output tokens per call",
        "route": route, "planning_cost_by_route": by_route,
        "planning_cost_bound_usd_any_route": max(bounds) if bounds and None not in bounds else None,
        "output_limit": limits[route["provider_order"][0]] if limits else None,
        "planning_cost_bound_note": (
            f"provider-enforced output limit of at most {OUTPUT_TOKEN_CEILING_PER_CALL} tokens on every route"
            if enforced else
            f"assumes at most {OUTPUT_TOKEN_CEILING_PER_CALL} output tokens per call; not a hard spending ceiling, "
            "because the production chat model enforces no provider-side output limit on this route; "
            "live verifier execution is refused until one is enforced"),
        "configuration": configuration(),
    }


def check_authorization(authorization: dict, *, adapter: str, dataset_hash: str, route: dict) -> dict:
    """Refuse provider calls unless the authorization is explicit and matches this run."""
    if not isinstance(authorization, dict) or authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise AuthorizationError("paid calls need an explicit authorization file "
                                 f"({AUTHORIZATION_SCHEMA}); none was supplied")
    problems = [field for field in ("authorization_id", "approved_by", "provider", "model")
                if not str(authorization.get(field) or "").strip()]
    if authorization.get("adapter") != adapter:
        problems.append(f"adapter {adapter}")
    if authorization.get("handoff_dataset_hash") != dataset_hash:
        problems.append("handoff_dataset_hash of this handoff")
    if not _positive_int(authorization.get("max_calls")):
        problems.append("max_calls as a positive integer")
    if not _positive_finite(authorization.get("max_cost_usd")):
        problems.append("max_cost_usd as a finite positive number")
    if problems:
        raise AuthorizationError("authorization is missing or does not match: " + ", ".join(problems))
    if route["provider_order"] != [authorization["provider"]]:
        raise AuthorizationError("the gateway route must be pinned to the single authorized provider "
                                 f"(set LLM_PROVIDER_ORDER_{GATEWAY_ROLE.upper()}={authorization['provider']}); "
                                 f"configured order is {route['provider_order']}")
    if route["models"][authorization["provider"]] != authorization["model"]:
        raise AuthorizationError(f"configured model {route['models'][authorization['provider']]!r} is not the "
                                 f"authorized model {authorization['model']!r}")
    from scientific_rag.costing import rate_for
    if rate_for(authorization["provider"], authorization["model"]) is None:
        raise AuthorizationError("the authorized model has no known price, so the cost ceiling cannot be enforced")
    return authorization


def _known_cost(attempts: list[dict]) -> float:
    """Sum of costs for attempts that reported usage and have a price: a lower bound."""
    from scientific_rag.costing import calculate_cost
    total = 0.0
    for attempt in attempts:
        if attempt.get("skipped") or attempt.get("input_tokens") is None or attempt.get("output_tokens") is None:
            continue
        cost, _ = calculate_cost(str(attempt["provider"]), str(attempt["model"]), int(attempt["input_tokens"]),
                                 int(attempt["output_tokens"]))
        total += cost or 0.0
    return round(total, 8)


def _usage(summary: dict, reserved: float) -> dict:
    complete = bool(summary.get("token_usage_complete")) and summary.get("estimated_cost_usd") is not None
    usage = {"cost_status": "complete" if complete else "incomplete",
             "provider_calls": summary.get("provider_calls", 0), "input_tokens": summary.get("input_tokens"),
             "output_tokens": summary.get("output_tokens"), "provider_models": sorted(summary.get("by_model") or {}),
             "failed_attempts": summary.get("failed_attempts", 0), "reserved_cost_usd": reserved}
    if complete:
        usage["cost_usd"] = summary["estimated_cost_usd"]
    else:
        usage["known_cost_lower_bound_usd"] = _known_cost(summary.get("attempts") or [])
    return usage


async def run_replay(items: list[dict], *, authorization: dict, dataset_hash: str, route: dict | None = None,
                     output_limit: dict | None = None,
                     verifier: Callable[..., Awaitable] = verify_claims) -> tuple[list[dict], dict]:
    """Run the production verifier on each claim within the authorized budget. Unscored.

    ``output_limit`` defaults to inspecting the production chat model; tests may pass a
    stand-in to exercise the execution path without a provider."""
    from scientific_rag.llm_gateway import begin_trace, end_trace, summarize_trace
    route = route or gateway_route()
    auth = check_authorization(authorization, adapter=ADAPTER, dataset_hash=dataset_hash, route=route)
    limit = output_limit if output_limit is not None else output_limit_status(auth["provider"], auth["model"])
    if not limit.get("enforced") or not isinstance(limit.get("limit"), int) or limit["limit"] > OUTPUT_TOKEN_CEILING_PER_CALL:
        raise AuthorizationError(
            "live verifier execution refused: the production chat model enforces no provider-side output-token "
            f"limit of at most {OUTPUT_TOKEN_CEILING_PER_CALL} tokens, so the per-call reservation is not a "
            f"spending ceiling (found: {limit.get('limit')!r})")
    allowed_model = f"{auth['provider']}:{auth['model']}"
    planned = [plan(item) for item in claim_inputs(items)]
    if sum(p["status"] == "planned" for p in planned) > auth["max_calls"] and not auth.get("allow_partial"):
        raise AuthorizationError("planned calls exceed the authorized max_calls; select a smaller pilot")
    budget = {"provider_calls_charged": 0, "cost_complete_usd": 0.0, "known_cost_lower_bound_usd": 0.0,
              "uncertain_reserved_usd": 0.0, "budget_charged_usd": 0.0}
    records, stop_reason = [], None
    for entry in planned:
        item = entry["item"]
        if entry["status"] != "planned":
            records.append(_record(item, "claim_verification", entry["status"], reason=entry["reason"]))
            continue
        reserved = call_cost_ceiling(entry["prompt_chars"], auth["provider"], auth["model"])
        if (stop_reason or budget["provider_calls_charged"] + 1 > auth["max_calls"]
                or budget["budget_charged_usd"] + reserved > auth["max_cost_usd"]):
            records.append(_record(item, "claim_verification", "not_executed", reason=stop_reason or CEILING_REASON))
            continue
        token = begin_trace()
        started = time.perf_counter()
        error = None
        try:
            batch = await verifier([entry["input"]], question=REPLAY_QUESTION, max_gateway_invocations=1)
        except Exception as exc:   # gateway, validation or programming error: recorded, never retried
            error = exc
        finally:
            summary = summarize_trace(end_trace(token))
        usage = {**_usage(summary, reserved), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        budget["provider_calls_charged"] += max(1, usage["provider_calls"])   # retries and fallbacks count
        if usage["cost_status"] == "complete":
            budget["cost_complete_usd"] += usage["cost_usd"]
            budget["budget_charged_usd"] += usage["cost_usd"]
        else:   # unknown spend consumes the whole reservation; it is never treated as zero
            budget["uncertain_reserved_usd"] += reserved
            budget["known_cost_lower_bound_usd"] += usage["known_cost_lower_bound_usd"]
            budget["budget_charged_usd"] += reserved
            stop_reason = USAGE_REASON
        if any(model != allowed_model for model in usage["provider_models"]):
            records.append(_record(item, "claim_verification", "invalid", usage=usage,
                                   reason="answered by a provider/model outside the authorization"))
            stop_reason = "stopped: a call was answered by an unauthorized provider/model"
            continue
        if error is not None:
            kind = "invalid_response" if "response invalid" in str(error) else "gateway_failure"
            records.append(_record(item, "claim_verification", "failed", usage=usage,
                                   reason=f"{kind} ({type(error).__name__})"))
            continue
        assessment = batch.assessments[0]
        records.append(_record(item, "claim_verification", "completed", usage=usage, prediction={
            "verdict": assessment.verdict.value, "reason": assessment.reason,
            "instruction_guard_applied": (not isolate_verification_evidence(entry["input"]).safe
                                          and assessment.reason == INSTRUCTION_LIKE_EVIDENCE_REASON)}))
    records.sort(key=lambda record: record["input_id"])
    budget = {key: round(value, 8) if isinstance(value, float) else value for key, value in budget.items()}
    return records, {"authorization_id": auth["authorization_id"], "budget": budget, "stopped": stop_reason,
                     "output_limit": limit}
