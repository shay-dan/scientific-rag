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

Paid calls: ``dry_run`` plans every call and estimates tokens and a cost ceiling without
calling any provider. ``run_replay`` refuses to call a provider unless an explicit
authorization (``scientific-rag-paid-evaluation-authorization/v1``) names this adapter,
the handoff, a single pinned provider and model, a call ceiling and a cost ceiling. It
never starts a call whose conservative cost ceiling would exceed the remaining budget,
and stops if usage cannot be measured.
"""
from __future__ import annotations

import math
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
    """Plan every call and estimate tokens and a cost ceiling. Makes no provider call."""
    from scientific_rag.costing import calculate_cost
    planned = [plan(item) for item in claim_inputs(items)]
    calls = [p for p in planned if p["status"] == "planned"]
    input_tokens = sum(max(1, math.ceil(p["prompt_chars"] / 4)) for p in calls)
    input_ceiling = sum(call_token_ceiling(p["prompt_chars"])[0] for p in calls)
    output_ceiling = OUTPUT_TOKEN_CEILING_PER_CALL * len(calls)
    route = route or gateway_route()
    costs = {}
    for provider in route["provider_order"]:
        model = route["models"][provider]
        estimate, source = calculate_cost(provider, model, input_tokens, output_ceiling // 10)
        ceiling, _ = calculate_cost(provider, model, input_ceiling, output_ceiling)
        costs[f"{provider}:{model}"] = {"cost_estimate_usd": estimate, "cost_ceiling_usd": ceiling, "rate_source": source}
    known = [c["cost_ceiling_usd"] for c in costs.values() if c["cost_ceiling_usd"] is not None]
    return {
        "adapter": ADAPTER, "provider_calls_made": 0, "inputs": len(planned),
        "status": dict(sorted(Counter(p["status"] for p in planned).items())),
        "not_executed_reasons": dict(sorted(Counter(p["reason"] for p in planned if p["status"] != "planned").items())),
        "planned_calls": len(calls),
        "prompt_chars": {"total": sum(p["prompt_chars"] for p in calls),
                         "max": max((p["prompt_chars"] for p in calls), default=0)},
        "isolation_applied": sum(p["isolation_applied"] for p in calls),
        "estimated_input_tokens": input_tokens, "input_token_ceiling": input_ceiling,
        "output_token_ceiling": output_ceiling,
        "token_estimate_method": "estimate: ceil(prompt characters / 4) input (production estimate_text_tokens rule) "
                                 f"and {OUTPUT_TOKEN_CEILING_PER_CALL // 10} output tokens per call; ceiling: "
                                 f"ceil(prompt characters / {INPUT_CHARS_PER_TOKEN_CEILING}) input and "
                                 f"{OUTPUT_TOKEN_CEILING_PER_CALL} output tokens per call",
        "route": route, "cost_ceiling_by_route": costs,
        "cost_ceiling_usd_any_route": max(known) if len(known) == len(costs) and known else None,
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
    for field in ("max_calls", "max_cost_usd"):
        value = authorization.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"positive {field}")
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


def _usage(summary: dict) -> dict:
    models = sorted(summary.get("by_model") or {})
    return {"cost_status": "measured" if summary.get("token_usage_complete") and summary.get("estimated_cost_usd") is not None
            else "usage_incomplete",
            "provider_calls": summary.get("provider_calls", 0), "input_tokens": summary.get("input_tokens"),
            "output_tokens": summary.get("output_tokens"), "estimated_cost_usd": summary.get("estimated_cost_usd"),
            "provider_models": models, "failed_attempts": summary.get("failed_attempts", 0)}


async def run_replay(items: list[dict], *, authorization: dict, dataset_hash: str, route: dict | None = None,
                     verifier: Callable[..., Awaitable] = verify_claims) -> tuple[list[dict], dict]:
    """Run the production verifier on each claim within the authorized ceilings. Unscored."""
    from scientific_rag.llm_gateway import begin_trace, end_trace, summarize_trace
    route = route or gateway_route()
    auth = check_authorization(authorization, adapter=ADAPTER, dataset_hash=dataset_hash, route=route)
    allowed_model = f"{auth['provider']}:{auth['model']}"
    planned = [plan(item) for item in claim_inputs(items)]
    if sum(p["status"] == "planned" for p in planned) > auth["max_calls"] and not auth.get("allow_partial"):
        raise AuthorizationError("planned calls exceed the authorized max_calls; select a smaller pilot")
    records, spent, calls, stop_reason = [], 0.0, 0, None
    for entry in planned:
        item = entry["item"]
        if entry["status"] != "planned":
            records.append(_record(item, "claim_verification", entry["status"], reason=entry["reason"]))
            continue
        next_ceiling = call_cost_ceiling(entry["prompt_chars"], auth["provider"], auth["model"])
        if stop_reason or calls >= auth["max_calls"] or spent + next_ceiling > auth["max_cost_usd"]:
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
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        usage = {**_usage(summary), "latency_ms": elapsed}
        calls += max(1, usage["provider_calls"])   # gateway retries and fallbacks count against max_calls
        if usage["cost_status"] == "measured":
            spent += usage["estimated_cost_usd"]
        else:
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
    return records, {"authorization_id": auth["authorization_id"], "provider_calls": calls,
                     "estimated_cost_usd": round(spent, 8), "stopped": stop_reason}
