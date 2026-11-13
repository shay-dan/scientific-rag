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
without calling any provider. ``run_replay`` refuses to call a provider unless an explicit
authorization (``scientific-rag-paid-evaluation-authorization/v1``) names this adapter,
the handoff, a single pinned priced provider and model, a positive integer call limit, a
finite positive cost limit and a positive integer ``max_output_tokens`` (parsed strictly:
no NaN or Infinity), and the gateway can enforce that output limit for the provider (R6D).
The limit is passed to the production verifier, which passes it to the gateway; the
gateway sets it on the provider request and refuses any outbound request without it.

Budget (R4 accounting, ``BudgetLedger``): one call and a conservative cost (input at one
token per 3 prompt characters, output at the authorized limit) are reserved before each
call, atomically, so concurrent or cancelled calls cannot overspend. A call with complete
provider usage is then charged its measured cost (``complete``); a call without complete
usage keeps its full reservation, is recorded ``incomplete`` with any known lower bound,
and stops the run. Missing usage is never zero. The output limit bounds the output tokens
the provider is asked for; it is not an absolute billing guarantee (see the R6D notes in
``docs/benchmark-r6c.md``).
"""
from __future__ import annotations

import json
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
# Default output limit per call: one assessment (reason <= 500 characters) plus room for
# reasoning-model tokens that some providers bill as output. The authorization sets the limit.
OUTPUT_TOKEN_CEILING_PER_CALL = 2_000
OUTPUT_TOKEN_ESTIMATE_PER_CALL = 200
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


def call_token_ceiling(prompt_chars: int, max_output_tokens: int = OUTPUT_TOKEN_CEILING_PER_CALL) -> tuple[int, int]:
    return math.ceil(prompt_chars / INPUT_CHARS_PER_TOKEN_CEILING), max_output_tokens


def call_cost_ceiling(prompt_chars: int, provider: str, model: str,
                      max_output_tokens: int = OUTPUT_TOKEN_CEILING_PER_CALL) -> float | None:
    """Reservation for one call: conservative input tokens plus the full output limit."""
    from scientific_rag.costing import calculate_cost
    return calculate_cost(provider, model, *call_token_ceiling(prompt_chars, max_output_tokens))[0]


def output_limit_status(provider: str, max_output_tokens: int) -> dict:
    """Whether the gateway can enforce this output limit for the provider. No call is made."""
    from scientific_rag import llm_gateway
    try:
        limit = llm_gateway.validate_max_output_tokens(max_output_tokens)
        if limit is None:
            raise llm_gateway.InvalidOutputTokenLimit("max_output_tokens is required")
        parameters = llm_gateway.output_token_limit_parameters(provider)
    except llm_gateway.OutputTokenLimitError as error:
        return {"enforced": False, "limit": None, "parameters": None, "source": str(error)}
    return {"enforced": True, "limit": limit, "parameters": list(parameters),
            "source": "gateway sets the limit on the provider request and refuses outbound requests without it"}


class BudgetLedger:
    """Reserve-then-settle budget. ``reserve`` is atomic (no await), so concurrent calls cannot overspend."""

    def __init__(self, max_calls: int, max_cost_usd: float) -> None:
        self.max_calls, self.max_cost_usd = max_calls, max_cost_usd
        self.calls = 0
        self.complete = self.lower_bound = self.uncertain = self.unsettled = self.charged = 0.0

    def reserve(self, cost: float | None) -> bool:
        """Charge one call and its reserved cost, or refuse if either allowance would be exceeded."""
        if cost is None or self.calls + 1 > self.max_calls or self.charged + cost > self.max_cost_usd:
            return False
        self.calls += 1
        self.unsettled += cost
        self.charged += cost
        return True

    def settle(self, reserved: float, usage: dict) -> None:
        """Replace a reservation with measured cost when usage is complete; otherwise keep it charged."""
        self.calls += max(0, int(usage.get("provider_calls") or 0) - 1)   # gateway retries and fallbacks
        self.unsettled -= reserved
        if usage["cost_status"] == "complete":
            self.complete += usage["cost_usd"]
            self.charged += usage["cost_usd"] - reserved
        else:   # unknown spend consumes the whole reservation; it is never treated as zero
            self.uncertain += reserved
            self.lower_bound += usage.get("known_cost_lower_bound_usd") or 0.0

    def snapshot(self) -> dict:
        clean = lambda value: round(max(value, 0.0), 8)
        return {"provider_calls_charged": self.calls, "cost_complete_usd": clean(self.complete),
                "known_cost_lower_bound_usd": clean(self.lower_bound), "uncertain_reserved_usd": clean(self.uncertain),
                "unsettled_reserved_usd": clean(self.unsettled), "budget_charged_usd": clean(self.charged)}


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


def live_execution_status(route: dict, max_output_tokens: int, planned_calls: int) -> dict:
    """What a live run would still need. Informational; ``run_replay`` enforces the same gates."""
    from scientific_rag.costing import rate_for
    blockers = []
    order = route["provider_order"]
    if len(order) != 1:
        blockers.append(f"the gateway route must be pinned to one provider (LLM_PROVIDER_ORDER_{GATEWAY_ROLE.upper()}); "
                        f"configured order is {order}")
    provider = order[0] if order else None
    status = (output_limit_status(provider, max_output_tokens) if provider
              else {"enforced": False, "limit": None, "parameters": None, "source": "no provider configured"})
    if not status["enforced"]:
        blockers.append(f"provider {provider!r} cannot enforce a verified output-token limit: {status['source']}")
    if provider and rate_for(provider, route["models"][provider]) is None:
        blockers.append("the model has no known price, so the budget cannot be enforced")
    if not planned_calls:
        blockers.append("no call is planned")
    return {"eligible_after_authorization": not blockers, "blockers": blockers,
            "max_output_tokens": max_output_tokens, "output_limit": status,
            "still_required": "an owner authorization file naming this adapter, handoff, provider, model, "
                              "max_calls, max_cost_usd and max_output_tokens"}


def dry_run(items: list[dict], *, route: dict | None = None,
            max_output_tokens: int = OUTPUT_TOKEN_CEILING_PER_CALL) -> dict:
    """Plan every call; estimate tokens and a planning cost bound. Makes no provider call."""
    from scientific_rag.costing import calculate_cost
    from scientific_rag.llm_gateway import validate_max_output_tokens
    limit = validate_max_output_tokens(max_output_tokens) or OUTPUT_TOKEN_CEILING_PER_CALL
    planned = [plan(item) for item in claim_inputs(items)]
    calls = [p for p in planned if p["status"] == "planned"]
    input_tokens = sum(max(1, math.ceil(p["prompt_chars"] / 4)) for p in calls)
    input_reserve = sum(call_token_ceiling(p["prompt_chars"])[0] for p in calls)
    output_reserve = limit * len(calls)
    output_estimate = min(OUTPUT_TOKEN_ESTIMATE_PER_CALL, limit) * len(calls)
    route = route or gateway_route()
    by_route = {}
    for provider in route["provider_order"]:
        model = route["models"][provider]
        estimate, source = calculate_cost(provider, model, input_tokens, output_estimate)
        bound, _ = calculate_cost(provider, model, input_reserve, output_reserve)
        by_route[f"{provider}:{model}"] = {
            "cost_estimate_usd": estimate, "planning_cost_bound_usd": bound,
            "output_limit_enforced": output_limit_status(provider, limit)["enforced"], "rate_source": source}
    bounds = [entry["planning_cost_bound_usd"] for entry in by_route.values()]
    live = live_execution_status(route, limit, len(calls))
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
                                 f"and {min(OUTPUT_TOKEN_ESTIMATE_PER_CALL, limit)} output tokens per call; reservation: "
                                 f"ceil(prompt characters / {INPUT_CHARS_PER_TOKEN_CEILING}) input and "
                                 f"{limit} output tokens per call (the output limit)",
        "route": route, "planning_cost_by_route": by_route,
        "planning_cost_bound_usd_any_route": max(bounds) if bounds and None not in bounds else None,
        "output_limit": live["output_limit"], "live_execution": live,
        "planning_cost_bound_note": (
            f"reserves {limit} output tokens per call"
            + (", which the gateway enforces on the provider request for this route" if live["output_limit"]["enforced"]
               else ", which this route cannot enforce, so live execution is refused")
            + "; a planning bound, not an absolute billing guarantee: input tokens are estimated from characters, "
              "and provider pricing, provider-side token counting and unreported usage are outside its control"),
        "configuration": configuration(),
    }


def check_authorization(authorization: dict, *, adapter: str, dataset_hash: str, route: dict,
                        require_output_limit: bool = False) -> dict:
    """Refuse provider calls unless the authorization is explicit and matches this run.

    Order: authorization shape and budgets, handoff identity, pinned provider and model, known
    price, then (verifier only) an output-token limit the gateway can enforce for that provider."""
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
    if require_output_limit and not _positive_int(authorization.get("max_output_tokens")):
        problems.append("max_output_tokens as a positive integer")
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
    if require_output_limit:
        status = output_limit_status(authorization["provider"], authorization["max_output_tokens"])
        if not status["enforced"]:
            raise AuthorizationError(
                f"live verifier execution refused: provider {authorization['provider']!r} cannot enforce a "
                f"verified output-token limit ({status['source']})")
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
                     ledger: BudgetLedger | None = None,
                     verifier: Callable[..., Awaitable] = verify_claims) -> tuple[list[dict], dict]:
    """Run the production verifier on each claim within the authorized budget. Unscored.

    Gates before any provider call: authorization, handoff identity, pinned provider and model,
    an output limit the gateway can enforce, then a call and cost reservation for each call.
    ``ledger`` lets a caller share one budget across runs and read it after a cancellation."""
    from scientific_rag.llm_gateway import OutputTokenLimitError, begin_trace, end_trace, summarize_trace
    route = route or gateway_route()
    auth = check_authorization(authorization, adapter=ADAPTER, dataset_hash=dataset_hash, route=route,
                               require_output_limit=True)
    limit = auth["max_output_tokens"]
    allowed_model = f"{auth['provider']}:{auth['model']}"
    planned = [plan(item) for item in claim_inputs(items)]
    if sum(p["status"] == "planned" for p in planned) > auth["max_calls"] and not auth.get("allow_partial"):
        raise AuthorizationError("planned calls exceed the authorized max_calls; select a smaller pilot")
    ledger = ledger if ledger is not None else BudgetLedger(auth["max_calls"], auth["max_cost_usd"])
    records, stop_reason = [], None
    for entry in planned:
        item = entry["item"]
        if entry["status"] != "planned":
            records.append(_record(item, "claim_verification", entry["status"], reason=entry["reason"]))
            continue
        reserved = call_cost_ceiling(entry["prompt_chars"], auth["provider"], auth["model"], limit)
        # Reserved before the call and kept if it fails, is cancelled or reports no usage.
        if stop_reason or not ledger.reserve(reserved):
            records.append(_record(item, "claim_verification", "not_executed", reason=stop_reason or CEILING_REASON))
            continue
        token = begin_trace()
        started = time.perf_counter()
        error = None
        try:
            batch = await verifier([entry["input"]], question=REPLAY_QUESTION, max_gateway_invocations=1,
                                   max_output_tokens=limit)
        except Exception as exc:   # gateway, validation or programming error: recorded, never retried
            error = exc
        finally:
            summary = summarize_trace(end_trace(token))
        usage = {**_usage(summary, reserved), "max_output_tokens": limit,
                 "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        ledger.settle(reserved, usage)
        if usage["cost_status"] != "complete":
            stop_reason = USAGE_REASON
        if isinstance(usage["output_tokens"], int) and usage["output_tokens"] > limit * max(1, usage["provider_calls"]):
            usage["output_limit_exceeded"] = True
            stop_reason = "stopped: the provider reported more output tokens than the enforced output limit"
        limit_error = _cause(error, OutputTokenLimitError)
        if limit_error is not None:
            records.append(_record(item, "claim_verification", "failed", usage=usage,
                                   reason=f"output_limit_not_enforced ({type(limit_error).__name__})"))
            stop_reason = "stopped: the output-token limit could not be enforced on the provider request"
            continue
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
    return records, {"authorization_id": auth["authorization_id"], "budget": ledger.snapshot(), "stopped": stop_reason,
                     "output_limit": output_limit_status(auth["provider"], limit)}


def _cause(error: BaseException | None, kind: type) -> BaseException | None:
    visited: set[int] = set()
    while error is not None and id(error) not in visited:
        if isinstance(error, kind):
            return error
        visited.add(id(error))
        error = error.__cause__ or error.__context__
    return None
