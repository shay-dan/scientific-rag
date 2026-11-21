"""Consolidated routing and recovery decisions: WAIT / FALLBACK / SEND / STOP (Stage 20G).

One small, pure policy used by the opt-in consolidated gateway path
(``LLM_ROUTING_POLICY=consolidated``). It decides, for one logical model call:

* before a transmission, from local admission results: send on the preferred route,
  wait a bounded time for a fresh window, fall back to an approved compatible route, or stop;
* after a failed transmission, from how far the request got: whether fallback is allowed.

The policy never re-sends a request to the route that already received it, never routes
outside the approved candidates, and never treats a possibly-transmitted request as unsent.
Budgets (attempts, deadline, request/cost accounting) are consulted *before* each
transmission; the caller reserves every physical request before sending it.

Pure module: no I/O, clock, network or provider client.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Iterable, Sequence

from .provider_failures import (
    AUTHENTICATION_OR_CONFIGURATION_FAILURE, COMPLETION_BUDGET_EXHAUSTED, LOCAL_STRUCTURED_OUTPUT_FAILURE,
    PROVIDER_RATE_LIMIT, PROVIDER_STRUCTURED_GENERATION_FAILURE, TRANSIENT_INFRASTRUCTURE_FAILURE,
)

WAIT, FALLBACK, SEND, STOP = "wait", "fallback", "send", "stop"

# How far a failed attempt got. Every failure maps to exactly one of these.
NOT_TRANSMITTED = "not_transmitted"          # never left this process (local refusal/admission)
DEFINITIVE_RESPONSE = "definitive_response"  # transmitted; a non-2xx HTTP response was received
UNCERTAIN = "uncertain"                      # transmitted (or possibly); outcome unknown
COMPLETED_INVALID = "completed_invalid"      # 2xx response received; local parsing/validation failed
TRANSMISSION_STATES = (NOT_TRANSMITTED, DEFINITIVE_RESPONSE, UNCERTAIN, COMPLETED_INVALID)

# Local admission refusal reasons (provider_throttle) that a later window can satisfy.
WAITABLE_REASONS = frozenset({"tokens", "requests", "cooldown", "concurrency"})
EXCEEDS_WINDOW = "exceeds_window"            # the request can never fit the route's whole window
ADMISSION_UNAVAILABLE = "unavailable"        # admission store unreachable: fail closed

# HTTP statuses/codes meaning "this route cannot take a request this large" (not a model fault).
_CAPACITY_STATUSES = frozenset({413})
_CAPACITY_CODES = frozenset({"context_length_exceeded", "request_too_large"})


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    quota_group: str
    context_tokens: int
    max_output_tokens: int
    structured_output: bool
    bounded_output: bool
    reasoning_efforts: tuple[str, ...] = ()
    priced: bool = True

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class RouteRequest:
    role: str
    estimated_input_tokens: int
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    structured: bool = False


@dataclass(frozen=True)
class Capacity:
    """Local admission result for a route: admitted, or refused with a reason and a wait."""
    admitted: bool
    reason: str | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class PolicyConfig:
    max_wait_seconds: float = 65.0            # cumulative bounded wait per logical call
    prefer_primary_within_seconds: float = 5.0
    max_physical_attempts: int = 2            # transmissions per logical call (primary + one fallback)
    min_attempt_seconds: float = 10.0         # do not start an attempt with less time than this
    fallback_on_invalid_output: bool = False  # a billed but invalid answer is an outcome, not an outage

    def __post_init__(self) -> None:
        if min(self.max_wait_seconds, self.prefer_primary_within_seconds, self.min_attempt_seconds) < 0:
            raise ValueError("routing policy durations cannot be negative")
        if type(self.max_physical_attempts) is not int or self.max_physical_attempts < 1:
            raise ValueError("max_physical_attempts must be a positive integer")


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    route: Route | None = None
    wait_seconds: float = 0.0


@dataclass
class CallState:
    """Per-logical-call memory the policy reads; the gateway updates it."""
    attempts: int = 0
    waited_seconds: float = 0.0
    excluded_routes: set[str] = field(default_factory=set)        # route labels already transmitted to
    exhausted_quota_groups: set[str] = field(default_factory=set)  # e.g. after a provider 429


def incompatibility(route: Route, request: RouteRequest) -> str | None:
    """Why ``route`` cannot carry ``request`` unchanged, or None. Constraints are never dropped."""
    if request.max_output_tokens is not None:
        if not route.bounded_output:
            return "output_limit_unsupported"
        if request.max_output_tokens > route.max_output_tokens:
            return "output_limit_exceeds_route"
    if request.reasoning_effort is not None and request.reasoning_effort not in route.reasoning_efforts:
        return "reasoning_effort_unsupported"
    if request.structured and not route.structured_output:
        return "structured_output_unsupported"
    output = request.max_output_tokens or 0
    if request.estimated_input_tokens + output > route.context_tokens:
        return "context_exceeded"
    if not route.priced:
        return "unpriced_route"
    return None


def viable_routes(routes: Sequence[Route], request: RouteRequest, state: CallState,
                  can_reserve: Callable[[Route], bool]) -> tuple[list[Route], dict[str, str]]:
    """Routes still allowed for this call, in preference order, and why the others are not."""
    viable: list[Route] = []
    rejected: dict[str, str] = {}
    for route in routes:
        reason = incompatibility(route, request)
        if reason is None and route.label in state.excluded_routes:
            reason = "already_attempted"
        if reason is None and route.quota_group in state.exhausted_quota_groups:
            reason = "quota_group_exhausted"
        if reason is None and not can_reserve(route):
            reason = "budget_exhausted"
        if reason is None:
            viable.append(route)
        else:
            rejected[route.label] = reason
    return viable, rejected


def _stop_reason(rejected: dict[str, str]) -> str:
    reasons = set(rejected.values())
    if reasons and reasons <= {"already_attempted", "quota_group_exhausted"}:
        return "no_fallback_route"      # every approved route was tried or shares the exhausted quota
    if reasons == {"budget_exhausted"}:
        return "budget_exhausted"
    if reasons and all(item.endswith(("_unsupported", "_exceeds_route", "_exceeded", "unpriced_route"))
                       for item in reasons):
        return "no_compatible_route"
    return "no_admissible_route"


def _action(route: Route, routes: Sequence[Route]) -> str:
    return SEND if routes and route.label == routes[0].label else FALLBACK


def _waitable(capacity: Capacity, state: CallState, remaining_seconds: float, config: PolicyConfig) -> float | None:
    if capacity.reason not in WAITABLE_REASONS or capacity.retry_after_seconds is None:
        return None
    wait = max(0.0, float(capacity.retry_after_seconds))
    if state.waited_seconds + wait > config.max_wait_seconds:
        return None
    if remaining_seconds - wait < config.min_attempt_seconds:
        return None
    return wait


async def _probe(probe: Callable[[Route], Any], route: Route) -> Capacity:
    result = probe(route)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, Capacity):
        raise TypeError("admission probe must return a Capacity")
    return result


async def decide_admission(routes: Sequence[Route], request: RouteRequest, state: CallState, *,
                     remaining_seconds: float, probe: Callable[[Route], Any],
                     can_reserve: Callable[[Route], bool] = lambda _route: True,
                     config: PolicyConfig = PolicyConfig()) -> Decision:
    """Choose the next transmission, a bounded wait, or a classified stop.

    ``routes`` are the approved candidates in preference order (``routes[0]`` is primary).
    ``probe`` performs one local admission check (sync or async); it is called lazily, so an
    admission is only taken on a route the decision then uses (preferred first, then
    fallbacks in order). The function itself performs no I/O.
    """
    if state.attempts >= config.max_physical_attempts:
        return Decision(STOP, "attempt_limit")
    if remaining_seconds < config.min_attempt_seconds:
        return Decision(STOP, "deadline_exhausted")
    viable, rejected = viable_routes(routes, request, state, can_reserve)
    if not viable:
        return Decision(STOP, _stop_reason(rejected))
    waits: list[tuple[float, Route]] = []
    refusals: list[str] = []
    first = viable[0]
    capacity = await _probe(probe, first)
    if capacity.admitted:
        return Decision(_action(first, routes), "admitted", first)
    refusals.append(capacity.reason or "refused")
    wait = _waitable(capacity, state, remaining_seconds, config)
    if wait is not None:
        if wait <= config.prefer_primary_within_seconds or len(viable) == 1:
            return Decision(WAIT, f"{capacity.reason}_window", first, wait)
        waits.append((wait, first))
    for route in viable[1:]:
        capacity = await _probe(probe, route)
        if capacity.admitted:
            return Decision(FALLBACK, f"primary_{refusals[0]}", route)
        refusals.append(capacity.reason or "refused")
        wait = _waitable(capacity, state, remaining_seconds, config)
        if wait is not None:
            waits.append((wait, route))
    if waits:
        wait, route = min(waits, key=lambda item: item[0])
        return Decision(WAIT, "capacity_window", route, wait)
    if ADMISSION_UNAVAILABLE in refusals:
        return Decision(STOP, "admission_unavailable")
    if all(reason == EXCEEDS_WINDOW for reason in refusals):
        return Decision(STOP, "exceeds_provider_window")
    if any(reason in WAITABLE_REASONS for reason in refusals):
        return Decision(STOP, "capacity_wait_exceeds_budget")
    return Decision(STOP, "no_admissible_route")


@dataclass(frozen=True)
class FailureDisposition:
    may_fallback: bool
    reason: str
    exhaust_quota_group: bool = False


def failure_disposition(transmission: str, failure_class: str | None, *, http_status: int | None = None,
                        provider_error_code: str | None = None,
                        config: PolicyConfig = PolicyConfig()) -> FailureDisposition:
    """Whether a failed attempt may be followed by a fallback attempt on another route.

    The failed route itself is always excluded afterwards (no same-route replay).
    """
    if transmission not in TRANSMISSION_STATES:
        raise ValueError(f"unknown transmission state {transmission!r}")
    if transmission == NOT_TRANSMITTED:
        # A local refusal of an outbound request (contract/limit guard) is a configuration fault.
        return FailureDisposition(False, "local_refusal")
    if transmission == UNCERTAIN:
        # An uncertain outcome is never replayed on any route, even by configuration.
        return FailureDisposition(False, "uncertain_transmission")
    if transmission == COMPLETED_INVALID:
        return FailureDisposition(config.fallback_on_invalid_output, "invalid_output")
    if http_status in _CAPACITY_STATUSES or provider_error_code in _CAPACITY_CODES:
        return FailureDisposition(True, "route_capacity_rejected", exhaust_quota_group=True)
    if failure_class == PROVIDER_RATE_LIMIT:
        return FailureDisposition(True, "provider_rate_limit", exhaust_quota_group=True)
    if failure_class == TRANSIENT_INFRASTRUCTURE_FAILURE:
        return FailureDisposition(True, "provider_unavailable")
    if failure_class in (PROVIDER_STRUCTURED_GENERATION_FAILURE, COMPLETION_BUDGET_EXHAUSTED,
                         LOCAL_STRUCTURED_OUTPUT_FAILURE):
        return FailureDisposition(config.fallback_on_invalid_output, "invalid_output")
    if failure_class == AUTHENTICATION_OR_CONFIGURATION_FAILURE:
        return FailureDisposition(False, "configuration_rejected")
    return FailureDisposition(False, "unclassified_provider_failure")


def config_from_environment(environ: dict[str, str]) -> PolicyConfig:
    def number(name: str, default: float) -> float:
        value = environ.get(name)
        return default if value in (None, "") else float(value)

    def flag(name: str) -> bool:
        return environ.get(name, "").strip().lower() in ("1", "true", "yes")

    return PolicyConfig(
        max_wait_seconds=number("LLM_ROUTING_MAX_WAIT_SECONDS", 65.0),
        prefer_primary_within_seconds=number("LLM_ROUTING_PREFER_PRIMARY_WITHIN_SECONDS", 5.0),
        max_physical_attempts=int(number("LLM_ROUTING_MAX_PHYSICAL_ATTEMPTS", 2)),
        min_attempt_seconds=number("LLM_ROUTING_MIN_ATTEMPT_SECONDS", 10.0),
        fallback_on_invalid_output=flag("LLM_ROUTING_FALLBACK_ON_INVALID_OUTPUT"),
    )


def parse_route_allowlist(value: str | None) -> tuple[str, ...]:
    """``provider:model`` labels, comma separated. Empty means no route is approved."""
    labels = tuple(item.strip() for item in (value or "").split(",") if item.strip())
    for label in labels:
        provider, _, model = label.partition(":")
        if not provider or not model:
            raise ValueError(f"routing allowlist entries must be provider:model, got {label!r}")
    if len(set(labels)) != len(labels):
        raise ValueError("routing allowlist has duplicate routes")
    return labels


def ordered_candidates(allowlist: Iterable[str], declared: dict[str, Route]) -> list[Route]:
    """Approved routes in allowlist order; an approved but undeclared route is refused loudly."""
    routes = []
    for label in allowlist:
        if label not in declared:
            raise ValueError(f"approved route {label} has no declared capabilities")
        routes.append(declared[label])
    return routes


__all__ = [
    "ADMISSION_UNAVAILABLE", "COMPLETED_INVALID", "Capacity", "CallState", "DEFINITIVE_RESPONSE", "Decision",
    "EXCEEDS_WINDOW", "FALLBACK", "FailureDisposition", "NOT_TRANSMITTED", "PolicyConfig", "Route",
    "RouteRequest", "SEND", "STOP", "TRANSMISSION_STATES", "UNCERTAIN", "WAIT", "WAITABLE_REASONS",
    "config_from_environment", "decide_admission", "failure_disposition", "incompatibility",
    "ordered_candidates", "parse_route_allowlist", "viable_routes",
]
