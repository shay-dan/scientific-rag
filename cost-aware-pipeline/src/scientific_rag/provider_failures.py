"""Typed, sanitized classification of provider failures (R7D-C).

``describe_provider_failure`` turns a raised provider/gateway error into a small dictionary of
structured facts: HTTP status, the provider's error code and type, a request identity, a
failure class, transmission certainty and safe facts about any returned ``failed_generation``
(length, whether it parses as JSON, and which top-level schema fields it lacks).

It never returns message text, error bodies, prompts, generated text or credentials. The only
text it inspects is the provider's error ``message``, and only to confirm one narrow case:
explicit completion-budget exhaustion reported with ``json_validate_failed``. Other
``json_validate_failed`` responses and other HTTP 400s are not treated as truncation.

Pure module: no network, no provider client, no dependency beyond the standard library.
"""
from __future__ import annotations

import json
import re
from typing import Any

COMPLETION_BUDGET_EXHAUSTED = "completion_budget_exhausted"
PROVIDER_STRUCTURED_GENERATION_FAILURE = "provider_structured_generation_failure"
LOCAL_STRUCTURED_OUTPUT_FAILURE = "local_structured_output_failure"
AUTHENTICATION_OR_CONFIGURATION_FAILURE = "authentication_or_configuration_failure"
PROVIDER_RATE_LIMIT = "provider_rate_limit"
TRANSIENT_INFRASTRUCTURE_FAILURE = "transient_infrastructure_failure"
UNKNOWN_PROVIDER_FAILURE = "unknown_provider_failure"
# Harness-level classes (no provider error object): see the R7D decision table.
LOCAL_EVIDENCE_VALIDATION_FAILURE = "local_evidence_validation_failure"
PRE_TRANSMISSION_CAPACITY_PAUSE = "pre_transmission_capacity_pause"

FAILURE_CLASSES = (
    COMPLETION_BUDGET_EXHAUSTED,
    PROVIDER_STRUCTURED_GENERATION_FAILURE,
    LOCAL_STRUCTURED_OUTPUT_FAILURE,
    AUTHENTICATION_OR_CONFIGURATION_FAILURE,
    PROVIDER_RATE_LIMIT,
    TRANSIENT_INFRASTRUCTURE_FAILURE,
    UNKNOWN_PROVIDER_FAILURE,
)

TRANSMISSION_CONFIRMED = "confirmed"
TRANSMISSION_UNCERTAIN = "uncertain"
TRANSMISSION_NOT_SENT = "not_transmitted"

JSON_VALIDATE_FAILED = "json_validate_failed"
# Narrow: the provider's own wording for an exhausted completion budget (observed on Groq,
# 2026-11, D002). Anything else with json_validate_failed is a structured-generation failure.
_COMPLETION_BUDGET_MESSAGE = re.compile(
    r"max(?:imum)?[ _]completion[ _]tokens reached|truncated to fit max_completion_tokens", re.IGNORECASE)
_LOCAL_OUTPUT_NAMES = ("InvalidStructured", "ValidationError", "OutputParserException")
_NOT_TRANSMITTED_NAMES = ("OutputTokenLimit", "UnsupportedOutputTokenLimit", "ReasoningEffort")
_TRANSIENT_NAMES = ("Timeout", "ConnectError", "ConnectionError", "APIConnectionError", "ReadError",
                    "RemoteProtocolError", "WriteError", "PoolTimeout")
# R7D-C review P1.3: identifier-shaped text is not necessarily safe (a credential or private
# text can look like a code). Only known provider identifiers are kept; anything else becomes
# "other". Request ids are kept only in the provider's ``req_`` + ULID-like form.
KNOWN_ERROR_CODES = frozenset({
    "json_validate_failed", "invalid_api_key", "rate_limit_exceeded", "model_not_found", "model_decommissioned",
    "context_length_exceeded", "request_too_large", "insufficient_quota", "invalid_request_error",
    "unknown_url", "permission_denied", "organization_restricted", "service_unavailable",
    "internal_server_error", "bad_request", "not_found",
})
KNOWN_ERROR_TYPES = frozenset({
    "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
    "rate_limit_error", "tokens", "requests", "api_error", "server_error", "service_unavailable",
    "internal_server_error", "overloaded_error",
})
_REQUEST_ID = re.compile(r"^req_[0-9a-z]{16,40}$")


def _chain(error: BaseException):
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _names(error: BaseException) -> list[str]:
    return [cls.__name__ for cls in type(error).__mro__]


def _status_error(error: BaseException) -> BaseException | None:
    for item in _chain(error):
        status = getattr(item, "status_code", None)
        if status is None:
            status = getattr(getattr(item, "response", None), "status_code", None)
        if type(status) is int and status >= 400:   # a successful response is not a provider error
            return item
    return None


def _status(item: BaseException) -> int:
    status = getattr(item, "status_code", None)
    return status if type(status) is int else getattr(item.response, "status_code")


def _error_body(item: BaseException) -> dict:
    body = getattr(item, "body", None)
    if isinstance(body, dict):
        return body.get("error") if isinstance(body.get("error"), dict) else body
    return {}


def _known(value: Any, allowed: frozenset) -> str | None:
    """A known provider identifier, ``other`` for any other present value, or None."""
    if value is None or value == "":
        return None
    return value if isinstance(value, str) and value in allowed else "other"


def _request_id(item: BaseException) -> str | None:
    value = getattr(item, "request_id", None)
    if not isinstance(value, str):
        headers = getattr(getattr(item, "response", None), "headers", None) or {}
        try:
            value = headers.get("x-request-id")
        except AttributeError:
            value = None
    return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None


def _expected_fields(schema: Any) -> list[str] | None:
    try:
        properties = schema.model_json_schema().get("properties") if schema is not None else None
    except Exception:   # noqa: BLE001 - diagnostics must never raise
        return None
    return sorted(properties) if isinstance(properties, dict) else None


def _failed_generation_facts(body: dict, schema: Any) -> dict:
    generation = body.get("failed_generation")
    facts = {"failed_generation_returned": isinstance(generation, str) and bool(generation),
             "failed_generation_chars": len(generation) if isinstance(generation, str) else None,
             "failed_generation_parses_as_json": None, "missing_top_level_fields": None}
    if not isinstance(generation, str) or not generation:
        return facts
    try:
        parsed = json.loads(generation)
    except ValueError:
        facts["failed_generation_parses_as_json"] = False
        return facts
    facts["failed_generation_parses_as_json"] = True
    expected = _expected_fields(schema)
    if isinstance(parsed, dict) and expected is not None:
        # Field names come from our own schema, never from the generated text.
        facts["missing_top_level_fields"] = [name for name in expected if name not in parsed]
    return facts


def describe_provider_failure(error: BaseException, response_schema: Any = None) -> dict:
    """Classify ``error`` into one failure class with sanitized, structured facts."""
    description = {
        "failure_class": UNKNOWN_PROVIDER_FAILURE, "exception_type": type(error).__name__,
        "http_status": None, "provider_error_code": None, "provider_error_type": None, "request_id": None,
        "failed_generation_returned": False, "failed_generation_chars": None,
        "failed_generation_parses_as_json": None, "missing_top_level_fields": None,
        "transmission": TRANSMISSION_UNCERTAIN,
    }
    chain = list(_chain(error))
    names = [name for item in chain for name in _names(item)]
    if any(marker in name for name in names for marker in _NOT_TRANSMITTED_NAMES):
        sent = "BoundedRequestLimitExceeded" in names   # a first request had already been sent
        description.update(failure_class=AUTHENTICATION_OR_CONFIGURATION_FAILURE,
                           transmission=TRANSMISSION_CONFIRMED if sent else TRANSMISSION_NOT_SENT)
        return description
    status_error = _status_error(error)
    if status_error is None:
        if any(marker in name for name in names for marker in _LOCAL_OUTPUT_NAMES):
            description.update(failure_class=LOCAL_STRUCTURED_OUTPUT_FAILURE, transmission=TRANSMISSION_CONFIRMED)
        elif any(marker in name for name in names for marker in _TRANSIENT_NAMES) \
                or any(isinstance(item, (TimeoutError, ConnectionError)) for item in chain):
            description["failure_class"] = TRANSIENT_INFRASTRUCTURE_FAILURE
        return description
    status = _status(status_error)
    body = _error_body(status_error)
    raw_code = body.get("code") if body.get("code") is not None else getattr(status_error, "code", None)
    raw_kind = body.get("type") if body.get("type") is not None else getattr(status_error, "type", None)
    code = _known(raw_code, KNOWN_ERROR_CODES)
    kind = _known(raw_kind, KNOWN_ERROR_TYPES)
    description.update(http_status=status, provider_error_code=code, provider_error_type=kind,
                       request_id=_request_id(status_error), transmission=TRANSMISSION_CONFIRMED)
    description.update(_failed_generation_facts(body, response_schema))
    message = body.get("message") if isinstance(body.get("message"), str) else ""
    param = body.get("param")
    if status == 429:
        failure_class = PROVIDER_RATE_LIMIT
    elif status == 408 or 500 <= status <= 599:
        failure_class = TRANSIENT_INFRASTRUCTURE_FAILURE
    elif status in (401, 403, 404, 405, 413, 415, 422):
        failure_class = AUTHENTICATION_OR_CONFIGURATION_FAILURE
    elif status == 400 and code == JSON_VALIDATE_FAILED:
        failure_class = (COMPLETION_BUDGET_EXHAUSTED if _COMPLETION_BUDGET_MESSAGE.search(message)
                         else PROVIDER_STRUCTURED_GENERATION_FAILURE)
    elif status == 400 and (param is not None or code is not None):
        failure_class = AUTHENTICATION_OR_CONFIGURATION_FAILURE   # unsupported parameter/schema/value
    else:
        failure_class = UNKNOWN_PROVIDER_FAILURE                  # e.g. HTTP 400 with no JSON body
    description["failure_class"] = failure_class
    return description


__all__ = [
    "AUTHENTICATION_OR_CONFIGURATION_FAILURE", "COMPLETION_BUDGET_EXHAUSTED", "FAILURE_CLASSES",
    "JSON_VALIDATE_FAILED", "LOCAL_EVIDENCE_VALIDATION_FAILURE", "LOCAL_STRUCTURED_OUTPUT_FAILURE",
    "PRE_TRANSMISSION_CAPACITY_PAUSE", "PROVIDER_RATE_LIMIT", "PROVIDER_STRUCTURED_GENERATION_FAILURE",
    "TRANSIENT_INFRASTRUCTURE_FAILURE", "TRANSMISSION_CONFIRMED", "TRANSMISSION_NOT_SENT",
    "TRANSMISSION_UNCERTAIN", "UNKNOWN_PROVIDER_FAILURE", "describe_provider_failure",
]
