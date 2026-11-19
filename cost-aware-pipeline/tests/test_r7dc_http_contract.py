"""R7D-C: production-like offline HTTP contract for the bounded structured-output route.

The real gateway, ``ChatOpenAI`` and OpenAI SDK serialize every request; ``httpx.MockTransport``
stands in for Groq through the gateway's ``_outbound_transport`` hook. Zero provider calls, no
credential (a placeholder key). Requires the production dependencies (Docker/CI suite).

Includes the observed D002 failure: HTTP 400 ``json_validate_failed`` with an explicit
completion-budget message and a valid-but-incomplete ``failed_generation``.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from app.benchmark.generation_evaluation import canonical_sha256  # noqa: E402
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch  # noqa: E402
from scientific_rag import llm_gateway  # noqa: E402

KEY = "gsk_" + "placeholder0credential" * 2
PRIVATE_TEXT = "PRIVATE-EVIDENCE passage about a 12% response rate"
TRUNCATION = ("max completion tokens reached before generating a valid document: the output was truncated "
              "to fit max_completion_tokens and is missing required content.")
FAILED_GENERATION = json.dumps({"claims": [{"text": PRIVATE_TEXT, "source_id": "S1", "evidence_quote": PRIVATE_TEXT}]})
ENV = {"GROQ_API_KEY": KEY, "OPENROUTER_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": "",
       "LLM_PROVIDER_ORDER_EXPERT": "groq", "GROQ_MODEL_EXPERT": "openai/gpt-oss-120b"}
ANSWER = {"claims": [{"text": "The response rate was 12%.", "source_id": "S1",
                      "evidence_quote": "The response rate was 12% in the treated group."}],
          "limitations": [], "unanswered_parts": []}
# Pinned at the R7D-C base commit (796e0d2): the provider-facing schemas must not change.
STRUCTURED_ANSWER_SCHEMA_SHA256 = "194e586d95e1f6a0f07310ef27446ed5449370334d57769c7b42cb85c993f471"
REPAIR_BATCH_SCHEMA_SHA256 = "83bc2fa3e090ce798dec8b1c02f6b2e44ce3dbf5950d116774ee2685923219a4"


def completion(content: dict, *, usage=None, finish_reason="stop") -> dict:
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": usage if usage is not None else {
                "prompt_tokens": 1116, "completion_tokens": 1420, "total_tokens": 2536,
                "completion_tokens_details": {"reasoning_tokens": 610}}}


class Wire:
    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else httpx.Response(200, json=completion(ANSWER))

    def transport(self):
        return httpx.MockTransport(self.handler)

    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def invoke(wire: Wire, schema=StructuredEvidenceAnswerV2, *, capture=True, env=None, **kwargs):
    async def go():
        token = llm_gateway.begin_trace(capture_rate_limits=capture)
        error = result = None
        try:
            result = await llm_gateway.ainvoke_structured("expert", 0, "frozen prompt", schema, **kwargs)
        except Exception as caught:   # noqa: BLE001 - the contract inspects the raised error
            error = caught
        finally:
            attempts = llm_gateway.end_trace(token)
        return result, error, attempts
    with patch.dict(os.environ, {**ENV, **(env or {})}), \
            patch.object(llm_gateway, "_outbound_transport", wire.transport, create=True):
        return asyncio.run(go())


class SerializedRequestContractTests(unittest.TestCase):
    def test_initial_generation_request_carries_every_authorized_setting(self):
        wire = Wire()
        result, error, attempts = invoke(wire, max_output_tokens=2000, reasoning_effort="low")
        self.assertIsNone(error)
        self.assertIsInstance(result, StructuredEvidenceAnswerV2)
        self.assertEqual(1, len(wire.requests))                        # exactly one outbound request
        request, body = wire.requests[0], wire.bodies()[0]
        self.assertEqual(("POST", "api.groq.com", "/openai/v1/chat/completions"),
                         (request.method, request.url.host, request.url.path))
        self.assertEqual(f"Bearer {KEY}", request.headers["authorization"])
        self.assertEqual("openai/gpt-oss-120b", body["model"])
        self.assertEqual(0, body["temperature"])
        limits = {name: body[name] for name in ("max_completion_tokens", "max_tokens") if name in body}
        self.assertTrue(limits and all(value == 2000 for value in limits.values()), limits)
        self.assertEqual("low", body["reasoning_effort"])
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertFalse(body.get("stream", False))
        self.assertIn(body.get("n"), (None, 1))
        response_format = body["response_format"]
        self.assertEqual("json_schema", response_format["type"])
        self.assertEqual(set(StructuredEvidenceAnswerV2.model_json_schema()["properties"]),
                         set(response_format["json_schema"]["schema"]["properties"]))
        diagnostics = attempts[0]["diagnostics"]
        self.assertEqual(("low", 2000, 610, "stop"),
                         (diagnostics["reasoning_effort"], diagnostics["max_output_tokens"],
                          diagnostics["reasoning_tokens"], diagnostics["finish_reason"]))
        self.assertTrue(diagnostics["usage_complete"])

    def test_repair_request_carries_the_same_settings_and_the_repair_schema(self):
        wire = Wire(httpx.Response(200, json=completion(
            {"repairs": [{"target": "claims.1", "action": "drop", "replacement": None, "unresolved_reason": None}]})))
        result, error, _ = invoke(wire, V2RepairBatch, max_output_tokens=3000, reasoning_effort="low")
        self.assertIsNone(error)
        body = wire.bodies()[0]
        self.assertEqual(("low", "json_schema"), (body["reasoning_effort"], body["response_format"]["type"]))
        self.assertEqual({"repairs"}, set(body["response_format"]["json_schema"]["schema"]["properties"]))
        self.assertTrue(all(body[name] == 3000 for name in ("max_completion_tokens", "max_tokens") if name in body))

    def test_without_reasoning_effort_the_request_is_unchanged(self):
        wire = Wire()
        invoke(wire, max_output_tokens=2000)
        self.assertNotIn("reasoning_effort", wire.bodies()[0])
        # Unbounded production call: the client is built exactly as before, with no reasoning setting.
        wire, built = Wire(), []

        def factory(**kwargs):
            built.append(dict(kwargs))
            return ChatOpenAI(**kwargs, http_async_client=httpx.AsyncClient(transport=wire.transport()))
        with patch.object(llm_gateway, "ChatOpenAI", factory):
            invoke(wire)
        self.assertEqual(1, len(wire.requests))
        self.assertNotIn("reasoning_effort", wire.bodies()[0])
        self.assertEqual({"model", "api_key", "base_url", "timeout", "max_retries", "temperature",
                          "include_response_headers"}, set(built[0]))

    def test_unsupported_reasoning_routes_are_refused_before_any_request(self):
        for env, kwargs in (({"LLM_PROVIDER_ORDER_EXPERT": "openrouter", "OPENROUTER_API_KEY": "k"},
                             {"max_output_tokens": 2000, "reasoning_effort": "low"}),
                            ({"GROQ_MODEL_EXPERT": "llama-3.3-70b-versatile"},
                             {"max_output_tokens": 2000, "reasoning_effort": "low"}),
                            ({}, {"max_output_tokens": 2000, "reasoning_effort": "minimal"}),
                            ({}, {"reasoning_effort": "low"})):      # reasoning needs a verified bounded request
            wire = Wire()
            with self.subTest(env=env, kwargs=kwargs):
                _, error, _ = invoke(wire, env=env, **kwargs)
                self.assertIsInstance(error, llm_gateway.UnsupportedReasoningEffort)
                self.assertEqual([], wire.requests)

    def test_a_dropped_reasoning_effort_is_refused_before_transmission(self):
        wire = Wire()
        dropping = lambda **kwargs: ChatOpenAI(**{k: v for k, v in kwargs.items() if k != "reasoning_effort"})
        with patch.object(llm_gateway, "ChatOpenAI", dropping):
            _, error, _ = invoke(wire, max_output_tokens=2000, reasoning_effort="low")
        self.assertIsInstance(error, llm_gateway.OutputTokenLimitError)
        self.assertEqual([], wire.requests)

    def test_provider_facing_schemas_are_unchanged(self):
        self.assertEqual(STRUCTURED_ANSWER_SCHEMA_SHA256,
                         canonical_sha256(StructuredEvidenceAnswerV2.model_json_schema()))
        self.assertEqual(REPAIR_BATCH_SCHEMA_SHA256, canonical_sha256(V2RepairBatch.model_json_schema()))


class ProviderErrorContractTests(unittest.TestCase):
    def failing(self, status, body=None, *, content=None, headers=None):
        headers = {"x-request-id": "req_01jabcdefghjkmnpqrstvwxyz0", **(headers or {})}
        if content is not None:
            return Wire(httpx.Response(status, content=content, headers=headers))
        return Wire(httpx.Response(status, json=body, headers=headers))

    def test_d002_completion_budget_truncation(self):
        wire = self.failing(400, {"error": {"message": TRUNCATION, "type": "invalid_request_error",
                                            "code": "json_validate_failed", "failed_generation": FAILED_GENERATION}})
        with self.assertLogs("ResearchAgent.LLMGateway", level=logging.INFO) as logs:
            _, error, attempts = invoke(wire, max_output_tokens=2000, reasoning_effort="medium")
        self.assertEqual(1, len(wire.requests))                        # no SDK retry, no fallback
        self.assertIsNotNone(error)
        description = llm_gateway.describe_provider_failure(error, StructuredEvidenceAnswerV2)
        self.assertEqual("completion_budget_exhausted", description["failure_class"])
        self.assertEqual(["limitations", "unanswered_parts"], description["missing_top_level_fields"])
        diagnostics = attempts[0]["diagnostics"]
        self.assertEqual((400, "json_validate_failed", "invalid_request_error", "completion_budget_exhausted",
                          "req_01jabcdefghjkmnpqrstvwxyz0", True, len(FAILED_GENERATION), True, "medium", 2000, False, "confirmed"),
                         (diagnostics["http_status"], diagnostics["provider_error_code"],
                          diagnostics["provider_error_type"], diagnostics["failure_class"],
                          diagnostics["request_id"], diagnostics["failed_generation_returned"],
                          diagnostics["failed_generation_chars"], diagnostics["failed_generation_parses_as_json"],
                          diagnostics["reasoning_effort"], diagnostics["max_output_tokens"],
                          diagnostics["usage_complete"], diagnostics["transmission"]))
        self.assertIsNone(attempts[0]["input_tokens"])                 # no invented usage
        trace_text = json.dumps(attempts) + "\n".join(logs.output)
        for forbidden in (PRIVATE_TEXT, TRUNCATION, KEY):
            self.assertNotIn(forbidden, trace_text)

    def test_error_statuses_are_classified_from_the_real_sdk_errors(self):
        cases = (
            (self.failing(400, content=b""), "unknown_provider_failure"),
            (self.failing(400, {"error": {"message": "'reasoning_effort' is not supported", "type":
                                          "invalid_request_error", "param": "reasoning_effort"}}),
             "authentication_or_configuration_failure"),
            (self.failing(400, {"error": {"message": "Failed to generate JSON.", "type": "invalid_request_error",
                                          "code": "json_validate_failed", "failed_generation": "{"}}),
             "provider_structured_generation_failure"),
            (self.failing(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error",
                                          "code": "invalid_api_key"}}), "authentication_or_configuration_failure"),
            (self.failing(403, {"error": {"message": "Forbidden"}}), "authentication_or_configuration_failure"),
            (self.failing(429, {"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
                          headers={"retry-after": "7", "x-ratelimit-remaining-tokens": "0"}), "provider_rate_limit"),
            (self.failing(500, {"error": {"message": "internal"}}), "transient_infrastructure_failure"),
            (self.failing(503, {"error": {"message": "unavailable"}}), "transient_infrastructure_failure"),
        )
        for wire, expected in cases:
            with self.subTest(expected=expected, status=wire.responses[0].status_code):
                _, error, attempts = invoke(wire, max_output_tokens=2000, reasoning_effort="low")
                self.assertEqual(1, len(wire.requests))
                self.assertEqual(expected, llm_gateway.describe_provider_failure(error)["failure_class"])
                self.assertEqual(expected, attempts[0]["diagnostics"]["failure_class"])

    def test_connection_failures_are_transient_with_uncertain_transmission(self):
        def reset(request):
            raise httpx.ConnectError("connection reset by peer", request=request)
        with patch.dict(os.environ, ENV), \
                patch.object(llm_gateway, "_outbound_transport", lambda: httpx.MockTransport(reset)):
            async def go():
                token = llm_gateway.begin_trace(capture_rate_limits=True)
                try:
                    await llm_gateway.ainvoke_structured("expert", 0, "p", StructuredEvidenceAnswerV2,
                                                         max_output_tokens=2000, reasoning_effort="low")
                except Exception as error:   # noqa: BLE001
                    return error, llm_gateway.end_trace(token)
            error, attempts = asyncio.run(go())
        description = llm_gateway.describe_provider_failure(error)
        self.assertEqual(("transient_infrastructure_failure", "uncertain"),
                         (description["failure_class"], description["transmission"]))
        self.assertEqual(1, len(attempts))

    def test_local_pydantic_failure_after_a_successful_response(self):
        wire = Wire(httpx.Response(200, json=completion({"claims": [], "limitations": [], "unanswered_parts": []})))
        _, error, attempts = invoke(wire, max_output_tokens=2000, reasoning_effort="low")
        self.assertIsInstance(error, llm_gateway.InvalidStructuredResponse)
        self.assertEqual("local_structured_output_failure", attempts[0]["diagnostics"]["failure_class"])
        self.assertEqual((1116, 1420), (attempts[0]["input_tokens"], attempts[0]["output_tokens"]))

    def test_production_traces_keep_their_shape_without_capture(self):
        wire = self.failing(400, {"error": {"message": TRUNCATION, "code": "json_validate_failed",
                                            "failed_generation": FAILED_GENERATION}})
        _, _, attempts = invoke(wire, capture=False, max_output_tokens=2000)
        self.assertNotIn("diagnostics", attempts[0])
        self.assertNotIn("rate_limit_headers", attempts[0])


class AuthenticatedPreflightTests(unittest.TestCase):
    def preflight(self, handler, key=KEY, model="openai/gpt-oss-120b"):
        from app.benchmark.provider_preflight import authenticated_model_preflight
        calls = []

        def recorder(request):
            calls.append(request)
            return handler(request)
        result = asyncio.run(authenticated_model_preflight("groq", model, api_key=key,
                                                           transport=httpx.MockTransport(recorder)))
        self.assertNotIn(key, json.dumps(result)) if key else None
        return result, calls

    def test_authenticated_model_listing(self):
        result, calls = self.preflight(lambda request: httpx.Response(200, json={"object": "list", "data": [
            {"id": "openai/gpt-oss-120b", "object": "model"}, {"id": "openai/gpt-oss-20b", "object": "model"}]}))
        self.assertEqual(1, len(calls))
        self.assertEqual(("GET", "/openai/v1/models"), (calls[0].method, calls[0].url.path))
        self.assertEqual({"credential": "present", "http_status": 200, "authenticated": True, "model_listed": True,
                          "generation_compatibility": "not_tested", "inference_requests": 0},
                         {key: result[key] for key in ("credential", "http_status", "authenticated", "model_listed",
                                                       "generation_compatibility", "inference_requests")})

    def test_unlisted_model_and_rejected_credentials(self):
        result, _ = self.preflight(lambda request: httpx.Response(200, json={"data": [{"id": "other"}]}))
        self.assertEqual((True, False), (result["authenticated"], result["model_listed"]))
        for status in (401, 403):
            result, _ = self.preflight(lambda request, status=status: httpx.Response(status, json={}))
            self.assertEqual((False, status), (result["authenticated"], result["http_status"]))

    def test_missing_or_implausible_credentials_send_nothing(self):
        for key in ("", "x", "has space inside the credential value", "line\nbreak" * 4):
            with self.subTest(length=len(key)):
                result, calls = self.preflight(lambda request: httpx.Response(200, json={"data": []}), key=key)
                self.assertEqual([], calls)
                self.assertIn(result["credential"], {"missing", "implausible"})
        # No fixed length: a longer future key format is accepted as plausible.
        result, calls = self.preflight(lambda request: httpx.Response(200, json={"data": []}),
                                       key="gsk_" + "a" * 120)
        self.assertEqual(("present", 1), (result["credential"], len(calls)))


if __name__ == "__main__":
    unittest.main()
