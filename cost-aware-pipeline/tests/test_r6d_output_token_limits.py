"""R6D: optional, provider-enforced output-token limits (tests written before the change).

No live or paid request: provider HTTP is served by ``httpx.MockTransport`` and the R6C
adapter tests use a fake gateway. The outbound tests read the JSON body that the real
``ChatOpenAI`` client would send to Groq/OpenRouter.

Covered: limit validation; the limit reaching the outbound request (and being absent when
not requested); unsupported providers failing closed; bounded fallback staying bounded; a
dropped or unverifiable limit refusing the request; unchanged default construction; the
production verifier pass-through; R6C budget reservations for success, failure, partial
usage, repeated failure, cancellation, concurrency and refusals; export compatibility.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from app import claim_verification  # noqa: E402
from app.benchmark import production_verifier_replay as verifier  # noqa: E402
from app.benchmark.prediction_export import build_export, verify_export  # noqa: E402
from app.claim_verification import ClaimAssessment, ClaimVerificationBatch, ClaimVerificationInput  # noqa: E402
from scientific_rag import llm_gateway  # noqa: E402
from tests.test_r6c_production_adapters import FC1, FC2, ROUTE, FakeGateway, authorization, fixed_items  # noqa: E402

LIMIT = 64
KEYS = ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER_ORDER",
        "LLM_PROVIDER_ORDER_GATEKEEPER")
LIMIT_NAMES = ("max_completion_tokens", "max_tokens")
HOSTS = {"groq": "api.groq.com", "openrouter": "openrouter.ai"}
INVALID_LIMITS = (0, -1, 1.5, 2.0, True, False, float("nan"), float("inf"), "64", [64])


def provider_env(order: str, **keys) -> dict:
    env = {key: "" for key in KEYS}
    env["LLM_PROVIDER_ORDER_GATEKEEPER"] = order
    env.update(keys)
    return env


class Wire:
    """Mock provider transport that records every outbound request body."""

    def __init__(self, fail_hosts=()):
        self.requests, self.fail_hosts = [], set(fail_hosts)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.host, json.loads(request.content)))
        if request.url.host in self.fail_hosts:
            return httpx.Response(503, json={"error": {"message": "service unavailable"}})
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})

    def transport(self):
        return httpx.MockTransport(self.handler)

    def limits(self):
        return [{name: body[name] for name in LIMIT_NAMES if name in body} for _, body in self.requests]


def bounded(env, wire, **kwargs):
    async def go():
        token = llm_gateway.begin_trace()
        try:
            result = await llm_gateway.ainvoke("gatekeeper", 0, lambda model: model, "hello", **kwargs)
        finally:
            attempts = llm_gateway.end_trace(token)
        return result, attempts
    with patch.dict(os.environ, env), patch.object(llm_gateway, "_outbound_transport", wire.transport, create=True):
        return asyncio.run(go())


class LimitValidationTests(unittest.TestCase):
    def test_valid_and_missing_limits(self):
        self.assertIsNone(llm_gateway.validate_max_output_tokens(None))
        for value in (1, 64, 2000):
            self.assertEqual(llm_gateway.validate_max_output_tokens(value), value)

    def test_invalid_limits_are_rejected(self):
        for value in INVALID_LIMITS:
            with self.subTest(value=value), self.assertRaises(llm_gateway.InvalidOutputTokenLimit):
                llm_gateway.validate_max_output_tokens(value)

    def test_invalid_limit_refuses_before_any_model_is_built(self):
        env = provider_env("groq", GROQ_API_KEY="k")
        for value in INVALID_LIMITS:
            wire = Wire()
            with self.subTest(value=value), self.assertRaises(llm_gateway.InvalidOutputTokenLimit):
                bounded(env, wire, max_output_tokens=value)
            self.assertEqual(wire.requests, [])

    def test_support_table(self):
        support = llm_gateway.output_token_limit_support()
        self.assertEqual(set(support), {"groq", "openrouter", "gemini"})
        self.assertTrue(support["groq"] and support["openrouter"])
        self.assertIsNone(support["gemini"])


class OutboundRequestTests(unittest.TestCase):
    def test_limit_reaches_the_outbound_request_for_each_supported_provider(self):
        for provider, key in (("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")):
            wire = Wire()
            with self.subTest(provider=provider):
                result, attempts = bounded(provider_env(provider, **{key: "k"}), wire, max_output_tokens=LIMIT)
                self.assertEqual(result.content, "ok")
                self.assertEqual([host for host, _ in wire.requests], [HOSTS[provider]])
                sent = wire.limits()[0]
                self.assertTrue(sent, "no output-token limit in the outbound request")
                self.assertTrue(all(value == LIMIT for value in sent.values()), sent)
                self.assertTrue(set(sent) <= set(llm_gateway.output_token_limit_support()[provider]))
                self.assertEqual((attempts[0]["input_tokens"], attempts[0]["output_tokens"]), (11, 3))
                print(f"R6D outbound limit parameter for {provider}: {sorted(sent)}")

    def test_limit_is_absent_when_not_requested(self):
        """Default requests carry no limit, and the default client construction is used."""
        for provider, key in (("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")):
            wire, built = Wire(), []

            def factory(**kwargs):
                built.append(dict(kwargs))
                return ChatOpenAI(**kwargs, http_async_client=httpx.AsyncClient(transport=wire.transport()))
            with self.subTest(provider=provider), patch.dict(os.environ, provider_env(provider, **{key: "k"})), \
                    patch.object(llm_gateway, "ChatOpenAI", factory):
                result = asyncio.run(llm_gateway.ainvoke("gatekeeper", 0, lambda model: model, "hello"))
                self.assertEqual(result.content, "ok")
                self.assertEqual(wire.limits(), [{}])
                expected = {"model", "api_key", "base_url", "timeout", "max_retries", "temperature"}
                self.assertEqual(set(built[0]), expected | ({"default_headers"} if provider == "openrouter" else set()))
                self.assertEqual(built[0]["max_retries"], 0)

    def test_gemini_default_construction_is_unchanged(self):
        built = []
        with patch.dict(os.environ, provider_env("gemini", GOOGLE_API_KEY="k")), \
                patch.object(llm_gateway, "ChatGoogleGenerativeAI", lambda **kwargs: built.append(kwargs) or "model"):
            self.assertEqual(llm_gateway.create_chat_model("gemini", "gatekeeper", 0), "model")
        self.assertFalse({"max_output_tokens", "max_tokens"} & set(built[0]))
        self.assertTrue({"model", "google_api_key", "timeout", "max_retries"} <= set(built[0]))

    def test_unsupported_provider_fails_closed_without_a_request(self):
        for order, keys in (("gemini", {"GOOGLE_API_KEY": "k"}),
                            ("groq,gemini", {"GROQ_API_KEY": "k", "GOOGLE_API_KEY": "k"})):
            wire = Wire()
            with self.subTest(order=order), self.assertRaises(llm_gateway.UnsupportedOutputTokenLimit):
                bounded(provider_env(order, **keys), wire, max_output_tokens=LIMIT)
            self.assertEqual(wire.requests, [])
        with patch.dict(os.environ, provider_env("gemini", GOOGLE_API_KEY="k")), \
                self.assertRaises(llm_gateway.UnsupportedOutputTokenLimit):
            llm_gateway.create_chat_model("gemini", "gatekeeper", 0, max_output_tokens=LIMIT)

    def test_bounded_fallback_stays_bounded(self):
        wire = Wire(fail_hosts={HOSTS["groq"]})
        env = provider_env("groq,openrouter", GROQ_API_KEY="k", OPENROUTER_API_KEY="k")
        result, attempts = bounded(env, wire, max_output_tokens=LIMIT)
        self.assertEqual(result.content, "ok")
        self.assertEqual([host for host, _ in wire.requests], [HOSTS["groq"], HOSTS["openrouter"]])
        for sent in wire.limits():
            self.assertTrue(sent and all(value == LIMIT for value in sent.values()), sent)
        self.assertEqual([attempt["success"] for attempt in attempts], [False, True])

    def test_failed_bounded_request_is_not_retried_by_the_sdk(self):
        wire = Wire(fail_hosts={HOSTS["groq"]})
        with self.assertRaises(Exception):
            bounded(provider_env("groq", GROQ_API_KEY="k"), wire, max_output_tokens=LIMIT)
        self.assertEqual(len(wire.requests), 1)
        self.assertTrue(wire.limits()[0])

    def test_a_dropped_limit_is_refused_before_the_request_is_sent(self):
        """If the client would send no limit, the gateway guard blocks the request."""
        wire = Wire()
        dropping = lambda **kwargs: ChatOpenAI(**{k: v for k, v in kwargs.items() if k != "max_tokens"})
        with patch.object(llm_gateway, "ChatOpenAI", dropping), \
                self.assertRaises(llm_gateway.OutputTokenLimitNotEnforced):
            bounded(provider_env("groq,openrouter", GROQ_API_KEY="k", OPENROUTER_API_KEY="k"), wire,
                    max_output_tokens=LIMIT)
        self.assertEqual(wire.requests, [])   # nothing sent, and no fallback attempt

    def test_a_response_without_a_verified_request_is_refused(self):
        class NoHttpChain:
            async def ainvoke(self, payload, config=None):
                return "answer produced without a verified outbound request"
        wire = Wire()
        with patch.dict(os.environ, provider_env("groq", GROQ_API_KEY="k")), \
                patch.object(llm_gateway, "_outbound_transport", wire.transport, create=True), \
                self.assertRaises(llm_gateway.OutputTokenLimitNotEnforced):
            asyncio.run(llm_gateway.ainvoke("gatekeeper", 0, lambda model: NoHttpChain(), "hello",
                                            max_output_tokens=LIMIT))

    def test_structured_requests_carry_the_limit(self):
        wire = Wire()

        async def go():
            return await llm_gateway.ainvoke_structured("gatekeeper", 0, "prompt", ClaimVerificationBatch,
                                                        max_output_tokens=LIMIT)
        with patch.dict(os.environ, provider_env("groq", GROQ_API_KEY="k")), \
                patch.object(llm_gateway, "_outbound_transport", wire.transport, create=True):
            try:
                asyncio.run(go())
            except llm_gateway.OutputTokenLimitError:
                raise
            except Exception:
                pass   # the mock reply is not a structured answer; only the outbound request matters here
        self.assertEqual(len(wire.requests), 1)
        self.assertTrue(wire.limits()[0] and all(v == LIMIT for v in wire.limits()[0].values()))


class SelfReviewRegressionTests(unittest.TestCase):
    """Defects found in the adversarial self-review of the guard."""

    def guard(self):
        return llm_gateway._OutputLimitGuard("groq", LIMIT)

    def test_guard_accepts_only_the_exact_limit(self):
        for body in ({"max_completion_tokens": LIMIT}, {"max_tokens": LIMIT},
                     {"max_tokens": LIMIT, "max_completion_tokens": LIMIT, "n": 1}):
            guard = self.guard()
            guard.check(json.dumps(body).encode())
            self.assertEqual(guard.verified_requests, 1)
        for body in ({}, {"max_tokens": None}, {"max_tokens": LIMIT + 1}, {"max_tokens": float(LIMIT)},
                     {"max_tokens": True}, {"max_tokens": LIMIT, "max_completion_tokens": LIMIT * 10},
                     {"max_tokens": str(LIMIT)}, [LIMIT]):
            guard = self.guard()
            with self.subTest(body=body), self.assertRaises(llm_gateway.OutputTokenLimitNotEnforced):
                guard.check(json.dumps(body).encode())
            self.assertEqual(guard.verified_requests, 0)
        with self.assertRaises(llm_gateway.OutputTokenLimitNotEnforced):
            self.guard().check(b"not json")

    def test_multiple_completions_are_refused(self):
        with self.assertRaises(llm_gateway.OutputTokenLimitNotEnforced):
            self.guard().check(json.dumps({"max_tokens": LIMIT, "n": 3}).encode())

    def test_bounded_clients_do_not_follow_redirects_or_retry(self):
        with patch.dict(os.environ, provider_env("groq", GROQ_API_KEY="k")):
            model = llm_gateway.create_chat_model("groq", "gatekeeper", 0, max_output_tokens=LIMIT)
        self.assertFalse(model.http_async_client.follow_redirects)
        self.assertFalse(model.http_client.follow_redirects)
        self.assertEqual(model.max_retries, 0)
        asyncio.run(llm_gateway._close_bounded_clients(model))

    def test_redirect_is_not_followed_as_a_second_request(self):
        calls = []

        def redirecting(request):
            calls.append(str(request.url))
            return httpx.Response(307, headers={"location": "https://elsewhere.invalid/v1/chat/completions"})
        with patch.dict(os.environ, provider_env("groq", GROQ_API_KEY="k")), \
                patch.object(llm_gateway, "_outbound_transport", lambda: httpx.MockTransport(redirecting)), \
                self.assertRaises(Exception):
            asyncio.run(llm_gateway.ainvoke("gatekeeper", 0, lambda model: model, "hello", max_output_tokens=LIMIT))
        self.assertEqual(len(calls), 1)

    def test_limit_errors_and_logs_carry_no_key_or_prompt(self):
        wire = Wire()
        dropping = lambda **kwargs: ChatOpenAI(**{k: v for k, v in kwargs.items() if k != "max_tokens"})
        with patch.object(llm_gateway, "ChatOpenAI", dropping), self.assertLogs("ResearchAgent.LLMGateway", "INFO") as logs:
            try:
                bounded(provider_env("groq", GROQ_API_KEY="secret-key-value"), wire, max_output_tokens=LIMIT)
            except llm_gateway.OutputTokenLimitNotEnforced as error:
                text = str(error) + " ".join(logs.output)
        self.assertNotIn("secret-key-value", text)
        self.assertNotIn("hello", text)


class OfflineCliDryRunTests(unittest.TestCase):
    def test_offline_dry_run_reports_eligibility_without_any_call(self):
        """A pinned, supported, priced route is eligible for live execution after authorization."""
        import subprocess
        import tempfile
        from tests.test_benchmark_r6b import make_handoff
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = make_handoff(tmp)
            env = {**os.environ, **provider_env("groq")}   # no API key configured at all
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/benchmark_r6c.py"), "verifier-dry-run", str(archive), "--group",
                 "eval_inputs", "--out", str(tmp / "plan.json"), "--max-output-tokens", "2000"],
                capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((tmp / "plan.json").read_text())
        self.assertEqual(plan["provider_calls_made"], 0)
        live = plan["live_execution"]
        self.assertEqual((live["eligible_after_authorization"], live["blockers"]), (True, []))
        self.assertEqual(live["output_limit"]["limit"], 2000)
        print("R6D offline dry run:", json.dumps({"planned_calls": plan["planned_calls"], "route": plan["route"],
                                                   "planning_cost_bound_usd": plan["planning_cost_bound_usd_any_route"],
                                                   "live_execution": live}, sort_keys=True))


class ProductionVerifierPassThroughTests(unittest.TestCase):
    ITEM = ClaimVerificationInput(target="claims.1", claim_text="c", source_id="S1", evidence_quote="e",
                                  matched_source_title="t", matched_passage_context="e")

    def call(self, **kwargs):
        reply = ClaimVerificationBatch(assessments=[ClaimAssessment(target="claims.1", verdict="supported", reason="r")])
        gateway = AsyncMock(return_value=reply)
        with patch.object(claim_verification, "ainvoke_structured", gateway):
            batch = asyncio.run(claim_verification.verify_claims([self.ITEM], question="q", **kwargs))
        return batch, gateway.call_args.kwargs

    def test_default_call_is_unchanged(self):
        batch, kwargs = self.call()
        self.assertEqual(set(kwargs), {"role", "temperature", "prompt", "response_schema"})
        self.assertEqual(batch.assessments[0].verdict.value, "supported")

    def test_limit_is_passed_through_when_given(self):
        batch, kwargs = self.call(max_output_tokens=LIMIT)
        self.assertEqual(kwargs["max_output_tokens"], LIMIT)
        self.assertEqual(set(kwargs), {"role", "temperature", "prompt", "response_schema", "max_output_tokens"})
        self.assertEqual(batch.model_dump(), self.call()[0].model_dump())


class BoundedGateway(FakeGateway):
    def __init__(self, *args, cancel_on=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.limits, self.cancel_on = [], set(cancel_on)

    async def __call__(self, *, role, temperature, prompt, response_schema, max_output_tokens=None):
        self.limits.append(max_output_tokens)
        if len(self.limits) in self.cancel_on:
            raise asyncio.CancelledError()
        return await super().__call__(role=role, temperature=temperature, prompt=prompt, response_schema=response_schema)


def replay(items, report, gateway, ledger=None, route=ROUTE, **auth):
    auth.setdefault("max_output_tokens", 2000)
    with patch.object(claim_verification, "ainvoke_structured", gateway):
        return asyncio.run(verifier.run_replay(items, authorization=authorization(report, **auth),
                                               dataset_hash=report["handoff"]["dataset_hash"], route=route,
                                               ledger=ledger))


class VerifierReplayBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixed, self.report = fixed_items()
        self.prompt_chars = [verifier.plan(i)["prompt_chars"] for i in verifier.claim_inputs(self.fixed)]

    def reserved(self, index=0, limit=2000):
        return verifier.call_cost_ceiling(self.prompt_chars[index], "groq", "openai/gpt-oss-120b", limit)

    def test_successful_bounded_requests(self):
        gateway = BoundedGateway()
        records, run = replay(self.fixed, self.report, gateway)
        self.assertEqual(gateway.limits, [2000, 2000])
        self.assertEqual([r["status"] for r in records], ["completed", "completed"])
        usage = records[0]["usage"]
        self.assertEqual((usage["cost_status"], usage["max_output_tokens"]), ("complete", 2000))
        budget = run["budget"]
        self.assertEqual((budget["provider_calls_charged"], budget["uncertain_reserved_usd"]), (2, 0.0))
        self.assertEqual(budget["budget_charged_usd"], budget["cost_complete_usd"])
        self.assertEqual(run["output_limit"]["limit"], 2000)

    def test_authorization_must_carry_a_valid_output_limit(self):
        for value in (None, 0, -5, 1.5, True, float("nan"), float("inf"), "2000"):
            gateway = BoundedGateway()
            with self.subTest(value=value), self.assertRaisesRegex(verifier.AuthorizationError, "max_output_tokens"):
                replay(self.fixed, self.report, gateway, max_output_tokens=value)
            self.assertEqual(gateway.limits, [])

    def test_provider_that_cannot_enforce_the_limit_is_refused(self):
        route = {"role": "gatekeeper", "provider_order": ["gemini"], "models": {"gemini": "gemini-3.5-flash-lite"}}
        gateway = BoundedGateway()
        with self.assertRaisesRegex(verifier.AuthorizationError, "output-token limit"):
            replay(self.fixed, self.report, gateway, route=route, provider="gemini", model="gemini-3.5-flash-lite")
        self.assertEqual(gateway.limits, [])

    def test_no_provider_call_when_authorization_validation_fails(self):
        for override in ({"adapter": "other"}, {"handoff_dataset_hash": "f" * 64}, {"model": "other"},
                         {"max_calls": 0}, {"max_cost_usd": float("nan")}, {"schema": "x"}):
            gateway = BoundedGateway()
            with self.subTest(override=override), self.assertRaises(verifier.AuthorizationError):
                replay(self.fixed, self.report, gateway, **override)
            self.assertEqual(gateway.limits, [])

    def test_failure_before_usage_charges_the_reservation_and_stops(self):
        gateway = BoundedGateway(fail_on={1}, usage=None)
        records, run = replay(self.fixed, self.report, gateway)
        self.assertEqual([r["status"] for r in records], ["failed", "not_executed"])
        self.assertEqual(len(gateway.limits), 1)
        self.assertEqual(records[0]["usage"]["cost_status"], "incomplete")
        self.assertEqual(run["budget"]["uncertain_reserved_usd"], self.reserved())
        self.assertEqual(run["budget"]["budget_charged_usd"], self.reserved())
        self.assertEqual(run["budget"]["provider_calls_charged"], 1)

    def test_partial_usage_is_incomplete_not_complete(self):
        records, run = replay(self.fixed, self.report, BoundedGateway(usage=(1000, None)))
        self.assertEqual(records[0]["usage"]["cost_status"], "incomplete")
        self.assertNotIn("cost_usd", records[0]["usage"])
        self.assertEqual(run["budget"]["cost_complete_usd"], 0.0)
        self.assertEqual(run["budget"]["budget_charged_usd"], self.reserved())

    def test_repeated_failures_each_consume_a_call(self):
        gateway = BoundedGateway(fail_on={1, 2})
        records, run = replay(self.fixed, self.report, gateway)
        self.assertEqual([r["status"] for r in records], ["failed", "failed"])
        self.assertEqual(run["budget"]["provider_calls_charged"], 2)
        self.assertGreater(run["budget"]["budget_charged_usd"], 0)
        records, run = replay(self.fixed, self.report, BoundedGateway(fail_on={1, 2}), max_calls=1, allow_partial=True)
        self.assertEqual([r["status"] for r in records], ["failed", "not_executed"])
        self.assertEqual(run["budget"]["provider_calls_charged"], 1)

    def test_cancellation_keeps_the_reservation_charged(self):
        auth = authorization(self.report, max_output_tokens=2000)
        ledger = verifier.BudgetLedger(auth["max_calls"], auth["max_cost_usd"])
        with self.assertRaises(asyncio.CancelledError):
            replay(self.fixed, self.report, BoundedGateway(cancel_on={1}), ledger=ledger)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["provider_calls_charged"], 1)
        self.assertEqual(snapshot["unsettled_reserved_usd"], self.reserved())
        self.assertEqual(snapshot["budget_charged_usd"], self.reserved())

    def test_reservation_scales_with_the_authorized_limit_and_blocks_calls(self):
        self.assertGreater(self.reserved(limit=4000), self.reserved(limit=2000))
        budget = self.reserved(limit=2000) + 1e-9   # fits a 2,000-token reservation, not a 4,000-token one
        gateway = BoundedGateway()
        records, run = replay(self.fixed, self.report, gateway, max_output_tokens=4000, max_cost_usd=budget)
        self.assertEqual(gateway.limits, [])
        self.assertTrue(all(r["status"] == "not_executed" for r in records))
        self.assertEqual(run["budget"]["budget_charged_usd"], 0.0)

    def test_concurrent_runs_share_one_ledger_without_overspending(self):
        auth = authorization(self.report, max_output_tokens=2000, max_calls=2, allow_partial=True)
        ledger = verifier.BudgetLedger(2, 1.0)
        gateway = BoundedGateway()

        async def both():
            run = lambda: verifier.run_replay(self.fixed, authorization=auth, route=ROUTE, ledger=ledger,
                                              dataset_hash=self.report["handoff"]["dataset_hash"])
            return await asyncio.gather(run(), run())
        with patch.object(claim_verification, "ainvoke_structured", gateway):
            asyncio.run(both())
        self.assertEqual(len(gateway.limits), 2)
        self.assertEqual(ledger.snapshot()["provider_calls_charged"], 2)

    def test_reported_output_above_the_limit_stops_the_run(self):
        records, run = replay(self.fixed, self.report, BoundedGateway(usage=(1000, 2500)))
        self.assertTrue(records[0]["usage"]["output_limit_exceeded"])
        self.assertEqual(records[1]["status"], "not_executed")
        self.assertIn("output", run["stopped"])

    def test_export_contract_is_unchanged(self):
        records, _ = replay(self.fixed, self.report, BoundedGateway())
        export = build_export(records=records, protocol={}, import_report=self.report, run_id="t",
                              git={"sha": "a" * 40, "source": "explicit"}, predictor={"name": verifier.ADAPTER})
        verify_export(export)
        self.assertEqual(export["schema"], "scientific-rag-benchmark-predictions/v1")
        self.assertEqual(set(records[0]), {"input_id", "task", "stage", "status", "input_hash", "usage", "prediction",
                                           "record_hash"})
        self.assertEqual(set(records[0]["prediction"]), {"verdict", "reason", "instruction_guard_applied"})


class DryRunEligibilityTests(unittest.TestCase):
    def test_pinned_supported_route_is_eligible_and_makes_no_call(self):
        fixed, _ = fixed_items()
        with patch.object(claim_verification, "ainvoke_structured", side_effect=AssertionError("provider called")) as gw:
            plan = verifier.dry_run(fixed, route=ROUTE, max_output_tokens=2000)
        gw.assert_not_called()
        self.assertEqual(plan["provider_calls_made"], 0)
        live = plan["live_execution"]
        self.assertTrue(live["eligible_after_authorization"], live)
        self.assertEqual(live["max_output_tokens"], 2000)
        self.assertTrue(live["output_limit"]["enforced"])
        self.assertIn("not an absolute billing guarantee", plan["planning_cost_bound_note"])

    def test_unpinned_or_unsupported_routes_are_not_eligible(self):
        fixed, _ = fixed_items()
        gemini = {"role": "gatekeeper", "provider_order": ["gemini"], "models": {"gemini": "gemini-3.5-flash-lite"}}
        both = {"role": "gatekeeper", "provider_order": ["groq", "openrouter"],
                "models": {"groq": "openai/gpt-oss-120b", "openrouter": "qwen/qwen3.6-35b-a3b"}}
        for route in (gemini, both):
            live = verifier.dry_run(fixed, route=route, max_output_tokens=2000)["live_execution"]
            self.assertFalse(live["eligible_after_authorization"])
            self.assertTrue(live["blockers"])


if __name__ == "__main__":
    unittest.main()
