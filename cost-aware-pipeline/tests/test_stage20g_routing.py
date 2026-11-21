"""Consolidated routing through the real gateway, SDK serialization and throttle (Stage 20G).

Offline: an httpx MockTransport stands in for providers, an in-memory Redis reproduces the
admission script's semantics, and a fake clock drives bounded waits. Any unmocked network
access fails the test.
"""
import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.weak_model_synthesis import StructuredEvidenceAnswerV2
from scientific_rag import llm_gateway, provider_throttle

GROQ_HOST, OPENROUTER_HOST = "api.groq.com", "openrouter.ai"
ANSWER = {"claims": [{"text": "The measured outcome was reported.", "source_id": "S1",
                      "evidence_quote": "The measured outcome was reported."}],
          "limitations": [], "unanswered_parts": []}


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeAdmissionRedis:
    """Python twin of provider_throttle.ADMIT_SCRIPT / RELEASE_SCRIPT on a fake clock."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock, self.values, self.expires = clock, {}, {}

    def _ms(self) -> int:
        return int(round(self.clock.now * 1000))

    def _live(self, key):
        if key in self.expires and self.expires[key] <= self._ms():
            self.values.pop(key, None)
            self.expires.pop(key, None)
        return self.values.get(key)

    async def eval(self, script, _numkeys, *args):
        if script == provider_throttle.RELEASE_SCRIPT:
            key, owner = args
            if self._live(key) == owner:
                self.values.pop(key, None)
                return 1
            return 0
        assert script == provider_throttle.ADMIT_SCRIPT
        prefix, rpm, tpm, estimate, owner, lease_ms, gatekeeper = args
        now = self._ms()
        minute = now // 60000
        requests, tokens = f"{prefix}:requests:{minute}", f"{prefix}:tokens:{minute}"
        lease, cooldown = f"{prefix}:gatekeeper:lease", f"{prefix}:cooldown"
        wait_ms = 60000 - (now % 60000)
        if self._live(cooldown) is not None:
            return [0, self.expires[cooldown] - now, "cooldown", minute]
        if rpm > 0 and int(self._live(requests) or 0) >= rpm:
            return [0, wait_ms, "requests", minute]
        if tpm > 0 and int(self._live(tokens) or 0) + estimate > tpm:
            return [0, wait_ms, "tokens", minute]
        if gatekeeper and self._live(lease) is not None:
            return [0, max(1, self.expires[lease] - now), "concurrency", minute]
        if gatekeeper:
            self.values[lease], self.expires[lease] = owner, now + lease_ms
        self.values[requests] = int(self._live(requests) or 0) + 1
        self.values[tokens] = int(self._live(tokens) or 0) + estimate
        self.expires[requests] = self.expires[tokens] = now + 120000
        return [1, 0, "admitted", minute]

    async def exists(self, key):
        return int(self._live(key) is not None)

    async def incrby(self, key, amount):
        self.values[key] = int(self._live(key) or 0) + amount
        return self.values[key]

    async def set(self, key, value, px=None):
        self.values[key] = value
        if px is not None:
            self.expires[key] = self._ms() + px

    def preload_tokens(self, provider: str, tokens: int) -> None:
        minute = self._ms() // 60000
        key = f"scientific_rag:llm:{provider}:tokens:{minute}"
        self.values[key], self.expires[key] = tokens, self._ms() + 120000


def completion(content: str, model: str = "openai/gpt-oss-120b") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200}})


def provider_error(status: int, code: str, kind: str, **extra) -> httpx.Response:
    headers = {"retry-after": "2"} if status == 429 else {}
    return httpx.Response(status, headers=headers,
                          json={"error": {"message": "simulated", "type": kind, "code": code, **extra}})


class FakeProviders:
    def __init__(self, **behaviour) -> None:
        self.behaviour = behaviour   # host -> callable(request) -> Response | raises
        self.sent: list[tuple[str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.sent.append((request.url.host, body))
        return self.behaviour[request.url.host](request)

    def hosts(self) -> list[str]:
        return [host for host, _ in self.sent]


def ok(_request):
    return completion(json.dumps(ANSWER))


class RoutedGatewayTests(unittest.IsolatedAsyncioTestCase):
    ROUTES = "groq:openai/gpt-oss-120b,openrouter:openai/gpt-oss-120b"

    async def asyncSetUp(self):
        # Inside a minute, 31.804 s before the next window: the recorded E001 shape.
        self.clock = FakeClock(now=60 * 1000 + 28.196)
        self.redis = FakeAdmissionRedis(self.clock)
        self.events: list[dict] = []
        environment = {
            "LLM_ROUTING_POLICY": "consolidated", "LLM_ROUTING_ROUTES": self.ROUTES,
            "LLM_THROTTLE_NAMESPACE": "scientific_rag:llm",
            "GROQ_API_KEY": "placeholder-groq", "OPENROUTER_API_KEY": "placeholder-openrouter",
            "LLM_GROQ_TPM_BUDGET": "6000", "LLM_GROQ_RPM_BUDGET": "20", "LLM_REQUEST_TIMEOUT_SECONDS": "150",
            "LLM_OPENROUTER_MAX_PRICE_PROMPT": "0.15", "LLM_OPENROUTER_MAX_PRICE_COMPLETION": "0.6",
        }
        self.enterContext(patch.dict(os.environ, environment))
        for name in [key for key in os.environ if key.startswith("LLM_ROUTING_ROUTES_")]:
            self.enterContext(patch.dict(os.environ, {name: ""}))
        self.enterContext(patch.object(llm_gateway, "_monotonic", self.clock.monotonic))
        self.enterContext(patch.object(llm_gateway, "_sleep", self.clock.sleep))
        self.enterContext(patch.object(httpx.HTTPTransport, "handle_request",
                                       side_effect=AssertionError("unmocked network")))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                                       side_effect=AssertionError("unmocked network")))

    async def call(self, providers: FakeProviders, *, routes: str | None = None, prompt: str = "test",
                   budget=None, structured: bool = True, role: str = "expert"):
        transport = httpx.MockTransport(providers.handler)
        env = {"LLM_ROUTING_ROUTES": routes} if routes is not None else {}
        token = llm_gateway.begin_trace(self.redis, capture_rate_limits=True)
        try:
            with patch.dict(os.environ, env), patch.object(llm_gateway, "_outbound_transport", lambda: transport), \
                    llm_gateway.routing_controls(budget=budget, observer=self.events.append):
                if structured:
                    return await llm_gateway.ainvoke_structured(role, 0, prompt, StructuredEvidenceAnswerV2,
                                                                max_output_tokens=2000, reasoning_effort="low")
                return await llm_gateway.ainvoke(role, 0, TextChain, {"question": prompt,
                                                                      "context": "--- [S1: source] text"})
        finally:
            self.trace = llm_gateway.end_trace(token)

    def actions(self):
        return [(event["action"], event["reason"]) for event in self.events if event["event"] == "routing_decision"]

    # --- A. local capacity -------------------------------------------------------------------
    async def test_e001_tpm_exhaustion_waits_for_the_window_reset_then_sends_once(self):
        self.redis.preload_tokens("groq", 3500)       # six earlier Groq calls consumed the window
        providers = FakeProviders(**{GROQ_HOST: ok})
        result = await self.call(providers, routes="groq:openai/gpt-oss-120b", prompt="x" * 6000)
        self.assertIsInstance(result, StructuredEvidenceAnswerV2)
        self.assertEqual([GROQ_HOST], providers.hosts())
        self.assertEqual([("wait", "tokens_window"), ("send", "admitted")], self.actions())
        self.assertAlmostEqual(31.804, self.clock.slept[0], places=3)

    async def test_e001_legacy_policy_fails_without_transmission(self):
        self.redis.preload_tokens("groq", 3500)
        providers = FakeProviders(**{GROQ_HOST: ok})
        with patch.dict(os.environ, {"LLM_ROUTING_POLICY": "legacy", "LLM_PROVIDER_ORDER": "groq",
                                     "LLM_PROVIDER_ORDER_EXPERT": "groq"}), \
                self.assertRaises(provider_throttle.ProviderThrottled) as raised:
            await self.call(providers, prompt="x" * 6000)
        self.assertEqual("tokens", raised.exception.reason)
        self.assertEqual([], providers.hosts())

    async def test_rpm_exhaustion_waits(self):
        with patch.dict(os.environ, {"LLM_GROQ_RPM_BUDGET": "1"}):
            await provider_throttle.admit(self.redis, "groq", "planner", "x", 20)
            providers = FakeProviders(**{GROQ_HOST: ok})
            await self.call(providers, routes="groq:openai/gpt-oss-120b")
        self.assertEqual(("wait", "requests_window"), self.actions()[0])
        self.assertEqual([GROQ_HOST], providers.hosts())

    async def test_request_larger_than_the_whole_window_stops_without_waiting(self):
        providers = FakeProviders(**{GROQ_HOST: ok})
        with self.assertRaises(llm_gateway.RoutingStopped) as raised:
            await self.call(providers, routes="groq:openai/gpt-oss-120b", prompt="x" * 20000)
        self.assertEqual("exceeds_provider_window", raised.exception.reason)
        self.assertEqual(([], []), (providers.hosts(), self.clock.slept))

    async def test_deadline_exhaustion_refuses_a_wait_it_cannot_afford(self):
        self.redis.preload_tokens("groq", 3500)
        providers = FakeProviders(**{GROQ_HOST: ok})
        with patch.dict(os.environ, {"LLM_REQUEST_TIMEOUT_SECONDS": "30"}), \
                self.assertRaises(llm_gateway.RoutingStopped) as raised:
            await self.call(providers, routes="groq:openai/gpt-oss-120b", prompt="x" * 6000)
        self.assertEqual("capacity_wait_exceeds_budget", raised.exception.reason)
        self.assertEqual([], providers.hosts())

    # --- B. fallback ---------------------------------------------------------------------------
    async def test_oversize_primary_falls_back_to_the_compatible_route(self):
        providers = FakeProviders(**{OPENROUTER_HOST: ok})
        result = await self.call(providers, prompt="x" * 20000)
        self.assertIsInstance(result, StructuredEvidenceAnswerV2)
        self.assertEqual([OPENROUTER_HOST], providers.hosts())
        body = providers.sent[0][1]
        self.assertEqual("openai/gpt-oss-120b", body["model"])
        self.assertEqual("low", body["reasoning_effort"])
        self.assertEqual(2000, body.get("max_completion_tokens", body.get("max_tokens")))
        self.assertEqual({"require_parameters": True, "allow_fallbacks": False,
                          "max_price": {"prompt": 0.15, "completion": 0.6}}, body["provider"])
        self.assertEqual("json_schema", body["response_format"]["type"])
        self.assertEqual(("fallback", "primary_exceeds_window"), self.actions()[0])

    async def test_incompatible_fallback_model_is_never_built_or_sent(self):
        from scientific_rag import provider_capabilities
        declared = {**provider_capabilities.ROUTE_CAPABILITIES,
                    ("gemini", "gemini-3.8-flash"): {"context_tokens": 1_000_000, "max_output_tokens": 65536,
                                                     "structured_output": True, "quota_group": "gemini"}}
        providers = FakeProviders()
        with patch.object(llm_gateway, "ROUTE_CAPABILITIES", declared), \
                patch.dict(os.environ, {"GOOGLE_API_KEY": "placeholder-gemini"}), \
                patch.object(llm_gateway, "ChatGoogleGenerativeAI", side_effect=AssertionError("built")), \
                self.assertRaises(llm_gateway.RoutingStopped) as raised:
            await self.call(providers, routes="groq:openai/gpt-oss-120b,gemini:gemini-3.8-flash", prompt="x" * 20000)
        self.assertEqual("exceeds_provider_window", raised.exception.reason)
        self.assertEqual([], providers.hosts())

    async def test_exhausted_fallback_budget_stops_before_transmission(self):
        class Budget:
            def can_reserve(self, provider, _model):
                return provider == "groq"
        providers = FakeProviders()
        with self.assertRaises(llm_gateway.RoutingStopped):
            await self.call(providers, prompt="x" * 20000, budget=Budget())
        self.assertEqual([], providers.hosts())

    async def test_no_fallback_outside_the_allowlist(self):
        providers = FakeProviders(**{GROQ_HOST: lambda r: provider_error(503, "service_unavailable", "server_error"),
                                     OPENROUTER_HOST: ok})
        with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "groq,openrouter,gemini"}), \
                self.assertRaises(Exception) as raised:
            await self.call(providers, routes="groq:openai/gpt-oss-120b")
        self.assertEqual([GROQ_HOST], providers.hosts(), "no SDK retry and no unapproved fallback")
        self.assertEqual("no_fallback_route", raised.exception.routing_reason)

    # --- C. transmission safety ----------------------------------------------------------------
    async def test_provider_429_is_definitive_and_falls_back_once(self):
        providers = FakeProviders(**{GROQ_HOST: lambda r: provider_error(429, "rate_limit_exceeded", "tokens"),
                                     OPENROUTER_HOST: ok})
        await self.call(providers)
        self.assertEqual([GROQ_HOST, OPENROUTER_HOST], providers.hosts(), "exactly one request per route")
        failed = [event for event in self.events if event["event"] == "attempt_failed"]
        self.assertEqual(("definitive_response", 429, True), (failed[0]["transmission"], failed[0]["http_status"],
                                                             failed[0]["may_fallback"]))

    async def test_provider_5xx_falls_back(self):
        providers = FakeProviders(**{GROQ_HOST: lambda r: provider_error(503, "service_unavailable", "server_error"),
                                     OPENROUTER_HOST: ok})
        await self.call(providers)
        self.assertEqual([GROQ_HOST, OPENROUTER_HOST], providers.hosts())

    async def test_timeout_after_transmission_is_uncertain_and_never_replayed(self):
        def timeout(request):
            raise httpx.ReadTimeout("simulated", request=request)
        providers = FakeProviders(**{GROQ_HOST: timeout, OPENROUTER_HOST: ok})
        with self.assertRaises(Exception) as raised:
            await self.call(providers)
        self.assertEqual([GROQ_HOST], providers.hosts())
        self.assertEqual("uncertain_transmission", raised.exception.routing_reason)

    async def test_provider_structured_generation_failure_stops(self):
        providers = FakeProviders(**{GROQ_HOST: lambda r: provider_error(
            400, "json_validate_failed", "invalid_request_error", failed_generation="{}"), OPENROUTER_HOST: ok})
        with self.assertRaises(Exception) as raised:
            await self.call(providers)
        self.assertEqual([GROQ_HOST], providers.hosts())
        self.assertEqual("invalid_output", raised.exception.routing_reason)

    async def test_valid_json_failing_the_schema_is_completed_invalid(self):
        providers = FakeProviders(**{GROQ_HOST: lambda r: completion(json.dumps({"claims": "not a list"})),
                                     OPENROUTER_HOST: ok})
        with self.assertRaises(llm_gateway.InvalidStructuredResponse) as raised:
            await self.call(providers)
        self.assertEqual([GROQ_HOST], providers.hosts())
        self.assertEqual("invalid_output", raised.exception.routing_reason)
        failed = [event for event in self.events if event["event"] == "attempt_failed"]
        self.assertEqual("completed_invalid", failed[0]["transmission"])

    async def test_failed_local_output_validation_after_a_response(self):
        providers = FakeProviders(**{GROQ_HOST: lambda r: completion("too short"), OPENROUTER_HOST: ok})
        with self.assertRaises(llm_gateway.InvalidStructuredResponse) as raised:
            await self.call(providers, structured=False)
        self.assertEqual([GROQ_HOST], providers.hosts())
        self.assertEqual("invalid_output", raised.exception.routing_reason)

    async def test_transport_refusal_is_final(self):
        from scientific_rag.provider_failures import TransmissionRefused

        def refuse(_request):
            raise TransmissionRefused("budget")
        providers = FakeProviders(**{GROQ_HOST: refuse, OPENROUTER_HOST: ok})
        with self.assertRaises(Exception):
            await self.call(providers)
        self.assertEqual([GROQ_HOST], providers.hosts())
        self.assertNotIn(OPENROUTER_HOST, providers.hosts())

    async def test_concurrent_routed_calls_each_take_their_own_admission(self):
        providers = FakeProviders(**{GROQ_HOST: ok})
        transport = httpx.MockTransport(providers.handler)
        token = llm_gateway.begin_trace(self.redis)
        try:
            with patch.dict(os.environ, {"LLM_ROUTING_ROUTES": "groq:openai/gpt-oss-120b",
                                         "LLM_GROQ_TPM_BUDGET": "0", "LLM_GROQ_RPM_BUDGET": "3",
                                         "LLM_REQUEST_TIMEOUT_SECONDS": "30"}), \
                    patch.object(llm_gateway, "_outbound_transport", lambda: transport):
                results = await asyncio.gather(*[
                    llm_gateway.ainvoke_structured("planner", 0, "x", StructuredEvidenceAnswerV2,
                                                   max_output_tokens=2000, reasoning_effort="low")
                    for _ in range(5)], return_exceptions=True)
        finally:
            llm_gateway.end_trace(token)
        sent = sum(1 for item in results if isinstance(item, StructuredEvidenceAnswerV2))
        self.assertEqual(3, sent)
        self.assertEqual(3, len(providers.hosts()), "admissions never exceed the shared request window")


class TextChain:
    def __init__(self, model):
        self.model = model

    async def ainvoke(self, payload, config=None):
        message = await self.model.ainvoke(payload["question"], config=config)
        return message.content


if __name__ == "__main__":
    unittest.main()
