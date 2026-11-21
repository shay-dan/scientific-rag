"""Stage 20G campaign accounting with routed fallback (offline; no credentials or network)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.benchmark import e2e_v2
from app.benchmark.e2e_runtime import guarded_embedding_client, refuse_proxies, routed_chat
from app.benchmark.e2e_v2 import Campaign, ExecutionStopped, GuardedTransport
from app.weak_model_synthesis import StructuredEvidenceAnswerV2
from scientific_rag import legacy_pipeline, llm_gateway
from tests.test_stage20g_routing import ANSWER, FakeAdmissionRedis, FakeClock, completion, provider_error

KEYS = {"groq": "placeholder-groq", "openrouter": "placeholder-openrouter"}


def chat_request(route_name="groq", **changes):
    provider = route_name
    route = e2e_v2.CHAT_ROUTES[provider]
    body = {"model": route["model"], "temperature": 0, "max_tokens": 2000, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "test"}]}
    if provider == "openrouter":
        body["provider"] = e2e_v2.OPENROUTER_PREFERENCES
    body.update(changes)
    return httpx.Request("POST", f"https://{route['host']}{route['path']}", json=body)


class ArithmeticAndProtocolTests(unittest.TestCase):
    def test_worst_case_ceilings_are_exact_and_unchanged_by_the_capped_fallback(self):
        self.assertEqual(6_115_200, e2e_v2.reservation("groq"))
        self.assertEqual(6_115_200, e2e_v2.reservation("openrouter"))
        self.assertEqual(1_638_400, e2e_v2.reservation("embedding"))
        self.assertEqual(816_320_000, e2e_v2.maximum_reservation_nanousd(True))
        self.assertEqual(816_320_000, e2e_v2.maximum_reservation_nanousd(False))
        self.assertEqual(816_320_000, e2e_v2.protocol()["campaign_limit_nanousd"])

    def test_fingerprint_binds_protocol_manifest_and_code(self):
        base = dict(manifest_sha256=e2e_v2.EXPECTED_MANIFEST_SHA256, code_sha="a" * 40,
                    image_id="sha256:" + "1" * 64)
        with_fallback = e2e_v2.fingerprint(e2e_v2.protocol(fallback=True), **base)
        self.assertEqual(64, len(with_fallback))
        self.assertNotEqual(with_fallback, e2e_v2.fingerprint(e2e_v2.protocol(fallback=False), **base))
        self.assertNotEqual(with_fallback, e2e_v2.fingerprint(e2e_v2.protocol(), **{**base, "code_sha": "b" * 40}))
        self.assertNotEqual(with_fallback, e2e_v2.fingerprint(e2e_v2.protocol(), **{**base, "manifest_sha256": "0" * 64}))
        self.assertNotEqual(with_fallback, e2e_v2.fingerprint(e2e_v2.protocol(), **{**base, "image_id": "sha256:" + "2" * 64}))
        with self.assertRaises(ValueError):
            e2e_v2.fingerprint(e2e_v2.protocol(), **{**base, "image_id": "mutable-tag"})
        self.assertNotEqual("422aabe98dd6ed242ea7c2b363d72a80b666404528073583e8a1cd3c082e308e", with_fallback)
        self.assertEqual("stage20-e2e-v2", e2e_v2.protocol()["campaign_id"])
        self.assertFalse(e2e_v2.protocol()["predecessor"]["resumed"])

    def test_proxy_environment_is_refused(self):
        with self.assertRaises(RuntimeError):
            refuse_proxies({"HTTPS_PROXY": "http://proxy:3128"})
        refuse_proxies({})


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.events, self.now = [], 0.0
        self.campaign = Campaign(self.events.append, clock=lambda: self.now)
        self.campaign.begin_case("E001")

    def test_openrouter_contract_requires_the_pinned_preferences(self):
        for request in (chat_request("openrouter", provider={"allow_fallbacks": True}),
                        chat_request("openrouter", reasoning_effort="high"),
                        chat_request("groq", provider=e2e_v2.OPENROUTER_PREFERENCES)):
            with self.subTest(body=json.loads(request.content).get("provider")):
                campaign = Campaign(lambda _event: None)
                campaign.begin_case("E001")
                with self.assertRaises(ExecutionStopped):
                    campaign.admit(request, "openrouter" if "openrouter.ai" == request.url.host else "groq")
                self.assertEqual(0, sum(campaign.summary()["requests_reserved"].values()))

    def test_no_fallback_protocol_refuses_the_openrouter_destination(self):
        campaign = Campaign(lambda _event: None, fallback=False)
        campaign.begin_case("E001")
        self.assertFalse(campaign.can_reserve("openrouter", e2e_v2.OPENROUTER_MODEL))
        with self.assertRaises(ExecutionStopped) as raised:
            campaign.admit(chat_request("openrouter"), "openrouter")
        self.assertEqual("campaign", raised.exception.scope)

    def test_per_case_ceiling_fails_the_case_not_the_campaign(self):
        for _ in range(20):
            self.campaign.returned(self.campaign.admit(chat_request("groq" if _ % 2 else "openrouter"),
                                                       "groq" if _ % 2 else "openrouter"), 200)
        self.assertFalse(self.campaign.can_reserve("groq", e2e_v2.GROQ_MODEL))
        with self.assertRaises(ExecutionStopped) as raised:
            self.campaign.admit(chat_request(), "groq")
        self.assertEqual(("per_case_request_ceiling", "case"), (raised.exception.reason, raised.exception.scope))
        self.assertIsNone(self.campaign.stop_reason)
        self.campaign.end_case("failed")
        self.campaign.begin_case("E002")
        self.assertTrue(self.campaign.can_reserve("groq", e2e_v2.GROQ_MODEL))

    def test_concurrent_admission_never_exceeds_the_shared_chat_pool(self):
        def worker(n):
            provider = "groq" if n % 2 else "openrouter"
            try:
                self.campaign.returned(self.campaign.admit(chat_request(provider), provider), 200)
                return True
            except ExecutionStopped:
                return False
        with ThreadPoolExecutor(max_workers=12) as pool:
            self.assertEqual(20, sum(pool.map(worker, range(40))))
        summary = self.campaign.summary()
        self.assertEqual(20, summary["requests_reserved"]["chat"])
        self.assertEqual(20 * 6_115_200, summary["reserved_nanousd"])

    def test_definitive_error_is_settled_without_stopping_and_uncertain_stays_charged(self):
        self.campaign.returned(self.campaign.admit(chat_request(), "groq"), 429)
        ticket = self.campaign.admit(chat_request(), "groq")
        self.campaign.uncertain(ticket)
        self.assertIsNone(self.campaign.stop_reason)
        self.assertEqual(2 * 6_115_200, self.campaign.summary()["reserved_nanousd"])
        self.assertEqual(0, self.campaign.summary()["pending"])

    def test_unsettled_ticket_stops_entire_campaign_without_discarding_reservation(self):
        ticket = self.campaign.admit(chat_request(), "groq")
        outcome = self.campaign.end_case("completed")
        self.assertEqual(("stopped", "unsettled_at_case_end"), (outcome["status"], outcome["reason"]))
        self.assertEqual("unsettled_at_case_end", self.campaign.stop_reason)
        self.assertEqual(1, self.campaign.summary()["pending"])
        self.assertEqual(ticket.reserved_nanousd, self.campaign.summary()["reserved_nanousd"])
        with self.assertRaises(ExecutionStopped):
            self.campaign.begin_case("E002")

    def test_end_case_rejects_late_success_even_without_another_provider_call(self):
        self.now = 601
        outcome = self.campaign.end_case("completed")
        self.assertEqual(("stopped", "question_deadline"), (outcome["status"], outcome["reason"]))
        self.assertIsNone(self.campaign.stop_reason)
        self.campaign.begin_case("E002")
        self.now = 3001
        outcome = self.campaign.end_case("completed")
        self.assertEqual(("stopped", "campaign_deadline"), (outcome["status"], outcome["reason"]))
        with self.assertRaises(ExecutionStopped):
            self.campaign.begin_case("E003")

    def test_question_deadline_is_case_scoped(self):
        self.now = 600
        with self.assertRaises(ExecutionStopped) as raised:
            self.campaign.admit(chat_request(), "groq")
        self.assertEqual(("question_deadline", "case"), (raised.exception.reason, raised.exception.scope))
        self.assertIsNone(self.campaign.stop_reason)

    def test_campaign_deadline_and_journal_failure_are_campaign_scoped(self):
        def broken(_event):
            raise OSError("cannot persist")
        self.campaign._append = broken
        with self.assertRaises(ExecutionStopped):
            self.campaign.admit(chat_request(), "groq")
        self.assertEqual(("journal_failure", 0), (self.campaign.stop_reason, self.campaign.summary()["reserved_nanousd"]))
        clock = [0.0]
        campaign = Campaign(lambda _event: None, clock=lambda: clock[0])
        clock[0] = 3000.0
        with self.assertRaises(ExecutionStopped):
            campaign.begin_case("E001")
        self.assertEqual("campaign_deadline", campaign.stop_reason)

    def test_usage_beyond_the_allowance_stops_the_campaign(self):
        ticket = self.campaign.admit(chat_request(), "groq")
        self.campaign.returned(ticket, 200, {"prompt_tokens": 10, "completion_tokens": 2001})
        self.assertEqual("provider_usage_exceeded_allowance", self.campaign.stop_reason)


class RoutedCampaignTests(unittest.IsolatedAsyncioTestCase):
    """The real gateway, SDK serialization and throttle, through guarded campaign transports."""

    async def asyncSetUp(self):
        self.clock = FakeClock(now=60 * 1000 + 28.196)
        self.redis = FakeAdmissionRedis(self.clock)
        self.events: list[dict] = []
        self.sent: list[tuple[str, str]] = []   # (host, journal event immediately before the send)
        self.enterContext(patch.object(llm_gateway, "_monotonic", self.clock.monotonic))
        self.enterContext(patch.object(llm_gateway, "_sleep", self.clock.sleep))
        self.enterContext(patch.dict(os.environ, {"LLM_THROTTLE_NAMESPACE": "scientific_rag:llm"}))
        self.enterContext(patch.object(httpx.HTTPTransport, "handle_request",
                                       side_effect=AssertionError("unmocked network")))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                                       side_effect=AssertionError("unmocked network")))

    def campaign(self, **kwargs):
        return Campaign(self.events.append, clock=self.clock.monotonic, **kwargs)

    def transports(self, campaign, behaviour):
        def handler(request):
            self.sent.append((request.url.host, self.events[-1]["event"]))
            return behaviour[request.url.host](request)
        mock = httpx.MockTransport(handler)
        providers = ["groq"] + (["openrouter"] if campaign.fallback else [])
        return {name: GuardedTransport(campaign, name, sync=mock, async_=mock) for name in providers}

    async def expert(self, campaign, behaviour, prompt="test"):
        token = llm_gateway.begin_trace(self.redis)
        try:
            with routed_chat(campaign, self.transports(campaign, behaviour), keys=KEYS):
                return await llm_gateway.ainvoke_structured("expert", 0, prompt, StructuredEvidenceAnswerV2)
        finally:
            llm_gateway.end_trace(token)

    def assert_reserved_before_every_send(self, campaign):
        self.assertTrue(self.sent)
        self.assertTrue(all(event == "reserved_before_transport" for _host, event in self.sent))
        self.assertEqual(len(self.sent), sum(campaign.summary()["requests_by_provider"].values()))

    async def test_e001_regression_waits_reserves_once_and_completes(self):
        campaign = self.campaign(fallback=False)
        campaign.begin_case("E001")
        self.redis.preload_tokens("groq", 3500)
        result = await self.expert(campaign, {"api.groq.com": lambda r: completion(json.dumps(ANSWER))},
                                   prompt="x" * 6000)
        self.assertIsInstance(result, StructuredEvidenceAnswerV2)
        routing = [(e["kind"], e.get("action")) for e in self.events if e["event"] == "routing"]
        self.assertEqual([("routing_decision", "wait"), ("routing_decision", "send"),
                          ("attempt_succeeded", None)], routing)
        self.assertEqual({"groq": 1, "openrouter": 0, "embedding": 0}, campaign.summary()["requests_by_provider"])
        self.assert_reserved_before_every_send(campaign)
        self.assertIsNone(campaign.stop_reason)

    async def test_provider_429_falls_back_with_both_requests_reserved(self):
        campaign = self.campaign()
        campaign.begin_case("E001")
        await self.expert(campaign, {
            "api.groq.com": lambda r: provider_error(429, "rate_limit_exceeded", "tokens"),
            "openrouter.ai": lambda r: completion(json.dumps(ANSWER))})
        self.assertEqual({"groq": 1, "openrouter": 1, "embedding": 0}, campaign.summary()["requests_by_provider"])
        self.assertEqual(2 * 6_115_200, campaign.summary()["reserved_nanousd"])
        self.assert_reserved_before_every_send(campaign)

    async def test_exhausted_fallback_budget_is_refused_before_admission(self):
        campaign = self.campaign()
        campaign.begin_case("E001")
        for _ in range(19):
            campaign.returned(campaign.admit(chat_request(), "groq"), 200)
        with self.assertRaises(Exception):
            await self.expert(campaign, {
                "api.groq.com": lambda r: provider_error(429, "rate_limit_exceeded", "tokens"),
                "openrouter.ai": lambda r: completion(json.dumps(ANSWER))})
        self.assertEqual(["api.groq.com"], [host for host, _ in self.sent])
        self.assertEqual(20, campaign.summary()["requests_reserved"]["chat"])
        self.assertIsNone(campaign.stop_reason)

    async def test_uncertain_transmission_is_charged_and_never_replayed(self):
        campaign = self.campaign()
        campaign.begin_case("E001")

        def timeout(request):
            raise httpx.ReadTimeout("simulated", request=request)
        with self.assertRaises(Exception) as raised:
            await self.expert(campaign, {"api.groq.com": timeout,
                                         "openrouter.ai": lambda r: completion(json.dumps(ANSWER))})
        self.assertEqual("uncertain_transmission", raised.exception.routing_reason)
        self.assertEqual(["api.groq.com"], [host for host, _ in self.sent])
        self.assertIn("transport_uncertain", [event["event"] for event in self.events])
        self.assertEqual(6_115_200, campaign.summary()["reserved_nanousd"])

    async def test_journal_failure_prevents_transmission_and_any_fallback(self):
        campaign = self.campaign()
        campaign.begin_case("E001")

        def broken(event):
            if event.get("event") == "reserved_before_transport":
                raise OSError("disk full")
            self.events.append(event)
        campaign._append = broken
        with self.assertRaises(Exception):
            await self.expert(campaign, {"api.groq.com": lambda r: completion(json.dumps(ANSWER)),
                                         "openrouter.ai": lambda r: completion(json.dumps(ANSWER))})
        self.assertEqual([], self.sent)
        self.assertEqual("journal_failure", campaign.stop_reason)

    async def test_real_sdks_both_chat_call_forms_and_gemini_embeddings(self):
        campaign = self.campaign()
        campaign.begin_case("E001")
        behaviour = {"api.groq.com": lambda r: completion(json.dumps(ANSWER)),
                     "generativelanguage.googleapis.com": lambda r: httpx.Response(200, json={
                         "embeddings": [{"values": [0.0] * 768}
                                        for _ in json.loads(r.content).get("requests", [None])]})}
        token = llm_gateway.begin_trace(self.redis)
        try:
            with routed_chat(campaign, self.transports(campaign, behaviour), keys=KEYS):
                first = await llm_gateway.ainvoke_structured("expert", 0, "test", StructuredEvidenceAnswerV2)
                second = await legacy_pipeline.ainvoke_with_fallback("planner", 0, lambda model: model, "test")
            self.assertIsInstance(first, StructuredEvidenceAnswerV2)
            self.assertTrue(second.content)
        finally:
            llm_gateway.end_trace(token)

        def handler(request):
            self.sent.append((request.url.host, self.events[-1]["event"]))
            return behaviour[request.url.host](request)
        mock = httpx.MockTransport(handler)
        client = guarded_embedding_client(campaign, (GuardedTransport(campaign, "embedding", sync=mock),
                                                     GuardedTransport(campaign, "embedding", async_=mock)),
                                          api_key="placeholder-gemini")
        try:
            from google.genai import types
            result = await asyncio.to_thread(
                client.models.embed_content, model=e2e_v2.EMBED_MODEL,
                contents=[types.Content(parts=[types.Part.from_text(text="one")]),
                          types.Content(parts=[types.Part.from_text(text="two")])],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768))
            self.assertEqual(2, len(result.embeddings))
        finally:
            client.close()
        self.assertEqual({"groq": 2, "openrouter": 0, "embedding": 1}, campaign.summary()["requests_by_provider"])
        self.assert_reserved_before_every_send(campaign)
        self.assertIsNone(campaign.stop_reason)
        campaign.end_case("completed")


if __name__ == "__main__":
    unittest.main()
