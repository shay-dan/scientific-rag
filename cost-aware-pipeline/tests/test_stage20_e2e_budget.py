"""Offline request-budget and pinned SDK tests; no credentials or network."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

from app.benchmark.e2e_budget import (
    Campaign, ExecutionStopped, GuardedTransport, PrivateJournal, protocol,
    GROQ_MODEL, EMBED_MODEL, RESERVATION, CASES,
)
from app.benchmark.e2e_execution import bounded_chat, bounded_embedding_client
from app.weak_model_synthesis import StructuredEvidenceAnswerV2
from scientific_rag import llm_gateway, legacy_pipeline


def chat_request(**changes):
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions",
                         json={"model": GROQ_MODEL, "temperature": 0, "max_tokens": 2000,
                               "reasoning_effort": "low", "messages": [{"role": "user", "content": "test"}],
                               **changes})


def embed_request(**changes):
    return httpx.Request("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents",
                         json={"requests": [{"model": f"models/{EMBED_MODEL}",
                                            "content": {"parts": [{"text": "test"}]},
                                            "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 768,
                                            **changes}]})


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.now = 0
        self.campaign = Campaign(self.events.append, clock=lambda: self.now)
        self.campaign.begin_case("E001")

    def test_exact_reservation_arithmetic(self):
        self.assertEqual(protocol()["maximum_modeled_reservation_nanousd"], 816_320_000)
        self.assertEqual(protocol()["campaign_limit_nanousd"], 850_000_000)

    def test_reserve_before_send_and_no_refund(self):
        seen = []
        def handler(_request):
            seen.append(self.events[-1]["event"])
            return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 200}})
        wire = GuardedTransport(self.campaign, "groq", sync=httpx.MockTransport(handler))
        with httpx.Client(transport=wire) as client:
            self.assertEqual(client.send(chat_request()).status_code, 200)
        self.assertEqual(seen, ["reserved_before_transport"])
        self.assertEqual(self.campaign.summary()["reserved_nanousd"], RESERVATION["groq"])
        self.assertEqual(self.events[-1]["event"], "transport_returned")

    def test_mixed_budget_and_full_campaign_limit(self):
        for case in CASES:
            if case != "E001":
                self.campaign.begin_case(case)
            for provider, req, count in (("groq", chat_request(), 20), ("embedding", embed_request(), 25)):
                for _ in range(count):
                    ticket = self.campaign.admit(req, provider)
                    self.campaign.returned(ticket, 200)
            self.campaign.end_case()
        self.assertEqual(self.campaign.summary()["requests_reserved"], {"groq": 100, "embedding": 125})
        self.assertEqual(self.campaign.summary()["reserved_nanousd"], 816_320_000)
        with self.assertRaises(ExecutionStopped):
            self.campaign.begin_case("E006")

    def test_concurrent_admission_never_exceeds_limit(self):
        def worker(_n):
            try:
                ticket = self.campaign.admit(chat_request(), "groq")
                self.campaign.returned(ticket, 200)
                return True
            except ExecutionStopped:
                return False
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(worker, range(40)))
        self.assertEqual(sum(results), 20)
        self.assertEqual(self.campaign.summary()["stop_reason"], "per_case_request_ceiling")
        with self.assertRaises(ExecutionStopped):
            self.campaign.admit(embed_request(), "embedding")

    def test_wrong_route_or_limits_never_transmitted(self):
        for provider, req in (
            ("groq", chat_request(max_tokens=2001)),
            ("groq", chat_request(reasoning_effort="high")),
            ("groq", chat_request(model="other")),
            ("groq", chat_request(messages=[{"content": "x" * 33000}])),
            ("embedding", embed_request(outputDimensionality=3072)),
            ("embedding", embed_request(content={"parts": [{"text": "x" * 8192}]})),
            ("embedding", embed_request(model="models/another")),
            ("groq", httpx.Request("POST", "https://example.org/")),
        ):
            with self.subTest(provider=provider):
                campaign = Campaign(lambda _event: None)
                campaign.begin_case("E001")
                with self.assertRaises(ExecutionStopped):
                    campaign.admit(req, provider)
                self.assertEqual(sum(campaign.summary()["requests_reserved"].values()), 0)

    def test_provider_error_latches_across_providers(self):
        calls = []
        def handler(_request):
            calls.append(True)
            return httpx.Response(429)
        with httpx.Client(transport=GuardedTransport(
            self.campaign, "groq", sync=httpx.MockTransport(handler)
        )) as client:
            self.assertEqual(client.send(chat_request()).status_code, 429)
            with self.assertRaises(ExecutionStopped):
                client.send(chat_request())
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ExecutionStopped):
            self.campaign.admit(embed_request(), "embedding")

    def test_uncertain_transmission_charged_and_terminal(self):
        def handler(request):
            raise httpx.ReadTimeout("simulated", request=request)
        with httpx.Client(transport=GuardedTransport(
            self.campaign, "embedding", sync=httpx.MockTransport(handler)
        )) as client:
            with self.assertRaises(httpx.ReadTimeout):
                client.send(embed_request())
        self.assertEqual(self.campaign.summary()["reserved_nanousd"], RESERVATION["embedding"])
        self.assertEqual(self.campaign.summary()["stop_reason"], "uncertain_transmission")

    def test_question_deadline(self):
        self.now = 600
        with self.assertRaises(ExecutionStopped):
            self.campaign.admit(chat_request(), "groq")
        self.assertEqual(self.campaign.summary()["stop_reason"], "question_deadline")

    def test_journal_failure_prevents_transport(self):
        def broken(_event):
            raise OSError("cannot persist")
        self.campaign._append = broken
        with self.assertRaises(ExecutionStopped):
            self.campaign.admit(chat_request(), "groq")
        self.assertEqual(self.campaign.summary()["stop_reason"], "journal_failure")
        self.assertEqual(self.campaign.summary()["reserved_nanousd"], 0)

    def test_exclusive_journal_refuses_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wire.jsonl"
            journal = PrivateJournal(path)
            journal.append({"event": "test"})
            journal.close()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                PrivateJournal(path)


class ActualSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_chat_gateway_both_call_forms_and_gemini_embedding(self):
        events = []
        campaign = Campaign(events.append)
        campaign.begin_case("E001")
        sent = []
        answer = {"claims": [{"text": "The measured outcome was reported.",
                              "source_id": "S1", "evidence_quote": "The measured outcome was reported."}],
                  "limitations": [], "unanswered_parts": []}
        def handler(request):
            sent.append((request.url.host, json.loads(request.content)))
            if request.url.host == "api.groq.com":
                return httpx.Response(200, json={
                    "id": "test", "object": "chat.completion", "created": 0, "model": GROQ_MODEL,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": json.dumps(answer)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
                })
            count = len(json.loads(request.content).get("requests", [None]))
            return httpx.Response(200, json={"embeddings": [{"values": [0.0] * 768} for _ in range(count)]})
        def factory(provider):
            mock = httpx.MockTransport(handler)
            return GuardedTransport(campaign, provider, sync=mock, async_=mock)
        # A missed SDK injection is a hard test failure, never a real network call.
        with patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("unmocked network")), \
             patch.object(httpx.AsyncHTTPTransport, "handle_async_request", side_effect=AssertionError("unmocked network")), \
             bounded_chat(campaign, factory, api_key="placeholder-test-key"):
            first = await llm_gateway.ainvoke_structured("expert", 0, "test", StructuredEvidenceAnswerV2)
            self.assertIsInstance(first, StructuredEvidenceAnswerV2)
            second = await legacy_pipeline.ainvoke_with_fallback(
                "planner", 0, lambda model: model, "test"
            )
            self.assertTrue(second.content)
            client = bounded_embedding_client(campaign, factory, api_key="placeholder-test-key")
            try:
                from google.genai import types
                result = await asyncio.to_thread(
                    client.models.embed_content, model=EMBED_MODEL,
                    contents=[types.Content(parts=[types.Part.from_text(text="test one")]),
                              types.Content(parts=[types.Part.from_text(text="test two")])],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
                )
                self.assertEqual(len(result.embeddings), 2)
            finally:
                client.close()
        self.assertEqual([host for host, _ in sent],
                         ["api.groq.com", "api.groq.com", "generativelanguage.googleapis.com"])
        self.assertEqual(campaign.summary()["requests_reserved"], {"groq": 2, "embedding": 1})
        self.assertIsNone(campaign.summary()["stop_reason"])
        campaign.end_case()


if __name__ == "__main__":
    unittest.main()
