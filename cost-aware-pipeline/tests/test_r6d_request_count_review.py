"""R6D review reproduction (PR #16 review at 8010f22): one bounded gateway attempt could send
several billable provider requests while one provider attempt was recorded and one call reserved.

A chain that invokes the same bounded model more than once is run through the real
``ChatOpenAI`` client against ``httpx.MockTransport``; the transport counts the requests that
would actually be sent. No live or paid request.
"""
import asyncio
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

from app.benchmark import production_verifier_replay as verifier  # noqa: E402
from app.claim_verification import ClaimAssessment, ClaimVerificationBatch, ClaimVerificationError  # noqa: E402
from scientific_rag import llm_gateway  # noqa: E402
from tests.test_r6c_production_adapters import ROUTE, authorization, fixed_items  # noqa: E402
from tests.test_r6d_output_token_limits import HOSTS, LIMIT, Wire, provider_env  # noqa: E402


class Sequential:
    """Invokes the same model twice, one request after the other."""

    def __init__(self, model):
        self.model = model

    async def ainvoke(self, payload, config=None):
        await self.model.ainvoke(payload, config=config)
        return await self.model.ainvoke(payload, config=config)


class Concurrent(Sequential):
    async def ainvoke(self, payload, config=None):
        results = await asyncio.gather(self.model.ainvoke(payload, config=config),
                                       self.model.ainvoke(payload, config=config), return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        return results[0]


class SwallowsSecondFailure(Sequential):
    """Returns the first answer even if its second request is refused."""

    async def ainvoke(self, payload, config=None):
        first = await self.model.ainvoke(payload, config=config)
        try:
            await self.model.ainvoke(payload, config=config)
        except Exception:
            pass
        return first


class RetriesAfterFailure(Sequential):
    """Hand-rolled retry: a second request after the first one failed."""

    async def ainvoke(self, payload, config=None):
        try:
            return await self.model.ainvoke(payload, config=config)
        except Exception:
            return await self.model.ainvoke(payload, config=config)


def run_chain(chain, env, wire, *, limit=LIMIT):
    """Return (result or raised error, recorded provider attempts)."""
    async def go():
        token = llm_gateway.begin_trace()
        try:
            outcome = await llm_gateway.ainvoke("gatekeeper", 0, chain, "hello", max_output_tokens=limit)
        except Exception as error:
            outcome = error
        return outcome, llm_gateway.end_trace(token)
    with patch.dict(os.environ, env), patch.object(llm_gateway, "_outbound_transport", wire.transport):
        return asyncio.run(go())


GROQ = provider_env("groq", GROQ_API_KEY="k")
BOTH = provider_env("groq,openrouter", GROQ_API_KEY="k", OPENROUTER_API_KEY="k")


class OneRequestPerAttemptTests(unittest.TestCase):
    def assert_one_request(self, wire, attempts, outcome):
        self.assertEqual(len(wire.requests), 1,
                         f"{len(wire.requests)} outbound provider requests for {len(attempts)} recorded attempt(s)")
        self.assertIsInstance(outcome, llm_gateway.OutputTokenLimitError, outcome)
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["success"])

    def test_one_valid_bounded_request_is_unaffected(self):
        wire = Wire()
        outcome, attempts = run_chain(lambda model: model, GROQ, wire)
        self.assertEqual(outcome.content, "ok")
        self.assertEqual((len(wire.requests), len(attempts), attempts[0]["success"]), (1, 1, True))

    def test_two_sequential_requests(self):
        wire = Wire()
        outcome, attempts = run_chain(Sequential, GROQ, wire)
        self.assert_one_request(wire, attempts, outcome)
        # the first request's usage is kept on the recorded attempt
        self.assertEqual((attempts[0]["input_tokens"], attempts[0]["output_tokens"]), (11, 3))

    def test_two_concurrent_requests(self):
        wire = Wire()
        outcome, attempts = run_chain(Concurrent, GROQ, wire)
        self.assert_one_request(wire, attempts, outcome)

    def test_a_swallowed_refusal_still_fails_the_attempt(self):
        wire = Wire()
        outcome, attempts = run_chain(SwallowsSecondFailure, GROQ, wire)
        self.assert_one_request(wire, attempts, outcome)
        self.assertEqual((attempts[0]["input_tokens"], attempts[0]["output_tokens"]), (11, 3))

    def test_second_request_after_a_failed_first_request(self):
        wire = Wire(fail_hosts={HOSTS["groq"]})
        outcome, attempts = run_chain(RetriesAfterFailure, GROQ, wire)
        self.assert_one_request(wire, attempts, outcome)

    def test_refusal_does_not_fall_back_or_retry(self):
        for chain, fail in ((Sequential, ()), (RetriesAfterFailure, {HOSTS["groq"]})):
            wire = Wire(fail_hosts=fail)
            with self.subTest(chain=chain.__name__):
                outcome, attempts = run_chain(chain, BOTH, wire)
                self.assert_one_request(wire, attempts, outcome)
                self.assertEqual([host for host, _ in wire.requests], [HOSTS["groq"]])   # openrouter never tried
                self.assertEqual([attempt["provider"] for attempt in attempts], ["groq"])

    def test_the_sync_client_shares_the_one_request_allowance(self):
        class SyncThenAsync(Sequential):
            async def ainvoke(self, payload, config=None):
                self.model.invoke(payload)
                return await self.model.ainvoke(payload, config=config)
        wire = Wire()
        outcome, attempts = run_chain(SyncThenAsync, GROQ, wire)
        self.assert_one_request(wire, attempts, outcome)

    def test_each_fallback_attempt_gets_its_own_single_request(self):
        """Ordinary bounded fallback is unchanged: one request per provider attempt."""
        wire = Wire(fail_hosts={HOSTS["groq"]})
        outcome, attempts = run_chain(lambda model: model, BOTH, wire)
        self.assertEqual(outcome.content, "ok")
        self.assertEqual([host for host, _ in wire.requests], [HOSTS["groq"], HOSTS["openrouter"]])
        self.assertEqual(len(attempts), 2)

    def test_unbounded_production_calls_may_still_send_several_requests(self):
        wire = Wire()
        factory = lambda **kwargs: ChatOpenAI(**kwargs, http_async_client=httpx.AsyncClient(transport=wire.transport()))
        with patch.dict(os.environ, GROQ), patch.object(llm_gateway, "ChatOpenAI", factory):
            result = asyncio.run(llm_gateway.ainvoke("gatekeeper", 0, Sequential, "hello"))
        self.assertEqual(result.content, "ok")
        self.assertEqual(len(wire.requests), 2)
        self.assertEqual(wire.limits(), [{}, {}])


class VerifierBudgetAccountingTests(unittest.TestCase):
    """R6C ledger stays consistent with real outbound requests when a chain misbehaves."""

    def test_calls_charged_match_outbound_requests(self):
        fixed, report = fixed_items()
        wire = Wire()

        async def double_request_verifier(items, *, question, max_gateway_invocations, max_output_tokens):
            try:
                await llm_gateway.ainvoke("gatekeeper", 0, Sequential, "prompt", max_output_tokens=max_output_tokens)
            except Exception as error:
                raise ClaimVerificationError(f"semantic verification gateway failed: {error}") from error
            return ClaimVerificationBatch(assessments=[ClaimAssessment(target="claims.1", verdict="supported", reason="r")])

        async def go():
            return await verifier.run_replay(fixed, authorization=authorization(report), route=ROUTE,
                                             dataset_hash=report["handoff"]["dataset_hash"],
                                             verifier=double_request_verifier)
        with patch.dict(os.environ, GROQ), patch.object(llm_gateway, "_outbound_transport", wire.transport):
            records, run = asyncio.run(go())
        self.assertEqual(len(wire.requests), run["budget"]["provider_calls_charged"],
                         f"{len(wire.requests)} outbound requests but {run['budget']['provider_calls_charged']} call(s) charged")
        self.assertEqual(len(wire.requests), 1)
        self.assertEqual([record["status"] for record in records], ["failed", "not_executed"])
        self.assertIn("output_limit_not_enforced", records[0]["reason"])
        self.assertNotIn("prediction", records[0])
        self.assertEqual(records[0]["usage"]["cost_status"], "complete")   # the first request's measured usage is kept
        self.assertGreater(run["budget"]["cost_complete_usd"], 0)
        self.assertEqual(run["budget"]["budget_charged_usd"], run["budget"]["cost_complete_usd"])
        self.assertIn("output-token limit", run["stopped"])


if __name__ == "__main__":
    unittest.main()
