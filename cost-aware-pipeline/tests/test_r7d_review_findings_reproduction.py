"""R7D PR #24 review findings: stop precedence and infrastructure termination.

Finding 1: a safety stop observed after a call (uncertain billing, quota, model drift,
unexpected retry, output beyond the limit) must take precedence over model-quality
classification. An ``InvalidStructuredResponse`` whose accounting is not acceptable is not a
scorable model failure.

Finding 2: a non-429 provider/network failure with complete usage must stop incremental
execution, and an infrastructure-interrupted pair must never count as a completed pair.

Deterministic and offline: fake structured gateway, synthetic provider trace summaries,
real private checkpoint writes in a temporary directory. No provider request.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark.generation_evaluation import build_public_report, run_paid_evaluation
from app.benchmark.incremental_generation_evaluation import (
    AtomicCheckpoint,
    CheckpointError,
    _pair_status,
    run_incremental_evaluation,
)
from scientific_rag.llm_gateway import InvalidStructuredResponse
from tests import test_r7d_execution_safety_reproduction as repro
from tests import test_r7d_generation_evaluation as base

GROQ = "groq:openai/gpt-oss-120b"


def _drift(value):
    value["by_model"] = {"groq:another-model": {}}


def _incomplete(value):
    value.update(token_usage_complete=False, estimated_cost_usd=None, input_tokens=None, output_tokens=None)


def _retry(value):
    value["provider_calls"] = 2


def _over_limit(value):
    value["output_tokens"] = 5_000


def _quota(value):
    value["attempts"][0]["status_code"] = 429


# (stop code, summary mutation) for each post-call safety stop.
SAFETY_STOPS = (
    ("billing_unknown", _incomplete),
    ("model_drift", _drift),
    ("unexpected_retry", _retry),
    ("output_limit", _over_limit),
    ("rate_limit", _quota),
)


class FaultyProvider(repro.FakeProvider):
    """Fails one chosen request (1-based) with ``error`` and mutates its trace summary."""

    def __init__(self, *, fail_at, error, mutate=None, **kwargs):
        super().__init__(**kwargs)
        self.fail_at, self.error, self.mutate = fail_at, error, mutate

    async def gateway(self, role, temperature, prompt, schema, *, max_output_tokens):
        result = await super().gateway(role, temperature, prompt, schema, max_output_tokens=max_output_tokens)
        if len(self.requests) == self.fail_at:
            raise self.error
        return result

    def summary(self, *args):
        value = super().summary(*args)
        if len(self.requests) == self.fail_at and self.mutate is not None:
            self.mutate(value)
        return value


class _IncrementalHarness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = AtomicCheckpoint(Path(self.temporary.name) / "r7d-checkpoint.private.json")
        self.config = base._config()

    async def execute(self, manifest, provider, *, pairs=1, resume=False):
        authorization = base.GateAndReportTests()._authorization(manifest, self.config)
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", side_effect=provider.summary)):
            return await run_incremental_evaluation(
                manifest, self.config, authorization=authorization, checkpoint_store=self.store, pairs=pairs,
                gateway=provider.gateway, resume_after_rate_limit=resume,
                provenance_resolver=lambda: {"sha": "a" * 40, "source": "test"}, clock=lambda: 1_000.0)

    def assert_not_scored(self, checkpoint):
        record = checkpoint["records"][0]
        self.assertNotEqual("completed", record["status"])
        self.assertEqual(0, checkpoint["execution"]["stages"][-1]["completed_pairs"])
        report = build_public_report(checkpoint["records"], scores=None, manifest_sha256="f" * 64,
                                     config_sha256="e" * 64)
        self.assertEqual([False], [row["scorable"] for row in report["scenarios"]])
        self.assertEqual(0, report["arms"]["v15"]["failures"] + report["arms"]["v16"]["failures"])


class StopPrecedenceTests(_IncrementalHarness):
    async def test_invalid_structured_response_with_unknown_billing_on_the_second_arm_is_not_scored(self):
        provider = FaultyProvider(fail_at=2, error=InvalidStructuredResponse("invalid structured output"),
                                  mutate=_incomplete)
        checkpoint = await self.execute(base._manifest(), provider, pairs=2)
        self.assertEqual("billing_unknown", checkpoint["execution"]["stop_code"])
        first, second = checkpoint["records"][0]["arms"]
        self.assertEqual("completed", first["status"])
        self.assertEqual(("interrupted", "execution_stop"), (second["status"], second["failure_class"]))
        self.assertEqual({"phase": "initial", "code": "billing_unknown", "transmitted": True, "resumable": False},
                         second["interruption"])
        self.assertEqual(2, len(provider.requests))            # the second pair was never started
        self.assertEqual(1, len(checkpoint["records"]))
        self.assert_not_scored(checkpoint)
        with self.assertRaisesRegex(CheckpointError, r"stopped \(billing_unknown\)"):
            await self.execute(base._manifest(), provider, pairs=1)
        self.assertEqual(2, len(provider.requests))

    async def test_every_post_call_safety_stop_outranks_a_model_output_error(self):
        for code, mutate in SAFETY_STOPS:
            for fail_at in (1, 2):
                with self.subTest(code=code, arm=fail_at):
                    self.setUp()
                    provider = FaultyProvider(fail_at=fail_at, mutate=mutate,
                                              error=InvalidStructuredResponse("invalid structured output"))
                    checkpoint = await self.execute(base._manifest(), provider, pairs=2)
                    self.assertEqual(code, checkpoint["execution"]["stop_code"])
                    arm = checkpoint["records"][0]["arms"][fail_at - 1]
                    self.assertEqual("interrupted", arm["status"])
                    self.assertNotEqual("model_output", arm.get("failure_class"))
                    self.assertEqual(fail_at, len(provider.requests))
                    self.assert_not_scored(checkpoint)

    async def test_safety_stop_on_a_repair_response_outranks_its_model_output_error(self):
        provider = FaultyProvider(fail_at=2, error=InvalidStructuredResponse("invalid structured output"),
                                  mutate=_incomplete, initial_quotes=["This sentence is not in the passage."])
        checkpoint = await self.execute(base._manifest(), provider, pairs=1)
        arm = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("interrupted", "execution_stop", "repair"),
                         (arm["status"], arm["failure_class"], arm["failure_phase"]))
        self.assertEqual("billing_unknown", checkpoint["execution"]["stop_code"])
        self.assert_not_scored(checkpoint)

    async def test_model_output_error_with_clean_accounting_stays_a_scorable_failure(self):
        provider = FaultyProvider(fail_at=2, error=InvalidStructuredResponse("invalid structured output"))
        checkpoint = await self.execute(base._manifest(), provider, pairs=1)
        self.assertIsNone(checkpoint["execution"]["stop_code"])
        record = checkpoint["records"][0]
        self.assertEqual(["completed", "failed"], [arm["status"] for arm in record["arms"]])
        self.assertEqual("model_output", record["arms"][1]["failure_class"])
        self.assertEqual("completed", record["status"])
        self.assertEqual(1, checkpoint["execution"]["stages"][-1]["completed_pairs"])


class InfrastructureTerminationTests(_IncrementalHarness):
    async def test_a_non_429_provider_failure_with_complete_usage_stops_the_campaign(self):
        for fail_at in (1, 2):
            with self.subTest(arm=fail_at):
                self.setUp()
                provider = FaultyProvider(fail_at=fail_at, error=ConnectionResetError("connection reset by peer"))
                checkpoint = await self.execute(base._manifest(), provider, pairs=3)
                self.assertEqual(fail_at, len(provider.requests))   # no later request was sent
                self.assertEqual("provider_failure", checkpoint["execution"]["stop_code"])
                self.assertEqual(1, len(checkpoint["records"]))
                arm = checkpoint["records"][0]["arms"][fail_at - 1]
                self.assertEqual(("interrupted", "infrastructure"), (arm["status"], arm["failure_class"]))
                self.assertEqual("interrupted", checkpoint["records"][0]["status"])
                self.assert_not_scored(checkpoint)
                with self.assertRaisesRegex(CheckpointError, r"stopped \(provider_failure\)"):
                    await self.execute(base._manifest(), provider, pairs=1)
                with self.assertRaisesRegex(CheckpointError, r"stopped \(provider_failure\)"):
                    await self.execute(base._manifest(), provider, pairs=1, resume=True)
                self.assertEqual(fail_at, len(provider.requests))

    async def test_a_server_error_status_is_infrastructure_not_a_model_failure(self):
        def server_error(value):
            value["attempts"][0]["status_code"] = 503
        provider = FaultyProvider(fail_at=1, error=RuntimeError("service unavailable"), mutate=server_error)
        checkpoint = await self.execute(base._manifest(), provider, pairs=2)
        self.assertEqual(1, len(provider.requests))
        self.assertEqual("provider_failure", checkpoint["execution"]["stop_code"])
        self.assert_not_scored(checkpoint)

    def test_pair_status_counts_only_two_scorable_arms_as_completed(self):
        def arm(status):
            return {"prompt_version": "v15", "status": status}
        self.assertEqual("completed", _pair_status([arm("completed"), arm("failed")]))
        self.assertEqual("completed", _pair_status([arm("failed"), arm("failed")]))
        self.assertEqual("interrupted", _pair_status([arm("completed"), arm("interrupted")]))
        self.assertEqual("interrupted", _pair_status([arm("interrupted"), arm("completed")]))
        self.assertEqual("interrupted", _pair_status([arm("interrupted")]))
        self.assertEqual("partial", _pair_status([arm("completed")]))
        self.assertEqual("partial", _pair_status([]))


class OneShotRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def run_one_shot(self, provider):
        manifest, config = base._manifest(), base._config()
        authorization = base.GateAndReportTests()._authorization(manifest, config)
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", side_effect=provider.summary),
              patch("app.evaluation_report.resolve_git_provenance",
                    return_value={"sha": "a" * 40, "source": "test"})):
            return await run_paid_evaluation(manifest, config, authorization=authorization, gateway=provider.gateway)

    async def test_one_shot_safety_stop_outranks_a_model_output_error(self):
        provider = FaultyProvider(fail_at=2, error=InvalidStructuredResponse("invalid structured output"),
                                  mutate=_incomplete)
        records, execution = await self.run_one_shot(provider)
        self.assertIn("usage was incomplete", execution["stopped"])
        self.assertEqual(2, len(provider.requests))
        self.assertEqual(["completed", "interrupted"], [arm["status"] for arm in records[0]["arms"]])
        self.assertEqual("execution_stop", records[0]["arms"][1]["failure_class"])
        self.assertTrue(all(record["status"] == "not_executed" for record in records[1:]))

    async def test_one_shot_infrastructure_failure_stops_later_requests(self):
        provider = FaultyProvider(fail_at=1, error=ConnectionResetError("connection reset by peer"))
        records, execution = await self.run_one_shot(provider)
        self.assertEqual(1, len(provider.requests))
        self.assertIsNotNone(execution["stopped"])
        self.assertEqual(["interrupted", "not_executed"], [arm["status"] for arm in records[0]["arms"]])
        self.assertEqual("infrastructure", records[0]["arms"][0]["failure_class"])
        self.assertTrue(all(record["status"] == "not_executed" for record in records[1:]))


if __name__ == "__main__":
    unittest.main()
