"""R7D execution-safety correction: durable arms, calibrated admission, scoring integrity.

Deterministic and offline: fake structured gateway, synthetic provider trace summaries,
injected clock and sleep, real private checkpoint writes in a temporary directory.
No provider request and no private benchmark artifact.
"""
import asyncio
import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark.generation_evaluation import (
    ADMISSION_MARGIN,
    EvaluationStopped,
    build_blind_scoring_packet,
    build_public_report,
    canonical_sha256,
    dry_run,
    evaluate_pair,
    schema_bytes,
)
from app.benchmark.incremental_generation_evaluation import (
    UNRESOLVED_MESSAGE,
    AtomicCheckpoint,
    CheckpointError,
    _capacity_check,
    admission_estimate,
    run_incremental_evaluation,
)
from app.benchmark.production_verifier_replay import BudgetLedger
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch
from tests import test_r7d_execution_safety_reproduction as repro
from tests import test_r7d_generation_evaluation as base

ANSWER, REPAIR = "StructuredEvidenceAnswerV2", "V2RepairBatch"
INVALID_QUOTE = "This sentence is not in the passage."


class SimulatedCrash(BaseException):
    """Process death at a chosen checkpoint write (nothing after it runs)."""


class CrashingCheckpoint(AtomicCheckpoint):
    def __init__(self, path, predicate):
        super().__init__(path)
        self.predicate, self.fired, self.previous = predicate, False, None

    def write(self, value):
        if not self.fired and self.predicate(value, self.previous):
            self.fired = True
            raise SimulatedCrash()
        self.previous = copy.deepcopy(value)
        return super().write(value)


def _call(schema, tokens, request, *, complete=True, provider_calls=1):
    return {"schema": schema, "input_tokens": tokens, "request_bytes": request,
            "cost_status": "complete" if complete else "incomplete", "provider_calls": provider_calls}


def _state(headers, *, observed_at=1_000.0):
    from app.benchmark.incremental_generation_evaluation import _rate_limit_state
    return {"execution": {"calls": [{}], "rate_limit_state": _rate_limit_state(headers, observed_at)}}


class AdmissionEstimatorTests(unittest.TestCase):
    def test_uncalibrated_estimate_is_one_token_per_request_byte_and_not_an_upper_bound(self):
        estimate = admission_estimate([], prompt_bytes=5_000, schema_name=ANSWER, schema_size=1_000,
                                      max_output_tokens=2_000)
        self.assertEqual("uncalibrated_one_token_per_byte", estimate["method"])
        self.assertFalse(estimate["upper_bound"])
        self.assertEqual(6_000, estimate["estimated_input_tokens"])
        self.assertEqual(8_000, estimate["required"]["tokens"])
        self.assertEqual(2_000, estimate["required"]["output-tokens"])

    def test_same_schema_uses_its_own_highest_observed_ratio_with_margin(self):
        calls = [_call(ANSWER, 2_000, 8_000), _call(ANSWER, 3_000, 10_000), _call(REPAIR, 1_000, 10_000)]
        answer = admission_estimate(calls, prompt_bytes=9_000, schema_name=ANSWER, schema_size=1_000,
                                    max_output_tokens=2_000)
        repair = admission_estimate(calls, prompt_bytes=9_000, schema_name=REPAIR, schema_size=1_000,
                                    max_output_tokens=2_000)
        self.assertEqual("calibrated_same_schema", answer["method"])
        self.assertEqual(round(0.30 * 10_000 * (1 + ADMISSION_MARGIN)), answer["estimated_input_tokens"])
        self.assertEqual(round(0.10 * 10_000 * (1 + ADMISSION_MARGIN)), repair["estimated_input_tokens"])

    def test_unobserved_schema_counts_its_overhead_at_one_token_per_byte(self):
        calls = [_call(ANSWER, 2_500, 10_000)]
        estimate = admission_estimate(calls, prompt_bytes=8_000, schema_name=REPAIR, schema_size=1_300,
                                      max_output_tokens=2_000)
        self.assertEqual("calibrated_text_unobserved_schema_one_token_per_byte", estimate["method"])
        self.assertEqual(round(0.25 * 8_000 * (1 + ADMISSION_MARGIN)) + 1_300, estimate["estimated_input_tokens"])

    def test_unusable_observations_never_calibrate(self):
        calls = [_call(ANSWER, 100, 10_000, complete=False), _call(ANSWER, 100, 10_000, provider_calls=2),
                 _call(ANSWER, None, 10_000), _call(ANSWER, 0, 10_000), _call(ANSWER, 100, 0), "not a call"]
        estimate = admission_estimate(calls, prompt_bytes=5_000, schema_name=ANSWER, schema_size=1_000,
                                      max_output_tokens=2_000)
        self.assertEqual("uncalibrated_one_token_per_byte", estimate["method"])

    def test_a_denser_observation_raises_the_calibrated_ratio(self):
        sparse = admission_estimate([_call(ANSWER, 2_000, 10_000)], prompt_bytes=9_000, schema_name=ANSWER,
                                    schema_size=1_000, max_output_tokens=2_000)
        dense = admission_estimate([_call(ANSWER, 2_000, 10_000), _call(ANSWER, 4_000, 10_000)],
                                   prompt_bytes=9_000, schema_name=ANSWER, schema_size=1_000, max_output_tokens=2_000)
        self.assertGreater(dense["estimated_input_tokens"], sparse["estimated_input_tokens"])


class CapacityCheckTests(unittest.TestCase):
    def estimate(self, input_tokens=3_000, output=2_000):
        return {"required": {"requests": 1, "tokens": input_tokens + output, "input-tokens": input_tokens,
                             "output-tokens": output}}

    def test_combined_token_limit(self):
        self.assertIsNone(_capacity_check(_state(repro._headers(5_000)), self.estimate(), now=1_000.0,
                                          allow_elapsed_reset=False))
        refusal = _capacity_check(_state(repro._headers(4_999)), self.estimate(), now=1_000.0, allow_elapsed_reset=False)
        self.assertEqual(("rate_limit", ["tokens"]), (refusal["code"], refusal["blocked"]))

    def test_separate_input_and_output_token_limits(self):
        split = {"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999",
                 "x-ratelimit-reset-requests": "1h",
                 "x-ratelimit-limit-input-tokens": "6000", "x-ratelimit-remaining-input-tokens": "2999",
                 "x-ratelimit-reset-input-tokens": "5s",
                 "x-ratelimit-limit-output-tokens": "4000", "x-ratelimit-remaining-output-tokens": "4000",
                 "x-ratelimit-reset-output-tokens": "5s"}
        refusal = _capacity_check(_state(split), self.estimate(), now=1_000.0, allow_elapsed_reset=False)
        self.assertEqual(["input-tokens"], refusal["blocked"])
        output_short = dict(split, **{"x-ratelimit-remaining-input-tokens": "6000",
                                      "x-ratelimit-remaining-output-tokens": "1999"})
        refusal = _capacity_check(_state(output_short), self.estimate(), now=1_000.0, allow_elapsed_reset=False)
        self.assertEqual(["output-tokens"], refusal["blocked"])
        both_fit = dict(split, **{"x-ratelimit-remaining-input-tokens": "3000"})
        self.assertIsNone(_capacity_check(_state(both_fit), self.estimate(), now=1_000.0, allow_elapsed_reset=False))
        input_only = {key: value for key, value in split.items() if "output-tokens" not in key}
        refusal = _capacity_check(_state(input_only), self.estimate(), now=1_000.0, allow_elapsed_reset=False)
        self.assertEqual("rate_limit_capacity_unknown", refusal["code"])

    def test_account_specific_limit_below_requirement_fails_closed_after_reset(self):
        headers = dict(repro._headers(0), **{"x-ratelimit-limit-tokens": "4500"})
        refusal = _capacity_check(_state(headers), self.estimate(), now=1_100.0, allow_elapsed_reset=True)
        self.assertEqual("rate_limit_capacity_unknown", refusal["code"])

    def test_request_allowance_and_missing_metadata(self):
        no_requests = dict(repro._headers(8_000), **{"x-ratelimit-remaining-requests": "0"})
        self.assertEqual(["requests"], _capacity_check(_state(no_requests), self.estimate(), now=1_000.0,
                                                       allow_elapsed_reset=False)["blocked"])
        self.assertEqual("rate_limit_capacity_unknown",
                         _capacity_check(_state({}), self.estimate(), now=1_000.0, allow_elapsed_reset=False)["code"])


class ExecutionSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "r7d-checkpoint.private.json"
        self.store = AtomicCheckpoint(self.path)
        self.config = base._config()
        self.now = [1_000.0]
        self.sleeps = []

    async def _sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now[0] += seconds

    async def execute(self, manifest, provider, *, store=None, resume=False, wait=False, sleep=None, pairs=1):
        authorization = base.GateAndReportTests()._authorization(manifest, self.config)
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", side_effect=provider.summary)):
            return await run_incremental_evaluation(
                manifest, self.config, authorization=authorization, checkpoint_store=store or self.store,
                pairs=pairs, gateway=provider.gateway, resume_after_rate_limit=resume,
                provenance_resolver=lambda: {"sha": "a" * 40, "source": "test"}, clock=lambda: self.now[0],
                wait_for_token_reset=wait, sleep=sleep or self._sleep)

    def schemas(self, provider):
        return [schema.__name__ for _, schema in provider.requests]

    def assert_reconciled(self, checkpoint, provider):
        budget = checkpoint["execution"]["budget"]
        self.assertEqual(len(provider.requests), budget["provider_calls_charged"])
        self.assertEqual(len(provider.requests), len(checkpoint["execution"]["calls"]))
        self.assertAlmostEqual(sum(call["cost_usd"] for call in checkpoint["execution"]["calls"]
                                   if call["cost_status"] == "complete"), budget["cost_complete_usd"])

    def repair_pause_provider(self):
        """v15 needs a repair; only 1,000 tokens remain after the initial call."""
        return repro.FakeProvider(remaining_after=(1_000, repro.TPM_LIMIT), initial_quotes=[INVALID_QUOTE])

    # --- durable pending repair -------------------------------------------------------------

    async def test_pause_before_repair_keeps_a_durable_pending_repair(self):
        manifest = base._manifest()
        first = await self.execute(manifest, self.repair_pause_provider())
        active = first["active_arm"]
        self.assertEqual(("D001", "v15", "repair_pending"), (active["scenario_id"], active["prompt_version"],
                                                              active["phase"]))
        self.assertEqual(REPAIR, active["pending_operation"]["schema"])
        self.assertEqual(active["initial"]["output"], active["attempts"][0]["result"])
        self.assertEqual([], first["records"][0]["arms"])
        self.assertEqual("repair", first["execution"]["stop_context"]["phase"])
        self.assertIsNone(first["active_attempt"])
        self.assertEqual(0o600, stat.S_IMODE(self.path.stat().st_mode))

    async def test_resume_before_reset_sends_nothing(self):
        manifest, provider = base._manifest(), self.repair_pause_provider()
        await self.execute(manifest, provider)
        self.now[0] += 3
        checkpoint = await self.execute(manifest, provider, resume=True)
        self.assertEqual([ANSWER], self.schemas(provider))
        self.assertEqual("rate_limit", checkpoint["execution"]["stop_code"])
        self.assertEqual("repair_pending", checkpoint["active_arm"]["phase"])

    async def test_resume_after_reset_sends_only_the_repair_and_reconciles(self):
        manifest, provider = base._manifest(), self.repair_pause_provider()
        await self.execute(manifest, provider)
        self.now[0] += 60
        checkpoint = await self.execute(manifest, provider, resume=True)
        self.assertEqual([ANSWER, REPAIR, ANSWER], self.schemas(provider))
        continuation = checkpoint["execution"]["rate_limit_continuations"][0]
        self.assertEqual(("operator_resume_after_reset", "repair"), (continuation["kind"], continuation["next_phase"]))
        self.assertEqual("completed", checkpoint["records"][0]["status"])
        self.assert_reconciled(checkpoint, provider)

    async def test_a_resumed_arm_keeps_the_one_repair_maximum(self):
        manifest = base._manifest()
        provider = self.repair_pause_provider()
        original = provider.gateway

        async def still_invalid(role, temperature, prompt, schema, *, max_output_tokens):
            if schema is V2RepairBatch:
                provider.requests.append((prompt, schema))
                return V2RepairBatch(repairs=[{"target": "claims.1", "action": "replace",
                                               "replacement": base._claim(quote=INVALID_QUOTE)}])
            return await original(role, temperature, prompt, schema, max_output_tokens=max_output_tokens)
        provider.gateway = still_invalid
        await self.execute(manifest, provider)
        self.now[0] += 60
        checkpoint = await self.execute(manifest, provider, resume=True)
        v15 = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("failed", "model_output", "validation_failed_after_repair"),
                         (v15["status"], v15["failure_class"], v15["failure_kind"]))
        self.assertEqual(1, self.schemas(provider).count(REPAIR))

    # --- crashes -----------------------------------------------------------------------------

    async def test_crash_before_the_pending_repair_is_saved_rederives_it_without_repeating(self):
        manifest, provider = base._manifest(), repro.FakeProvider(initial_quotes=[INVALID_QUOTE])
        crashing = CrashingCheckpoint(self.path, lambda value, _: (value.get("active_arm") or {}).get("phase")
                                      == "repair_pending")
        with self.assertRaises(SimulatedCrash):
            await self.execute(manifest, provider, store=crashing)
        saved = AtomicCheckpoint(self.path).load()
        self.assertEqual("initial_pending", saved["active_arm"]["phase"])
        self.assertEqual(1, len(saved["active_arm"]["attempts"]))
        checkpoint = await self.execute(manifest, provider)
        self.assertEqual([ANSWER, REPAIR, ANSWER], self.schemas(provider))
        self.assertTrue(checkpoint["execution"]["stages"][-1]["resumed_active_arm"])
        self.assert_reconciled(checkpoint, provider)

    async def test_crash_after_the_repair_response_completes_the_arm_from_saved_results(self):
        manifest, provider = base._manifest(), repro.FakeProvider(initial_quotes=[INVALID_QUOTE])

        def arm_recorded(value, _):
            return value.get("active_arm") is None and any(arm.get("prompt_version") == "v15"
                                                            for record in value["records"] for arm in record["arms"])
        with self.assertRaises(SimulatedCrash):
            await self.execute(manifest, provider, store=CrashingCheckpoint(self.path, arm_recorded))
        self.assertEqual(2, len(AtomicCheckpoint(self.path).load()["active_arm"]["attempts"]))
        checkpoint = await self.execute(manifest, provider)
        self.assertEqual([ANSWER, REPAIR, ANSWER], self.schemas(provider))
        v15 = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("completed", True, 2), (v15["status"], v15["repair_attempted"], v15["physical_calls"]))
        self.assert_reconciled(checkpoint, provider)

    async def test_crash_with_a_possibly_transmitted_request_fails_closed(self):
        manifest = base._manifest()
        for phase_quotes, expected in (([], [ANSWER]), ([INVALID_QUOTE], [ANSWER, REPAIR])):
            with self.subTest(phase=expected[-1]):
                self.path.unlink(missing_ok=True)
                provider = repro.FakeProvider(initial_quotes=list(phase_quotes))
                target = len(expected)
                seen = []

                def after_request(value, previous):
                    if previous is not None and previous.get("active_attempt") and value.get("active_attempt") is None:
                        seen.append(1)
                        return len(seen) == target
                    return False
                with self.assertRaises(SimulatedCrash):
                    await self.execute(manifest, provider, store=CrashingCheckpoint(self.path, after_request))
                self.assertEqual("reserved", AtomicCheckpoint(self.path).load()["active_attempt"]["state"])
                with self.assertRaisesRegex(CheckpointError, "unresolved provider attempt"):
                    await self.execute(manifest, provider)
                with self.assertRaisesRegex(CheckpointError, "unresolved provider attempt"):
                    await self.execute(manifest, provider, resume=True)
                self.assertEqual(expected, self.schemas(provider))

    async def test_cancellation_during_a_repair_request_is_refused_on_resume(self):
        manifest = base._manifest()
        provider = repro.FakeProvider(initial_quotes=[INVALID_QUOTE])
        original, summary = provider.gateway, provider.summary

        async def cancel_repair(role, temperature, prompt, schema, *, max_output_tokens):
            if schema is V2RepairBatch:
                provider.requests.append((prompt, schema))
                raise asyncio.CancelledError()
            return await original(role, temperature, prompt, schema, max_output_tokens=max_output_tokens)

        def incomplete_for_repair(*args):
            value = summary(*args)
            if provider.requests[-1][1] is V2RepairBatch:
                value.update(token_usage_complete=False, estimated_cost_usd=None, input_tokens=None, output_tokens=None)
            return value
        provider.gateway, provider.summary = cancel_repair, incomplete_for_repair
        with self.assertRaises(asyncio.CancelledError):
            await self.execute(manifest, provider)
        self.assertEqual("cancelled", AtomicCheckpoint(self.path).load()["execution"]["stop_code"])
        with self.assertRaisesRegex(CheckpointError, "unresolved provider attempt"):
            await self.execute(manifest, provider)
        self.assertEqual([ANSWER, REPAIR], self.schemas(provider))

    # --- saved-state integrity ------------------------------------------------------------------

    async def test_tampered_pending_repair_state_is_refused_without_a_call(self):
        manifest = base._manifest()
        tampers = {
            "pending fingerprint": lambda arm: arm["pending_operation"].update(prompt_sha256="0" * 64),
            "validation errors": lambda arm: arm["initial"].update(validation_errors={"claims.1": ["edited"]}),
            "initial output": lambda arm: [item.update(text="An edited claim with a 99% rate.")
                                           for item in (arm["attempts"][0]["result"]["claims"][0],
                                                        arm["initial"]["output"]["claims"][0])],
            "unaudited attempt": lambda arm: arm["attempts"][0].update(prompt_sha256="1" * 64),
        }
        for name, tamper in tampers.items():
            with self.subTest(name):
                self.path.unlink(missing_ok=True)
                provider = self.repair_pause_provider()
                await self.execute(manifest, provider)
                checkpoint = AtomicCheckpoint(self.path).load()
                tamper(checkpoint["active_arm"])
                AtomicCheckpoint(self.path).write(checkpoint)   # valid checksum; tampered content
                self.now[0] += 60
                with self.assertRaises(CheckpointError):
                    await self.execute(manifest, provider, resume=True)
                self.assertEqual([ANSWER], self.schemas(provider))

    async def test_v1_checkpoint_is_refused_and_left_unchanged(self):
        legacy = {"schema": "scientific-rag-generation-eval-checkpoint/v1", "records": []}
        legacy["checkpoint_sha256"] = canonical_sha256(legacy)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        self.path.chmod(0o600)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(CheckpointError, "earlier accepted code version"):
            await self.execute(base._manifest(), repro.FakeProvider())
        self.assertEqual(before, self.path.read_bytes())

    # --- calibrated admission ------------------------------------------------------------------

    async def test_admission_is_calibrated_for_initial_and_repair_requests(self):
        manifest = repro._large_evidence_manifest()
        provider = repro.FakeProvider(remaining_after=(repro.TPM_LIMIT,), initial_quotes=[INVALID_QUOTE])
        checkpoint = await self.execute(manifest, provider)
        methods = [call["admission"]["method"] for call in checkpoint["execution"]["calls"]]
        self.assertEqual(["uncalibrated_one_token_per_byte",
                          "calibrated_text_unobserved_schema_one_token_per_byte",
                          "calibrated_same_schema"], methods)
        self.assertTrue(all(call["admission"]["upper_bound"] is False for call in checkpoint["execution"]["calls"]))
        self.assertEqual("completed", checkpoint["records"][0]["status"])

    async def test_insufficient_calibration_for_a_repair_fails_closed_without_a_call(self):
        manifest = repro._large_evidence_manifest()
        # The repair schema has not been observed yet, so its overhead counts at one token per
        # byte; with the calibrated text and the output limit that exceeds this account limit.
        limit = 4_500
        provider = repro.FakeProvider(remaining_after=(limit,), initial_quotes=[INVALID_QUOTE])
        provider_headers = repro._headers
        with patch.object(repro, "_headers", lambda remaining: dict(provider_headers(remaining),
                                                                    **{"x-ratelimit-limit-tokens": str(limit)})):
            first = await self.execute(manifest, provider)
            self.assertEqual("rate_limit", first["execution"]["stop_code"])
            self.now[0] += 60
            resumed = await self.execute(manifest, provider, resume=True)
        self.assertEqual("rate_limit_capacity_unknown", resumed["execution"]["stop_code"])
        admission = resumed["execution"]["stop_context"]["admission"]
        self.assertEqual("calibrated_text_unobserved_schema_one_token_per_byte", admission["method"])
        self.assertGreater(admission["required"]["tokens"], limit)
        self.assertEqual([ANSWER], self.schemas(provider))
        self.assertEqual("repair_pending", resumed["active_arm"]["phase"])
        self.assertEqual([], resumed["records"][0]["arms"])   # never counted as a failed answer

    async def test_an_underestimate_surfaces_as_a_terminal_429_without_retry(self):
        manifest = base._manifest()
        provider = repro.FakeProvider()
        summary = provider.summary

        def rejects_second(*args):
            value = summary(*args)
            if len(provider.requests) == 2:   # denser than calibrated: the provider rejects it
                value.update(token_usage_complete=False, estimated_cost_usd=None, input_tokens=None,
                             output_tokens=None)
                value["attempts"][0]["status_code"] = 429
            return value
        original = provider.gateway

        async def gateway(role, temperature, prompt, schema, *, max_output_tokens):
            result = await original(role, temperature, prompt, schema, max_output_tokens=max_output_tokens)
            if len(provider.requests) == 2:
                raise RuntimeError("rate limit exceeded")
            return result
        provider.gateway, provider.summary = gateway, rejects_second
        checkpoint = await self.execute(manifest, provider)
        self.assertEqual("provider_rate_limit", checkpoint["execution"]["stop_context"]["kind"])
        arm = checkpoint["records"][0]["arms"][1]
        self.assertEqual(("interrupted", "infrastructure"), (arm["status"], arm["failure_class"]))
        with self.assertRaisesRegex(CheckpointError, "pre-transmission capacity stop"):
            await self.execute(manifest, provider, resume=True)
        self.assertEqual(2, len(provider.requests))
        self.assertGreater(checkpoint["execution"]["budget"]["uncertain_reserved_usd"], 0)

    # --- opt-in in-process wait ------------------------------------------------------------------

    async def test_wait_is_disabled_by_default(self):
        await self.execute(base._manifest(), self.repair_pause_provider())
        self.assertEqual([], self.sleeps)

    async def test_opt_in_wait_covers_a_short_token_reset_and_rechecks(self):
        manifest, provider = base._manifest(), self.repair_pause_provider()
        checkpoint = await self.execute(manifest, provider, wait=True)
        self.assertEqual(1, len(self.sleeps))
        self.assertAlmostEqual(8.5, self.sleeps[0])
        self.assertEqual([ANSWER, REPAIR, ANSWER], self.schemas(provider))
        self.assertIsNone(checkpoint["execution"]["stop_code"])
        self.assertEqual("in_process_token_reset_wait", checkpoint["execution"]["rate_limit_continuations"][0]["kind"])
        self.assertTrue(checkpoint["execution"]["stages"][-1]["wait_for_token_reset"])

    async def test_opt_in_wait_never_covers_daily_or_long_resets(self):
        for name, headers in (("requests per day", {"x-ratelimit-remaining-requests": "0"}),
                              ("long token reset", {"x-ratelimit-reset-tokens": "90s"})):
            with self.subTest(name):
                self.path.unlink(missing_ok=True)
                self.sleeps.clear()
                provider = self.repair_pause_provider()
                original = repro._headers
                with patch.object(repro, "_headers", lambda remaining: dict(original(remaining), **headers)):
                    checkpoint = await self.execute(base._manifest(), provider, wait=True)
                self.assertEqual([], self.sleeps)
                self.assertEqual("rate_limit", checkpoint["execution"]["stop_code"])

    async def test_cancellation_during_the_wait_keeps_a_resumable_pending_repair(self):
        manifest, provider = base._manifest(), self.repair_pause_provider()

        async def cancelled_sleep(seconds):
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.execute(manifest, provider, wait=True, sleep=cancelled_sleep)
        saved = AtomicCheckpoint(self.path).load()
        self.assertEqual(("rate_limit", "preflight_capacity"),
                         (saved["execution"]["stop_code"], saved["execution"]["stop_context"]["kind"]))
        self.assertIn("waiting_until_unix", saved["execution"]["stop_context"])
        self.assertEqual("repair_pending", saved["active_arm"]["phase"])
        with self.assertRaisesRegex(CheckpointError, "stopped"):
            await self.execute(manifest, provider)   # the established continuation is required
        self.now[0] += 60
        checkpoint = await self.execute(manifest, provider, resume=True)
        self.assertEqual([ANSWER, REPAIR, ANSWER], self.schemas(provider))
        self.assertEqual("completed", checkpoint["records"][0]["status"])

    # --- failure classes ---------------------------------------------------------------------------

    async def test_budget_stop_before_a_repair_is_an_interruption_not_a_failure(self):
        manifest, provider = base._manifest(), repro.FakeProvider(initial_quotes=[INVALID_QUOTE])
        original = BudgetLedger.reserve
        count = [0]

        def second_reservation_refused(self, cost):
            count[0] += 1
            return original(self, cost) if count[0] == 1 else False
        with patch.object(BudgetLedger, "reserve", second_reservation_refused):
            checkpoint = await self.execute(manifest, provider)
        self.assertEqual([], checkpoint["records"][0]["arms"])   # not a failed (or any) terminal outcome
        self.assertEqual("repair_pending", checkpoint["active_arm"]["phase"])   # untransmitted, preserved
        self.assertEqual("budget", checkpoint["execution"]["stop_code"])
        with self.assertRaisesRegex(CheckpointError, r"stopped \(budget\)"):
            await self.execute(manifest, provider, resume=True)
        self.assertEqual([ANSWER], self.schemas(provider))

    async def test_incomplete_billing_after_a_repair_response_is_a_terminal_interruption(self):
        manifest, provider = base._manifest(), repro.FakeProvider(initial_quotes=[INVALID_QUOTE])
        summary = provider.summary

        def incomplete_repair(*args):
            value = summary(*args)
            if provider.requests[-1][1] is V2RepairBatch:
                value.update(token_usage_complete=False, estimated_cost_usd=None)
            return value
        provider.summary = incomplete_repair
        checkpoint = await self.execute(manifest, provider)
        arm = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("interrupted", True), (arm["status"], arm["interruption"]["transmitted"]))
        self.assertEqual("billing_unknown", checkpoint["execution"]["stop_code"])
        with self.assertRaises(CheckpointError):
            await self.execute(manifest, provider, resume=True)

    async def test_model_output_failures_are_valid_terminal_outcomes(self):
        from scientific_rag.llm_gateway import InvalidStructuredResponse

        async def invalid(*args, **kwargs):
            raise InvalidStructuredResponse("Provider returned invalid structured output")

        async def bad_repair_target(role, temperature, prompt, schema, *, max_output_tokens):
            if schema is V2RepairBatch:
                return V2RepairBatch(repairs=[{"target": "claims.2", "action": "drop", "unresolved_reason": "x"}])
            return StructuredEvidenceAnswerV2(claims=[base._claim(quote=INVALID_QUOTE)])
        for gateway, kind in ((invalid, "invalid_structured_output"), (bad_repair_target, "invalid_repair_response")):
            with self.subTest(kind):
                pair = await evaluate_pair(base._manifest()["scenarios"][0], config=self.config, gateway=gateway)
                self.assertTrue(all(arm["status"] == "failed" and arm["failure_class"] == "model_output"
                                    and arm["failure_kind"] == kind for arm in pair["arms"]))

    # --- paired scoring -----------------------------------------------------------------------------

    def test_only_valid_terminal_model_outcomes_enter_pairing(self):
        def arm(version, status, **extra):
            return {"prompt_version": version, "status": status, "repair_attempted": False, "physical_calls": 1,
                    "initial_output": {"claims": []}, "initial_structurally_valid": True, "usage": {}, **extra}
        records = [
            {"scenario_id": "D001", "arms": [arm("v15", "completed"), arm("v16", "failed",
                                                                            failure_class="model_output",
                                                                            failure_kind="invalid_repair_response",
                                                                            initial_structurally_valid=False)]},
            {"scenario_id": "D002", "arms": [arm("v15", "interrupted", failure_class="infrastructure",
                                                 failure_kind="gateway_failure"), arm("v16", "completed")]},
            {"scenario_id": "D003", "arms": [arm("v15", "interrupted", failure_class="execution_stop",
                                                 failure_kind="capacity_pause")]},
        ]
        report = build_public_report(records, scores=None, manifest_sha256="f" * 64, config_sha256="e" * 64)
        self.assertEqual([True, False, False], [row["scorable"] for row in report["scenarios"]])
        self.assertEqual({"pairs": 2, "arms_completed": 1, "arms_interrupted": 2, "arms_not_executed": 1},
                         report["excluded_from_pairing"])
        self.assertEqual({"structurally_valid": 1}, report["initial_generation"]["v15"])
        self.assertEqual({"structurally_invalid": 1}, report["initial_generation"]["v16"])
        self.assertEqual({"completed_without_repair": 1}, report["post_repair"]["v15"])
        self.assertEqual({"invalid_repair_response": 1}, report["post_repair"]["v16"])
        self.assertEqual((0, 1), (report["arms"]["v15"]["failures"], report["arms"]["v16"]["failures"]))
        self.assertEqual(2, report["arms"]["v15"]["interrupted"])
        with self.assertRaisesRegex(ValueError, "incomplete scenario"):
            build_public_report(records, scores={"D002": {"winner": "v16"}}, manifest_sha256="f" * 64,
                                config_sha256="e" * 64)
        manifest = base._manifest()
        rest = [{"scenario_id": item["scenario_id"], "arms": []} for item in manifest["scenarios"][3:]]
        packet, _ = build_blind_scoring_packet(manifest, records + rest)
        self.assertEqual([True, False, False], [case["scorable"] for case in packet["cases"][:3]])

    # --- frozen experiment contract --------------------------------------------------------------------

    def test_frozen_prompts_order_and_repair_maximum_are_unchanged(self):
        plan = dry_run(base._manifest(), self.config)
        contracts = {(item["scenario_id"], item["prompt_version"]): item["prompt_sha256"]
                     for item in plan["prompt_contracts"]}
        self.assertEqual("38a41de1fe4bcbe66009aac178bad647d26b6d2eac7f320763cf78a02f1ec1fe", contracts[("D001", "v15")])
        self.assertEqual("38f9c51b578023f5ecc5f569d0449eb026de77fe8274e4daf0a5b5af9eb2c15e", contracts[("D001", "v16")])
        self.assertEqual({"v15": 10, "v16": 10}, plan["first_arm_counts"])
        self.assertEqual((80, 40), (plan["max_physical_calls"], plan["max_repair_calls"]))
        self.assertEqual((0.25, False), (plan["capacity_admission"]["margin"],
                                         "upper bound" in plan["capacity_admission"]["method"].replace("not an upper bound", "")))
        self.assertGreater(schema_bytes(V2RepairBatch), 0)

    def test_dry_run_warns_when_uncalibrated_admission_exceeds_the_free_reference(self):
        plan = dry_run(repro._large_evidence_manifest(), self.config)
        self.assertTrue(plan["capacity_admission"]["warnings"])
        self.assertTrue(plan["eligible_after_exact_owner_authorization"])   # informational, not a blocker

    async def test_one_shot_runner_stop_is_an_interruption(self):
        async def stopped(*args, **kwargs):
            raise EvaluationStopped("authorized call or cost ceiling reached before the call", code="budget")
        pair = await evaluate_pair(base._manifest()["scenarios"][0], config=self.config, gateway=stopped)
        self.assertEqual(["interrupted", "not_executed"], [arm["status"] for arm in pair["arms"]])
        self.assertEqual("interrupted", pair["status"])


if __name__ == "__main__":
    unittest.main()
