"""R7D-C independent review of PR #25 at f1c58f6: reproductions and regressions.

P0.1 duplicate/stale checkpoint execution under one authorization; P0.2 baseline mutation
while a campaign is open; P0.3 safety stops overriding completion-budget scoring; P1.1
real-provider entry points that bypass cumulative accounting; P1.2 reservation basis and
fail-closed settlement; P1.3 adversarial diagnostic sanitization; P2 open-campaign
observability and reasoning-token telemetry.

Zero provider calls: scripted gateways, synthetic traces, real private files in temporary
directories.
"""
import asyncio
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from app.benchmark import generation_evaluation as harness
from app.benchmark.generation_evaluation import schema_bytes
from app.benchmark.incremental_generation_evaluation import (
    AtomicCheckpoint,
    CampaignLocked,
    CheckpointError,
    run_incremental_evaluation,
)
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch
from scientific_rag import provider_failures as pf
from tests import test_r7d_generation_evaluation as base
from tests import test_r7dc_generation_reliability as r7dc
from tests.test_r7dc_generation_reliability import (
    PRIVATE_TEXT,
    SECRET,
    FakeStatusError,
    ScriptedProvider,
    _CampaignHarness,
    _call,
    _historical_checkpoint,
    _trace_patches,
    ledger_module,
    truncation_error,
    v2_authorization,
    v2_config,
)


class SimulatedCrash(BaseException):
    """Process death after the reservation was written and before settlement."""


class MutatingProvider(ScriptedProvider):
    """Scripted provider whose trace summary for request N is changed by ``mutations[N]``."""

    def __init__(self, actions=(), mutations=None):
        super().__init__(actions)
        self.mutations = mutations or {}

    def summary(self, *args):
        value = super().summary(*args)
        mutate = self.mutations.get(len(self.requests))
        if mutate is not None:
            mutate(value)
        return value


def _retry(value):
    value["provider_calls"] = 2


def _drift(value):
    value["by_model"] = {"groq:another-model": {}}


def _over_ceiling(value):
    value.update(token_usage_complete=True, estimated_cost_usd=0.0015, input_tokens=700, output_tokens=2400)


def _cost(amount):
    def mutate(value):
        value.update(token_usage_complete=True, estimated_cost_usd=amount, input_tokens=700, output_tokens=600)
    return mutate


class _Corrections(_CampaignHarness):
    def authorization(self, ledger_store, config=None, **kwargs):
        config = config or v2_config()
        baseline = ledger_module().program_baseline(ledger_store.load())
        return v2_authorization(self.manifest, config, baseline_sha256=baseline["sha256"], **kwargs)


# --------------------------------------------------------------------------- P0.1

class DuplicateExecutionTests(_Corrections):
    async def test_same_authorization_with_a_new_checkpoint_path_is_refused(self):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        await self.run_v2(provider, self.checkpoint("A.json"), ledger_store, authorization=authorization)
        self.assertEqual(2, len(provider.requests))
        with self.assertRaisesRegex(ledger_module().ProgramLedgerError, "authoritative checkpoint"):
            await self.run_v2(provider, self.checkpoint("B.json"), ledger_store, authorization=authorization)
        self.assertEqual(2, len(provider.requests))
        self.assertFalse((self.directory / "B.json").exists())

    async def test_concurrent_execution_through_two_paths_is_refused(self):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        await self.run_v2(provider, self.checkpoint("A.json"), ledger_store, authorization=authorization)
        with ledger_store.campaign_lock(authorization["authorization_id"]):   # another process runs A
            with self.assertRaises(CampaignLocked):
                await self.run_v2(provider, self.checkpoint("A.json"), ledger_store, authorization=authorization)
            with self.assertRaises((CampaignLocked, ledger_module().ProgramLedgerError)):
                await self.run_v2(provider, self.checkpoint("B.json"), ledger_store, authorization=authorization)
        self.assertEqual(2, len(provider.requests))

    async def test_an_older_copy_of_the_checkpoint_is_refused_at_any_path(self):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        path = self.directory / "A.json"
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        old = path.read_bytes()
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        self.assertEqual(4, len(provider.requests))
        copy = self.directory / "A-copy.json"
        copy.write_bytes(old)
        os.chmod(copy, 0o600)
        with self.assertRaisesRegex(ledger_module().ProgramLedgerError, "authoritative checkpoint"):
            await self.run_v2(provider, AtomicCheckpoint(copy), ledger_store, authorization=authorization)
        path.write_bytes(old)                                            # restored backup at the same path
        with self.assertRaisesRegex(ledger_module().ProgramLedgerError, "stale"):
            await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        self.assertEqual(4, len(provider.requests))

    async def test_close_requires_the_current_authoritative_checkpoint(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        path = self.directory / "A.json"
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        old = path.read_bytes()
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        copy = self.directory / "A-copy.json"
        shutil.copyfile(path, copy)
        os.chmod(copy, 0o600)
        with self.assertRaisesRegex(module.ProgramLedgerError, "authoritative checkpoint"):
            module.close_campaign(ledger_store, AtomicCheckpoint(copy))
        current = path.read_bytes()
        path.write_bytes(old)
        with self.assertRaisesRegex(module.ProgramLedgerError, "stale"):
            module.close_campaign(ledger_store, AtomicCheckpoint(path))
        path.write_bytes(current)
        closed = module.close_campaign(ledger_store, AtomicCheckpoint(path))
        self.assertEqual(4, closed["consumed"]["provider_calls_charged"])

    async def test_crash_after_registration_is_recoverable_only_through_the_registered_path(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        config = v2_config()
        authorization = self.authorization(ledger_store, config)
        module.open_campaign(ledger_store, authorization=authorization, config=config,
                             manifest_sha256=self.manifest["manifest_sha256"],
                             checkpoint_path=self.directory / "A.json")
        provider = ScriptedProvider()
        with self.assertRaisesRegex(module.ProgramLedgerError, "authoritative checkpoint"):
            await self.run_v2(provider, self.checkpoint("B.json"), ledger_store, authorization=authorization)
        checkpoint = await self.run_v2(provider, self.checkpoint("A.json"), ledger_store, authorization=authorization)
        self.assertEqual(2, checkpoint["execution"]["budget"]["provider_calls_charged"])

    async def test_a_lost_checkpoint_after_progress_is_never_recreated_but_can_be_closed_from_its_mark(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        path = self.directory / "A.json"
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        path.unlink()
        with self.assertRaisesRegex(module.ProgramLedgerError, "missing"):
            await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        closed = module.close_lost_campaign(ledger_store, authorization["authorization_id"])
        self.assertTrue(closed["closed_from_ledger_mark"])
        self.assertEqual(2, closed["consumed"]["provider_calls_charged"])
        self.assertGreater(closed["consumed"]["budget_charged_usd"], 0.0)


# --------------------------------------------------------------------------- P0.2

class BaselineConsistencyTests(_Corrections):
    async def test_historical_imports_are_refused_while_a_campaign_is_open(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        await self.run_v2(ScriptedProvider(), self.checkpoint("A.json"), ledger_store)
        history = self.directory / "history.private.json"
        _historical_checkpoint(history, authorization_id="late-history", calls=[_call(complete=False, reserved=0.001)])
        with self.assertRaisesRegex(module.ProgramLedgerError, "open"):
            module.import_campaign(ledger_store, history)
        self.assertEqual(0, module.program_baseline(ledger_store.load())["closed_campaigns"])

    async def test_unresolved_reservations_survive_a_crash_and_campaign_closure(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        store = self.checkpoint("A.json")
        provider = ScriptedProvider()
        with patch("scientific_rag.llm_gateway.begin_trace", side_effect=SimulatedCrash()), \
                self.assertRaises(SimulatedCrash):
            config = v2_config()
            authorization = self.authorization(ledger_store, config)
            with patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)), \
                    patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"):
                await run_incremental_evaluation(
                    self.manifest, config, authorization=authorization, checkpoint_store=store, pairs=1,
                    gateway=provider.gateway, provenance_resolver=lambda: {"sha": config["code_sha"], "source": "t"},
                    clock=lambda: 1_000.0, program_ledger=ledger_store)
        reserved = store.load()["active_attempt"]["reserved_cost_usd"]
        with self.assertRaisesRegex(CheckpointError, "unresolved"):
            await self.run_v2(provider, store, ledger_store, authorization=authorization)
        closed = module.close_campaign(ledger_store, store)
        self.assertTrue(closed["unresolved_attempt"])
        self.assertEqual(1, closed["consumed"]["provider_calls_charged"])
        self.assertAlmostEqual(reserved, closed["consumed"]["uncertain_reserved_usd"], places=8)
        self.assertEqual([], provider.requests)


# --------------------------------------------------------------------------- P0.3

class SafetyStopPrecedenceTests(_Corrections):
    async def check(self, mutate, expected_stop):
        ledger_store = self.initialized_ledger()
        provider = MutatingProvider(["truncate"], {1: mutate} if mutate else {})
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store, pairs=2)
        arm = checkpoint["records"][0]["arms"][0]
        return checkpoint, arm, provider

    async def test_truncation_with_an_execution_integrity_failure_is_never_scored(self):
        for label, mutate, code in (("retry", _retry, "unexpected_retry"), ("drift", _drift, "model_drift"),
                                    ("output ceiling", _over_ceiling, "output_limit")):
            with self.subTest(label=label):
                self.setUp()
                checkpoint, arm, provider = await self.check(mutate, code)
                self.assertEqual(code, checkpoint["execution"]["stop_code"])
                self.assertEqual(("interrupted", harness.EXECUTION_STOP), (arm["status"], arm["failure_class"]))
                self.assertNotEqual("completion_budget_exhausted", arm.get("failure_kind"))
                self.assertEqual("interrupted", checkpoint["records"][0]["status"])
                self.assertEqual(("excluded", "stop_terminal"),
                                 (arm["outcome_dimensions"]["scoring"], arm["outcome_dimensions"]["continuation"]))
                self.assertEqual(1, len(provider.requests))
                self.assertEqual(1, checkpoint["execution"]["budget"]["provider_calls_charged"]
                                 - (1 if label == "retry" else 0))

    async def test_truncation_with_only_incomplete_usage_is_a_scored_failure_with_its_reservation(self):
        checkpoint, arm, provider = await self.check(None, None)
        self.assertIsNone(checkpoint["execution"]["stop_code"])
        self.assertEqual(("failed", "completion_budget_exhausted"), (arm["status"], arm["failure_kind"]))
        self.assertEqual("completed", checkpoint["records"][0]["status"])
        self.assertEqual("uncertain_reserved", arm["outcome_dimensions"]["accounting"])
        self.assertGreater(checkpoint["execution"]["budget"]["uncertain_reserved_usd"], 0.0)
        self.assertEqual(4, len(provider.requests))


# --------------------------------------------------------------------------- P1.1

class RealProviderEntryPointTests(_Corrections):
    async def test_the_real_gateway_cannot_run_without_v2_and_the_program_ledger(self):
        from scientific_rag import llm_gateway

        async def must_not_run(*args, **kwargs):
            raise AssertionError("the real provider gateway was reached")
        config = base._config()
        authorization = base.GateAndReportTests()._authorization(self.manifest, config)
        with patch.object(llm_gateway, "ainvoke_structured", must_not_run), \
                patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)), \
                patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"), \
                patch("app.evaluation_report.resolve_git_provenance", return_value={"sha": "a" * 40, "source": "t"}):
            for gateway in (None, llm_gateway.ainvoke_structured):
                with self.subTest(gateway=gateway), self.assertRaisesRegex(harness.AuthorizationError, "program ledger"):
                    await harness.run_paid_evaluation(self.manifest, config, authorization=authorization,
                                                      gateway=gateway)
                with self.subTest(gateway=gateway), self.assertRaisesRegex(harness.AuthorizationError, "program ledger"):
                    await run_incremental_evaluation(
                        self.manifest, config, authorization=authorization, checkpoint_store=self.checkpoint(),
                        pairs=1, gateway=gateway, provenance_resolver=lambda: {"sha": "a" * 40, "source": "t"})
        self.assertFalse((self.directory / "r7d-checkpoint.private.json").exists())

    async def test_an_internal_error_before_transmission_fails_closed(self):
        def broken_clock():
            raise RuntimeError("clock failure")
        provider = ScriptedProvider()
        config = base._config()
        authorization = base.GateAndReportTests()._authorization(self.manifest, config)
        patches = _trace_patches(provider)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            checkpoint = await run_incremental_evaluation(
                self.manifest, config, authorization=authorization, checkpoint_store=self.checkpoint(), pairs=1,
                gateway=provider.gateway, provenance_resolver=lambda: {"sha": "a" * 40, "source": "t"},
                clock=broken_clock)
        self.assertEqual("internal_error", checkpoint["execution"]["stop_code"])
        self.assertEqual([], provider.requests)
        self.assertEqual(0, checkpoint["execution"]["budget"]["provider_calls_charged"])

    async def test_injected_test_gateways_keep_historical_v1_behavior(self):
        checkpoint = await self.run_v1(ScriptedProvider(), self.checkpoint())
        self.assertEqual(2, checkpoint["execution"]["budget"]["provider_calls_charged"])


# --------------------------------------------------------------------------- P1.2

class ReservationIntegrityTests(_Corrections):
    async def test_v2_reservations_include_the_schema_and_request_framing(self):
        from scientific_rag.costing import calculate_cost
        ledger_store = self.initialized_ledger()
        provider = MutatingProvider(["needs_repair"])
        store = self.checkpoint()
        checkpoint = await self.run_v2(provider, store, ledger_store)
        calls = checkpoint["execution"]["calls"]
        for call, schema in zip(calls, (StructuredEvidenceAnswerV2, V2RepairBatch)):
            basis = call["prompt_bytes"] + schema_bytes(schema) + harness.RESERVATION_FRAMING_TOKENS
            self.assertEqual(basis, call["reservation_input_tokens"])
            expected, _ = calculate_cost("groq", "openai/gpt-oss-120b", basis, 2000)
            self.assertAlmostEqual(expected, call["reserved_cost_usd"], places=12)

    async def test_v1_reservation_basis_is_unchanged(self):
        from scientific_rag.costing import calculate_cost
        checkpoint = await self.run_v1(ScriptedProvider(), self.checkpoint())
        call = checkpoint["execution"]["calls"][0]
        expected, _ = calculate_cost("groq", "openai/gpt-oss-120b", call["prompt_bytes"], 2000)
        self.assertAlmostEqual(expected, call["reserved_cost_usd"], places=12)

    async def test_measured_cost_above_its_reservation_stops_and_is_preserved(self):
        ledger_store = self.initialized_ledger()
        provider = MutatingProvider(["ok"], {1: _cost(0.05)})
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store, pairs=2)
        self.assertEqual("reservation_exceeded", checkpoint["execution"]["stop_code"])
        self.assertAlmostEqual(0.05, checkpoint["execution"]["budget"]["cost_complete_usd"], places=8)
        self.assertEqual("interrupted", checkpoint["records"][0]["arms"][0]["status"])
        self.assertEqual(1, len(provider.requests))

    async def test_settlement_beyond_the_cumulative_ceiling_stops_and_is_preserved(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        history = self.directory / "history.private.json"
        _historical_checkpoint(history, authorization_id="history", calls=[_call(complete=True, reserved=1.49, cost=1.49)])
        module.import_campaign(ledger_store, history)
        provider = MutatingProvider(["ok"], {1: _cost(0.009)})
        store = self.checkpoint()
        checkpoint = await self.run_v2(provider, store, ledger_store, pairs=2)
        self.assertIn(checkpoint["execution"]["stop_code"], {"reservation_exceeded", "cost_ceiling_exceeded"})
        self.assertGreater(checkpoint["execution"]["budget"]["budget_charged_usd"], 1.49)
        closed = module.close_campaign(ledger_store, store)
        self.assertAlmostEqual(0.009, closed["consumed"]["cost_complete_usd"], places=8)


# --------------------------------------------------------------------------- P1.3

class AdversarialSanitizationTests(unittest.TestCase):
    def test_codes_types_request_ids_and_nested_causes_never_carry_secrets(self):
        body = {"message": f"{SECRET} {PRIVATE_TEXT}", "type": SECRET[:40], "code": "gsk_" + "AbCd1234" * 4,
                "failed_generation": PRIVATE_TEXT}
        error = FakeStatusError(400, body, request_id=SECRET[:60])
        try:
            try:
                raise ValueError(f"inner {SECRET} {PRIVATE_TEXT}")
            except ValueError as inner:
                raise error from inner
        except FakeStatusError as outer:
            description = pf.describe_provider_failure(outer, StructuredEvidenceAnswerV2)
        text = json.dumps(description)
        for forbidden in (SECRET[:20], PRIVATE_TEXT, "gsk_AbCd"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(("other", "other", None),
                         (description["provider_error_code"], description["provider_error_type"],
                          description["request_id"]))
        known = pf.describe_provider_failure(truncation_error(), StructuredEvidenceAnswerV2)
        self.assertEqual(("json_validate_failed", "invalid_request_error"),
                         (known["provider_error_code"], known["provider_error_type"]))
        groq_id = FakeStatusError(429, {"code": "rate_limit_exceeded"}, request_id="req_01jabcdefghjkmnpqrstvwxyz0")
        self.assertEqual("req_01jabcdefghjkmnpqrstvwxyz0", pf.describe_provider_failure(groq_id)["request_id"])


class LocalFailureSanitizationTests(_Corrections):
    async def test_local_structured_output_errors_are_not_stored_verbatim(self):
        ledger_store = self.initialized_ledger()
        store = self.checkpoint()
        await self.run_v2(ScriptedProvider(["local_invalid"]), store, ledger_store)
        text = store.path.read_text()
        self.assertNotIn(PRIVATE_TEXT, json.dumps(store.load()["records"][0]["arms"][0].get("error")))
        self.assertNotIn(SECRET, text)


# --------------------------------------------------------------------------- P2

class ObservabilityTests(_Corrections):
    async def test_summary_shows_open_campaign_consumption_separately(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        history = self.directory / "history.private.json"
        _historical_checkpoint(history, authorization_id="history", calls=[_call(complete=False, reserved=0.001)])
        module.import_campaign(ledger_store, history)
        await self.run_v2(ScriptedProvider(), self.checkpoint(), ledger_store)
        summary = module.accounting_summary(ledger_store.load())
        self.assertEqual(1, summary["closed_baseline"]["provider_calls_charged"])
        self.assertEqual(2, summary["current_experiment"]["provider_calls_charged"])
        self.assertEqual("ledger_mark", summary["current_experiment"]["basis"])
        self.assertEqual(3, summary["cumulative"]["provider_calls_charged"])
        self.assertEqual(77, summary["cumulative"]["remaining_calls"])
        self.assertFalse(summary["diagnostics"]["counted_against_benchmark_ceilings"])

    async def test_reasoning_tokens_are_recorded_only_when_reported(self):
        def reported(value):
            value["attempts"][0]["diagnostics"] = {"reasoning_tokens": 610, "finish_reason": "stop"}
        ledger_store = self.initialized_ledger()
        checkpoint = await self.run_v2(MutatingProvider([], {1: reported}), self.checkpoint(), ledger_store)
        first, second = checkpoint["execution"]["calls"]
        self.assertEqual((610, "stop"), (first["reasoning_tokens"], first["finish_reason"]))
        self.assertEqual((None, None), (second["reasoning_tokens"], second["finish_reason"]))


if __name__ == "__main__":
    unittest.main()
