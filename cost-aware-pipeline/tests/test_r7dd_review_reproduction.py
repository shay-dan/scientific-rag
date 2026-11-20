"""PR #26 independent review: reproductions of P0.1, P0.2, P1.1, P1.2 and P1.3.

Offline and deterministic: the scripted gateway from ``test_r7dd_acceptance`` and a
fault-injecting checkpoint store that stops the process (a BaseException) at a chosen durable
write. No provider is reached. Each test states the corrected behaviour; against ``d1d5a95``
these fail.
"""
import asyncio
import copy

from app.benchmark import acceptance_evaluation as acceptance
from app.benchmark.generation_evaluation import AuthorizationError, canonical_sha256
from app.benchmark.incremental_generation_evaluation import AtomicCheckpoint, CheckpointError
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch
from tests import test_r7dd_acceptance as base


class ProcessStopped(BaseException):
    """Simulates the process dying (not an ordinary exception: nothing may handle it)."""


class StoppingCheckpoint(AtomicCheckpoint):
    """Raises ProcessStopped instead of performing the first write matching ``when``."""

    def __init__(self, path, when):
        super().__init__(path)
        self.when = when

    def write(self, value):
        if self.when(value):
            self.when = lambda _value: False
            raise ProcessStopped()
        return super().write(value)


def finalizing(count):
    return lambda value: len(value.get("records") or []) >= count


def recording_repair_pending(value):
    return (value.get("active_arm") or {}).get("phase") == "repair_pending"


class FaultProvider(base.AcceptanceProvider):
    """Adds ``process_stop`` / ``cancel`` (during transmission) and ``schema_invalid`` actions."""

    async def gateway(self, role, temperature, prompt, schema, *, max_output_tokens, **settings):
        action = self.plan.get(len(self.requests) + 1)
        if action in ("process_stop", "cancel", "schema_invalid"):
            self.requests.append({"schema": schema, "max_output_tokens": max_output_tokens, "prompt": prompt,
                                  **settings})
            if action == "process_stop":
                raise ProcessStopped()
            if action == "cancel":
                raise asyncio.CancelledError()
            return {"claims": "not a list"}
        return await super().gateway(role, temperature, prompt, schema, max_output_tokens=max_output_tokens,
                                     **settings)


def _schemas(provider):
    return [request["schema"] for request in provider.requests]


class P01ContinuationNeverRetransmits(base._Harness):
    async def interrupted(self, provider, when, **kwargs):
        path = self.store.path
        self.store = StoppingCheckpoint(path, when)
        with self.assertRaises(ProcessStopped):
            await self.execute(provider, **kwargs)
        self.store = AtomicCheckpoint(path)

    async def test_1_settled_valid_initial_before_finalization(self):
        provider = FaultProvider()
        await self.interrupted(provider, finalizing(1))
        self.assertEqual(1, len(provider.requests))
        checkpoint = await self.execute(provider, operations=1)
        self.assertEqual(1, len(provider.requests), "the settled initial generation was transmitted again")
        record = checkpoint["records"][0]
        self.assertEqual((1, "completed", 1), (record["operation"], record["status"], record["physical_calls"]))

    async def test_2_settled_invalid_initial_before_pending_repair_recorded(self):
        provider = FaultProvider({1: "needs_repair"})
        await self.interrupted(provider, recording_repair_pending)
        self.assertEqual([StructuredEvidenceAnswerV2], _schemas(provider))
        checkpoint = await self.execute(provider, operations=1)
        self.assertEqual([StructuredEvidenceAnswerV2, V2RepairBatch], _schemas(provider),
                         "the settled invalid initial generation was transmitted again")
        self.assertEqual(("completed", 2), (checkpoint["records"][0]["status"],
                                            checkpoint["records"][0]["physical_calls"]))

    async def test_3_settled_conditional_repair_before_finalization(self):
        provider = FaultProvider({1: "needs_repair"})
        await self.interrupted(provider, finalizing(1))
        self.assertEqual(2, len(provider.requests))
        checkpoint = await self.execute(provider, operations=1)
        self.assertEqual(2, len(provider.requests), "the settled conditional repair was transmitted again")
        self.assertEqual(("completed", 2), (checkpoint["records"][0]["status"],
                                            checkpoint["records"][0]["physical_calls"]))

    async def test_4_settled_dedicated_repair_before_finalization(self):
        provider = FaultProvider()
        await self.execute(provider, operations=4)
        await self.interrupted(provider, finalizing(5))
        self.assertEqual(5, len(provider.requests))
        checkpoint = await self.execute(provider, operations=1)
        self.assertEqual(5, len(provider.requests), "the settled dedicated repair was transmitted again")
        record = checkpoint["records"][4]
        self.assertEqual((5, "dedicated_repair", True, 1),
                         (record["operation"], record["kind"], record["repair_schema_parsed"], record["physical_calls"]))

    async def test_5a_transmitted_attempt_without_a_result_is_refused(self):
        provider = FaultProvider({1: "process_stop"})
        with self.assertRaises(ProcessStopped):
            await self.execute(provider)
        with self.assertRaises(CheckpointError):
            await self.execute(provider)
        self.assertEqual(1, len(provider.requests))

    async def test_5b_transmitted_attempt_with_schema_invalid_output_is_refused(self):
        provider = FaultProvider({1: "schema_invalid"})
        await self.interrupted(provider, finalizing(1))
        with self.assertRaises(CheckpointError):
            await self.execute(provider)
        self.assertEqual(1, len(provider.requests))

    async def test_5c_tampered_active_operation_identity_is_refused(self):
        for field, value in (("configuration", "B"), ("prompt_version", "v16"), ("operation", 2),
                             ("kind", "dedicated_repair")):
            with self.subTest(field=field):
                self.setUp()
                provider = FaultProvider()
                await self.interrupted(provider, finalizing(1))
                checkpoint = self.store.load()
                checkpoint["active_arm"][field] = value
                self.store.write(checkpoint)
                with self.assertRaises(CheckpointError):
                    await self.execute(provider)
                self.assertEqual(1, len(provider.requests))

    async def test_6_cancellation_after_transmission_is_terminal(self):
        provider = FaultProvider({1: "cancel"})
        with self.assertRaises(asyncio.CancelledError):
            await self.execute(provider)
        self.assertEqual("cancelled", self.store.load()["execution"]["stop_code"])
        with self.assertRaises(CheckpointError):
            await self.execute(provider)
        self.assertEqual(1, len(provider.requests))


class P02ReferencesAreFrozen(base._Harness):
    def modified_references(self):
        value = base.references()
        value["predicates"]["A001"][0]["any_of"] = ["changed after generation"]
        value["references_sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "references_sha256"})
        return value

    def test_reference_changes_change_the_commitment_not_the_prompts(self):
        plain = acceptance.acceptance_dry_run(self.dataset, self.config, base.references())
        changed = acceptance.acceptance_dry_run(self.dataset, self.config, self.modified_references())
        self.assertEqual([item["initial_prompt_sha256"] for item in plain["operations"]],
                         [item["initial_prompt_sha256"] for item in changed["operations"]])
        self.assertNotEqual(plain["plan_sha256"], changed["plan_sha256"])

    async def test_authorization_must_bind_the_reference_hash(self):
        for value in (None, "0" * 64):
            with self.subTest(references_sha256=value):
                self.setUp()
                provider = FaultProvider()
                authorization = self.authorization()
                if value is None:
                    authorization.pop("references_sha256", None)
                else:
                    authorization["references_sha256"] = value
                with self.assertRaises(AuthorizationError):
                    await self.execute(provider, authorization=authorization)
                self.assertEqual([], provider.requests)

    async def test_report_refuses_post_hoc_reference_changes(self):
        checkpoint = await self.execute(FaultProvider())
        with self.assertRaises((CheckpointError, acceptance.AcceptanceError)):
            acceptance.build_acceptance_report(checkpoint, self.dataset, self.modified_references(), self.config)


class P11ReportBindings(base._Harness):
    async def asyncSetUp(self):
        self.checkpoint = await self.execute(FaultProvider())

    def refused(self, checkpoint=None, cfg=None):
        with self.assertRaises((CheckpointError, acceptance.AcceptanceError)):
            acceptance.build_acceptance_report(checkpoint or self.checkpoint, self.dataset, base.references(),
                                               cfg or self.config)

    def test_configuration_drift(self):
        self.refused(cfg=base.config(max_cost_usd=0.1))

    def test_dataset_drift(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["manifest_sha256"] = "1" * 64
        self.refused(checkpoint)

    def test_duplicate_and_unscheduled_records(self):
        duplicate = copy.deepcopy(self.checkpoint)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        self.refused(duplicate)
        outside = copy.deepcopy(self.checkpoint)
        outside["records"][0]["operation"] = 99
        self.refused(outside)

    def test_mismatched_prompt_version_or_configuration(self):
        for field, value in (("prompt_version", "v16"), ("configuration", "B")):
            with self.subTest(field=field):
                checkpoint = copy.deepcopy(self.checkpoint)
                checkpoint["records"][0][field] = value
                self.refused(checkpoint)


class P11TerminalFailureNeverSelects(base._Harness):
    async def test_interrupted_final_operation_blocks_selection(self):
        provider = base.AcceptanceProvider()
        original = provider.gateway

        async def gateway(role, temperature, prompt, schema, **kwargs):
            if len(provider.requests) == 13:                  # the 14th (final) request
                provider.requests.append({"schema": schema, "prompt": prompt, **kwargs})
                raise ConnectionError("simulated connection reset after transmission")
            return await original(role, temperature, prompt, schema, **kwargs)
        provider.gateway = gateway
        checkpoint = await self.execute(provider)
        self.assertEqual(14, len(checkpoint["records"]))
        self.assertEqual("interrupted", checkpoint["records"][-1]["status"])
        report = acceptance.build_acceptance_report(checkpoint, self.dataset, base.references(), self.config)
        self.assertEqual("inconclusive", report["selection"]["decision"])
        cell = report["cells"]["A/v15"]
        self.assertEqual(2 * len(base.references()["predicates"]["A001"]) and cell["predicates_evaluated"],
                         cell["predicates_evaluated"])
        completed = [record for record in checkpoint["records"]
                     if record["kind"] == "arm" and record["configuration"] == "A" and record["prompt_version"] == "v15"
                     and record["status"] == "completed"]
        expected = sum(len(base.references()["predicates"][record["scenario_id"]]) for record in completed)
        self.assertEqual(expected, cell["predicates_evaluated"], "an interrupted arm was scored as an observation")


class P12PredicateDefects(base._Harness):
    def results(self, answer, scenario="A001"):
        return {item["id"]: item["passed"]
                for item in acceptance.evaluate_predicates(answer, base.references()["predicates"][scenario])}

    def test_absent_answer_earns_no_credit(self):
        for scenario in ("A001", "A002", "A003"):
            with self.subTest(scenario=scenario):
                self.assertFalse(any(self.results(None, scenario).values()))

    def test_asserting_an_unsupported_endpoint_is_not_abstention(self):
        answer = copy.deepcopy(base.VALID["KESTREL-3"])
        answer["limitations"] = []
        answer["unanswered_parts"] = []
        answer["claims"].append(base.claim("Diastolic pressure fell by 6.1 mmHg with velarozan.", "S1", "x" * 8))
        self.assertFalse(self.results(answer)["A001-A1"])
        cardiovascular = copy.deepcopy(base.VALID["ORCHID-1"])
        cardiovascular["unanswered_parts"] = []
        cardiovascular["limitations"] = []
        cardiovascular["claims"].append(base.claim("Tralvekimab lowered cardiovascular events.", "S1", "x" * 8))
        self.assertFalse(self.results(cardiovascular, "A003")["A003-A1"])

    def test_swapped_treatment_results_fail_numeric_checks(self):
        answer = copy.deepcopy(base.VALID["KESTREL-3"])
        answer["claims"][0] = base.claim("Placebo lowered systolic blood pressure by 14.2 mmHg versus 4.1 mmHg "
                                         "with velarozan.", "S1", "x" * 8)
        results = self.results(answer)
        self.assertEqual((False, False), (results["A001-F1"], results["A001-F2"]))
        swapped = copy.deepcopy(base.VALID["ORCHID-1"])
        swapped["claims"][0] = base.claim("At week 16, PASI 90 was reached by 71% with placebo versus 4% with "
                                          "tralvekimab.", "S1", "x" * 8)
        self.assertFalse(self.results(swapped, "A003")["A003-F1"])


class P13MatrixIsEnforced(base._Harness):
    def test_out_of_scope_configurations_are_refused(self):
        bad = {
            "temperature": base.config(temperature=0.7),
            "other model": base.config(model="openai/gpt-oss-20b"),
            "medium reasoning on both": base.config(configurations=[
                {"label": "A", "reasoning_effort": "medium", "max_output_tokens": 2000},
                {"label": "B", "reasoning_effort": "medium", "max_output_tokens": 3000}]),
            "other ceilings": base.config(configurations=[
                {"label": "A", "reasoning_effort": "low", "max_output_tokens": 1000},
                {"label": "B", "reasoning_effort": "low", "max_output_tokens": 4000}]),
            "claim audit on": base.config(claim_audit_mode="audit"),
            "expanded budget": base.config(max_cost_usd=0.5),
            "other failure policy": base.config(completion_budget_failure_policy="stop"),
        }
        for label, value in bad.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                acceptance.validate_acceptance_config(value)

    def test_a_smaller_budget_is_permitted(self):
        acceptance.validate_acceptance_config(base.config(max_cost_usd=0.10))
