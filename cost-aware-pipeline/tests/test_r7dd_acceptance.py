"""R7D-D configuration acceptance harness: offline, deterministic, zero provider calls.

A scripted structured gateway returns genuinely valid (or deliberately repairable) answers
for the three synthetic scenarios; trace summaries are synthetic. Real private checkpoint and
acceptance-ledger files are written in temporary directories. The real SDK/HTTP serialization
of low reasoning with 2,000 and 3,000 completion tokens (initial and repair schemas) is
covered by ``test_r7dc_http_contract.py`` in the production suite.
"""
import asyncio
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark import acceptance_evaluation as acceptance
from app.benchmark import program_ledger as programs
from app.benchmark.generation_evaluation import AuthorizationError, canonical_sha256
from app.benchmark.incremental_generation_evaluation import AtomicCheckpoint, CheckpointError
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch
from tests import test_r7d_execution_safety_reproduction as repro
from tests.test_r7dc_generation_reliability import truncation_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "r7d-d"
CODE_SHA = "e025a3d36350edfaedb38988993c5f96f593a647"
GROQ = "groq:openai/gpt-oss-120b"


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def dataset():
    return load("synthetic-acceptance-v1.json")


def references():
    return load("synthetic-acceptance-references-v1.json")


def config(**updates):
    value = {"schema": acceptance.CONFIG_SCHEMA, "code_sha": CODE_SHA, "provider": "groq",
             "model": "openai/gpt-oss-120b", "temperature": 0, "prompt_versions": ["v15", "v16"],
             "model_context_tokens": 131072, "max_input_tokens": 100000, "max_calls": 26, "max_cost_usd": 0.15,
             "semantic_verification_mode": "off", "claim_recovery_mode": "strict", "claim_audit_mode": "off",
             "completion_budget_failure_policy": "score_as_failure",
             "configurations": [{"label": "A", "reasoning_effort": "low", "max_output_tokens": 2000},
                                {"label": "B", "reasoning_effort": "low", "max_output_tokens": 3000}]}
    value.update(updates)
    return value


def claim(text, source, quote):
    return {"text": text, "source_id": source, "evidence_quote": quote}


VALID = {
    "KESTREL-3": {"claims": [
        claim("Velarozan lowered systolic blood pressure by 14.2 mmHg versus 4.1 mmHg with placebo.", "S1",
              "mean systolic blood pressure fell by 14.2 mmHg (95% CI 12.9 to 15.5) with velarozan"),
        claim("Stroke occurred in 18 of 1,206 (1.5%) with velarozan versus 31 of 1,206 (2.6%) with placebo; the "
              "hazard ratio of 0.58 had a "
              "confidence interval up to 1.03, so the stroke effect was not statistically significant.", "S1",
              "the hazard ratio was 0.58 (95% CI 0.33 to 1.03)"),
        claim("The separate Arden registry, an observational cohort the authors cite, reported 32% lower stroke "
              "incidence among velarozan users.", "S2", "reported 32% lower stroke incidence among velarozan users")],
        "limitations": [claim("Diastolic results were not reported in the available evidence.", "S3",
                              "Diastolic blood pressure was a secondary outcome whose results will be reported separately")],
        "unanswered_parts": ["The diastolic blood-pressure reduction is not reported."]},
    "TWILIGHT": {"claims": [
        claim("In TWILIGHT, delirium occurred in 11.9% with melzorin and 14.4% with placebo (risk ratio 0.83, "
              "95% CI 0.56 to 1.22), which is not statistically significant.", "S1",
              "Delirium occurred in 11.9% of the melzorin group and 14.4% of the placebo group"),
        claim("The observational Northfield cohort found lower adjusted odds of delirium (0.62), but users were "
              "younger and less frail, so confounding is possible.", "S2",
              "adjusted odds ratio 0.62, 95% CI 0.51 to 0.75")],
        "limitations": [claim("No study has compared melzorin directly with the ward protocol.", "S3",
                              "No study has compared melzorin directly with the protocol")],
        "unanswered_parts": ["Whether melzorin is better than the ward protocol: no head-to-head data exist."]},
    "ORCHID-1": {"claims": [
        claim("At week 16, PASI 90 was reached by 71% with tralvekimab versus 4% with placebo.", "S1",
              "PASI 90 was achieved by 71% of patients receiving tralvekimab and 4% receiving placebo"),
        claim("Against bremilumab, PASI 90 was 67% with tralvekimab versus 49% with bremilumab, a difference of "
              "18 percentage points.", "S2",
              "a difference of 18 percentage points (95% CI 11 to 25)"),
        claim("In the uncontrolled open-label extension, PASI 90 was maintained in 82% of those remaining at week 104.",
              "S3", "PASI 90 was maintained in 82% of patients who remained in the study"),
        claim("Serious adverse events occurred at 3.1 per 100 patient-years versus 3.4 with placebo, and oral "
              "candidiasis in 2.0% versus 0.4%.", "S4", "oral candidiasis in 2.0% and 0.4% of patients")],
        "limitations": [claim("No trial was powered for cardiovascular outcomes and no data exist beyond two years.",
                              "S5", "no data beyond two years are available")],
        "unanswered_parts": ["The effect on cardiovascular outcomes cannot be determined from the evidence."]},
}
MARKERS = {"KESTREL-3": "KESTREL-3", "TWILIGHT": "TWILIGHT", "ORCHID-1": "ORCHID-1"}


def _marker(prompt):
    return next(key for key, marker in MARKERS.items() if marker in prompt)


class AcceptanceProvider:
    """Scripted gateway: ``plan`` maps request number (1-based) to an action."""

    def __init__(self, plan=None, *, remaining=7600, cost=0.0009):
        self.plan = plan or {}
        self.cost = cost
        self.requests = []
        self.remaining = remaining
        self.mutations = {}

    async def gateway(self, role, temperature, prompt, schema, *, max_output_tokens, **settings):
        self.requests.append({"schema": schema, "max_output_tokens": max_output_tokens, "prompt": prompt, **settings})
        action = self.plan.get(len(self.requests), "ok")
        if action == "truncate":
            raise truncation_error()
        if schema is V2RepairBatch:
            return V2RepairBatch.model_validate({"repairs": [
                {"target": "claims.2", "action": "drop", "replacement": None,
                 "unresolved_reason": "The claim could not be supported by a verbatim quote."}]})
        answer = copy.deepcopy(VALID[_marker(prompt)])
        if action == "needs_repair":
            answer["claims"].append(claim("An extra unsupported sentence about the trial.", "S1",
                                          "this quote is not in the passage"))
            answer["claims"][1], answer["claims"][-1] = answer["claims"][-1], answer["claims"][1]
        return StructuredEvidenceAnswerV2.model_validate(answer)

    def summary(self, *_args):
        number = len(self.requests)
        value = {"token_usage_complete": True, "estimated_cost_usd": self.cost, "provider_calls": 1,
                 "input_tokens": 1300, "output_tokens": 1100, "by_model": {GROQ: {}},
                 "attempts": [{"provider": "groq", "model": "openai/gpt-oss-120b", "status_code": None,
                               "rate_limit_headers": repro._headers(self.remaining),
                               "diagnostics": {"reasoning_tokens": 400, "finish_reason": "stop"}}]}
        if self.plan.get(number) == "truncate":
            value.update(token_usage_complete=False, estimated_cost_usd=None, input_tokens=None, output_tokens=None)
            value["attempts"][0]["status_code"] = 400
        if self.plan.get(number) == "missing_usage":
            value.update(token_usage_complete=False, estimated_cost_usd=None, input_tokens=None, output_tokens=None)
        mutate = self.mutations.get(number)
        if mutate:
            mutate(value)
        return value


class _Harness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "private"
        self.directory.mkdir(mode=0o700)
        self.dataset, self.config = dataset(), config()
        self.ledger = programs.ProgramLedgerStore(self.directory / "acceptance-ledger.private.json")
        programs.initialize_program_ledger(self.ledger, program_id="r7d-d-acceptance", max_calls=26,
                                           max_cost_usd=0.15, now=1_000.0)
        self.store = AtomicCheckpoint(self.directory / "acceptance-checkpoint.private.json")
        self.clock = [1_000.0]

    def authorization(self, cfg=None, **updates):
        cfg = cfg or self.config
        plan = acceptance.acceptance_dry_run(self.dataset, cfg, references())
        value = {"schema": acceptance.AUTHORIZATION_SCHEMA, "authorization_id": "r7d-d-a1", "approved_by": "owner",
                 "accepted_code_sha": cfg["code_sha"], "dataset_sha256": self.dataset["dataset_sha256"],
                 "config_sha256": plan["config_sha256"], "schedule_sha256": plan["schedule_sha256"],
                 "plan_sha256": plan["plan_sha256"], "provider": "groq", "model": "openai/gpt-oss-120b",
                 "reasoning_effort": "low", "max_output_tokens": [2000, 3000], "max_physical_requests": 26,
                 "max_cost_usd": cfg["max_cost_usd"], "completion_budget_failure_policy": "score_as_failure",
                 "selection_rules_sha256": plan["selection_rules_sha256"],
                 "references_sha256": plan["references_sha256"], "program_id": "r7d-d-acceptance",
                 "program_baseline_sha256": programs.program_baseline(self.ledger.load())["sha256"]}
        value.update(updates)
        return value

    async def execute(self, provider, *, cfg=None, authorization=None, **kwargs):
        cfg = cfg or self.config
        with patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)), \
                patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"), \
                patch("scientific_rag.llm_gateway.begin_trace", return_value=object()), \
                patch("scientific_rag.llm_gateway.end_trace", return_value=[]), \
                patch("scientific_rag.llm_gateway.summarize_trace", side_effect=provider.summary):
            return await acceptance.run_acceptance(
                self.dataset, cfg, authorization=authorization or self.authorization(cfg),
                checkpoint_store=self.store, program_ledger=self.ledger, gateway=provider.gateway,
                provenance_resolver=lambda: {"sha": cfg["code_sha"], "source": "test"},
                clock=lambda: self.clock[0], **kwargs)


class ContractTests(unittest.TestCase):
    def test_schedule_is_frozen_balanced_and_bounded(self):
        schedule = acceptance.acceptance_schedule(config())
        self.assertEqual(14, len(schedule))
        arms = [item for item in schedule if item["kind"] == "arm"]
        self.assertEqual(12, len(arms))
        for scenario in ("A001", "A002", "A003"):
            cells = sorted((item["configuration"], item["prompt_version"]) for item in arms if item["target"] == scenario)
            self.assertEqual([("A", "v15"), ("A", "v16"), ("B", "v15"), ("B", "v16")], cells)
        self.assertEqual([("RF1", "B"), ("RF2", "A")], [(item["target"], item["configuration"])
                                                        for item in schedule if item["kind"] == "dedicated_repair"])
        plan = acceptance.acceptance_dry_run(dataset(), config(), references())
        self.assertEqual((0, 12, 12, 2, 26), (plan["provider_calls_made"], plan["initial_generations"],
                                              plan["conditional_repairs_max"], plan["dedicated_repairs"],
                                              plan["max_physical_requests"]))
        self.assertEqual(plan["schedule_sha256"], canonical_sha256(schedule))

    def test_matched_configurations_share_evidence_prompts_and_schemas(self):
        plan = acceptance.acceptance_dry_run(dataset(), config(), references())
        arms = [item for item in plan["operations"] if item["kind"] == "arm"]
        for scenario in ("A001", "A002", "A003"):
            for version in ("v15", "v16"):
                matched = [item for item in arms if item["target"] == scenario and item["prompt_version"] == version]
                self.assertEqual({"A", "B"}, {item["configuration"] for item in matched})
                self.assertEqual(1, len({item["initial_prompt_sha256"] for item in matched}))
                self.assertEqual(1, len({item["question_context_sha256"] for item in matched}))
                self.assertEqual({2000, 3000}, {item["max_output_tokens"] for item in matched})
                self.assertEqual({"low"}, {item["reasoning_effort"] for item in matched})
        self.assertEqual({"initial": "194e586d95e1f6a0f07310ef27446ed5449370334d57769c7b42cb85c993f471",
                          "repair": "83bc2fa3e090ce798dec8b1c02f6b2e44ce3dbf5950d116774ee2685923219a4"},
                         plan["schemas"])

    def test_dry_run_is_zero_call_and_within_the_proposed_budget(self):
        from scientific_rag import llm_gateway

        async def must_not_run(*args, **kwargs):
            raise AssertionError("provider reached during a dry run")
        with patch.object(llm_gateway, "ainvoke_structured", must_not_run):
            plan = acceptance.acceptance_dry_run(dataset(), config(), references())
        with self.assertRaisesRegex(acceptance.AcceptanceError, "references"):
            acceptance.acceptance_dry_run(dataset(), config())
        self.assertEqual([], plan["blockers"])
        self.assertTrue(plan["eligible_after_exact_owner_authorization"])
        self.assertLess(plan["deterministic_reservation_usd"], acceptance.PROPOSED_DIAGNOSTIC_BUDGET_USD)
        self.assertLess(plan["deterministic_reservation_usd"] + plan["typical_conditional_repair_estimate_usd"],
                        acceptance.PROPOSED_DIAGNOSTIC_BUDGET_USD)
        self.assertIn("not a guaranteed upper bound", plan["cost_basis"])

    def test_invalid_configurations_and_datasets_are_refused(self):
        bad = {
            "medium reasoning only on one side": config(configurations=[
                {"label": "A", "reasoning_effort": "low", "max_output_tokens": 2000},
                {"label": "B", "reasoning_effort": "medium", "max_output_tokens": 3000}]),
            "unsupported route": config(model="llama-3.3-70b-versatile"),
            "wrong ceiling": config(max_calls=27),
            "extra field": dict(config(), retries=1),
        }
        for label, value in bad.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                acceptance.validate_acceptance_config(value)
        tampered = dataset()
        tampered["scenarios"][0]["predicates"] = []
        with self.assertRaises(acceptance.AcceptanceError):
            acceptance.validate_dataset(tampered)

    def test_reference_predicates_are_isolated_from_prompts_but_bound_by_the_plan(self):
        plain = acceptance.acceptance_dry_run(dataset(), config(), references())
        modified = references()
        modified["predicates"]["A001"][0]["any_of"] = ["changed"]
        modified["references_sha256"] = canonical_sha256({k: v for k, v in modified.items() if k != "references_sha256"})
        with_references = acceptance.acceptance_dry_run(dataset(), config(), modified)
        # Generation is independent of scoring: every prompt hash is unchanged ...
        self.assertEqual([item["initial_prompt_sha256"] for item in plain["operations"]],
                         [item["initial_prompt_sha256"] for item in with_references["operations"]])
        self.assertEqual([item.get("repair_prompt_sha256") for item in plain["operations"]],
                         [item.get("repair_prompt_sha256") for item in with_references["operations"]])
        # ... while the experiment/scoring commitment changes, so a post hoc edit cannot match.
        self.assertNotEqual(plain["plan_sha256"], with_references["plan_sha256"])
        self.assertEqual(references()["references_sha256"], plain["references_sha256"])
        dataset_text = (DATA / "synthetic-acceptance-v1.json").read_text()
        for scenario, items in references()["predicates"].items():
            for item in items:
                self.assertNotIn(item["id"], dataset_text)              # no predicate reaches the dataset
        for operation in plain["operations"]:
            self.assertNotIn("predicate", json.dumps(operation))
        import inspect
        self.assertNotIn("references", inspect.signature(acceptance.run_acceptance).parameters)

    def test_no_private_benchmark_material(self):
        text = (DATA / "synthetic-acceptance-v1.json").read_text() + (DATA / "synthetic-acceptance-references-v1.json").read_text()
        value = dataset()
        self.assertTrue(value["synthetic"])
        self.assertEqual(["A001", "A002", "A003"], [item["scenario_id"] for item in value["scenarios"]])
        for forbidden in ("3dc9e107", "r7d-20-manifest", '"D0', "D001", "development"):
            self.assertNotIn(forbidden, text)
        source = Path(acceptance.__file__).read_text()
        self.assertNotIn("r7d-20-manifest", source)

    def test_predicates_detect_attribution_numbers_and_unqualified_subgroups(self):
        predicates = references()["predicates"]
        good = {item["id"]: item["passed"] for item in acceptance.evaluate_predicates(VALID["KESTREL-3"], predicates["A001"])}
        self.assertTrue(all(good.values()), good)
        bad = copy.deepcopy(VALID["KESTREL-3"])
        bad["claims"].append(claim("KESTREL-3 showed 32% lower stroke incidence.", "S2", "x" * 8))
        bad["claims"].append(claim("Diastolic pressure fell by 6.1 mmHg.", "S1", "x" * 8))
        results = {item["id"]: item["passed"] for item in acceptance.evaluate_predicates(bad, predicates["A001"])}
        self.assertEqual((False, False), (results["A001-X1"], results["A001-X2"]))
        subgroup = copy.deepcopy(VALID["TWILIGHT"])
        subgroup["claims"].append(claim("Melzorin works in patients with cognitive impairment.", "S1", "x" * 8))
        results = {item["id"]: item["passed"] for item in acceptance.evaluate_predicates(subgroup, predicates["A002"])}
        self.assertFalse(results["A002-C1"])
        absent = acceptance.evaluate_predicates(None, predicates["A003"])
        self.assertFalse(any(item["passed"] or item["critical_violation"] for item in absent))
        for scenario, answer in (("A001", "KESTREL-3"), ("A002", "TWILIGHT"), ("A003", "ORCHID-1")):
            results = acceptance.evaluate_predicates(VALID[answer], predicates[scenario])
            self.assertTrue(all(item["passed"] for item in results), (scenario, results))


class ExecutionTests(_Harness):
    async def test_full_run_uses_each_configurations_settings_without_retry(self):
        provider = AcceptanceProvider()
        checkpoint = await self.execute(provider)
        self.assertEqual(14, len(checkpoint["records"]))
        self.assertEqual(14, len(provider.requests))                  # 12 generations + 2 dedicated repairs
        schedule = acceptance.acceptance_schedule(self.config)
        limits = {"A": 2000, "B": 3000}
        dedicated = [item for item in schedule if item["kind"] == "dedicated_repair"]
        self.assertEqual(2, sum(request["schema"] is V2RepairBatch for request in provider.requests))
        for record in checkpoint["records"]:
            for attempt in record["attempts"]:
                self.assertEqual(1, attempt["provider_calls"])
        for request, operation in zip(provider.requests, schedule):
            self.assertEqual((limits[operation["configuration"]], "low"),
                             (request["max_output_tokens"], request["reasoning_effort"]))
            self.assertIs(request["schema"], V2RepairBatch if operation in dedicated else StructuredEvidenceAnswerV2)
        self.assertEqual(14, checkpoint["execution"]["budget"]["provider_calls_charged"])
        report = acceptance.build_acceptance_report(checkpoint, self.dataset, references(), self.config)
        self.assertEqual([], report["missing_operations"])
        for cell in report["cells"].values():
            self.assertEqual((3, 3, 3, 0), (cell["planned_arms"], cell["executed"], cell["final_valid"],
                                            cell["critical_violations"]))
            self.assertEqual(cell["output_tokens"] - cell["reasoning_tokens"], cell["non_reasoning_completion_tokens"])
        self.assertTrue(all(item["repair_schema_parsed"] for item in report["dedicated_repairs"].values()))
        self.assertEqual("A", report["selection"]["decision"])
        with self.assertRaisesRegex(CheckpointError, "refusing repeated execution"):
            await self.execute(provider)
        self.assertEqual(14, len(provider.requests))

    async def test_every_generation_repairing_reaches_exactly_26_requests(self):
        provider = AcceptanceProvider({number: "needs_repair" for number in range(1, 40)})
        checkpoint = await self.execute(provider)
        self.assertEqual(26, len(provider.requests))
        self.assertEqual(26, checkpoint["execution"]["budget"]["provider_calls_charged"])
        self.assertTrue(all(record["status"] == "completed" for record in checkpoint["records"]))

    async def test_authorization_refusal_sends_nothing(self):
        provider = AcceptanceProvider()
        for field, value in (("plan_sha256", "0" * 64), ("max_output_tokens", [2000, 2000]),
                             ("reasoning_effort", "medium"), ("max_physical_requests", 27),
                             ("accepted_code_sha", "b" * 40), ("schema", "scientific-rag-generation-eval-authorization/v2"),
                             ("program_baseline_sha256", None), ("selection_rules_sha256", "1" * 64)):
            with self.subTest(field=field), self.assertRaises(AuthorizationError):
                await self.execute(provider, authorization=self.authorization(**{field: value}))
        with self.assertRaisesRegex(AuthorizationError, "program ledger"):
            await acceptance.run_acceptance(self.dataset, self.config, authorization=self.authorization(),
                                            checkpoint_store=self.store, program_ledger=None)
        self.assertEqual([], provider.requests)
        self.assertFalse(self.store.path.exists())

    async def test_acceptance_accounting_is_separate_and_history_is_preserved(self):
        benchmark = programs.ProgramLedgerStore(self.directory / "benchmark-ledger.private.json")
        programs.initialize_program_ledger(benchmark, program_id="r7d-b-program", max_calls=80, max_cost_usd=1.5,
                                           now=1.0)
        before = benchmark.path.read_bytes()
        checkpoint = await self.execute(AcceptanceProvider(), operations=3)
        self.assertEqual(before, benchmark.path.read_bytes())
        summary = programs.accounting_summary(self.ledger.load(), checkpoint, checkpoint_path=self.store.path)
        self.assertEqual(3, summary["cumulative"]["provider_calls_charged"])
        self.assertEqual(23, summary["cumulative"]["remaining_calls"])
        await self.execute(AcceptanceProvider())                         # staged continuation, same campaign
        closed = programs.close_campaign(self.ledger, self.store)
        self.assertEqual(14, closed["consumed"]["provider_calls_charged"])

    async def test_budget_exhaustion_stops_before_transmission(self):
        cfg = config(max_cost_usd=0.034)
        programs_path = self.directory / "small-ledger.private.json"
        self.ledger = programs.ProgramLedgerStore(programs_path)
        programs.initialize_program_ledger(self.ledger, program_id="r7d-d-acceptance", max_calls=26,
                                           max_cost_usd=0.034, now=1.0)
        provider = AcceptanceProvider({number: "needs_repair" for number in range(1, 40)}, cost=0.0018)
        checkpoint = await self.execute(provider, cfg=cfg)
        self.assertEqual("budget", checkpoint["execution"]["stop_code"])
        self.assertLess(len(provider.requests), 26)
        self.assertLessEqual(checkpoint["execution"]["budget"]["budget_charged_usd"], 0.034 + 1e-9)
        report = acceptance.build_acceptance_report(checkpoint, self.dataset, references(), cfg)
        self.assertTrue(report["missing_operations"])
        self.assertEqual("inconclusive", report["selection"]["decision"])

    async def test_quota_interruption_resumes_without_repeating(self):
        provider = AcceptanceProvider()

        def exhausted(value):
            value["attempts"][0]["rate_limit_headers"] = repro._headers(0)
        provider.mutations[2] = exhausted
        checkpoint = await self.execute(provider)
        self.assertEqual("rate_limit", checkpoint["execution"]["stop_code"])
        self.assertEqual(2, len(provider.requests))
        self.clock[0] += 120                                           # the token window has reset
        checkpoint = await self.execute(provider, resume_after_rate_limit=True)
        self.assertEqual(14, len(checkpoint["records"]))
        self.assertEqual(14, len(provider.requests))
        operations = [record["operation"] for record in checkpoint["records"]]
        self.assertEqual(sorted(set(operations)), operations)

    async def test_truncation_is_a_scored_completion_budget_failure_and_missing_usage_stops(self):
        provider = AcceptanceProvider({3: "truncate"})
        checkpoint = await self.execute(provider, operations=4)
        record = checkpoint["records"][2]
        self.assertEqual(("failed", "completion_budget_exhausted"), (record["status"], record["failure_kind"]))
        self.assertGreater(checkpoint["execution"]["budget"]["uncertain_reserved_usd"], 0.0)
        self.assertIsNone(checkpoint["execution"]["stop_code"])
        report = acceptance.build_acceptance_report(checkpoint, self.dataset, references(), self.config)
        # An accounted, predeclared truncation is a scored outcome, not an accounting defect.
        self.assertEqual((False, 1), (report["accounting"]["accounting_defect"],
                                      report["accounting"]["accounted_truncations"]))
        self.assertFalse([block for block in report["selection"]["global_blocks"] if "accounting" in block])
        self.setUp()
        provider = AcceptanceProvider({2: "missing_usage"})
        checkpoint = await self.execute(provider)
        self.assertEqual("billing_unknown", checkpoint["execution"]["stop_code"])
        self.assertEqual(2, len(provider.requests))
        report = acceptance.build_acceptance_report(checkpoint, self.dataset, references(), self.config)
        self.assertEqual(list(range(2, 15)) if not checkpoint["records"][1:] else list(range(3, 15)),
                         report["missing_operations"])
        self.assertIn("accounting defect: billing_unknown", report["selection"]["global_blocks"])
        self.assertEqual("inconclusive", report["selection"]["decision"])

    async def test_report_denominators_include_unexecuted_operations(self):
        checkpoint = await self.execute(AcceptanceProvider(), operations=5)
        report = acceptance.build_acceptance_report(checkpoint, self.dataset, references(), self.config)
        self.assertEqual(list(range(6, 15)), report["missing_operations"])
        self.assertEqual(12, sum(cell["planned_arms"] for cell in report["cells"].values()))
        self.assertEqual(8, sum(len(cell["unexecuted"]) for cell in report["cells"].values()))
        self.assertFalse(report["dedicated_repairs"]["A"]["executed"])
        self.assertEqual("inconclusive", report["selection"]["decision"])


class InterruptionRegressionTests(_Harness):
    """A process stop at any durable step never causes a second transmission (P0.1)."""

    async def stop_and_continue(self, provider, when):
        from tests.test_r7dd_review_reproduction import ProcessStopped, StoppingCheckpoint
        path = self.store.path
        self.store = StoppingCheckpoint(path, when)
        with self.assertRaises(ProcessStopped):                     # the stop really happened
            await self.execute(provider)
        self.store = AtomicCheckpoint(path)
        return await self.execute(provider)

    async def test_stop_before_every_finalization_keeps_exactly_14_requests(self):
        for count in range(1, 15):
            with self.subTest(finalized=count):
                self.setUp()
                provider = AcceptanceProvider()
                checkpoint = await self.stop_and_continue(provider, lambda value, n=count:
                                                          len(value.get("records") or []) >= n)
                self.assertEqual((14, 14, 14), (len(provider.requests), len(checkpoint["records"]),
                                                checkpoint["execution"]["budget"]["provider_calls_charged"]))

    async def test_stop_around_every_repair_keeps_exactly_26_requests(self):
        always = {number: "needs_repair" for number in range(1, 40)}
        for operation in range(1, 15):
            for point in ("repair_pending", "finalized"):
                with self.subTest(operation=operation, point=point):
                    self.setUp()
                    provider = AcceptanceProvider(always)
                    if point == "repair_pending":
                        def when(value, n=operation):
                            active = value.get("active_arm") or {}
                            return active.get("operation") == n and active.get("phase") == "repair_pending"
                    else:
                        def when(value, n=operation):
                            return len(value.get("records") or []) >= n
                    checkpoint = await self.stop_and_continue(provider, when)
                    self.assertEqual(26, len(provider.requests))
                    self.assertEqual(26, checkpoint["execution"]["budget"]["provider_calls_charged"])
                    self.assertTrue(all(record["status"] == "completed" for record in checkpoint["records"]))

    async def test_tampered_call_audit_is_refused_before_transmission(self):
        provider = AcceptanceProvider()
        await self.execute(provider, operations=2)
        for label, tamper in (("cost", lambda c: c["execution"]["calls"][0].update(cost_usd=0.0)),
                              ("dropped call", lambda c: c["execution"]["calls"].pop()),
                              ("budget", lambda c: c["execution"]["budget"].update(provider_calls_charged=1))):
            with self.subTest(label=label):
                checkpoint = self.store.load()
                original = copy.deepcopy(checkpoint)
                tamper(checkpoint)
                self.store.write(checkpoint)
                with self.assertRaisesRegex(CheckpointError, "call audit"):
                    await self.execute(provider)
                self.assertEqual(2, len(provider.requests))
                self.store.write(original)


class SelectionRuleTests(unittest.TestCase):
    def report(self, **overrides):
        cell = {"planned_arms": 3, "executed": 3, "final_valid": 3, "pydantic_failures": 0,
                "completion_budget_failures": 0, "provider_structured_generation_failures": 0, "interrupted": 0,
                "critical_violations": 0, "predicates_evaluated": 30, "predicates_passed": 30,
                "min_headroom_tokens": 900}
        cells = {key: dict(cell) for key in ("A/v15", "A/v16", "B/v15", "B/v16")}
        for key, values in overrides.items():
            cells[key].update(values)
        return {"cells": cells, "dedicated_repairs": {"A": {"repair_schema_parsed": True},
                                                      "B": {"repair_schema_parsed": True}},
                "stop_code": None, "unresolved_attempt": False, "missing_operations": []}

    def select(self, report):
        return acceptance.select_configuration(report, {"A": 2000, "B": 3000})

    def test_both_pass_prefers_2000(self):
        self.assertEqual("A", self.select(self.report())["decision"])

    def test_3000_preferred_only_for_a_meaningful_advantage(self):
        report = self.report(**{"A/v16": {"final_valid": 2, "completion_budget_failures": 1, "min_headroom_tokens": 0}})
        self.assertEqual("B", self.select(report)["decision"])
        report = self.report(**{"A/v15": {"min_headroom_tokens": 150}})
        self.assertEqual("B", self.select(report)["decision"])
        report = self.report(**{"B/v15": {"predicates_passed": 31, "predicates_evaluated": 31}})
        self.assertEqual("A", self.select(report)["decision"])        # one more predicate is not meaningful

    def test_gates_and_inconclusive_outcomes(self):
        report = self.report(**{"A/v15": {"final_valid": 1}, "B/v15": {"final_valid": 1}})
        self.assertEqual("inconclusive", self.select(report)["decision"])
        report = self.report()
        report["stop_code"] = "model_drift"
        self.assertEqual("inconclusive", self.select(report)["decision"])
        report = self.report()
        report["dedicated_repairs"]["A"]["repair_schema_parsed"] = False
        self.assertEqual("B", self.select(report)["decision"])
        report = self.report(**{"B/v15": {"critical_violations": 2}})
        self.assertEqual("A", self.select(report)["decision"])
        self.assertIn("none", self.select(self.report())["statistical_claim"])


class CommandLineTests(unittest.TestCase):
    def test_dry_run_writes_a_private_plan_and_refuses_outputs_in_the_repository(self):
        spec = importlib.util.spec_from_file_location("acceptance_cli", ROOT / "scripts" / "acceptance_r7d_d.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text(json.dumps(config()))
            out = Path(directory) / "plan.json"
            stream = io.StringIO()
            with patch.object(sys, "argv", ["acceptance_r7d_d.py", "dry-run", "--config", str(cfg), "--out", str(out)]), \
                    redirect_stdout(stream):
                self.assertEqual(0, cli.main())
            self.assertEqual(0o600, os.stat(out).st_mode & 0o777)
            self.assertEqual(26, json.loads(stream.getvalue())["max_physical_requests"])
            inside = ROOT / "evaluation" / "r7d-d" / "plan.private.json"
            with patch.object(sys, "argv", ["acceptance_r7d_d.py", "dry-run", "--config", str(cfg), "--out",
                                            str(inside)]), self.assertRaisesRegex(SystemExit, "outside"):
                cli.main()
            self.assertFalse(inside.exists())

    def test_report_exports_answers_for_owner_review_and_refuses_drifted_references(self):
        spec = importlib.util.spec_from_file_location("acceptance_cli", ROOT / "scripts" / "acceptance_r7d_d.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        harness = ExecutionTests("test_full_run_uses_each_configurations_settings_without_retry")
        harness.setUp()
        checkpoint = asyncio.run(harness.execute(AcceptanceProvider()))
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text(json.dumps(config()))
            out, answers = Path(directory) / "report.json", Path(directory) / "answers.json"
            argv = ["acceptance_r7d_d.py", "report", "--config", str(cfg), "--checkpoint", str(harness.store.path),
                    "--out", str(out), "--answers-out", str(answers)]
            before = harness.store.path.read_bytes()
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(0, cli.main())
            self.assertEqual(before, harness.store.path.read_bytes())                 # read-only
            exported = json.loads(answers.read_text())
            self.assertEqual(12, len(exported["arms"]))
            self.assertEqual(0o600, os.stat(answers).st_mode & 0o777)
            self.assertNotIn("post_repair_output", out.read_text())                    # metrics file has no text
            self.assertEqual("recommendation", json.loads(out.read_text())["selection"]["status"])
            drifted = references()
            drifted["predicates"]["A001"][0]["any_of"] = ["x"]
            drifted["references_sha256"] = canonical_sha256({k: v for k, v in drifted.items()
                                                              if k != "references_sha256"})
            refs = Path(directory) / "refs.json"
            refs.write_text(json.dumps(drifted))
            with patch.object(sys, "argv", argv + ["--references", str(refs)]), \
                    self.assertRaisesRegex(SystemExit, "references_sha256"):
                cli.main()
        self.assertEqual(14, len(checkpoint["records"]))
        harness.doCleanups()


if __name__ == "__main__":
    unittest.main()
