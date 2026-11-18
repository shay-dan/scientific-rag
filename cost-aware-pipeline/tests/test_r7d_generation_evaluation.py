import asyncio
import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark.generation_evaluation import (
    AUTHORIZATION_SCHEMA,
    AuthorizationError,
    EvaluationStopped,
    ManifestError,
    build_blind_scoring_packet,
    build_public_report,
    canonical_sha256,
    check_paid_authorization,
    dry_run,
    evaluate_pair,
    manifest_hash,
    question_context_hash,
    run_paid_evaluation,
    unblind_owner_scores,
    validate_manifest,
)
from app.benchmark.production_verifier_replay import BudgetLedger
from app.benchmark.incremental_generation_evaluation import (
    AtomicCheckpoint,
    CampaignLocked,
    CheckpointError,
    run_incremental_evaluation,
)
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch


def _claim(text="The response rate was 12% in the treated group.", quote="The response rate was 12% in the treated group."):
    return {"text": text, "source_id": "S1", "evidence_quote": quote}


def _manifest():
    scenarios = []
    categories = ("numeric_fidelity", "attribution", "study_design", "coverage")
    for index in range(20):
        evidence = [{
            "source_id": "S1",
            "title": f"Development source {index + 1}",
            "passage": "The response rate was 12% in the treated group.",
            "section": "Results",
            "production_notes": "fixed development evidence",
        }]
        for item in evidence:
            item["evidence_sha256"] = canonical_sha256(item["passage"])
        scenario = {
            "scenario_id": f"D{index + 1:03d}",
            "category": categories[index % len(categories)],
            "question": "What response rate was reported?",
            "evidence": evidence,
        }
        scenario["question_context_sha256"] = question_context_hash(scenario)
        scenarios.append(scenario)
    value = {
        "schema": "scientific-rag-generation-eval-manifest/v1",
        "selection": {"split": "development", "version": "reviewed-selection-v1", "frozen": True},
        "provenance": {"dataset": "owner-reviewed-development-set", "prepared_by": "owner"},
        "scenarios": scenarios,
    }
    value["manifest_sha256"] = manifest_hash(value)
    return value


def _config(**updates):
    value = {
        "schema": "scientific-rag-generation-eval-config/v1",
        "code_sha": "a" * 40,
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "temperature": 0,
        "model_context_tokens": 131072,
        "max_input_tokens": 100000,
        "max_output_tokens": 2000,
        "max_calls": 80,
        "max_cost_usd": 1.5,
        "semantic_verification_mode": "off",
        "claim_recovery_mode": "strict",
        "claim_audit_mode": "off",
        "prompt_versions": ["v15", "v16"],
    }
    value.update(updates)
    return value


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_is_exactly_twenty_label_free_development_scenarios(self):
        manifest = _manifest()
        validated = validate_manifest(manifest)
        self.assertEqual(20, len(validated["scenarios"]))
        self.assertEqual(manifest["manifest_sha256"], manifest_hash(manifest))

    def test_manifest_rejects_wrong_count_duplicate_ids_and_non_development_split(self):
        for mutate in (
            lambda value: value["scenarios"].pop(),
            lambda value: value["scenarios"][1].update(scenario_id="D001"),
            lambda value: value["selection"].update(split="holdout"),
        ):
            value = _manifest()
            mutate(value)
            value["manifest_sha256"] = manifest_hash(value)
            with self.assertRaises(ManifestError):
                validate_manifest(value)

    def test_manifest_rejects_labels_references_and_unfrozen_selection(self):
        for field, content in (("label", "supported"), ("reference_answer", "gold"), ("gold_claims", [])):
            value = _manifest()
            value["scenarios"][0][field] = content
            value["manifest_sha256"] = manifest_hash(value)
            with self.assertRaises(ManifestError):
                validate_manifest(value)
        value = _manifest()
        value["selection"]["frozen"] = False
        value["manifest_sha256"] = manifest_hash(value)
        with self.assertRaises(ManifestError):
            validate_manifest(value)
        value = _manifest()
        value["scenarios"][0]["evidence"][0]["label"] = "supported"
        value["manifest_sha256"] = manifest_hash(value)
        with self.assertRaises(ManifestError):
            validate_manifest(value)

    def test_manifest_rejects_passage_and_question_context_tampering(self):
        value = _manifest()
        value["scenarios"][0]["evidence"][0]["passage"] += " changed"
        value["manifest_sha256"] = manifest_hash(value)
        with self.assertRaises(ManifestError):
            validate_manifest(value)
        value = _manifest()
        value["scenarios"][0]["question_context_sha256"] = "0" * 64
        value["manifest_sha256"] = manifest_hash(value)
        with self.assertRaises(ManifestError):
            validate_manifest(value)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_pair_keeps_missing_arm_as_not_executed(self):
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise EvaluationStopped("authorized evaluation stopped")

        for execution_index, expected_order in ((0, ["v15", "v16"]), (1, ["v16", "v15"])):
            result = await evaluate_pair(
                _manifest()["scenarios"][execution_index], config=_config(), gateway=gateway,
                execution_index=execution_index,
            )
            self.assertEqual(expected_order, [arm["prompt_version"] for arm in result["arms"]])
            # R7D correction: a runner stop is an interruption, never a failed model answer.
            self.assertEqual(["interrupted", "not_executed"], [arm["status"] for arm in result["arms"]])
            self.assertEqual(0, result["arms"][1]["physical_calls"])
        self.assertEqual(2, calls)

    async def test_execution_order_is_counterbalanced_across_twenty_scenarios(self):
        first_versions = []

        async def gateway(*args, **kwargs):
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        manifest = _manifest()
        for index, scenario in enumerate(manifest["scenarios"]):
            result = await evaluate_pair(
                scenario, config=_config(), gateway=gateway, execution_index=index,
            )
            first_versions.append(result["arms"][0]["prompt_version"])
            self.assertEqual({"v15", "v16"}, {arm["prompt_version"] for arm in result["arms"]})
            self.assertEqual(2, result["physical_calls"])
        self.assertEqual(10, first_versions.count("v15"))
        self.assertEqual(10, first_versions.count("v16"))
        self.assertEqual(["v15", "v16", "v15", "v16"], first_versions[:4])

    async def test_pair_uses_identical_evidence_and_only_prompt_version_differs(self):
        calls = []

        async def gateway(role, temperature, prompt, schema, *, max_output_tokens):
            calls.append((role, temperature, prompt, schema, max_output_tokens))
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        result = await evaluate_pair(_manifest()["scenarios"][0], config=_config(), gateway=gateway)
        self.assertEqual(["v15", "v16"], [arm["prompt_version"] for arm in result["arms"]])
        self.assertEqual(2, len(calls))
        self.assertEqual(calls[0][0:2], calls[1][0:2])
        self.assertEqual(calls[0][3:], calls[1][3:])
        self.assertNotEqual(calls[0][2], calls[1][2])
        self.assertEqual(result["arms"][0]["evidence_sha256"], result["arms"][1]["evidence_sha256"])
        self.assertIn("initial_output", result["arms"][0])
        self.assertIn("post_repair_output", result["arms"][0])

    async def test_invalid_generation_gets_exactly_one_repair_and_captures_both_outputs(self):
        calls = []
        invalid = StructuredEvidenceAnswerV2(claims=[_claim(quote="not in evidence")])
        repaired = _claim()

        async def gateway(role, temperature, prompt, schema, *, max_output_tokens):
            calls.append(schema)
            return invalid if schema is StructuredEvidenceAnswerV2 else V2RepairBatch(
                repairs=[{"target": "claims.1", "action": "replace", "replacement": repaired}]
            )

        result = await evaluate_pair(_manifest()["scenarios"][0], config=_config(), gateway=gateway)
        self.assertEqual(4, len(calls))
        self.assertTrue(all(arm["repair_attempted"] for arm in result["arms"]))
        self.assertTrue(all(arm["status"] == "completed" for arm in result["arms"]))
        self.assertTrue(all(arm["initial_output"] != arm["post_repair_output"] for arm in result["arms"]))

    async def test_gateway_failure_is_captured_without_hidden_retry(self):
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("provider timeout")

        result = await evaluate_pair(_manifest()["scenarios"][0], config=_config(), gateway=gateway)
        self.assertEqual(2, calls)
        # R7D correction: provider/network failures are infrastructure interruptions, not
        # model-output failures, so they can never count against a prompt version.
        self.assertTrue(all(arm["status"] == "interrupted" for arm in result["arms"]))
        self.assertTrue(all(arm["failure_kind"] == "gateway_failure" for arm in result["arms"]))
        self.assertTrue(all(arm["failure_class"] == "infrastructure" for arm in result["arms"]))

    async def test_no_retrieval_mining_or_embedding_dependency_is_needed(self):
        async def gateway(*args, **kwargs):
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        # The adapter accepts only a fixed scenario and a gateway. There is no callback for
        # retrieval, mining, planning, or embedding to invoke.
        result = await evaluate_pair(_manifest()["scenarios"][0], config=_config(), gateway=gateway)
        self.assertEqual(2, result["physical_calls"])

    async def test_paid_entry_refuses_before_gateway_without_future_authorization(self):
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1

        with self.assertRaises(AuthorizationError):
            await run_paid_evaluation(_manifest(), _config(), authorization=None, gateway=gateway)
        self.assertEqual(0, calls)

    async def test_missing_usage_stops_after_one_call_and_keeps_reservation(self):
        manifest, config = _manifest(), _config()
        authorization = GateAndReportTests()._authorization(manifest, config)
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        incomplete = {"token_usage_complete": False, "estimated_cost_usd": None,
                      "provider_calls": 1, "input_tokens": None, "output_tokens": None,
                      "by_model": {"groq:openai/gpt-oss-120b": {}}}
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", return_value=incomplete),
              patch("app.evaluation_report.resolve_git_provenance", return_value={"sha": "a" * 40, "source": "test"})):
            records, execution = await run_paid_evaluation(
                manifest, config, authorization=authorization, gateway=gateway,
            )
        self.assertEqual(1, calls)
        self.assertIn("usage was incomplete", execution["stopped"])
        self.assertGreater(execution["budget"]["uncertain_reserved_usd"], 0)
        self.assertEqual("budget_or_accounting_stop", records[0]["arms"][0]["failure_kind"])
        self.assertEqual("not_executed", records[1]["status"])
        self.assertEqual({"v15", "v16"}, {arm["prompt_version"] for arm in records[1]["arms"]})
        self.assertTrue(all(arm["status"] == "not_executed" for arm in records[1]["arms"]))

    async def test_paid_entry_refuses_runtime_code_not_accepted_by_owner(self):
        manifest, config = _manifest(), _config()
        authorization = GateAndReportTests()._authorization(manifest, config)
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1

        with patch("app.evaluation_report.resolve_git_provenance",
                   return_value={"sha": "b" * 40, "source": "test"}):
            with self.assertRaises(AuthorizationError):
                await run_paid_evaluation(manifest, config, authorization=authorization, gateway=gateway)
        self.assertEqual(0, calls)

    async def test_paid_entry_rejects_fallback_routes_and_wrong_model_before_gateway(self):
        manifest, config = _manifest(), _config()
        authorization = GateAndReportTests()._authorization(manifest, config)
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1

        # provider_order() returns a tuple. Extra fallback providers, a different
        # provider, or a drifted expert model must fail before any provider call.
        for route, model in (
            (("groq", "openrouter"), "openai/gpt-oss-120b"),
            (("openrouter",), "openai/gpt-oss-120b"),
            (("groq",), "openai/gpt-oss-20b"),
        ):
            with (patch("scientific_rag.llm_gateway.provider_order", return_value=route),
                  patch("scientific_rag.llm_gateway.model_for", return_value=model),
                  patch("app.evaluation_report.resolve_git_provenance",
                        return_value={"sha": "a" * 40, "source": "test"})):
                with self.assertRaisesRegex(AuthorizationError, "gateway route is not pinned"):
                    await run_paid_evaluation(
                        manifest, config, authorization=authorization, gateway=gateway,
                    )
        self.assertEqual(0, calls)

    async def test_cancellation_keeps_reserved_cost_and_propagates(self):
        manifest, config = _manifest(), _config()
        authorization = GateAndReportTests()._authorization(manifest, config)
        ledger = BudgetLedger(80, 1.5)

        async def gateway(*args, **kwargs):
            raise asyncio.CancelledError()

        incomplete = {"token_usage_complete": False, "estimated_cost_usd": None,
                      "provider_calls": 1, "input_tokens": None, "output_tokens": None,
                      "by_model": {"groq:openai/gpt-oss-120b": {}}}
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", return_value=incomplete),
              patch("app.evaluation_report.resolve_git_provenance", return_value={"sha": "a" * 40, "source": "test"})):
            with self.assertRaises(asyncio.CancelledError):
                await run_paid_evaluation(
                    manifest, config, authorization=authorization, gateway=gateway, ledger=ledger,
                )
        self.assertGreater(ledger.snapshot()["uncertain_reserved_usd"], 0)


class GateAndReportTests(unittest.TestCase):
    def _authorization(self, manifest, config):
        return {
            "schema": AUTHORIZATION_SCHEMA,
            "authorization_id": "future-owner-authorization",
            "approved_by": "owner",
            "manifest_sha256": manifest["manifest_sha256"],
            "accepted_code_sha": config["code_sha"],
            "config_sha256": canonical_sha256(config),
            "provider": config["provider"],
            "model": config["model"],
            "max_calls": 80,
            "max_cost_usd": 1.5,
            "model_context_tokens": 131072,
            "max_input_tokens": 100000,
            "max_output_tokens": 2000,
        }

    def test_dry_run_is_zero_call_and_plans_eighty_worst_case_calls(self):
        report = dry_run(_manifest(), _config())
        self.assertEqual(0, report["provider_calls_made"])
        self.assertEqual(40, report["generation_calls_planned"])
        self.assertEqual(80, report["max_physical_calls"])
        self.assertEqual(["v15", "v16"], report["single_arm_difference"]["prompt_versions"])
        self.assertEqual({"v15": 10, "v16": 10}, report["first_arm_counts"])
        self.assertEqual(["v15", "v16"], report["execution_schedule"][0]["execution_order"])
        self.assertEqual(["v16", "v15"], report["execution_schedule"][1]["execution_order"])
        self.assertEqual("194e586d95e1f6a0f07310ef27446ed5449370334d57769c7b42cb85c993f471",
                         report["schemas"]["structured_answer_sha256"])

    def test_authorization_is_required_and_binds_every_material_input(self):
        manifest, config = _manifest(), _config()
        with self.assertRaises(AuthorizationError):
            check_paid_authorization(None, manifest=manifest, config=config)
        authorization = self._authorization(manifest, config)
        self.assertEqual(authorization, check_paid_authorization(authorization, manifest=manifest, config=config))
        mutations = {
            "manifest_sha256": "0" * 64,
            "accepted_code_sha": "b" * 40,
            "config_sha256": "0" * 64,
            "provider": "openrouter",
            "model": "another-model",
            "max_calls": 81,
            "max_cost_usd": 1.51,
            "model_context_tokens": 131073,
            "max_input_tokens": 100001,
            "max_output_tokens": 2001,
        }
        for field, value in mutations.items():
            changed = copy.deepcopy(authorization)
            changed[field] = value
            with self.assertRaises(AuthorizationError, msg=field):
                check_paid_authorization(changed, manifest=manifest, config=config)

    def test_unknown_price_and_unenforceable_output_limit_are_refused(self):
        manifest = _manifest()
        for updates in (
            {"model": "unknown-model"},
            {"provider": "unsupported", "model": "unknown-model"},
        ):
            config = _config(**updates)
            authorization = self._authorization(manifest, config)
            with self.assertRaises(AuthorizationError):
                check_paid_authorization(authorization, manifest=manifest, config=config)

    def test_budget_reservation_is_conservative_for_failure_cancellation_and_missing_usage(self):
        for usage in (
            {"cost_status": "incomplete", "provider_calls": 1, "known_cost_lower_bound_usd": 0.0},
            {"cost_status": "incomplete", "provider_calls": 1, "known_cost_lower_bound_usd": 0.001},
        ):
            ledger = BudgetLedger(80, 1.5)
            self.assertTrue(ledger.reserve(0.1))
            ledger.settle(0.1, usage)
            snapshot = ledger.snapshot()
            self.assertEqual(0.1, snapshot["uncertain_reserved_usd"])
            self.assertEqual(0.1, snapshot["budget_charged_usd"])

    def test_config_rejects_call_cost_and_input_ceiling_expansion(self):
        for updates in ({"max_calls": 81}, {"max_cost_usd": 1.5001}, {"max_input_tokens": 0},
                        {"model_context_tokens": 101999}):
            with self.assertRaises(ValueError):
                dry_run(_manifest(), _config(**updates))

    def test_public_report_is_deterministic_complete_and_contains_no_sensitive_text(self):
        records = [{
            "scenario_id": "D002",
            "arms": [{"prompt_version": "v15", "status": "failed", "repair_attempted": False,
                      "physical_calls": 1, "usage": {"input_tokens": 5, "output_tokens": 0,
                                                      "latency_ms": 2, "cost_usd": 0.0}},
                     {"prompt_version": "v16", "status": "completed", "repair_attempted": True,
                      "physical_calls": 2, "usage": {"input_tokens": 9, "output_tokens": 3,
                                                      "latency_ms": 4, "cost_usd": 0.01}}],
        }, {
            "scenario_id": "D001",
            "arms": [{"prompt_version": version, "status": "completed", "repair_attempted": False,
                      "physical_calls": 1, "usage": {}} for version in ("v15", "v16")],
        }]
        scores = {"D002": {"winner": "v16", "metrics": {"numeric_fidelity": {"v15": 0, "v16": 2}}}}
        one = build_public_report(records, scores=scores, manifest_sha256="f" * 64, config_sha256="e" * 64)
        two = build_public_report(list(reversed(records)), scores=scores,
                                  manifest_sha256="f" * 64, config_sha256="e" * 64)
        self.assertEqual(one, two)
        self.assertEqual(2, one["attempted_scenarios"])
        self.assertEqual({"v15": 0, "tie": 0, "v16": 1, "unscored": 1}, one["paired_outcomes"])
        encoded = json.dumps(one)
        for forbidden in ("question", "passage", "reference_answer", "evidence_quote", "initial_output"):
            self.assertNotIn(forbidden, encoded)

    def test_blind_packet_separates_prompt_key_and_public_report_rejects_free_text_scores(self):
        manifest = _manifest()
        records = []
        for scenario in manifest["scenarios"]:
            records.append({"scenario_id": scenario["scenario_id"], "arms": [
                {"prompt_version": "v15", "status": "completed", "post_repair_output": {"claims": []}},
                {"prompt_version": "v16", "status": "completed", "post_repair_output": {"claims": []}},
            ]})
        packet, key = build_blind_scoring_packet(manifest, records)
        self.assertNotIn("v15", json.dumps(packet))
        self.assertNotIn("v16", json.dumps(packet))
        self.assertIn("v15", json.dumps(key))
        first = key["arms"][0]
        unblinded = unblind_owner_scores({"D001": {
            "winner": "A", "metrics": {"coverage": {"A": 2, "B": 1}}
        }}, key)
        self.assertEqual(first["A"], unblinded["D001"]["winner"])
        self.assertEqual({first["A"]: 2, first["B"]: 1}, unblinded["D001"]["metrics"]["coverage"])
        with self.assertRaises(ValueError):
            build_public_report(records[:1], scores={"D001": {
                "winner": "tie", "metrics": {"coverage": {"v15": "private prose", "v16": 1}}
            }}, manifest_sha256="f" * 64, config_sha256="e" * 64)

    def test_blind_packet_and_report_retain_incomplete_pair_without_scoring_it(self):
        manifest = _manifest()
        records = []
        for scenario in manifest["scenarios"]:
            records.append({"scenario_id": scenario["scenario_id"], "arms": [
                {"prompt_version": "v15", "status": "completed", "post_repair_output": {"claims": []},
                 "physical_calls": 1},
                {"prompt_version": "v16", "status": "not_executed", "post_repair_output": None,
                 "physical_calls": 0, "reason": "stopped"},
            ]})
        # A truncated persisted record is normalized to the same explicit two-arm shape.
        records[0]["arms"] = records[0]["arms"][:1]
        packet, _ = build_blind_scoring_packet(manifest, records)
        self.assertEqual(20, len(packet["cases"]))
        self.assertTrue(all(case["scorable"] is False for case in packet["cases"]))
        report = build_public_report(
            records, scores={}, manifest_sha256=manifest["manifest_sha256"], config_sha256="e" * 64,
        )
        self.assertEqual(20, report["attempted_scenarios"])
        self.assertEqual(20, report["paired_outcomes"]["unscored"])
        self.assertTrue(all(len(row["arms"]) == 2 for row in report["scenarios"]))
        self.assertEqual(20, report["arms"]["v15"]["attempted"])
        self.assertEqual(0, report["arms"]["v16"]["attempted"])
        with self.assertRaises(ValueError):
            build_public_report(
                records, scores={"D001": {"winner": "v15", "metrics": {}}},
                manifest_sha256=manifest["manifest_sha256"], config_sha256="e" * 64,
            )


class IncrementalExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = AtomicCheckpoint(Path(self.temporary.name) / "r7d-checkpoint.private.json")
        self.manifest = _manifest()
        self.config = _config()
        self.authorization = GateAndReportTests()._authorization(self.manifest, self.config)

    @staticmethod
    def _summary(*, headers=None, complete=True, status=None, provider_calls=1, cost=0.000021):
        attempt = {
            "provider": "groq", "model": "openai/gpt-oss-120b", "status_code": status,
            "rate_limit_headers": headers or {},
        }
        return {
            "token_usage_complete": complete,
            "estimated_cost_usd": cost if complete else None,
            "provider_calls": provider_calls,
            "input_tokens": 100 if complete else None,
            "output_tokens": 10 if complete else None,
            "by_model": {"groq:openai/gpt-oss-120b": {}},
            "attempts": [attempt],
        }

    async def _run(self, *, pairs, gateway, summaries, resume_after_rate_limit=False, clock=None):
        with (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
              patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
              patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
              patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
              patch("scientific_rag.llm_gateway.summarize_trace", side_effect=summaries)):
            return await run_incremental_evaluation(
                self.manifest, self.config, authorization=self.authorization,
                checkpoint_store=self.store, pairs=pairs, gateway=gateway,
                resume_after_rate_limit=resume_after_rate_limit,
                provenance_resolver=lambda: {"sha": "a" * 40, "source": "test"},
                **({"clock": clock} if clock is not None else {}),
            )

    @staticmethod
    def _headers():
        return {
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "29",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "8000",
            "x-ratelimit-reset-requests": "1h",
            "x-ratelimit-reset-tokens": "1m",
        }

    async def test_checkpoint_recovery_preserves_order_budget_and_prompt_fingerprints(self):
        calls = []

        async def gateway(role, temperature, prompt, schema, *, max_output_tokens):
            calls.append((prompt, schema))
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        first = await self._run(
            pairs=1, gateway=gateway,
            summaries=[self._summary(headers=self._headers()) for _ in range(2)],
        )
        first_revision = first["checkpoint_revision"]
        self.assertEqual(0o600, stat.S_IMODE(self.store.path.stat().st_mode))
        self.assertEqual(["D001"], [record["scenario_id"] for record in first["records"]])
        self.assertEqual(["v15", "v16"], [arm["prompt_version"] for arm in first["records"][0]["arms"]])
        original_fingerprints = copy.deepcopy(first["prompt_contracts"])
        second = await self._run(
            pairs=1, gateway=gateway,
            summaries=[self._summary(headers=self._headers()) for _ in range(2)],
        )
        self.assertEqual(["D001", "D002"], [record["scenario_id"] for record in second["records"]])
        self.assertEqual(["v16", "v15"], [arm["prompt_version"] for arm in second["records"][1]["arms"]])
        self.assertEqual(original_fingerprints, second["prompt_contracts"])
        self.assertEqual(4, second["execution"]["budget"]["provider_calls_charged"])
        self.assertAlmostEqual(0.000084, second["execution"]["budget"]["cost_complete_usd"])
        self.assertGreater(second["checkpoint_revision"], first_revision)
        self.assertEqual(4, len(calls))

    async def test_repeated_completed_campaign_is_refused_without_a_call(self):
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        checkpoint = await self._run(
            pairs=20, gateway=gateway,
            summaries=[self._summary(headers=self._headers()) for _ in range(40)],
        )
        packet, blind_key = build_blind_scoring_packet(self.manifest, checkpoint["records"])
        self.assertEqual(20, len(packet["cases"]))
        self.assertEqual(20, len(blind_key["arms"]))
        self.assertTrue(all(case["scorable"] for case in packet["cases"]))
        with self.assertRaisesRegex(CheckpointError, "no unexecuted pairs"):
            await self._run(pairs=1, gateway=gateway, summaries=[])
        self.assertEqual(40, calls)

    async def test_partial_pair_stops_when_remaining_capacity_is_unknown(self):
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        checkpoint = await self._run(
            pairs=1, gateway=gateway, summaries=[self._summary(headers={})],
        )
        record = checkpoint["records"][0]
        self.assertEqual("partial", record["status"])
        self.assertEqual(["v15"], [arm["prompt_version"] for arm in record["arms"]])
        self.assertEqual("rate_limit_capacity_unknown", checkpoint["execution"]["stop_code"])
        self.assertEqual(1, calls)

    async def test_explicit_resume_uses_elapsed_token_reset_instead_of_stale_remaining(self):
        now = [1_000.0]
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        exhausted = self._headers() | {
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "1m0.5s",
        }
        first = await self._run(
            pairs=1, gateway=gateway, summaries=[self._summary(headers=exhausted)],
            clock=lambda: now[0],
        )
        self.assertEqual("rate_limit", first["execution"]["stop_code"])
        self.assertEqual(
            1_060.5, first["execution"]["rate_limit_state"]["reset_at_unix"]["tokens"],
        )
        self.assertEqual(
            4_600.0, first["execution"]["rate_limit_state"]["reset_at_unix"]["requests"],
        )
        self.assertEqual(1, calls)

        now[0] += 61
        resumed = await self._run(
            pairs=1, gateway=gateway, summaries=[self._summary(headers=self._headers())],
            resume_after_rate_limit=True, clock=lambda: now[0],
        )
        self.assertEqual("completed", resumed["records"][0]["status"])
        self.assertEqual(2, resumed["execution"]["budget"]["provider_calls_charged"])
        self.assertEqual(1, len(resumed["execution"]["rate_limit_continuations"]))
        self.assertEqual(2, calls)

    async def test_explicit_resume_before_reset_remains_stopped_without_a_call(self):
        now = [1_000.0]
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        exhausted = self._headers() | {"x-ratelimit-remaining-tokens": "0"}
        await self._run(
            pairs=1, gateway=gateway, summaries=[self._summary(headers=exhausted)],
            clock=lambda: now[0],
        )
        now[0] += 30
        checkpoint = await self._run(
            pairs=1, gateway=gateway, summaries=[], resume_after_rate_limit=True,
            clock=lambda: now[0],
        )
        self.assertEqual("rate_limit", checkpoint["execution"]["stop_code"])
        self.assertIn("has not elapsed", checkpoint["execution"]["stopped"])
        self.assertEqual(1, calls)

    async def test_malformed_reset_metadata_refuses_explicit_resume(self):
        now = [1_000.0]
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        exhausted = self._headers() | {
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "tomorrow",
        }
        await self._run(
            pairs=1, gateway=gateway, summaries=[self._summary(headers=exhausted)],
            clock=lambda: now[0],
        )
        now[0] += 3_600
        checkpoint = await self._run(
            pairs=1, gateway=gateway, summaries=[], resume_after_rate_limit=True,
            clock=lambda: now[0],
        )
        self.assertEqual("rate_limit_capacity_unknown", checkpoint["execution"]["stop_code"])
        self.assertEqual(1, calls)

    async def test_cancellation_persists_attempt_and_refuses_unsafe_resume(self):
        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await self._run(
                pairs=1, gateway=gateway,
                summaries=[self._summary(complete=False, cost=None)],
            )
        checkpoint = self.store.load()
        self.assertEqual(1, checkpoint["execution"]["budget"]["provider_calls_charged"])
        self.assertGreater(checkpoint["execution"]["budget"]["uncertain_reserved_usd"], 0)
        self.assertIsNotNone(checkpoint["active_arm"])
        with self.assertRaisesRegex(CheckpointError, "unresolved provider attempt or arm"):
            await self._run(pairs=1, gateway=gateway, summaries=[])
        self.assertEqual(1, calls)

    async def test_429_stops_after_one_attempt_without_retrying_other_arm(self):
        class RateLimitError(Exception):
            status_code = 429

        calls = 0

        async def gateway(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise RateLimitError("rate limit exceeded")

        checkpoint = await self._run(
            pairs=1, gateway=gateway,
            summaries=[self._summary(headers={"retry-after": "60"}, complete=False, status=429)],
        )
        self.assertEqual("rate_limit", checkpoint["execution"]["stop_code"])
        self.assertEqual(1, calls)
        self.assertEqual(1, len(checkpoint["records"][0]["arms"]))
        self.assertEqual("interrupted", checkpoint["records"][0]["arms"][0]["status"])   # R7D correction
        with self.assertRaisesRegex(CheckpointError, "pre-transmission capacity stop"):
            await self._run(
                pairs=1, gateway=gateway, summaries=[], resume_after_rate_limit=True,
            )
        self.assertEqual(1, calls)

    async def test_checkpoint_rejects_prompt_and_billing_reconciliation_tampering(self):
        async def gateway(*args, **kwargs):
            return StructuredEvidenceAnswerV2(claims=[_claim()])

        await self._run(
            pairs=1, gateway=gateway,
            summaries=[self._summary(headers=self._headers()) for _ in range(2)],
        )
        valid = self.store.load()
        changed = copy.deepcopy(valid)
        changed["prompt_contracts"][0]["prompt_sha256"] = "0" * 64
        self.store.write(changed)
        with self.assertRaisesRegex(CheckpointError, "prompt_contracts"):
            await self._run(pairs=1, gateway=gateway, summaries=[])

        # Restore a valid checkpoint, then make the internally checksummed ledger disagree
        # with its immutable per-attempt audit.
        changed = copy.deepcopy(valid)
        changed["execution"]["budget"]["cost_complete_usd"] += 0.1
        changed["execution"]["budget"]["budget_charged_usd"] += 0.1
        self.store.write(changed)
        with self.assertRaisesRegex(CheckpointError, "complete usage"):
            await self._run(pairs=1, gateway=gateway, summaries=[])

        changed = copy.deepcopy(valid)
        changed["execution"]["budget"]["budget_charged_usd"] -= 0.00001
        self.store.write(changed)
        with self.assertRaisesRegex(ValueError, "does not reconcile with its components"):
            await self._run(pairs=1, gateway=gateway, summaries=[])

    async def test_campaign_lock_refuses_concurrent_execution(self):
        with self.store.locked():
            with self.assertRaises(CampaignLocked):
                with self.store.locked():
                    self.fail("concurrent lock unexpectedly acquired")


if __name__ == "__main__":
    unittest.main()
