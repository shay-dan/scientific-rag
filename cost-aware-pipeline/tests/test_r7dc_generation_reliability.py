"""R7D-C: typed provider failures, reasoning configuration and cumulative campaign accounting.

Written before the implementation (tests first). Deterministic and offline: a fake structured
gateway raises errors shaped like the OpenAI SDK's status errors (``status_code``, ``body``,
``code``, ``type``, ``request_id``), with synthetic provider trace summaries and real private
checkpoint and ledger writes in temporary directories. No provider request, no credential and
no private benchmark artifact. The real SDK/HTTP serialization is covered separately by
``test_r7dc_http_contract.py`` (production Docker suite).

Reproduced failure (owner D002 run at 796e0d2): HTTP 400 ``json_validate_failed`` with the
provider message "max completion tokens reached before generating a valid document ..." and a
syntactically valid but incomplete ``failed_generation``. The runner treated it as an
infrastructure failure with unknown billing and stopped the campaign terminally.
"""
import copy
import importlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.benchmark import generation_evaluation as harness
from app.benchmark.generation_evaluation import (
    build_public_report,
    canonical_sha256,
    check_paid_authorization,
    dry_run,
    validate_config,
)
from app.benchmark.incremental_generation_evaluation import (
    CHECKPOINT_SCHEMA,
    AtomicCheckpoint,
    CampaignLocked,
    CheckpointError,
    run_incremental_evaluation,
)
from app.weak_model_synthesis import StructuredEvidenceAnswerV2, V2RepairBatch
from tests import test_r7d_execution_safety_reproduction as repro
from tests import test_r7d_generation_evaluation as base

GROQ_ROUTE = "groq:openai/gpt-oss-120b"
SECRET = "gsk_" + "S3cr3tKeyMaterial" * 3
PRIVATE_TEXT = "PRIVATE-EVIDENCE: the treated group reported a 12% response rate"
TRUNCATION_MESSAGE = ("max completion tokens reached before generating a valid document: the output was "
                      "truncated to fit max_completion_tokens and is missing required content.")
INCOMPLETE_GENERATION = json.dumps({"claims": [{"text": PRIVATE_TEXT, "source_id": "S1",
                                                "evidence_quote": PRIVATE_TEXT}]})
HISTORICAL_V1_CONFIG_SHA256 = "e3b2a812adec5606efc6d6d44a55884d7f7a186761950a3bc9a6baa9c27ef965"


def _module(name):
    return importlib.import_module(name)


def failures():
    return _module("scientific_rag.provider_failures")


def ledger_module():
    return _module("app.benchmark.program_ledger")


class FakeStatusError(Exception):
    """Shaped like ``openai.APIStatusError``; ``str()`` echoes the body as the SDK does."""

    def __init__(self, status, body=None, *, request_id="req_01jabcdefghjkmnpqrstvwxyz0"):
        self.status_code = status
        self.body = body
        self.code = body.get("code") if isinstance(body, dict) else None
        self.type = body.get("type") if isinstance(body, dict) else None
        self.param = body.get("param") if isinstance(body, dict) else None
        self.message = (body or {}).get("message") if isinstance(body, dict) else f"Error code: {status}"
        self.request_id = request_id
        self.response = SimpleNamespace(status_code=status, headers={"x-request-id": request_id})
        super().__init__(f"Error code: {status} - {{'error': {body!r}}} key={SECRET}")


def truncation_error(generation=INCOMPLETE_GENERATION, message=TRUNCATION_MESSAGE):
    return FakeStatusError(400, {"message": message, "type": "invalid_request_error",
                                 "code": "json_validate_failed", "failed_generation": generation})


def unsupported_parameter_error():
    return FakeStatusError(400, {"message": "'reasoning_effort' is not supported with this model",
                                 "type": "invalid_request_error", "param": "reasoning_effort"})


class InvalidStructuredResponse(ValueError):
    """Name-compatible stand-in for the gateway's local structured-output failure."""


def v2_config(**updates):
    value = base._config(schema="scientific-rag-generation-eval-config/v2", reasoning_effort="low",
                         completion_budget_failure_policy="score_as_failure")
    value.update(updates)
    return value


def historical_config():
    return base._config(code_sha="796e0d265df3419702df07830dec24034d71c062")


# --------------------------------------------------------------------------- classification

class ProviderFailureClassificationTests(unittest.TestCase):
    def classify(self, error, schema=StructuredEvidenceAnswerV2):
        description = failures().describe_provider_failure(error, schema)
        serialized = json.dumps(description, sort_keys=True)
        for forbidden in (SECRET, PRIVATE_TEXT, TRUNCATION_MESSAGE):
            self.assertNotIn(forbidden, serialized, "provider diagnostics leaked private or secret text")
        return description

    def test_explicit_truncation_with_valid_but_incomplete_json_is_completion_budget_exhaustion(self):
        description = self.classify(truncation_error())
        self.assertEqual("completion_budget_exhausted", description["failure_class"])
        self.assertEqual((400, "json_validate_failed", "invalid_request_error"),
                         (description["http_status"], description["provider_error_code"],
                          description["provider_error_type"]))
        self.assertTrue(description["failed_generation_returned"])
        self.assertEqual(len(INCOMPLETE_GENERATION), description["failed_generation_chars"])
        self.assertTrue(description["failed_generation_parses_as_json"])
        self.assertEqual(["limitations", "unanswered_parts"], description["missing_top_level_fields"])
        self.assertEqual(("confirmed", "req_01jabcdefghjkmnpqrstvwxyz0"), (description["transmission"], description["request_id"]))

    def test_truncation_with_syntactically_truncated_json(self):
        description = self.classify(truncation_error(generation='{"claims": [{"text": "unterminated'))
        self.assertEqual("completion_budget_exhausted", description["failure_class"])
        self.assertFalse(description["failed_generation_parses_as_json"])
        self.assertIsNone(description["missing_top_level_fields"])

    def test_json_validation_failure_without_truncation_evidence_is_not_truncation(self):
        for message in ("Failed to generate JSON. Please adjust your prompt.", "", "output was truncated"):
            with self.subTest(message=message):
                description = self.classify(truncation_error(message=message))
                self.assertEqual("provider_structured_generation_failure", description["failure_class"])

    def test_truncation_wording_without_the_provider_code_is_not_truncation(self):
        body = {"message": TRUNCATION_MESSAGE, "type": "invalid_request_error", "code": "other_code"}
        self.assertNotEqual("completion_budget_exhausted", self.classify(FakeStatusError(400, body))["failure_class"])
        self.assertNotEqual("completion_budget_exhausted",
                            self.classify(FakeStatusError(500, dict(body, code="json_validate_failed")))["failure_class"])

    def test_configuration_authentication_rate_limit_and_transient_statuses(self):
        cases = {
            unsupported_parameter_error(): ("authentication_or_configuration_failure", "confirmed"),
            FakeStatusError(400, None): ("unknown_provider_failure", "confirmed"),
            FakeStatusError(401, {"message": "Invalid API Key", "code": "invalid_api_key"}):
                ("authentication_or_configuration_failure", "confirmed"),
            FakeStatusError(403, {"message": "forbidden"}): ("authentication_or_configuration_failure", "confirmed"),
            FakeStatusError(404, {"message": "model not found", "code": "model_not_found"}):
                ("authentication_or_configuration_failure", "confirmed"),
            FakeStatusError(429, {"message": "Rate limit reached", "code": "rate_limit_exceeded"}):
                ("provider_rate_limit", "confirmed"),
            FakeStatusError(500, {"message": "internal"}): ("transient_infrastructure_failure", "confirmed"),
            FakeStatusError(503, {"message": "unavailable"}): ("transient_infrastructure_failure", "confirmed"),
            ConnectionResetError("connection reset by peer"): ("transient_infrastructure_failure", "uncertain"),
            TimeoutError("timed out"): ("transient_infrastructure_failure", "uncertain"),
            RuntimeError("something else"): ("unknown_provider_failure", "uncertain"),
        }
        for error, expected in cases.items():
            with self.subTest(error=type(error).__name__, status=getattr(error, "status_code", None)):
                description = self.classify(error)
                self.assertEqual(expected, (description["failure_class"], description["transmission"]))

    def test_local_structured_output_failures(self):
        from pydantic import ValidationError
        try:
            StructuredEvidenceAnswerV2.model_validate({"claims": []})
        except ValidationError as error:
            pydantic_error = error
        for error in (InvalidStructuredResponse(f"invalid output {PRIVATE_TEXT}"), pydantic_error):
            with self.subTest(error=type(error).__name__):
                description = self.classify(error)
                self.assertEqual(("local_structured_output_failure", "confirmed"),
                                 (description["failure_class"], description["transmission"]))

    def test_an_unsent_guard_refusal_is_a_configuration_failure_that_was_not_transmitted(self):
        class OutputTokenLimitNotEnforced(ValueError):
            pass
        description = self.classify(OutputTokenLimitNotEnforced("not sent"))
        self.assertEqual(("authentication_or_configuration_failure", "not_transmitted"),
                         (description["failure_class"], description["transmission"]))

    def test_classification_follows_the_exception_chain(self):
        try:
            try:
                raise truncation_error()
            except FakeStatusError as inner:
                raise RuntimeError("wrapped by a framework") from inner
        except RuntimeError as outer:
            self.assertEqual("completion_budget_exhausted", self.classify(outer)["failure_class"])

    def test_every_failure_class_has_all_five_outcome_dimensions(self):
        table = harness.FAILURE_DECISION_TABLE
        self.assertEqual(set(failures().FAILURE_CLASSES) | {"local_evidence_validation_failure",
                                                             "pre_transmission_capacity_pause"}, set(table))
        for failure_class, row in table.items():
            with self.subTest(failure_class=failure_class):
                self.assertEqual({"generation_outcome", "scoring", "continuation", "transmission", "accounting"},
                                 set(row))
        self.assertEqual("excluded", table["transient_infrastructure_failure"]["scoring"])
        self.assertEqual("stop_terminal", table["unknown_provider_failure"]["continuation"])


# --------------------------------------------------------------------------- configuration

class ConfigurationContractTests(unittest.TestCase):
    def test_historical_v1_configuration_is_valid_and_its_commitment_is_unchanged(self):
        config = historical_config()
        validate_config(config)
        self.assertEqual(HISTORICAL_V1_CONFIG_SHA256, canonical_sha256(config))
        self.assertNotIn("reasoning_effort", harness.generation_settings(config))
        self.assertEqual({"max_output_tokens": 2000}, harness.generation_settings(config))

    def test_v2_binds_reasoning_effort_and_the_completion_budget_policy(self):
        for effort in ("low", "medium", "high"):
            with self.subTest(effort=effort):
                config = validate_config(v2_config(reasoning_effort=effort))
                self.assertEqual({"max_output_tokens": 2000, "reasoning_effort": effort},
                                 harness.generation_settings(config))
        self.assertNotEqual(canonical_sha256(v2_config(reasoning_effort="low")),
                            canonical_sha256(v2_config(reasoning_effort="medium")))
        self.assertNotEqual(canonical_sha256(v2_config(max_output_tokens=2000)),
                            canonical_sha256(v2_config(max_output_tokens=3000)))
        self.assertNotEqual(canonical_sha256(v2_config(completion_budget_failure_policy="stop")),
                            canonical_sha256(v2_config()))

    def test_v2_rejects_unsupported_or_missing_settings(self):
        invalid = {
            "unknown effort": v2_config(reasoning_effort="extreme"),
            "missing effort": {k: v for k, v in v2_config().items() if k != "reasoning_effort"},
            "null effort": v2_config(reasoning_effort=None),
            "unsupported provider": v2_config(provider="openrouter"),
            "unsupported model": v2_config(model="llama-3.3-70b-versatile"),
            "unknown policy": v2_config(completion_budget_failure_policy="retry"),
            "missing policy": {k: v for k, v in v2_config().items() if k != "completion_budget_failure_policy"},
            "unknown field": v2_config(repair_reasoning_effort="high"),
            "v1 with reasoning": base._config(reasoning_effort="low"),
        }
        for label, config in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_config(config)

    def test_v2_dry_run_rejects_a_completion_limit_that_cannot_fit_published_free_capacity(self):
        plan = dry_run(base._manifest(), v2_config(max_output_tokens=8000, model_context_tokens=131072))
        self.assertIn("output-token limit cannot fit the published Groq Free tokens-per-minute sample",
                      plan["blockers"])
        self.assertEqual({"reasoning_effort": "low", "max_output_tokens": 2000,
                          "applies_to": ["initial_generation", "repair"],
                          "completion_budget_failure_policy": "score_as_failure"},
                         dry_run(base._manifest(), v2_config())["generation_settings"])

    def test_v1_dry_run_shape_is_unchanged(self):
        plan = dry_run(base._manifest(), base._config())
        self.assertNotIn("generation_settings", plan)

    def test_v2_authorization_binds_reasoning_policy_and_program_baseline(self):
        manifest, config = base._manifest(), v2_config()
        authorization = v2_authorization(manifest, config, baseline_sha256="b" * 64)
        self.assertEqual(authorization, check_paid_authorization(authorization, manifest=manifest, config=config))
        for field, value in (("reasoning_effort", "medium"), ("completion_budget_failure_policy", "stop"),
                             ("program_baseline_sha256", None), ("program_id", ""),
                             ("schema", "scientific-rag-generation-eval-authorization/v1")):
            with self.subTest(field=field), self.assertRaises(harness.AuthorizationError):
                check_paid_authorization(dict(authorization, **{field: value}), manifest=manifest, config=config)
        with self.assertRaises(harness.AuthorizationError):   # a v1 authorization cannot authorize a v2 config
            check_paid_authorization(base.GateAndReportTests()._authorization(manifest, config),
                                     manifest=manifest, config=config)

    def test_the_one_shot_runner_refuses_v2_configurations(self):
        manifest, config = base._manifest(), v2_config()
        authorization = v2_authorization(manifest, config, baseline_sha256="b" * 64)
        with self.assertRaisesRegex(harness.AuthorizationError, "incremental"):
            import asyncio
            asyncio.run(harness.run_paid_evaluation(manifest, config, authorization=authorization,
                                                    gateway=lambda *a, **k: None))


def v2_authorization(manifest, config, *, baseline_sha256, program_id="r7d-b-program", authorization_id="r7d-c-a1"):
    return {
        "schema": "scientific-rag-generation-eval-authorization/v2",
        "authorization_id": authorization_id, "approved_by": "owner",
        "manifest_sha256": manifest["manifest_sha256"], "accepted_code_sha": config["code_sha"],
        "config_sha256": canonical_sha256(config), "provider": config["provider"], "model": config["model"],
        "max_calls": config["max_calls"], "max_cost_usd": config["max_cost_usd"],
        "model_context_tokens": config["model_context_tokens"], "max_input_tokens": config["max_input_tokens"],
        "max_output_tokens": config["max_output_tokens"], "reasoning_effort": config["reasoning_effort"],
        "completion_budget_failure_policy": config["completion_budget_failure_policy"],
        "program_id": program_id, "program_baseline_sha256": baseline_sha256,
    }


# --------------------------------------------------------------------------- scripted provider

class ScriptedProvider:
    """Each request takes the next action; records the authorized settings it received."""

    def __init__(self, actions=()):
        self.actions = list(actions)
        self.requests: list[dict] = []
        self.last_action = "ok"

    async def gateway(self, role, temperature, prompt, schema, *, max_output_tokens, **settings):
        action = self.actions.pop(0) if self.actions else "ok"
        self.last_action = action
        self.requests.append({"schema": schema, "max_output_tokens": max_output_tokens, **settings})
        if action == "truncate" or action == "truncate_with_usage":
            raise truncation_error()
        if action == "json_validate":
            raise truncation_error(message="Failed to generate JSON. Please adjust your prompt.")
        if action == "auth":
            raise FakeStatusError(401, {"message": "Invalid API Key", "code": "invalid_api_key"})
        if action == "quota_words_in_generation":
            raise truncation_error(generation=json.dumps({"claims": [{"text": "quota exceeded; rate limit"}]}))
        if action == "local_invalid":
            raise InvalidStructuredResponse(f"Provider returned invalid structured output: {PRIVATE_TEXT}")
        if schema is V2RepairBatch:
            return V2RepairBatch(repairs=[{"target": "claims.1", "action": "replace", "replacement": base._claim()}])
        if action == "needs_repair":
            return StructuredEvidenceAnswerV2(claims=[base._claim(quote="This sentence is not in the passage.")])
        return StructuredEvidenceAnswerV2(claims=[base._claim()])

    def summary(self, *_args):
        action = self.last_action
        failed = action in {"truncate", "json_validate", "auth", "quota_words_in_generation"}
        status = {"auth": 401}.get(action, 400 if failed else None)
        complete = not failed or action == "truncate_with_usage"
        return {
            "token_usage_complete": complete, "estimated_cost_usd": 0.0003 if complete else None,
            "provider_calls": 1, "input_tokens": 700 if complete else None,
            "output_tokens": 2000 if action == "truncate_with_usage" else (600 if complete else None),
            "by_model": {GROQ_ROUTE: {}},
            "attempts": [{"provider": "groq", "model": "openai/gpt-oss-120b", "status_code": status,
                          "rate_limit_headers": repro._headers(repro.TPM_LIMIT)}],
        }


def _trace_patches(provider):
    return (patch("scientific_rag.llm_gateway.provider_order", return_value=("groq",)),
            patch("scientific_rag.llm_gateway.model_for", return_value="openai/gpt-oss-120b"),
            patch("scientific_rag.llm_gateway.begin_trace", return_value=object()),
            patch("scientific_rag.llm_gateway.end_trace", return_value=[]),
            patch("scientific_rag.llm_gateway.summarize_trace", side_effect=provider.summary))


class _CampaignHarness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "private"
        self.directory.mkdir(mode=0o700)
        self.manifest = base._manifest()

    def checkpoint(self, name="r7d-checkpoint.private.json"):
        return AtomicCheckpoint(self.directory / name)

    async def run_v1(self, provider, store, *, pairs=1, config=None):
        config = config or base._config()
        authorization = base.GateAndReportTests()._authorization(self.manifest, config)
        patches = _trace_patches(provider)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return await run_incremental_evaluation(
                self.manifest, config, authorization=authorization, checkpoint_store=store, pairs=pairs,
                gateway=provider.gateway, provenance_resolver=lambda: {"sha": config["code_sha"], "source": "test"},
                clock=lambda: 1_000.0)

    def ledger_store(self, name="r7d-program-ledger.private.json"):
        return ledger_module().ProgramLedgerStore(self.directory / name)

    def initialized_ledger(self, **kwargs):
        store = self.ledger_store()
        ledger_module().initialize_program_ledger(store, program_id="r7d-b-program", max_calls=80,
                                                  max_cost_usd=1.5, now=1_000.0, **kwargs)
        return store

    async def run_v2(self, provider, store, ledger_store, *, pairs=1, config=None, authorization=None,
                     resume=False):
        config = config or v2_config()
        if authorization is None:
            baseline = ledger_module().program_baseline(ledger_store.load())
            authorization = v2_authorization(self.manifest, config, baseline_sha256=baseline["sha256"])
        patches = _trace_patches(provider)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return await run_incremental_evaluation(
                self.manifest, config, authorization=authorization, checkpoint_store=store, pairs=pairs,
                gateway=provider.gateway, provenance_resolver=lambda: {"sha": config["code_sha"], "source": "test"},
                clock=lambda: 1_000.0, program_ledger=ledger_store, resume_after_rate_limit=resume)

    def assert_no_leak(self, path: Path):
        text = path.read_text(encoding="utf-8")
        self.assertNotIn(SECRET, text)
        self.assertNotIn(TRUNCATION_MESSAGE, text)


# --------------------------------------------------------------------------- runner decisions

class D002ReproductionTests(_CampaignHarness):
    async def test_v1_configuration_keeps_the_historical_terminal_billing_unknown_stop(self):
        """Historical behavior (796e0d2), preserved for v1 configurations."""
        provider = ScriptedProvider(["ok", "ok", "truncate"])
        checkpoint = await self.run_v1(provider, self.checkpoint(), pairs=2)
        self.assertEqual("billing_unknown", checkpoint["execution"]["stop_code"])
        arm = checkpoint["records"][1]["arms"][0]
        self.assertEqual(("interrupted", "v16"), (arm["status"], arm["prompt_version"]))
        self.assertEqual("completion_budget_exhausted", arm["provider_failure_class"])

    async def test_confirmed_truncation_is_a_scored_terminal_outcome_under_the_predeclared_policy(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["ok", "ok", "truncate"])
        store = self.checkpoint()
        checkpoint = await self.run_v2(provider, store, ledger_store, pairs=3)
        self.assertIsNone(checkpoint["execution"]["stop_code"])
        self.assertEqual(6, len(provider.requests))   # D002's failed arm is never re-sent; D003 runs
        arm = checkpoint["records"][1]["arms"][0]
        self.assertEqual(("failed", "model_output", "completion_budget_exhausted", "initial"),
                         (arm["status"], arm["failure_class"], arm["failure_kind"], arm["failure_phase"]))
        self.assertEqual("completed", checkpoint["records"][1]["status"])
        self.assertEqual({"generation_outcome": "completion_budget_exhausted", "scoring": "scored_failure",
                          "continuation": "continue", "transmission": "confirmed",
                          "accounting": "uncertain_reserved"}, arm["outcome_dimensions"])
        self.assertEqual(["limitations", "unanswered_parts"], arm["provider_failure"]["missing_top_level_fields"])
        budget = checkpoint["execution"]["budget"]
        self.assertEqual(6, budget["provider_calls_charged"])
        self.assertGreater(budget["uncertain_reserved_usd"], 0.0)   # unknown usage is never zero
        self.assertAlmostEqual(5 * 0.0003, budget["cost_complete_usd"], places=8)
        report = build_public_report(checkpoint["records"], scores=None, manifest_sha256="f" * 64,
                                     config_sha256="e" * 64)
        self.assertEqual(3, sum(row["scorable"] for row in report["scenarios"]))
        self.assert_no_leak(store.path)

    async def test_truncation_with_complete_usage_settles_measured_cost(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["truncate_with_usage"])
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store)
        arm = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("failed", "completion_budget_exhausted"), (arm["status"], arm["failure_kind"]))
        self.assertEqual("complete", arm["outcome_dimensions"]["accounting"])
        self.assertEqual(0.0, checkpoint["execution"]["budget"]["uncertain_reserved_usd"])

    async def test_policy_stop_makes_truncation_a_terminal_unscored_stop(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["truncate"])
        store = self.checkpoint()
        config = v2_config(completion_budget_failure_policy="stop")
        checkpoint = await self.run_v2(provider, store, ledger_store, pairs=2, config=config)
        self.assertEqual("completion_budget_exhausted", checkpoint["execution"]["stop_code"])
        arm = checkpoint["records"][0]["arms"][0]
        self.assertEqual("interrupted", arm["status"])
        self.assertEqual("excluded", arm["outcome_dimensions"]["scoring"])
        self.assertEqual(1, len(provider.requests))
        with self.assertRaisesRegex(CheckpointError, r"stopped \(completion_budget_exhausted\)"):
            await self.run_v2(provider, store, ledger_store, config=config)
        with self.assertRaisesRegex(CheckpointError, r"stopped \(completion_budget_exhausted\)"):
            await self.run_v2(provider, store, ledger_store, config=config, resume=True)
        self.assertEqual(1, len(provider.requests))

    async def test_truncation_during_repair_is_a_scored_post_repair_failure(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["needs_repair", "truncate"])
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store)
        arm = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("failed", "completion_budget_exhausted", "repair"),
                         (arm["status"], arm["failure_kind"], arm["failure_phase"]))
        self.assertEqual([StructuredEvidenceAnswerV2, V2RepairBatch, StructuredEvidenceAnswerV2],
                         [request["schema"] for request in provider.requests])

    async def test_unconfirmed_json_validation_failure_and_authentication_failures_stop_unscored(self):
        for action, code in (("json_validate", "provider_structured_generation_failure"),
                             ("auth", "authentication_or_configuration_failure")):
            with self.subTest(action=action):
                self.setUp()
                ledger_store = self.initialized_ledger()
                provider = ScriptedProvider([action])
                checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store, pairs=2)
                self.assertEqual(code, checkpoint["execution"]["stop_code"])
                arm = checkpoint["records"][0]["arms"][0]
                self.assertEqual(("interrupted", code), (arm["status"], arm["provider_failure_class"]))
                self.assertEqual("excluded", arm["outcome_dimensions"]["scoring"])
                self.assertEqual(1, len(provider.requests))

    async def test_scientific_text_in_a_failed_generation_is_not_mistaken_for_a_quota_error(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["quota_words_in_generation"])
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store)
        self.assertNotEqual("rate_limit", checkpoint["execution"]["stop_code"])
        self.assertEqual("completion_budget_exhausted", checkpoint["records"][0]["arms"][0]["failure_kind"])

    async def test_local_structured_failure_with_complete_usage_is_scored_and_continues(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["local_invalid"])
        store = self.checkpoint()
        checkpoint = await self.run_v2(provider, store, ledger_store, pairs=2)
        arm = checkpoint["records"][0]["arms"][0]
        self.assertEqual(("failed", "invalid_structured_output"), (arm["status"], arm["failure_kind"]))
        self.assertEqual("local_structured_output_failure", arm["outcome_dimensions"]["generation_outcome"])
        self.assertEqual(4, len(provider.requests))

    async def test_generation_and_repair_receive_the_authorized_settings(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider(["needs_repair"])
        await self.run_v2(provider, self.checkpoint(), ledger_store, config=v2_config(max_output_tokens=3000))
        self.assertEqual(3, len(provider.requests))
        for request in provider.requests:
            self.assertEqual((3000, "low"), (request["max_output_tokens"], request["reasoning_effort"]))

    async def test_v1_gateways_receive_no_reasoning_setting(self):
        provider = ScriptedProvider()
        await self.run_v1(provider, self.checkpoint())
        self.assertTrue(provider.requests)
        self.assertTrue(all("reasoning_effort" not in request for request in provider.requests))

    async def test_configuration_drift_is_rejected_on_continuation(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider()
        store = self.checkpoint()
        first = await self.run_v2(provider, store, ledger_store)
        drifted = v2_config(reasoning_effort="medium")
        authorization = v2_authorization(self.manifest, drifted,
                                         baseline_sha256=first["program"]["baseline_sha256"])
        with self.assertRaisesRegex(CheckpointError, "binding mismatch"):
            await self.run_v2(provider, store, ledger_store, config=drifted, authorization=authorization)


# --------------------------------------------------------------------------- cumulative accounting

def _historical_checkpoint(path: Path, *, authorization_id, calls, active_attempt=None):
    """A checkpoint-v2 file shaped like the merged runner's, with the given call audit."""
    ledger = harness.BudgetLedger(80, 1.5)
    for call in calls:
        ledger.reserve(call["reserved_cost_usd"])
        ledger.settle(call["reserved_cost_usd"], call)
    if active_attempt is not None:
        ledger.reserve(active_attempt["reserved_cost_usd"])
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_revision": 0, "manifest_sha256": "3dc9e107" + "0" * 56,
        "config_sha256": HISTORICAL_V1_CONFIG_SHA256, "authorization_sha256": "c" * 64,
        "authorization_id": authorization_id, "code_sha": "796e0d265df3419702df07830dec24034d71c062",
        "provider": "groq", "model": "openai/gpt-oss-120b", "prompt_contracts": [], "records": [],
        "active_arm": None, "active_attempt": active_attempt,
        "execution": {"authorization_id": authorization_id, "budget": ledger.snapshot(), "code": {},
                      "stopped": "provider usage was incomplete; spend is uncertain", "stop_code": "billing_unknown",
                      "stop_context": None, "calls": calls, "rate_limit_state": {},
                      "rate_limit_continuations": [], "stages": []},
    }
    return AtomicCheckpoint(path).write(value)


def _call(*, complete, reserved, cost=None):
    return {"cost_status": "complete" if complete else "incomplete", "provider_calls": 1,
            "cost_usd": cost if complete else None, "known_cost_lower_bound_usd": cost or 0.0,
            "reserved_cost_usd": reserved, "input_tokens": None, "output_tokens": None}


class ProgramLedgerTests(_CampaignHarness):
    def import_owner_history(self, ledger_store):
        original = self.directory / "r7d-original.private.json"
        replacement = self.directory / "r7d-replacement.private.json"
        _historical_checkpoint(original, authorization_id="r7d-b-free-796e0d2-original",
                               calls=[_call(complete=False, reserved=0.00165105)])
        _historical_checkpoint(replacement, authorization_id="r7d-b-free-796e0d2-replacement", calls=[
            _call(complete=True, reserved=0.0018, cost=0.0010194),
            _call(complete=True, reserved=0.0018, cost=0.0008823),
            _call(complete=False, reserved=0.00190095)])
        module = ledger_module()
        module.import_campaign(ledger_store, original)
        module.import_campaign(ledger_store, replacement)
        return original, replacement

    def test_initialization_is_private_exclusive_and_bounded(self):
        store = self.initialized_ledger()
        self.assertEqual(0o600, stat.S_IMODE(store.path.stat().st_mode))
        module = ledger_module()
        with self.assertRaises(module.ProgramLedgerError):
            module.initialize_program_ledger(store, program_id="again", max_calls=80, max_cost_usd=1.5, now=1.0)
        for calls, cost in ((81, 1.5), (80, 1.51), (0, 1.5), (80, 0.0)):
            with self.subTest(calls=calls, cost=cost), self.assertRaises(module.ProgramLedgerError):
                module.initialize_program_ledger(self.ledger_store(f"other-{calls}-{cost}.json"),
                                                 program_id="p", max_calls=calls, max_cost_usd=cost, now=1.0)

    def test_owner_history_reconciles_to_the_documented_cumulative_totals(self):
        store = self.initialized_ledger()
        original, replacement = self.import_owner_history(store)
        before = {path: path.read_bytes() for path in (original, replacement)}
        module = ledger_module()
        baseline = module.program_baseline(store.load())
        budget = baseline["budget"]
        self.assertEqual(4, budget["provider_calls_charged"])
        self.assertAlmostEqual(0.0019017, budget["cost_complete_usd"], places=8)
        self.assertAlmostEqual(0.003552, budget["uncertain_reserved_usd"], places=8)
        self.assertAlmostEqual(0.0054537, budget["budget_charged_usd"], places=8)
        self.assertEqual(0.0, budget["unsettled_reserved_usd"])
        self.assertEqual({"calls": 76, "cost_usd": 1.4945463}, baseline["remaining"])
        # Import is idempotent and never modifies the historical checkpoints.
        module.import_campaign(store, replacement)
        self.assertEqual(baseline["sha256"], module.program_baseline(store.load())["sha256"])
        self.assertEqual(before, {path: path.read_bytes() for path in (original, replacement)})

    def test_import_rejects_tampered_conflicting_or_unreconciled_checkpoints(self):
        store = self.initialized_ledger()
        _, replacement = self.import_owner_history(store)
        module = ledger_module()
        value = json.loads(replacement.read_text())
        value["execution"]["calls"].append(_call(complete=True, reserved=0.001, cost=0.0005))
        conflicting = self.directory / "conflict.private.json"
        conflicting.write_text(json.dumps(value))
        os.chmod(conflicting, 0o600)
        with self.assertRaisesRegex(module.ProgramLedgerError, "checksum"):
            module.import_campaign(store, conflicting)
        AtomicCheckpoint(conflicting).write(value)   # valid checksum, but a different state of the campaign
        with self.assertRaisesRegex(module.ProgramLedgerError, "already recorded"):
            module.import_campaign(store, conflicting)
        value["authorization_id"] = value["execution"]["authorization_id"] = "unreconciled"
        AtomicCheckpoint(conflicting).write(value)    # calls no longer match the budget snapshot
        with self.assertRaisesRegex(module.ProgramLedgerError, "reconcile"):
            module.import_campaign(store, conflicting)

    def test_an_unresolved_potentially_transmitted_attempt_stays_charged_as_uncertain(self):
        store = self.initialized_ledger()
        path = self.directory / "unresolved.private.json"
        _historical_checkpoint(path, authorization_id="unresolved", calls=[],
                               active_attempt={"reserved_cost_usd": 0.002, "state": "reserved"})
        entry = ledger_module().import_campaign(store, path)
        self.assertTrue(entry["unresolved_attempt"])
        budget = ledger_module().program_baseline(store.load())["budget"]
        self.assertEqual(1, budget["provider_calls_charged"])
        self.assertAlmostEqual(0.002, budget["uncertain_reserved_usd"], places=8)

    async def test_a_new_campaign_starts_from_the_cumulative_baseline(self):
        ledger_store = self.initialized_ledger()
        self.import_owner_history(ledger_store)
        provider = ScriptedProvider()
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store, pairs=1)
        budget = checkpoint["execution"]["budget"]
        self.assertEqual(6, budget["provider_calls_charged"])              # 4 historical + 2 new
        self.assertAlmostEqual(0.003552, budget["uncertain_reserved_usd"], places=8)
        self.assertEqual(4, checkpoint["program"]["baseline"]["provider_calls_charged"])
        state = ledger_module().accounting_summary(ledger_store.load(), checkpoint,
                                                   checkpoint_path=self.checkpoint().path)
        self.assertEqual(2, state["current_experiment"]["provider_calls_charged"])
        self.assertEqual(6, state["cumulative"]["provider_calls_charged"])
        self.assertEqual(4, state["closed_baseline"]["provider_calls_charged"])
        self.assertEqual("not_available", state["actual_provider_billing"]["status"])

    async def test_v2_campaigns_require_the_program_ledger_and_an_exact_baseline(self):
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider()
        config = v2_config()
        authorization = v2_authorization(self.manifest, config, baseline_sha256="0" * 64)
        with self.assertRaisesRegex(harness.AuthorizationError, "baseline"):
            await self.run_v2(provider, self.checkpoint(), ledger_store, authorization=authorization)
        patches = _trace_patches(provider)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
                self.assertRaisesRegex(harness.AuthorizationError, "program ledger"):
            await run_incremental_evaluation(
                self.manifest, config, authorization=v2_authorization(self.manifest, config, baseline_sha256="0" * 64),
                checkpoint_store=self.checkpoint("other.json"), pairs=1, gateway=provider.gateway,
                provenance_resolver=lambda: {"sha": config["code_sha"], "source": "test"}, clock=lambda: 1_000.0)
        self.assertEqual([], provider.requests)

    async def test_cross_campaign_call_and_cost_exhaustion_stop_before_transmission(self):
        for calls, reserved, expected_requests in ((79, 0.00001, 1), (1, 1.4995, 0)):
            with self.subTest(calls=calls):
                self.setUp()
                ledger_store = self.initialized_ledger()
                path = self.directory / "history.private.json"
                _historical_checkpoint(path, authorization_id="history", calls=[
                    _call(complete=False, reserved=reserved / calls) for _ in range(calls)])
                ledger_module().import_campaign(ledger_store, path)
                provider = ScriptedProvider()
                checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store, pairs=1)
                self.assertEqual(expected_requests, len(provider.requests))
                self.assertEqual("budget", checkpoint["execution"]["stop_code"])
                self.assertLessEqual(checkpoint["execution"]["budget"]["provider_calls_charged"], 80)
                self.assertLessEqual(checkpoint["execution"]["budget"]["budget_charged_usd"], 1.5 + 1e-9)

    async def test_one_open_campaign_at_a_time_and_no_replayed_or_reopened_campaign(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        provider = ScriptedProvider()
        store = self.checkpoint()
        first = await self.run_v2(provider, store, ledger_store)
        second_config = v2_config(max_output_tokens=3000)
        authorization = v2_authorization(self.manifest, second_config, authorization_id="r7d-c-a2",
                                         baseline_sha256=first["program"]["baseline_sha256"])
        with self.assertRaisesRegex(module.ProgramLedgerError, "already open"):
            await self.run_v2(provider, self.checkpoint("second.json"), ledger_store, config=second_config,
                              authorization=authorization)
        module.close_campaign(ledger_store, store)
        module.close_campaign(ledger_store, store)            # idempotent for the same final checkpoint
        with self.assertRaisesRegex(module.ProgramLedgerError, "closed"):
            await self.run_v2(provider, store, ledger_store)   # a closed campaign never runs again
        with self.assertRaisesRegex(module.ProgramLedgerError, "closed|already recorded"):
            await self.run_v2(provider, self.checkpoint("replay.json"), ledger_store,
                              authorization=v2_authorization(self.manifest, v2_config(),
                                                             baseline_sha256=module.program_baseline(
                                                                 ledger_store.load())["sha256"]))
        baseline = module.program_baseline(ledger_store.load())
        self.assertEqual(2, baseline["budget"]["provider_calls_charged"])
        self.assertEqual(2, len(provider.requests))

    async def test_a_crash_between_opening_and_checkpoint_creation_can_be_completed(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        config = v2_config()
        baseline = module.program_baseline(ledger_store.load())
        authorization = v2_authorization(self.manifest, config, baseline_sha256=baseline["sha256"])
        module.open_campaign(ledger_store, authorization=authorization, config=config,
                             manifest_sha256=self.manifest["manifest_sha256"], checkpoint_path=self.checkpoint().path)
        provider = ScriptedProvider()
        checkpoint = await self.run_v2(provider, self.checkpoint(), ledger_store, authorization=authorization)
        self.assertEqual(2, checkpoint["execution"]["budget"]["provider_calls_charged"])

    async def test_closing_requires_the_campaign_lock_and_ledger_operations_are_exclusive(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        store = self.checkpoint()
        await self.run_v2(ScriptedProvider(), store, ledger_store)
        with store.locked(), self.assertRaises(CampaignLocked):
            module.close_campaign(ledger_store, store)
        with ledger_store.locked(), self.assertRaises(module.ProgramLedgerLocked):
            module.close_campaign(ledger_store, store)
        with ledger_store.locked(), self.assertRaises(module.ProgramLedgerLocked):
            module.import_campaign(ledger_store, store.path)

    def test_diagnostics_are_recorded_separately_and_idempotently(self):
        module = ledger_module()
        store = self.initialized_ledger()
        source = self.directory / "diagnostic.jsonl"
        source.write_text('{"request": 1}\n')
        entry = module.record_diagnostic(store, source, label="config-discovery-2026-11-18", requests=20,
                                         estimated_cost_usd=0.0123, uncertain_reserved_usd=0.004)
        again = module.record_diagnostic(store, source, label="config-discovery-2026-11-18", requests=20,
                                         estimated_cost_usd=0.0123, uncertain_reserved_usd=0.004)
        self.assertEqual(entry, again)
        with self.assertRaises(module.ProgramLedgerError):
            module.record_diagnostic(store, source, label="config-discovery-2026-11-18", requests=21,
                                     estimated_cost_usd=0.0123, uncertain_reserved_usd=0.004)
        ledger = store.load()
        self.assertEqual(0, module.program_baseline(ledger)["budget"]["provider_calls_charged"])
        summary = module.accounting_summary(ledger)
        self.assertEqual(20, summary["diagnostics"]["requests"])
        self.assertEqual({"provider_attempts", "closed_baseline", "current_experiment", "cumulative", "diagnostics",
                          "actual_provider_billing"}, set(summary))

    def test_a_tampered_ledger_is_refused(self):
        module = ledger_module()
        store = self.initialized_ledger()
        value = json.loads(store.path.read_text())
        value["ceilings"]["max_calls"] = 84
        store.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(module.ProgramLedgerError, "checksum"):
            store.load()


if __name__ == "__main__":
    unittest.main()
