"""R6C review reproductions (PR #15 review at d97957f): execution-safety defects.

1. Non-finite or non-integer authorization budgets (NaN, Infinity, fractional calls) were
   accepted, and NaN comparisons bypassed the ceilings; authorization JSON was parsed leniently.
2. Failed embedding requests were not charged against the call or cost budget (production
   counts embedding calls only after success), and the verifier's output-token reservation
   was presented as a hard ceiling although no provider-side output limit is enforced.
3. The retrieval CLI identified the application database from ``os.getenv`` only; a URL from
   Settings/.env was ignored, equivalent URL spellings were not recognized, and a missing
   application identity skipped the check.

No provider, network or paid call: verifier tests use a fake gateway; retrieval tests use fake
services and sessions; the CLI test points at an unreachable port. Synthetic handoffs only.
"""
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import claim_verification  # noqa: E402
from app.benchmark import production_retrieval_replay as retrieval  # noqa: E402
from app.benchmark import production_verifier_replay as verifier  # noqa: E402
from app.config import Settings  # noqa: E402
from tests.test_benchmark_r6b import _id, default_files, make_handoff  # noqa: E402
from tests.test_r6c_production_adapters import (  # noqa: E402
    FC1, ROUTE, FakeGateway, FakeResult, FakeService, authorization, fixed_items,
)

BAD_BUDGETS = {"max_cost_usd": [float("nan"), float("inf"), -float("inf"), 0, -1, True, "1"],
               "max_calls": [float("nan"), float("inf"), 1.5, True, 0, -1, "3"]}


def no_database():
    raise AssertionError("database touched")


def retrieval_auth(report, **overrides):
    return authorization(report, adapter=retrieval.ADAPTER, provider="gemini",
                         model=Settings().embedding_model, **overrides)


def three_pack_files():
    files = default_files()
    files["eval_inputs/fixed_evidence_questions.jsonl"] = [
        {"id": _id("fq", n), "question": f"does treatment {n} work",
         "evidence_pack": [{"evidence_index": 0, "section_label": "RESULTS", "text": f"treatment {n} worked well"}]}
        for n in (1, 2, 3)]
    return files


class IdentitySession:
    """Fake evaluation-database session answering the isolation queries."""

    def __init__(self, identity, revision="head", foreign=0):
        self.identity, self.revision, self.foreign = identity, revision, foreign

    async def execute(self, statement):
        text = str(statement)
        if "pg_postmaster_start_time" in text:
            return FakeResult(self.identity)
        if "alembic_version" in text:
            return FakeResult(self.revision)
        return FakeResult(self.foreign)


EVAL_IDENTITY = {"database": "scientific_rag_eval", "database_oid": 16500, "server_started": "2026-11-13 10:00:00+00"}
APP_IDENTITY = {"database": "scientific_rag", "database_oid": 16384, "server_started": "2026-11-13 10:00:00+00"}


async def identity_of(value):
    return value


class FiniteBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixed, self.report = fixed_items()

    def test_check_authorization_rejects_non_finite_and_non_integer_budgets(self):
        for field, values in BAD_BUDGETS.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(verifier.AuthorizationError):
                    verifier.check_authorization(authorization(self.report, **{field: value}), adapter=verifier.ADAPTER,
                                                 dataset_hash=self.report["handoff"]["dataset_hash"], route=ROUTE)

    def test_valid_finite_budgets_are_accepted(self):
        for cost in (0.05, 1, 2.5):
            auth = verifier.check_authorization(authorization(self.report, max_cost_usd=cost, max_calls=3),
                                                adapter=verifier.ADAPTER,
                                                dataset_hash=self.report["handoff"]["dataset_hash"], route=ROUTE)
            self.assertTrue(math.isfinite(auth["max_cost_usd"]))

    def test_verifier_execution_refuses_nan_budget_before_any_call(self):
        gateway = FakeGateway()
        with patch.object(claim_verification, "ainvoke_structured", gateway), self.assertRaises(verifier.AuthorizationError):
            asyncio.run(verifier.run_replay(self.fixed, authorization=authorization(self.report, max_cost_usd=float("nan")),
                                            dataset_hash=self.report["handoff"]["dataset_hash"], route=ROUTE))
        self.assertEqual(gateway.prompts, [])

    def test_retrieval_execution_refuses_non_finite_budgets_before_the_database(self):
        items = retrieval.retrieval_items(self.fixed)
        for field, value in (("max_cost_usd", float("inf")), ("max_calls", float("nan")), ("max_calls", 2.5)):
            with self.subTest(field=field), self.assertRaises(retrieval.AuthorizationError):
                asyncio.run(retrieval.run_retrieval(
                    items, session_factory=no_database, settings=Settings(), dataset_hash=self.report["handoff"]["dataset_hash"],
                    authorization=retrieval_auth(self.report, **{field: value}), evaluation_url="postgresql://h/eval",
                    application_url="postgresql://h/app", service_factory=FakeService))

    def test_authorization_files_are_parsed_strictly(self):
        spec = importlib.util.spec_from_file_location("benchmark_r6c_cli", ROOT / "scripts/benchmark_r6c.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp:
            for literal in ("NaN", "Infinity", "-Infinity"):
                path = Path(tmp) / "auth.json"
                path.write_text('{"max_cost_usd": %s, "max_calls": 1}' % literal)
                with self.subTest(literal), self.assertRaises(SystemExit) as refused:
                    cli._authorization(path)
                self.assertIn("refused", str(refused.exception))


class ProviderCallBudgetTests(unittest.TestCase):
    def test_failed_embedding_requests_consume_the_call_budget(self):
        """Production counts embedding calls only after success; failures must still be charged."""
        fixed, report = fixed_items(three_pack_files())
        items = retrieval.retrieval_items(fixed)
        attempts = []

        class ProductionLikeFailure(FakeService):
            async def add_documents(self, documents):
                attempts.append(1)
                raise ConnectionError("embedding provider unavailable")   # before any usage is recorded

        one_pack = retrieval.pack_embedding_plan(items[0], Settings())["requests"]
        for max_calls, allowed_attempts in ((1, 0), (one_pack, 1)):
            attempts.clear()
            with self.subTest(max_calls=max_calls):
                records, execution = run_with_fake_db(items, report, ProductionLikeFailure, max_calls=max_calls)
                self.assertLessEqual(len(attempts), allowed_attempts)
                self.assertLessEqual(execution["budget"]["provider_requests_charged"], max_calls)
                failed = [r for r in records if r["status"] == "failed"]
                for record in failed:
                    self.assertEqual(record["usage"]["cost_status"], "incomplete")
                    self.assertGreater(record["usage"]["reserved_cost_usd"], 0)
                self.assertTrue(all(r["status"] in {"failed", "not_executed"} for r in records))

    def test_failed_spend_is_charged_not_treated_as_zero(self):
        fixed, report = fixed_items(three_pack_files())
        items = retrieval.retrieval_items(fixed)

        class Failing(FakeService):
            async def add_documents(self, documents):
                raise ConnectionError("embedding provider unavailable")

        records, execution = run_with_fake_db(items, report, Failing, max_calls=100)
        budget = execution["budget"]
        self.assertEqual(budget["cost_estimated_usd"], 0.0)
        self.assertGreater(budget["uncertain_reserved_usd"], 0)
        self.assertEqual(budget["budget_charged_usd"], budget["uncertain_reserved_usd"])

    def test_verifier_live_execution_requires_a_provider_enforced_output_limit(self):
        """Since R6D the gateway can enforce a limit; a run without an authorized, enforceable one is refused."""
        fixed, report = fixed_items()
        gemini = {"role": "gatekeeper", "provider_order": ["gemini"], "models": {"gemini": "gemini-3.5-flash-lite"}}
        cases = ((ROUTE, authorization(report, max_output_tokens=None)),
                 (gemini, authorization(report, provider="gemini", model="gemini-3.5-flash-lite")))
        for route, auth in cases:
            gateway = FakeGateway()
            with patch.object(claim_verification, "ainvoke_structured", gateway):
                with self.assertRaisesRegex(verifier.AuthorizationError, "output"):
                    asyncio.run(verifier.run_replay(fixed, authorization=auth, route=route,
                                                    dataset_hash=report["handoff"]["dataset_hash"]))
            self.assertEqual(gateway.prompts, [])

    def test_default_production_model_has_no_output_limit(self):
        """Unchanged by R6D: without an explicit limit the production chat model sets none."""
        from scientific_rag import llm_gateway
        with patch.dict(os.environ, {"GROQ_API_KEY": "placeholder"}):
            chat = llm_gateway.create_chat_model("groq", "gatekeeper", 0)
        self.assertIsNone(getattr(chat, "max_tokens", None))
        self.assertFalse(verifier.output_limit_status("gemini", 2000)["enforced"])
        self.assertTrue(verifier.output_limit_status("groq", 2000)["enforced"])

    def test_dry_run_does_not_call_the_planning_bound_a_hard_ceiling(self):
        fixed, _ = fixed_items()
        plan = verifier.dry_run(fixed, route=ROUTE)
        self.assertNotIn("cost_ceiling_usd_any_route", plan)
        self.assertIsNotNone(plan["planning_cost_bound_usd_any_route"])
        self.assertIn("not an absolute billing guarantee", plan["planning_cost_bound_note"])
        gemini = {"role": "gatekeeper", "provider_order": ["gemini"], "models": {"gemini": "gemini-3.5-flash-lite"}}
        self.assertFalse(verifier.dry_run(fixed, route=gemini)["output_limit"]["enforced"])

    def test_unmeasured_verifier_usage_charges_the_reservation(self):
        fixed, report = fixed_items()
        with patch.object(claim_verification, "ainvoke_structured", FakeGateway(usage=None)):
            records, run = asyncio.run(verifier.run_replay(fixed, authorization=authorization(report),
                                                           dataset_hash=report["handoff"]["dataset_hash"], route=ROUTE))
        first = verifier.plan(next(i for i in fixed if i["input_id"] == FC1))
        reserved = verifier.call_cost_ceiling(first["prompt_chars"], "groq", "openai/gpt-oss-120b")
        self.assertEqual(records[0]["usage"]["cost_status"], "incomplete")
        self.assertEqual(records[0]["usage"]["reserved_cost_usd"], reserved)
        self.assertEqual(run["budget"]["uncertain_reserved_usd"], reserved)
        self.assertEqual(run["budget"]["cost_complete_usd"], 0.0)
        self.assertEqual(run["budget"]["budget_charged_usd"], reserved)


def run_with_fake_db(items, report, service, **auth):
    class Session:
        async def __aenter__(self):
            from tests.test_r6c_production_adapters import FakeSession
            return FakeSession(count=1)

        async def __aexit__(self, *exc):
            return False

    async def verified(*args, **kwargs):   # isolation is tested separately below
        return {"verified": True}

    async def go():
        with patch.object(retrieval, "verify_isolated_database", new=verified):
            return await retrieval.run_retrieval(
                items, session_factory=Session, settings=Settings(), authorization=retrieval_auth(report, **auth),
                dataset_hash=report["handoff"]["dataset_hash"], evaluation_url="postgresql://h/eval",
                application_url="postgresql://h/app", service_factory=service)
    return asyncio.run(go())


class DatabaseIsolationTests(unittest.TestCase):
    EVAL = "postgresql+asyncpg://eval:secret@localhost:5432/scientific_rag_eval"

    def check(self, application_url, app_identity=APP_IDENTITY, eval_identity=EVAL_IDENTITY):
        return asyncio.run(retrieval.verify_isolated_database(
            IdentitySession(eval_identity), evaluation_url=self.EVAL, application_url=application_url,
            migration_head="head", application_identity=lambda: identity_of(app_identity)))

    def test_missing_application_identity_is_refused(self):
        for missing in (None, "", "   "):
            with self.subTest(missing=missing), self.assertRaisesRegex(retrieval.IsolationError, "application database"):
                self.check(missing)

    def test_equivalent_url_spellings_of_the_evaluation_database_are_refused(self):
        for spelling in ("postgres://other:pw@127.0.0.1/scientific_rag_eval",
                         "postgresql://u@LOCALHOST:5432/scientific_rag_eval?sslmode=disable",
                         "postgresql+psycopg://u@[::1]:5432/scientific_rag_eval",
                         "postgresql://u@/scientific_rag_eval?host=/var/run/postgresql"):
            with self.subTest(spelling), self.assertRaisesRegex(retrieval.IsolationError, "application database"):
                self.check(spelling)

    def test_same_physical_database_under_another_name_is_refused(self):
        with self.assertRaisesRegex(retrieval.IsolationError, "same physical database"):
            self.check("postgresql://u@db-alias.internal:5432/scientific_rag_eval", app_identity=EVAL_IDENTITY)

    def test_unverifiable_application_database_is_refused(self):
        async def unreachable():
            raise ConnectionRefusedError("no route")
        with self.assertRaisesRegex(retrieval.IsolationError, "could not be verified"):
            asyncio.run(retrieval.verify_isolated_database(
                IdentitySession(EVAL_IDENTITY), evaluation_url=self.EVAL, application_url="postgresql://u@app-db/app",
                migration_head="head", application_identity=unreachable))
        with self.assertRaisesRegex(retrieval.IsolationError, "could not be verified"):
            self.check("postgresql://u@app-db/app", app_identity={"database": None})

    def test_a_dedicated_migrated_database_is_accepted(self):
        result = self.check("postgresql+asyncpg://app:pw@app-db:5432/scientific_rag")
        self.assertEqual(result["foreign_research_runs"], 0)
        self.assertEqual(result["evaluation_database"], EVAL_IDENTITY)
        self.assertEqual(result["application_database"], APP_IDENTITY)

    def test_cli_uses_the_effective_settings_url_not_only_the_environment(self):
        """DATABASE_URL comes from .env (Settings), not the process environment."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = make_handoff(tmp)
            from app.benchmark.handoff import import_handoff
            _, _, report = import_handoff(archive)
            (tmp / ".env").write_text("DATABASE_URL=postgresql+asyncpg://scientific_rag:pw@localhost:1/scientific_rag_eval\n")
            (tmp / "auth.json").write_text(json.dumps(retrieval_auth(report)))
            env = {key: value for key, value in os.environ.items() if key != "DATABASE_URL"}
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/benchmark_r6c.py"), "retrieval-run", str(archive), "--group",
                 "eval_inputs", "--out", str(tmp / "x.json"), "--authorization", str(tmp / "auth.json"),
                 "--evaluation-database-url", "postgresql://scientific_rag:pw@127.0.0.1:1/scientific_rag_eval"],
                capture_output=True, text=True, cwd=tmp, env=env, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refused", result.stderr)
            self.assertIn("application database", result.stderr)
            self.assertFalse((tmp / "x.json").exists())


if __name__ == "__main__":
    unittest.main()
