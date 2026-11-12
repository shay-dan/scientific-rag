"""R4 reproduction: evaluation-worker integrity against real PostgreSQL.

A ten-case dataset (five answered and judged, three abstentions, one research
failure, one research timeout) is run through the real evaluation worker. The
research service, Ragas and the fallback judge are deterministic fakes, so no
LLM or literature provider is contacted. One Ragas failure forces the fallback
judge, so the run mixes two evaluator engines.

Requires RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST (migrated).
"""
import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchRun
from app.services import PIPELINE_VERSION

RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"
RAGAS_SCORES = {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6, "context_recall": 0.5}
FALLBACK_SCORES = {"faithfulness": 0.2, "answer_relevancy": 0.3, "context_precision": 0.4, "context_recall": 0.1}


def _async_url(url: str) -> str:
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


class _FakeResearchService:
    """Creates real ResearchRun rows so evaluation results keep valid foreign keys."""

    def __init__(self, sessions, actor: str) -> None:
        self.sessions = sessions
        self.actor = actor

    async def run(self, session, question, mode, generate, *, actor, cache_mode):
        kind = question.split(":", 1)[0]
        if kind == "fail":
            raise RuntimeError("literature provider exploded")
        if kind == "timeout":
            raise asyncio.TimeoutError()
        status = "completed" if kind == "judged" else "insufficient_evidence"
        outcome = "accepted_sources" if kind == "judged" else "no_candidates"
        metadata = {
            "mode": "fast",
            "retrieval_version": PIPELINE_VERSION,
            "source_sufficiency": {"outcome": outcome, "sufficient": kind == "judged"},
            "llm_gateway": {"by_model": {"groq:fake-generator": {"provider": "groq", "model": "fake-generator", "attempts": 1}},
                            "by_provider": {"groq": {"attempts": 1, "successes": 1, "failures": 0}}},
            "cost_accounting": {"estimated_cost_usd": 0.001, "complete": True},
        }
        record = ResearchRun(id=uuid4(), question=question, requested_by=self.actor, status=status,
                             answer=f"Answer for {question}" if kind == "judged" else "No direct evidence.",
                             completed_at=datetime.now(timezone.utc), model_metadata=metadata)
        session.add(record)
        await session.commit()
        return SimpleNamespace(id=record.id, status=status, answer=record.answer,
                               source_sufficiency=metadata["source_sufficiency"],
                               llm_diagnostics=metadata["llm_gateway"], retrieval_version=PIPELINE_VERSION)


async def _fake_ragas(*, question, usage_tracker=None, **_kwargs):
    if usage_tracker is not None:
        usage_tracker.begin_judge_call()
    if question.startswith("judged:5"):
        if usage_tracker is not None:
            usage_tracker.fail_judge_call()
        raise RuntimeError("ragas structured output failed")
    if usage_tracker is not None:
        usage_tracker.record_openai_response(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=50, completion_tokens=10, completion_tokens_details=None)))
    return dict(RAGAS_SCORES)


async def _fake_diagnostics(**_kwargs):
    return {"total_excerpts": 0}


class _FakeGenai:
    class Client:
        def __init__(self, *args, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        @staticmethod
        def _generate(**_kwargs):
            return SimpleNamespace(
                parsed={**FALLBACK_SCORES, "rationale": "Fallback judge."},
                usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=0),
            )


@unittest.skipUnless(RUN and os.getenv("DATABASE_URL_TEST"), "Set RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST")
class R4EvaluationWorkerReproductionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(_async_url(os.environ["DATABASE_URL_TEST"]), pool_size=5)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.actor = f"r4-test-{uuid4().hex[:12]}"
        questions = ([f"judged:{i} question" for i in range(1, 6)]
                     + [f"abstain:{i} question" for i in range(1, 4)]
                     + ["fail:1 question", "timeout:1 question"])
        async with self.sessions() as session:
            self.dataset = EvaluationDataset(name=f"r4-repro-{uuid4().hex}", created_by=self.actor)
            session.add(self.dataset)
            await session.flush()
            session.add_all([EvaluationCase(dataset_id=self.dataset.id, question=q, reference_answer="Reference answer.",
                                            reference_claims=[], provenance="model_generated") for q in questions])
            self.run = EvaluationRun(dataset_id=self.dataset.id, status="queued", metrics={"mode": "fast"})
            session.add(self.run)
            await session.commit()

    async def asyncTearDown(self):
        async with self.sessions() as session:
            await session.execute(delete(EvaluationResult).where(EvaluationResult.evaluation_run_id == self.run.id))
            await session.execute(delete(EvaluationRun).where(EvaluationRun.id == self.run.id))
            await session.execute(delete(EvaluationCase).where(EvaluationCase.dataset_id == self.dataset.id))
            await session.execute(delete(EvaluationDataset).where(EvaluationDataset.id == self.dataset.id))
            await session.execute(delete(ResearchRun).where(ResearchRun.requested_by == self.actor))
            await session.commit()
        await self.engine.dispose()

    async def _run_worker(self):
        from app import worker

        ctx = {"service": _FakeResearchService(self.sessions, self.actor)}
        with patch.object(worker, "session_factory", self.sessions), \
                patch.object(worker, "score_with_ragas", _fake_ragas), \
                patch.object(worker, "diagnose_retrieved_contexts", _fake_diagnostics), \
                patch.object(worker, "diagnose_reference_claims", _fake_diagnostics), \
                patch.object(worker, "genai", _FakeGenai):
            await worker._run_evaluation_job(ctx, str(self.run.id))
        async with self.sessions() as session:
            run = await session.get(EvaluationRun, self.run.id)
            results = list((await session.execute(
                select(EvaluationResult).where(EvaluationResult.evaluation_run_id == self.run.id))).scalars())
            cases = {case.id: case for case in (await session.execute(
                select(EvaluationCase).where(EvaluationCase.dataset_id == self.dataset.id))).scalars()}
        return run, results, cases

    async def test_every_case_is_accounted_for_by_outcome(self):
        run, results, cases = await self._run_worker()
        self.assertEqual(len(results), 10)
        self.assertIn("outcomes", run.metrics, "run metrics do not break the population down by outcome")
        outcomes = run.metrics["outcomes"]
        self.assertEqual(sum(outcomes.values()), 10)
        self.assertEqual(outcomes["judged"], 5)
        self.assertEqual(outcomes["insufficient_evidence"], 3)
        self.assertEqual(outcomes["research_failed"], 1)
        self.assertEqual(outcomes["research_timeout"], 1)
        self.assertEqual(run.metrics["total_cases"], 10)
        by_question = {cases[item.evaluation_case_id].question: item for item in results}
        timeout = by_question["timeout:1 question"].judge_scores["case_record"]
        self.assertEqual(timeout["outcome"]["category"], "research_timeout")
        self.assertEqual(timeout["scores"], {})  # no quality score is invented for a failure

    async def test_ragas_and_fallback_judge_scores_are_not_averaged(self):
        run, results, cases = await self._run_worker()
        self.assertIsNone(run.metrics.get("faithfulness"), "Ragas and fallback-judge scores were averaged together")  # mixed engines: no single headline score
        self.assertTrue(run.metrics["mixed_engines"])
        groups = {group["engine"]: group for group in run.metrics["report"]["quality"]["groups"]}
        self.assertEqual(groups["ragas"]["metrics"]["faithfulness"]["n"], 4)
        self.assertEqual(groups["ragas"]["metrics"]["faithfulness"]["mean"], 0.8)
        self.assertEqual(groups["gemini_fallback"]["metrics"]["faithfulness"]["n"], 1)
        self.assertEqual(groups["gemini_fallback"]["metrics"]["faithfulness"]["mean"], 0.2)
        fallback = next(item for item in results if cases[item.evaluation_case_id].question.startswith("judged:5"))
        record = fallback.judge_scores["case_record"]
        self.assertEqual(record["judge"]["engine"], "gemini_fallback")
        self.assertEqual(record["judge"]["fallback"]["from_engine"], "ragas")
        self.assertEqual(record["judge"]["usage"]["fallback"]["judge"]["input_tokens"], 100)  # fallback cost visible

    async def test_run_records_reproducible_provenance(self):
        run, _results, _cases = await self._run_worker()
        self.assertIn("provenance", run.metrics, "evaluation runs record no provenance")
        provenance = run.metrics["provenance"]
        for key in ("git_sha", "pipeline_version", "dataset", "judge", "generator", "mode", "started_at", "report_schema"):
            self.assertIn(key, provenance)
        self.assertEqual(provenance["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(len(provenance["dataset"]["content_hash"]), 64)
        self.assertEqual(provenance["dataset"]["case_provenance"], {"model_generated": 10})
        self.assertIn("prompt_versions", provenance["judge"])
        built_sha = os.getenv("SCIENTIFIC_RAG_GIT_SHA", "")
        if len(built_sha) == 40:  # CI image built with --build-arg GIT_SHA
            self.assertEqual((provenance["git_sha"], provenance["git_sha_source"]), (built_sha, "build_arg"))
        else:
            self.assertIn(provenance["git_sha_source"], {"unavailable", "local_git_checkout_uncommitted_changes_not_detected"})
        self.assertEqual(run.status, "completed_with_errors")


    async def test_api_responses_keep_their_contract_and_add_the_report(self):
        await self._run_worker()
        from app import main as api

        admin = SimpleNamespace(role="admin", name="admin")
        with patch.object(api, "require_api_key", return_value=admin), patch.object(api, "_ui_admin", return_value=admin):
            async with self.sessions() as session:
                body = await api.get_evaluation_run(SimpleNamespace(), str(self.run.id), session)
            async with self.sessions() as session:
                exported = json.loads((await api.export_evaluation_run(SimpleNamespace(), str(self.run.id), session)).body)
        for key in ("id", "status", "mode", "case_count", "scored_count", "averages", "context_precision_views", "results"):
            self.assertIn(key, body)  # pre-R4 keys unchanged
        self.assertEqual(body["case_count"], 10)
        self.assertEqual(body["scored_count"], 5)
        self.assertEqual(set(body["averages"]), {"faithfulness", "answer_relevancy", "context_precision", "context_recall"})
        self.assertEqual(sum(body["report"]["population"]["outcomes"].values()), 10)
        self.assertEqual(sorted({item["outcome"] for item in body["results"]}),
                         ["insufficient_evidence", "judged", "research_failed", "research_timeout"])
        self.assertEqual(len(exported["report"]["cases"]), 10)
        self.assertEqual(exported["failed_count"], 5)  # legacy field: unjudged cases

if __name__ == "__main__":
    unittest.main()
