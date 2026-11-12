"""R4 review: an interrupted evaluation keeps its full population (real PostgreSQL).

The evaluation worker is interrupted (``asyncio.CancelledError``, as ARQ does on a
job timeout) part-way through the research phase or the judging phase. The report
served by the API must still cover every dataset case, distinguishing cases never
executed, research interrupted/failed/timed out, answered but not assessed, and
judged. Deterministic fakes; no LLM or literature provider is contacted.
Requires RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST (migrated).
"""
import asyncio
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchRun
from app.services import PIPELINE_VERSION

RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"
SCORES = {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6, "context_recall": 0.5}


def _async_url(url: str) -> str:
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


class _ScriptedResearch:
    """Outcome by call order (case order is by UUID, so it is not fixed by question)."""

    def __init__(self, actor: str, script: list[str]) -> None:
        self.actor, self.script, self.calls = actor, script, 0

    async def run(self, session, question, mode, generate, *, actor, cache_mode):
        kind = self.script[self.calls]
        self.calls += 1
        if kind == "cancel":
            raise asyncio.CancelledError()
        if kind == "fail":
            raise RuntimeError("literature provider exploded")
        if kind == "timeout":
            raise asyncio.TimeoutError()
        status = "completed" if kind == "answer" else "insufficient_evidence"
        metadata = {"retrieval_version": PIPELINE_VERSION,
                    "source_sufficiency": {"outcome": "accepted_sources" if kind == "answer" else "no_candidates"}}
        record = ResearchRun(id=uuid4(), question=question, requested_by=self.actor, status=status,
                             answer="An answer." if kind == "answer" else "No direct evidence.",
                             completed_at=datetime.now(timezone.utc), model_metadata=metadata)
        session.add(record)
        await session.commit()
        return SimpleNamespace(id=record.id, status=status, answer=record.answer,
                               source_sufficiency=metadata["source_sufficiency"], llm_diagnostics={})


async def _diagnostics(**_kwargs):
    return {"total_excerpts": 0}


@unittest.skipUnless(RUN and os.getenv("DATABASE_URL_TEST"), "Set RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST")
class R4InterruptedEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(_async_url(os.environ["DATABASE_URL_TEST"]), pool_size=5)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.actor = f"r4-interrupt-{uuid4().hex[:12]}"
        async with self.sessions() as session:
            self.dataset = EvaluationDataset(name=f"r4-interrupt-{uuid4().hex}", created_by=self.actor)
            session.add(self.dataset)
            await session.flush()
            session.add_all([EvaluationCase(dataset_id=self.dataset.id, question=f"Question {index}?",
                                            reference_answer="Reference.", reference_claims=[], provenance="synthetic")
                             for index in range(10)])
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

    async def _interrupted_report(self, script: list[str], ragas):
        from app import main as api
        from app import worker

        ctx = {"service": _ScriptedResearch(self.actor, script)}
        with patch.object(worker, "session_factory", self.sessions), patch.object(worker, "score_with_ragas", ragas), \
                patch.object(worker, "diagnose_retrieved_contexts", _diagnostics), \
                patch.object(worker, "diagnose_reference_claims", _diagnostics), \
                patch.object(worker, "genai", SimpleNamespace(Client=lambda **_kwargs: SimpleNamespace(models=None))):
            with self.assertRaises(asyncio.CancelledError):
                await worker._run_evaluation_job(ctx, str(self.run.id))
        admin = SimpleNamespace(role="admin", name="admin")
        with patch.object(api, "require_api_key", return_value=admin):
            async with self.sessions() as session:
                return await api.get_evaluation_run(SimpleNamespace(), str(self.run.id), session)

    async def test_interruption_during_research_keeps_every_case(self):
        async def never_called(**_kwargs):
            raise AssertionError("judging must not start")

        body = await self._interrupted_report(["answer", "answer", "fail", "timeout", "cancel"], never_called)
        population = body["report"]["population"]
        outcomes = population["outcomes"]
        self.assertEqual(population["total_cases"], 10, "cases never started left the population")
        self.assertEqual(outcomes.get("not_executed"), 5)
        self.assertEqual(outcomes.get("research_interrupted"), 1)
        self.assertEqual(outcomes["research_failed"], 1)
        self.assertEqual(outcomes["research_timeout"], 1)
        self.assertEqual(outcomes["missing_assessment"], 2)  # answered, never assessed
        self.assertEqual(outcomes["judged"], 0)
        self.assertEqual(population["answered"], 2)  # not-executed and interrupted cases are not answered
        self.assertEqual(body["report"]["quality"]["groups"], [])

    async def test_interruption_during_judging_keeps_every_case(self):
        calls = {"n": 0}

        async def ragas(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise asyncio.CancelledError()
            return dict(SCORES)

        body = await self._interrupted_report(["answer"] * 6 + ["abstain"] * 4, ragas)
        outcomes = body["report"]["population"]["outcomes"]
        self.assertEqual(body["report"]["population"]["total_cases"], 10)
        self.assertEqual(outcomes["judged"], 2)
        self.assertEqual(outcomes["missing_assessment"], 4)
        self.assertEqual(outcomes["insufficient_evidence"], 4)
        self.assertEqual(outcomes.get("not_executed", 0), 0)
        ragas_group = body["report"]["quality"]["groups"][0]
        self.assertEqual(ragas_group["metrics"]["faithfulness"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
