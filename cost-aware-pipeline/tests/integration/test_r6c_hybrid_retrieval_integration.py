"""R6C integration: production hybrid retrieval on frozen packs in a dedicated database.

Requires RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST (a PostgreSQL/pgvector server whose
user may create databases). Each test class creates a fresh database, migrates it with the
existing Alembic migrations (no new migration), runs the unchanged production
``PostgresVectorService`` (FTS + pgvector + MMR + RRF + parent/child), and drops it.

Only the embedding *provider* is replaced: ``LocalHashEmbeddingService`` overrides
``_embed`` with a deterministic 768-dimensional hashed bag of words and keeps the
production usage accounting, so no network or paid call is made. Everything else,
including SQL, ranking and parent expansion, is production code.
"""
import asyncio
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]

ENABLED = os.getenv("RUN_INTEGRATION_TESTS") == "1" and bool(os.getenv("DATABASE_URL_TEST"))

if ENABLED:
    import asyncpg
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.benchmark import production_retrieval_replay as retrieval
    from app.benchmark.handoff import import_handoff
    from app.benchmark.protocol import session_for_group
    from app.config import Settings
    from app.models import DocumentChunk, ParentDocument, ResearchRun, SourceDocument
    from app.pgvector_retrieval import PostgresVectorService
    from scientific_rag.costing import calculate_cost, estimate_text_tokens
    from tests.test_benchmark_r6b import _id, make_handoff

    class LocalHashEmbeddingService(PostgresVectorService):
        """Production service with a deterministic local embedder in place of the provider."""

        fail_on_question: str | None = None

        async def _embed(self, texts, task_type):
            if self.fail_on_question and any(self.fail_on_question in text for text in texts):
                raise ConnectionError("embedding provider unavailable")
            vectors = []
            for text in texts:
                vector = [0.0] * self.settings.embedding_dimensions
                for token in re.findall(r"[a-z0-9]+", text.lower()):
                    digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
                    vector[digest % len(vector)] += 1.0
                norm = math.sqrt(sum(v * v for v in vector)) or 1.0
                vectors.append([v / norm for v in vector])
            tokens = estimate_text_tokens(texts)
            cost, source = calculate_cost("gemini", self.settings.embedding_model, tokens)
            self.embedding_usage["calls"] += 1
            self.embedding_usage["input_tokens_estimated"] += tokens
            self.embedding_usage["estimated_cost_usd"] = round(self.embedding_usage["estimated_cost_usd"] + cost, 8)
            self.embedding_usage["rate_source"] = source
            return vectors


LONG_METHODS = " ".join(f"Methods paragraph {n} describes randomisation, blinding and allocation concealment."
                        for n in range(60))


def handoff_files():
    question = lambda n, text, segments: {"id": _id("fq", n), "question": text, "evidence_pack": segments}
    seg = lambda i, text, section="RESULTS": {"evidence_index": i, "section_label": section, "text": text}
    return {
        "eval_inputs/open_discovery_questions.jsonl": [{"id": _id("od", 1), "question": "Does statin therapy lower LDL?"}],
        "eval_inputs/fixed_evidence_questions.jsonl": [
            question(1, "Does statin therapy lower LDL cholesterol?", [
                seg(0, "Statin therapy lowered LDL cholesterol by 35 percent compared with placebo."),
                seg(1, "Participants were recruited from twelve outpatient clinics.", "METHODS"),
                seg(2, LONG_METHODS, "METHODS")]),
            question(2, "Is vitamin D associated with fracture risk?", [
                seg(0, "Vitamin D supplementation did not reduce hip fracture risk in older adults."),
                seg(1, "Statin therapy lowered LDL cholesterol in a separate cohort.", "BACKGROUND")]),
            question(3, "What did the locator-only trial report?", [
                {"evidence_index": 0, "paper_id": "pmcid:PMC1234567", "note": "not redistributed",
                 "full_text_locator": "https://example.invalid/PMC1234567", "full_text_sha256": "f" * 64}]),
        ],
        "eval_inputs/fixed_evidence_claims.jsonl": [
            {"id": _id("fc", 1), "claim": "c", "evidence": [{"evidence_index": 0, "text": "e"}]}],
    }


@unittest.skipUnless(ENABLED, "Set RUN_INTEGRATION_TESTS=1 and DATABASE_URL_TEST to run against PostgreSQL/pgvector.")
class ProductionHybridRetrievalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = os.environ["DATABASE_URL_TEST"]
        cls.db_name = f"scientific_rag_r6c_{uuid.uuid4().hex[:12]}"
        parts = urlsplit(base)
        cls.admin_url = base
        cls.plain_url = urlunsplit(parts._replace(path=f"/{cls.db_name}"))
        cls.async_url = cls.plain_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        cls.application_url = base.replace("postgresql://", "postgresql+asyncpg://", 1)

        async def create():
            connection = await asyncpg.connect(base)
            try:
                await connection.execute(f'CREATE DATABASE "{cls.db_name}"')
            finally:
                await connection.close()
        asyncio.run(create())
        migrated = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, capture_output=True,
                                  text=True, env={**os.environ, "DATABASE_URL": cls.async_url})
        if migrated.returncode != 0:
            raise RuntimeError(f"alembic upgrade failed: {migrated.stderr[-2000:]}")
        with tempfile.TemporaryDirectory() as tmp:
            _, inputs, cls.report = import_handoff(make_handoff(Path(tmp), handoff_files()))
        session = session_for_group(inputs, "eval_inputs")
        session.skip_discovery("not part of R6C")
        cls.items = retrieval.retrieval_items(session.load_fixed_evidence())
        cls.settings = Settings(google_api_key="integration-test-key-not-used")
        cls.genai_patch = patch("app.pgvector_retrieval.genai.Client")
        cls.genai_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.genai_patch.stop()

        async def drop():
            connection = await asyncpg.connect(cls.admin_url)
            try:
                await connection.execute(f'DROP DATABASE IF EXISTS "{cls.db_name}" WITH (FORCE)')
            finally:
                await connection.close()
        asyncio.run(drop())

    def authorization(self, **overrides):
        auth = {"schema": "scientific-rag-paid-evaluation-authorization/v1", "authorization_id": "integration",
                "approved_by": "test", "adapter": retrieval.ADAPTER,
                "handoff_dataset_hash": self.report["handoff"]["dataset_hash"], "provider": "gemini",
                "model": self.settings.embedding_model, "max_calls": 100, "max_cost_usd": 1.0}
        auth.update(overrides)
        return auth

    def run_adapter(self, items=None, service=None, **auth):
        service = service or LocalHashEmbeddingService

        async def go():
            engine = create_async_engine(self.async_url)
            try:
                return await retrieval.run_retrieval(
                    items or self.items, session_factory=async_sessionmaker(engine, expire_on_commit=False),
                    settings=self.settings, authorization=self.authorization(**auth),
                    dataset_hash=self.report["handoff"]["dataset_hash"], evaluation_url=self.async_url,
                    application_url=self.application_url, service_factory=service)
            finally:
                await engine.dispose()
        return asyncio.run(go())

    def table_counts(self):
        async def go():
            engine = create_async_engine(self.async_url)
            try:
                async with engine.connect() as connection:
                    return {model.__tablename__: (await connection.execute(select(func.count()).select_from(model))).scalar()
                            for model in (ResearchRun, SourceDocument, ParentDocument, DocumentChunk)}
            finally:
                await engine.dispose()
        return asyncio.run(go())

    def test_production_retrieval_ranks_within_each_frozen_pack(self):
        records, execution = self.run_adapter()
        by_id = {record["input_id"]: record for record in records}
        first, second = by_id[_id("fq", 1)], by_id[_id("fq", 2)]
        self.assertEqual((first["status"], second["status"]), ("completed", "completed"))
        self.assertEqual(first["prediction"]["ranked_evidence_ids"][0], f"{_id('fq', 1)}#0")
        self.assertEqual(second["prediction"]["ranked_evidence_ids"][0], f"{_id('fq', 2)}#0")
        for record in (first, second):
            self.assertTrue(all(eid.startswith(record["input_id"] + "#") for eid in record["prediction"]["ranked_evidence_ids"]))
            trace = record["prediction"]["trace"]
            self.assertGreater(trace["vector_count"], 0)      # pgvector ran
            self.assertLessEqual(trace["selected_count"], execution["configuration"]["top_n"])
            self.assertEqual(record["usage"]["cost_status"], "estimated")
            self.assertGreater(record["usage"]["provider_calls"], 0)
        self.assertGreater(first["prediction"]["trace"]["fulltext_count"], 0)   # PostgreSQL FTS ran
        # the long segment becomes several parents and many children; parent expansion returns it once
        self.assertEqual(first["prediction"]["ranked_evidence_ids"].count(f"{_id('fq', 1)}#2"), 1)
        self.assertEqual(by_id[_id("fq", 3)]["status"], "not_executed")
        self.assertEqual(execution["isolation"]["foreign_research_runs"], 0)

    def test_corpus_never_persists_and_runs_are_reproducible(self):
        first, _ = self.run_adapter()
        self.assertEqual(self.table_counts(), {"research_runs": 0, "source_documents": 0, "parent_documents": 0,
                                               "document_chunks": 0})
        second, _ = self.run_adapter()
        strip = lambda records: [(r["input_id"], r["status"], (r.get("prediction") or {}).get("ranked_evidence_ids"),
                                  (r.get("prediction") or {}).get("corpus_sha256")) for r in records]
        self.assertEqual(strip(first), strip(second))   # timings differ; rankings and corpus identity do not

    def test_refuses_a_database_with_foreign_research_runs(self):
        async def seed(add: bool):
            engine = create_async_engine(self.async_url)
            try:
                async with async_sessionmaker(engine)() as session:
                    if add:
                        session.add(ResearchRun(question="application run", requested_by="someone"))
                    else:
                        await session.execute(ResearchRun.__table__.delete())
                    await session.commit()
            finally:
                await engine.dispose()
        asyncio.run(seed(True))
        try:
            with self.assertRaisesRegex(retrieval.IsolationError, "not created by this adapter"):
                self.run_adapter()
        finally:
            asyncio.run(seed(False))

    def test_refuses_the_application_database(self):
        async def go():
            engine = create_async_engine(self.async_url)
            try:
                return await retrieval.run_retrieval(
                    self.items, session_factory=async_sessionmaker(engine), settings=self.settings,
                    authorization=self.authorization(), dataset_hash=self.report["handoff"]["dataset_hash"],
                    evaluation_url=self.async_url, application_url=self.async_url)
            finally:
                await engine.dispose()
        with self.assertRaisesRegex(retrieval.IsolationError, "application database"):
            asyncio.run(go())

    def test_failures_are_accounted_and_do_not_stop_other_packs(self):
        class Flaky(LocalHashEmbeddingService):
            fail_on_question = "vitamin D associated"
        records, _ = self.run_adapter(service=Flaky)
        by_id = {record["input_id"]: record for record in records}
        failed = by_id[_id("fq", 2)]
        self.assertEqual((failed["status"], failed["reason"]), ("failed", "retrieval failure (ConnectionError)"))
        self.assertGreater(failed["usage"]["provider_calls"], 0)   # document embeddings before the query failed
        self.assertEqual(by_id[_id("fq", 1)]["status"], "completed")
        self.assertEqual(self.table_counts()["research_runs"], 0)

    def test_cost_ceiling_prevents_starting_a_pack(self):
        records, execution = self.run_adapter(max_cost_usd=1e-9)
        self.assertTrue(all(record["status"] == "not_executed" for record in records))
        self.assertEqual(execution["embedding_calls"], 0)


if __name__ == "__main__":
    unittest.main()
