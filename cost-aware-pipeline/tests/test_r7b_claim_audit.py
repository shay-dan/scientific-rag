import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import ValidationError

from app.claim_verification import ClaimAssessment, ClaimVerificationBatch
from app.config import Settings
from app.cost_policy import budget_for
from app.job_lifecycle import _enqueue_arguments
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import PIPELINE_VERSION, ResearchService, effective_pipeline_version
from app.worker import run_research_job
from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2, V2RepairBatch, V2RepairItem
import scientific_rag.legacy_pipeline as legacy_pipeline


def _claim(text: str, source_id: str, quote: str) -> EvidenceClaim:
    return EvidenceClaim(text=text, source_id=source_id, evidence_quote=quote)


def _documents() -> list[Document]:
    return [
        Document(page_content="The first study reported 12.0 mg.",
                 metadata={"doc_id": "one", "title": "One"}),
        Document(page_content="The second study reported 8 mg.",
                 metadata={"doc_id": "two", "title": "Two"}),
    ]


def _answer() -> StructuredEvidenceAnswerV2:
    return StructuredEvidenceAnswerV2(claims=[
        _claim("The first result was 12 mg.", "S1", "reported 12.0 mg"),
        _claim("The second result was 9 mg.", "S2", "reported 8 mg"),
    ])


def _verdicts(second: str = "supported") -> ClaimVerificationBatch:
    return ClaimVerificationBatch(assessments=[
        ClaimAssessment(target="claims.1", verdict="supported", reason="First assessment."),
        ClaimAssessment(target="claims.2", verdict=second, reason="Second assessment."),
    ])


class ClaimAuditConfigurationTests(unittest.TestCase):
    def test_default_is_off_and_pipeline_identity_is_unchanged(self) -> None:
        self.assertEqual(Settings().claim_audit_mode, "off")
        self.assertEqual(effective_pipeline_version("off", "strict"), PIPELINE_VERSION)
        service = ResearchService(Settings(), redis=None)
        metadata = service._metadata(ResearchMode.fast, budget_for(ResearchMode.fast), "Starting")
        self.assertNotIn("claim_audit_mode", metadata)

    def test_audit_mode_is_persisted_without_changing_cache_identity(self) -> None:
        service = ResearchService(Settings(claim_audit_mode="audit"), redis=None)
        metadata = service._metadata(
            ResearchMode.fast, budget_for(ResearchMode.fast), "Starting",
            claim_audit_mode="audit",
        )
        self.assertEqual(metadata["claim_audit_mode"], "audit")
        self.assertEqual(metadata["retrieval_version"], PIPELINE_VERSION)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(claim_audit_mode="enforce")

    def test_queue_argument_shape_changes_only_for_audited_jobs(self) -> None:
        base = {
            "mode": "fast", "cache_mode": "use_cached_answer",
            "semantic_claim_verification_mode": "off",
        }
        off = SimpleNamespace(id=uuid4(), question="Question", model_metadata=base)
        audit = SimpleNamespace(
            id=uuid4(), question="Question",
            model_metadata={**base, "claim_audit_mode": "audit"},
        )
        self.assertEqual(len(_enqueue_arguments(off)), 6)
        self.assertEqual(_enqueue_arguments(audit)[-2:], ("strict", "audit"))


class _NoopLease:
    def __init__(self, *args, **kwargs):
        self.lost = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc_info):
        return False


class ClaimAuditPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def _queued_mode(self, record, supplied="__omitted__") -> str:
        service = ResearchService(Settings(claim_audit_mode="audit"), redis=None)
        service._execute = AsyncMock()
        session = SimpleNamespace(commit=AsyncMock())
        args = [record.id, "Question", ResearchMode.fast, ResearchCacheMode.use_cached_answer]
        if supplied != "__omitted__":
            args.extend([None, None, supplied])
        with patch("app.services.session_factory", return_value=_SessionContext(session)), \
             patch("app.services.claim_run", AsyncMock(return_value=record)), \
             patch("app.services.LeaseKeeper", _NoopLease):
            await service.run_queued(*args)
        return service._execute.await_args.kwargs["claim_audit_mode"]

    async def test_persisted_audit_wins_after_worker_restart(self) -> None:
        record = SimpleNamespace(
            id=uuid4(), status="running", attempt=2,
            model_metadata={"claim_audit_mode": "audit"},
        )
        self.assertEqual(await self._queued_mode(record, "off"), "audit")

    async def test_missing_persisted_mode_stays_off_despite_stale_job_argument(self) -> None:
        record = SimpleNamespace(id=uuid4(), status="running", attempt=1, model_metadata={})
        self.assertEqual(await self._queued_mode(record, "audit"), "off")

    async def test_cached_historical_result_is_reused_without_false_audit_metadata(self) -> None:
        now = datetime.now(timezone.utc)
        cached = SimpleNamespace(
            id=uuid4(), status="completed", answer="Cached answer", created_at=now,
            completed_at=now, model_metadata={
                "mode": "fast", "retrieval_version": PIPELINE_VERSION,
            },
        )
        service = ResearchService(Settings(claim_audit_mode="audit"), redis=object())
        service._find_cached = AsyncMock(return_value=cached)
        service.sources_for_run = AsyncMock(return_value=[])
        session = SimpleNamespace()
        with patch("app.services.acquire_identity_lock", AsyncMock()):
            response = await service.queue(
                session, "Question", ResearchMode.fast, actor="tester",
                cache_mode=ResearchCacheMode.use_cached_answer,
            )
        self.assertTrue(response.cached)
        self.assertNotIn("claim_audit_mode", cached.model_metadata)
        self.assertNotIn("claim_audit", cached.model_metadata)
        self.assertEqual(service._find_cached.await_args.args[-2:], ("off", "strict"))

    async def test_worker_forwards_persisted_audit_argument(self) -> None:
        service = SimpleNamespace(run_queued=AsyncMock())
        run_id = uuid4()
        await run_research_job(
            {"service": service}, str(run_id), "Question", "fast",
            "use_cached_answer", "off", "strict", "audit",
        )
        self.assertEqual(service.run_queued.await_args.kwargs["claim_audit_mode"], "audit")

    async def test_fresh_audit_report_is_persisted_in_internal_metadata(self) -> None:
        now = datetime.now(timezone.utc)
        report = {
            "mode": "audit", "status": "completed", "taxonomy_version": "numeric-quote-v1",
            "target_scope": "final_selected_answer",
            "findings": [],
            "summary": {"consistent": 0, "inconsistent": 0, "unverifiable": 0, "not_applicable": 0},
        }
        record = SimpleNamespace(
            id=uuid4(), question="Question", requested_by="tester", status="running",
            answer=None, error=None, model_metadata={
                "mode": "fast", "claim_audit_mode": "audit",
                "progress": {"stage": "Starting", "role": None, "model": None},
            }, created_at=now, completed_at=None, lease_owner=None, lease_expires_at=None,
        )
        session = SimpleNamespace(
            add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(), rollback=AsyncMock(),
            get=AsyncMock(return_value=record),
        )
        vector_service = SimpleNamespace(embedding_usage={}, retrieval_traces=[])
        pipeline_result = {
            "report": {
                "answer": "Audited answer", "raw_docs_used": [],
                "claim_validation": {"deterministic_audit": report},
            },
            "metadata": {
                "plan_executed": [], "cache": {}, "source_sufficiency": {},
                "source_diagnostics": {}, "pubmed_inspector": [], "arxiv_inspector": [],
            },
        }
        orchestrator = SimpleNamespace(execute_full_study=AsyncMock(return_value=pipeline_result))
        service = ResearchService(Settings(
            google_api_key="test", entrez_email="tester@example.com",
            claim_audit_mode="audit", cost_tracking_enabled=False,
        ), redis=None)
        service.cost_guard = None
        with patch("app.services.PostgresVectorService", return_value=vector_service), \
             patch("app.services.legacy_pipeline.ResearchAgent") as agent_type, \
             patch("app.services.legacy_pipeline.Orchestrator", return_value=orchestrator), \
             patch("app.services.legacy_pipeline.config.validate"), \
             patch("app.services.begin_trace", return_value="trace"), \
             patch("app.services.end_trace", return_value=[]), \
             patch("app.services.summarize_trace", return_value={"attempts": [], "estimated_cost_usd": None}):
            await service._execute(session, record, "Question", ResearchMode.fast, False)
        self.assertEqual(
            record.model_metadata["claim_validation"]["deterministic_audit"], report,
        )
        self.assertEqual(record.model_metadata["claim_audit_mode"], "audit")
        self.assertEqual(agent_type.call_args.kwargs["claim_audit_mode"], "audit")


class ClaimAuditProductionFlowTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, *, audit_mode="audit", semantic_mode="off", recovery_mode="strict"):
        agent = legacy_pipeline.ResearchAgent(
            SimpleNamespace(retrieve_context=AsyncMock(return_value=_documents())),
            max_synthesis_subquestions=1,
            semantic_verification_mode=semantic_mode,
            claim_recovery_mode=recovery_mode,
            claim_audit_mode=audit_mode,
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        return agent

    async def test_audit_off_is_backward_compatible_and_adds_no_calls(self) -> None:
        agent = self._agent(audit_mode="off")
        gateway = AsyncMock(return_value=_answer())
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            result = await agent.run_research("Question")
        self.assertNotIn("deterministic_audit", result["claim_validation"])
        gateway.assert_awaited_once()

    async def test_audit_does_not_change_answer_text_or_add_provider_calls(self) -> None:
        gateway = AsyncMock(return_value=_answer())
        agent = self._agent()
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            result = await agent.run_research("Question")
        self.assertIn("The first result was 12 mg.", result["answer"])
        self.assertIn("The second result was 9 mg.", result["answer"])
        audit = result["claim_validation"]["deterministic_audit"]
        self.assertEqual(audit["summary"]["consistent"], 1)
        self.assertEqual(audit["summary"]["inconsistent"], 1)
        gateway.assert_awaited_once()

    async def test_audit_and_semantic_shadow_remain_separate(self) -> None:
        gateway = AsyncMock(return_value=_answer())
        verifier = AsyncMock(return_value=_verdicts("contradicted"))
        agent = self._agent(semantic_mode="shadow")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway), \
             patch("scientific_rag.legacy_pipeline.verify_claims", new=verifier):
            result = await agent.run_research("Question")
        self.assertIn("The second result", result["answer"])
        validation = result["claim_validation"]
        self.assertIn("deterministic_audit", validation)
        self.assertEqual(validation["semantic_verification"]["contradicted_count"], 1)
        self.assertFalse(validation["semantic_verification"]["filtering_applied"])
        gateway.assert_awaited_once()
        verifier.assert_awaited_once()

    async def test_audit_does_not_change_semantic_enforce_filtering(self) -> None:
        gateway = AsyncMock(return_value=_answer())
        verifier = AsyncMock(return_value=_verdicts("contradicted"))
        agent = self._agent(semantic_mode="enforce")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway), \
             patch("scientific_rag.legacy_pipeline.verify_claims", new=verifier):
            result = await agent.run_research("Question")
        self.assertNotIn("The second result", result["answer"])
        validation = result["claim_validation"]
        self.assertEqual(len(validation["deterministic_audit"]["findings"]), 1)
        self.assertEqual(
            validation["deterministic_audit"]["findings"][0]["source_id"], "S1",
        )
        self.assertEqual(validation["semantic_verification"]["removed_target_ids"], ["claims.2"])
        gateway.assert_awaited_once()
        verifier.assert_awaited_once()

    async def test_enforce_audit_targets_reindex_to_the_final_selected_claims(self) -> None:
        gateway = AsyncMock(return_value=_answer())
        verifier = AsyncMock(return_value=ClaimVerificationBatch(assessments=[
            ClaimAssessment(target="claims.1", verdict="contradicted", reason="Removed."),
            ClaimAssessment(target="claims.2", verdict="supported", reason="Retained."),
        ]))
        agent = self._agent(semantic_mode="enforce")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway), \
             patch("scientific_rag.legacy_pipeline.verify_claims", new=verifier):
            result = await agent.run_research("Question")
        audit = result["claim_validation"]["deterministic_audit"]
        self.assertEqual(audit["target_scope"], "final_selected_answer")
        self.assertEqual(
            [(item["target"], item["source_id"]) for item in audit["findings"]],
            [("claims.1", "S2")],
        )
        self.assertNotIn("The first result", result["answer"])
        self.assertIn("The second result", result["answer"])
        gateway.assert_awaited_once()
        verifier.assert_awaited_once()

    async def test_audit_coexists_with_degrade_recovery_without_extra_calls(self) -> None:
        invalid = StructuredEvidenceAnswerV2(claims=[
            _answer().claims[0],
            _claim("The invalid result was 9 mg.", "S2", "invented 9 mg quote"),
        ])
        repair = V2RepairBatch(repairs=[V2RepairItem(
            target="claims.2", action="drop", unresolved_reason="unsupported",
        )])
        gateway = AsyncMock(side_effect=[invalid, repair])
        agent = self._agent(recovery_mode="degrade")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            result = await agent.run_research("Question")
        self.assertEqual(result["claim_validation"]["status"], "completed_with_removals")
        self.assertEqual(len(result["claim_validation"]["deterministic_audit"]["findings"]), 1)
        self.assertIn("Some statements were omitted", result["answer"])
        self.assertEqual(gateway.await_count, 2)


if __name__ == "__main__":
    unittest.main()
