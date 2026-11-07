import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from app.claim_verification import (
    ClaimAssessment,
    ClaimVerificationBatch,
    ClaimVerificationError,
    NoSemanticallySupportedClaimsError,
)
from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2
from app.config import Settings
from app.cost_policy import budget_for
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import ResearchService, effective_pipeline_version, to_response
from app.worker import run_research_job
from scientific_rag.legacy_pipeline import ResearchAgent


def _answer() -> StructuredEvidenceAnswerV2:
    return StructuredEvidenceAnswerV2(
        claims=[
            EvidenceClaim(
                text="The first study reported a measured clinical result.",
                source_id="S1",
                evidence_quote="reported a measured clinical result",
            ),
            EvidenceClaim(
                text="The second study reported an independent technical result.",
                source_id="S2",
                evidence_quote="reported an independent technical result",
            ),
        ],
        limitations=[EvidenceClaim(
            text="The first study used a retrospective design.",
            source_id="S1",
            evidence_quote="used a retrospective design",
        )],
        unanswered_parts=["Long-term comparative evidence remains unavailable."],
    )


def _documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "The first study reported a measured clinical result and used a retrospective design."
            ),
            metadata={"doc_id": "paper-1", "title": "Clinical study", "source": "pubmed"},
        ),
        Document(
            page_content="The second study reported an independent technical result.",
            metadata={"doc_id": "paper-2", "title": "Technical study", "source": "arxiv"},
        ),
    ]


def _verification_batch(**verdicts: str) -> ClaimVerificationBatch:
    defaults = {
        "claims.1": "supported",
        "claims.2": "supported",
        "limitations.1": "supported",
    }
    defaults.update({target.replace("_", "."): verdict for target, verdict in verdicts.items()})
    return ClaimVerificationBatch(assessments=[
        ClaimAssessment(target=target, verdict=verdict, reason=f"Reason for {target}.")
        for target, verdict in defaults.items()
    ])


class SemanticVerificationProductionFlowTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, mode: str) -> ResearchAgent:
        vector_service = SimpleNamespace(retrieve_context=AsyncMock(return_value=_documents()))
        agent = ResearchAgent(
            vector_service,
            max_synthesis_subquestions=1,
            semantic_verification_mode=mode,
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        return agent

    async def _run(self, mode: str, verifier_result=None, verifier_error=None):
        agent = self._agent(mode)
        verifier = AsyncMock(
            side_effect=verifier_error,
            return_value=verifier_result or _verification_batch(),
        )
        with patch(
            "scientific_rag.legacy_pipeline.ainvoke_structured",
            new=AsyncMock(return_value=_answer()),
        ), patch("app.claim_verification.ainvoke_structured", new=verifier):
            result = await agent.run_research("What do the studies report?")
        return result, verifier

    async def test_off_preserves_original_answer_and_never_calls_verifier(self) -> None:
        result, verifier = await self._run("off")
        self.assertIn("The first study reported", result["answer"])
        self.assertIn("The second study reported", result["answer"])
        self.assertNotIn("semantic_verification", result["claim_validation"])
        verifier.assert_not_awaited()

    async def test_success_audit_contains_no_prompt_or_credential_fields(self) -> None:
        result, _ = await self._run("enforce")
        audit = result["claim_validation"]["semantic_verification"]
        serialized = str(audit).casefold()
        self.assertNotIn("prompt", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("credential", serialized)

    async def test_shadow_all_supported_preserves_answer_and_records_audit(self) -> None:
        result, verifier = await self._run("shadow")
        audit = result["claim_validation"]["semantic_verification"]
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(audit["supported_count"], 3)
        self.assertFalse(audit["filtering_applied"])
        self.assertIn("The second study reported", result["answer"])
        verifier.assert_awaited_once()

    async def test_shadow_contradiction_is_recorded_without_filtering(self) -> None:
        result, _ = await self._run(
            "shadow", _verification_batch(claims_2="contradicted")
        )
        audit = result["claim_validation"]["semantic_verification"]
        self.assertEqual(audit["contradicted_count"], 1)
        self.assertEqual(audit["removed_target_ids"], [])
        self.assertEqual(audit["rejected_target_ids"], ["claims.2"])
        self.assertIn("The second study reported", result["answer"])

    async def test_shadow_provider_error_preserves_answer_and_records_safe_error(self) -> None:
        result, _ = await self._run("shadow", verifier_error=RuntimeError("secret provider detail"))
        audit = result["claim_validation"]["semantic_verification"]
        self.assertEqual(audit["status"], "error")
        self.assertEqual(audit["error_type"], "ClaimVerificationError")
        self.assertNotIn("secret provider detail", str(audit))
        self.assertIn("The second study reported", result["answer"])

    async def test_enforce_all_supported_preserves_answer_and_audit(self) -> None:
        result, _ = await self._run("enforce")
        audit = result["claim_validation"]["semantic_verification"]
        self.assertTrue(audit["filtering_applied"])
        self.assertEqual(audit["retained_target_ids"], [
            "claims.1", "claims.2", "limitations.1"
        ])
        self.assertIn("The second study reported", result["answer"])

    async def test_enforce_removes_contradicted_claim_and_keeps_stable_source_ids(self) -> None:
        result, _ = await self._run(
            "enforce", _verification_batch(claims_2="contradicted")
        )
        self.assertIn("The first study reported", result["answer"])
        self.assertNotIn("The second study reported", result["answer"])
        audit = result["claim_validation"]["semantic_verification"]
        self.assertEqual(audit["removed_target_ids"], ["claims.2"])
        self.assertIn("claims.2", [item["original_target_id"] for item in audit["assessments"]])
        self.assertEqual(result["claim_validation"]["claims"][0]["source_id"], "S1")

    async def test_enforce_removes_insufficient_claim(self) -> None:
        result, _ = await self._run(
            "enforce", _verification_batch(claims_2="insufficient_evidence")
        )
        self.assertNotIn("The second study reported", result["answer"])

    async def test_enforce_removes_rejected_limitation(self) -> None:
        result, _ = await self._run(
            "enforce", _verification_batch(limitations_1="contradicted")
        )
        self.assertNotIn("retrospective design", result["answer"])
        self.assertEqual(result["claim_validation"]["limitations"], [])

    async def test_enforce_all_claims_rejected_fails_safely(self) -> None:
        with self.assertRaises(NoSemanticallySupportedClaimsError):
            await self._run("enforce", _verification_batch(
                claims_1="contradicted", claims_2="insufficient_evidence"
            ))

    async def test_enforce_provider_error_fails_safely(self) -> None:
        with self.assertRaises(ClaimVerificationError):
            await self._run("enforce", verifier_error=RuntimeError("provider failed"))

    async def test_enforce_invalid_response_fails_safely(self) -> None:
        with self.assertRaises(ClaimVerificationError):
            await self._run("enforce", verifier_result={"assessments": []})

    async def test_enforce_timeout_fails_safely(self) -> None:
        vector_service = SimpleNamespace(retrieve_context=AsyncMock(return_value=_documents()))
        agent = ResearchAgent(
            vector_service,
            max_synthesis_subquestions=1,
            semantic_verification_mode="enforce",
            semantic_verification_timeout_seconds=0.01,
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])

        async def delayed_verification(*args, **kwargs):
            await asyncio.sleep(0.1)
            return _verification_batch()

        with patch(
            "scientific_rag.legacy_pipeline.ainvoke_structured",
            new=AsyncMock(return_value=_answer()),
        ), patch(
            "scientific_rag.legacy_pipeline.verify_claims",
            new=AsyncMock(side_effect=delayed_verification),
        ):
            with self.assertRaises(TimeoutError):
                await agent.run_research("What do the studies report?")

    async def test_exactly_one_surviving_claim_remains_structurally_valid(self) -> None:
        result, _ = await self._run(
            "enforce", _verification_batch(claims_2="contradicted")
        )
        self.assertEqual(len(result["claim_validation"]["claims"]), 1)
        self.assertEqual(result["claim_validation"]["remaining_errors"], {})

    async def test_post_filter_structural_validation_remains_mandatory(self) -> None:
        invalid_selected_answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="This field points to a source outside the evidence set.",
            source_id="S9",
            evidence_quote="quotation absent from the evidence set",
        )])
        with patch(
            "scientific_rag.legacy_pipeline.apply_semantic_verdicts",
            return_value=invalid_selected_answer,
        ):
            with self.assertRaisesRegex(RuntimeError, "mandatory structural evidence validation"):
                await self._run("enforce")

    async def test_shadow_budget_failure_skips_calls_and_preserves_answer(self) -> None:
        agent = self._agent("shadow")
        with patch(
            "scientific_rag.legacy_pipeline.ainvoke_structured",
            new=AsyncMock(return_value=_answer()),
        ), patch(
            "scientific_rag.legacy_pipeline.plan_verification_batches",
            return_value=tuple(((), "prompt") for _ in range(6)),
        ), patch("scientific_rag.legacy_pipeline.verify_claims", new=AsyncMock()) as verifier:
            result = await agent.run_research("What do the studies report?")
        audit = result["claim_validation"]["semantic_verification"]
        self.assertEqual(audit["status"], "error")
        self.assertEqual(audit["error_type"], "SemanticVerificationBudgetError")
        verifier.assert_not_awaited()


class _AsyncSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class SemanticVerificationServiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_semantic_audit_is_persisted_in_claim_validation_metadata(self) -> None:
        semantic_audit = {
            "mode": "enforce",
            "status": "completed",
            "total_assessed_targets": 2,
            "supported_count": 1,
            "contradicted_count": 1,
            "insufficient_evidence_count": 0,
            "assessments": [
                {"original_target_id": "claims.1", "verdict": "supported", "reason": "Supported."},
                {"original_target_id": "claims.2", "verdict": "contradicted", "reason": "Contradicted."},
            ],
            "retained_target_ids": ["claims.1"],
            "removed_target_ids": ["claims.2"],
            "rejected_target_ids": ["claims.2"],
            "filtering_applied": True,
            "batch_count": 1,
            "duration_ms": 25.0,
        }
        claim_validation = {
            "status": "completed",
            "contract": "structured_evidence_v2",
            "claims": [],
            "limitations": [],
            "unanswered_parts": [],
            "repair": None,
            "remaining_errors": {},
            "semantic_verification": semantic_audit,
        }
        now = datetime.now(timezone.utc)
        record = SimpleNamespace(
            id=uuid4(), question="Question", requested_by="tester", status="running",
            answer=None, error=None, model_metadata={
                "mode": "fast", "semantic_claim_verification_mode": "enforce",
                "progress": {"stage": "Starting", "role": None, "model": None},
            }, created_at=now, completed_at=None,
        )
        session = SimpleNamespace(
            add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(),
            rollback=AsyncMock(), get=AsyncMock(return_value=record),
        )
        vector_service = SimpleNamespace(embedding_usage={}, retrieval_traces=[])
        pipeline_result = {
            "report": {
                "answer": "Verified answer",
                "raw_docs_used": [],
                "claim_validation": claim_validation,
            },
            "metadata": {
                "plan_executed": [], "cache": {}, "source_sufficiency": {},
                "source_diagnostics": {}, "pubmed_inspector": [], "arxiv_inspector": [],
            },
        }
        orchestrator = SimpleNamespace(execute_full_study=AsyncMock(return_value=pipeline_result))
        settings = Settings(
            google_api_key="test", entrez_email="tester@example.com",
            semantic_claim_verification_mode="enforce", cost_tracking_enabled=False,
        )
        service = ResearchService(settings, redis=None)
        service.cost_guard = None
        with patch("app.services.PostgresVectorService", return_value=vector_service), \
             patch("app.services.legacy_pipeline.ResearchAgent") as agent_type, \
             patch("app.services.legacy_pipeline.Orchestrator", return_value=orchestrator), \
             patch("app.services.legacy_pipeline.config.validate"), \
             patch("app.services.begin_trace", return_value="trace"), \
             patch("app.services.end_trace", return_value=[]), \
             patch("app.services.summarize_trace", return_value={
                 "attempts": [], "estimated_cost_usd": None,
             }):
            response = await service._execute(
                session, record, "Question", ResearchMode.fast, False,
                semantic_verification_mode="enforce",
            )
        self.assertEqual(response.status, "completed")
        self.assertEqual(
            record.model_metadata["claim_validation"]["semantic_verification"],
            semantic_audit,
        )
        self.assertEqual(
            agent_type.call_args.kwargs["semantic_verification_mode"], "enforce"
        )

    async def test_enforce_failure_is_persisted_as_failed_at_verification_stage(self) -> None:
        now = datetime.now(timezone.utc)
        record = SimpleNamespace(
            id=uuid4(), question="Question", requested_by="tester", status="running",
            answer=None, error=None, model_metadata={
                "mode": "fast", "semantic_claim_verification_mode": "enforce",
                "progress": {"stage": "Starting", "role": None, "model": None},
            }, created_at=now, completed_at=None,
        )
        session = SimpleNamespace(
            add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(),
            rollback=AsyncMock(), get=AsyncMock(return_value=record),
        )
        vector_service = SimpleNamespace(embedding_usage={}, retrieval_traces=[])

        class FailingOrchestrator:
            def __init__(self, *args, progress_callback=None, **kwargs):
                self.progress_callback = progress_callback
                self.last_source_diagnostics = {}
                self.last_pubmed_inspector = []
                self.last_arxiv_inspector = []
                self.last_cache_diagnostics = {}
                self.last_mined_count = 0

            async def execute_full_study(self, *args, **kwargs):
                await self.progress_callback(
                    "Verifying answer claims against cited evidence", "gatekeeper"
                )
                raise NoSemanticallySupportedClaimsError(
                    "semantic enforcement rejected every answer claim"
                )

        settings = Settings(
            google_api_key="test", entrez_email="tester@example.com",
            semantic_claim_verification_mode="enforce", cost_tracking_enabled=False,
        )
        service = ResearchService(settings, redis=None)
        service.cost_guard = None
        with patch("app.services.PostgresVectorService", return_value=vector_service), \
             patch("app.services.legacy_pipeline.ResearchAgent"), \
             patch("app.services.legacy_pipeline.Orchestrator", FailingOrchestrator), \
             patch("app.services.legacy_pipeline.config.validate"), \
             patch("app.services.begin_trace", return_value="trace"), \
             patch("app.services.end_trace", return_value=[]), \
             patch("app.services.summarize_trace", return_value={
                 "attempts": [], "estimated_cost_usd": None,
             }):
            with self.assertRaises(NoSemanticallySupportedClaimsError):
                await service._execute(
                    session, record, "Question", ResearchMode.fast, False,
                    semantic_verification_mode="enforce",
                )
        self.assertEqual(record.status, "failed")
        self.assertEqual(
            record.model_metadata["failure_stage"],
            "Verifying answer claims against cited evidence",
        )
        self.assertNotEqual(record.status, "completed")

    async def test_queued_run_uses_persisted_mode_after_worker_configuration_change(self) -> None:
        record = SimpleNamespace(
            id=uuid4(),
            status="queued",
            model_metadata={"semantic_claim_verification_mode": "enforce"},
        )
        session = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: record)),
            commit=AsyncMock(),
        )
        service = ResearchService(Settings(semantic_claim_verification_mode="off"), redis=None)
        service._execute = AsyncMock()
        with patch("app.services.session_factory", return_value=_AsyncSessionContext(session)):
            await service.run_queued(
                record.id,
                "Queued research question",
                ResearchMode.fast,
                ResearchCacheMode.use_cached_answer,
                "shadow",
            )
        self.assertEqual(
            service._execute.await_args.kwargs["semantic_verification_mode"], "enforce"
        )

    async def test_legacy_queued_run_defaults_to_off_not_worker_environment(self) -> None:
        record = SimpleNamespace(id=uuid4(), status="queued", model_metadata={})
        session = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: record)),
            commit=AsyncMock(),
        )
        service = ResearchService(Settings(semantic_claim_verification_mode="enforce"), redis=None)
        service._execute = AsyncMock()
        with patch("app.services.session_factory", return_value=_AsyncSessionContext(session)):
            await service.run_queued(
                record.id,
                "Legacy queued research question",
                ResearchMode.fast,
                ResearchCacheMode.use_cached_answer,
            )
        self.assertEqual(service._execute.await_args.kwargs["semantic_verification_mode"], "off")

    async def test_worker_forwards_explicit_queued_mode(self) -> None:
        service = SimpleNamespace(run_queued=AsyncMock())
        run_id = uuid4()
        await run_research_job(
            {"service": service}, str(run_id), "Question", "fast",
            "use_cached_answer", "shadow",
        )
        self.assertEqual(service.run_queued.await_args.args[-1], "shadow")

    async def test_service_metadata_uses_effective_version_and_call_budget(self) -> None:
        service = ResearchService(Settings(semantic_claim_verification_mode="enforce"), redis=None)
        metadata = service._metadata(
            ResearchMode.fast,
            budget_for(ResearchMode.fast),
            "Starting research",
        )
        self.assertEqual(metadata["retrieval_version"], effective_pipeline_version("enforce"))
        self.assertEqual(metadata["estimated_max_generation_calls"], 10)
        self.assertEqual(metadata["semantic_claim_verification_mode"], "enforce")

    async def test_historical_record_without_semantic_metadata_remains_readable(self) -> None:
        record = SimpleNamespace(
            id=uuid4(), status="completed", answer="Historical answer", created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            model_metadata={"mode": "fast", "retrieval_version": "legacy-v13"},
        )
        response = to_response(record)
        self.assertEqual(response.answer, "Historical answer")
        self.assertEqual(response.retrieval_version, "legacy-v13")

    async def test_enforce_budget_failure_skips_calls_and_fails(self) -> None:
        vector_service = SimpleNamespace(retrieve_context=AsyncMock(return_value=_documents()))
        agent = ResearchAgent(
            vector_service,
            max_synthesis_subquestions=1,
            semantic_verification_mode="enforce",
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        with patch(
            "scientific_rag.legacy_pipeline.ainvoke_structured",
            new=AsyncMock(return_value=_answer()),
        ), patch(
            "scientific_rag.legacy_pipeline.plan_verification_batches",
            return_value=tuple(((), "prompt") for _ in range(6)),
        ), patch("scientific_rag.legacy_pipeline.verify_claims", new=AsyncMock()) as verifier:
            with self.assertRaisesRegex(ClaimVerificationError, "exceeding the limit"):
                await agent.run_research("What do the studies report?")
        verifier.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
