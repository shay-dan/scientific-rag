import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from langchain_core.documents import Document

from app.config import Settings
from app.cost_control import CostBudgetExceeded
from app.cost_policy import budget_for
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import PIPELINE_VERSION, ResearchService, effective_pipeline_version
import app.weak_model_synthesis as synthesis
import scientific_rag.legacy_pipeline as legacy_pipeline
from scientific_rag.llm_gateway import InvalidStructuredResponse


def _claim(text: str, source_id: str, quote: str) -> synthesis.EvidenceClaim:
    return synthesis.EvidenceClaim(text=text, source_id=source_id, evidence_quote=quote)


class ClaimRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = synthesis.build_evidence_blocks([
            Document(page_content="Alpha evidence supports the first result.",
                     metadata={"doc_id": "one", "title": "One"}),
            Document(page_content="Beta evidence supports the second result.",
                     metadata={"doc_id": "two", "title": "Two"}),
            Document(page_content="Gamma evidence supports the third result.",
                     metadata={"doc_id": "three", "title": "Three"}),
        ])
        self.original = synthesis.StructuredEvidenceAnswerV2(claims=[
            _claim("The first result is supported by alpha evidence.", "S1", "Alpha evidence supports the first result"),
            _claim("The second result has an invalid citation quote.", "S2", "invented quotation"),
            _claim("The third result is supported by gamma evidence.", "S3", "Gamma evidence supports the third result"),
        ])

    def test_partial_recovery_preserves_valid_claims_in_order(self) -> None:
        batch = synthesis.V2RepairBatch(repairs=[synthesis.V2RepairItem(
            target="claims.2", action="replace",
            replacement=_claim("The replacement still has an invalid quote.", "S2", "not in source"),
        )])
        recovered, audit, removed = synthesis.recover_v2_answer(self.original, batch, self.blocks)
        self.assertEqual([claim.source_id for claim in recovered.claims], ["S1", "S3"])
        self.assertEqual([item.reason for item in removed], ["replacement_invalid"])
        self.assertEqual(audit["claims_after"], 2)

    def test_incomplete_batch_is_deterministic(self) -> None:
        second_invalid = self.original.model_copy(deep=True)
        second_invalid.claims[2].evidence_quote = "also invented"
        batch = synthesis.V2RepairBatch(repairs=[
            synthesis.V2RepairItem(target="claims.2", action="replace", replacement=_claim(
                "The second result is supported by beta evidence.", "S2", "Beta evidence supports the second result"
            )),
            synthesis.V2RepairItem(target="claims.2", action="drop", unresolved_reason="duplicate"),
            synthesis.V2RepairItem(target="limitations.1", action="drop", unresolved_reason="unexpected"),
        ])
        recovered, audit, removed = synthesis.recover_v2_answer(second_invalid, batch, self.blocks)
        self.assertEqual([claim.source_id for claim in recovered.claims], ["S1", "S2"])
        self.assertEqual([item.target for item in removed], ["claims.3"])
        self.assertEqual(removed[0].reason, "repair_unaddressed")
        self.assertEqual(audit["duplicate_targets"], ["claims.2"])
        self.assertEqual(audit["unexpected_targets"], ["limitations.1"])

    def test_zero_survivors_is_explicit(self) -> None:
        original = synthesis.StructuredEvidenceAnswerV2(claims=[
            _claim("This answer claim has an invalid citation quote.", "S1", "invented quotation"),
        ])
        batch = synthesis.V2RepairBatch(repairs=[synthesis.V2RepairItem(
            target="claims.1", action="drop", unresolved_reason="not supported",
        )])
        recovered, audit, removed = synthesis.recover_v2_answer(original, batch, self.blocks)
        self.assertIsNone(recovered)
        self.assertEqual(audit["claims_after"], 0)
        self.assertEqual(removed[0].reason, "model_dropped")

    def test_all_limitations_may_be_removed_when_claims_survive(self) -> None:
        original = self.original.model_copy(deep=True)
        original.limitations = [_claim(
            "This limitation has an invalid citation quote.", "S1", "invented limitation quote"
        )]
        batch = synthesis.V2RepairBatch(repairs=[
            synthesis.V2RepairItem(target="claims.2", action="drop", unresolved_reason="unsupported"),
            synthesis.V2RepairItem(target="limitations.1", action="drop", unresolved_reason="unsupported"),
        ])
        recovered, audit, removed = synthesis.recover_v2_answer(original, batch, self.blocks)
        self.assertEqual([claim.source_id for claim in recovered.claims], ["S1", "S3"])
        self.assertEqual(recovered.limitations, [])
        self.assertEqual(audit["limitations_after"], 0)
        self.assertEqual(len(removed), 2)

    def test_strict_merge_behavior_is_unchanged(self) -> None:
        batch = synthesis.V2RepairBatch(repairs=[
            synthesis.V2RepairItem(target="claims.2", action="drop", unresolved_reason="first"),
            synthesis.V2RepairItem(target="claims.2", action="drop", unresolved_reason="duplicate"),
        ])
        with self.assertRaisesRegex(ValueError, "address exactly invalid fields"):
            synthesis.merge_v2_repairs(self.original, batch, self.blocks)


class RecoveryConfigurationTests(unittest.TestCase):
    def test_default_is_strict_and_keeps_v15_identity(self) -> None:
        self.assertEqual(Settings().claim_recovery_mode, "strict")
        self.assertEqual(effective_pipeline_version("off", "strict"), PIPELINE_VERSION)
        service = ResearchService(Settings(), redis=None)
        metadata = service._metadata(ResearchMode.fast, budget_for(ResearchMode.fast), "Starting")
        self.assertNotIn("claim_recovery_mode", metadata)

    def test_degrade_has_an_isolated_composable_identity(self) -> None:
        degraded = effective_pipeline_version("off", "degrade")
        shadow_degraded = effective_pipeline_version("shadow", "degrade")
        self.assertEqual(degraded, f"{PIPELINE_VERSION}-recovery-degrade-v1")
        self.assertIn("-semantic-shadow-v1", shadow_degraded)
        self.assertTrue(shadow_degraded.endswith("-recovery-degrade-v1"))

    def test_unknown_recovery_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Settings(claim_recovery_mode="permissive")


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


class RecoveryPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_stored_mode_wins_and_missing_mode_defaults_to_strict(self) -> None:
        session = SimpleNamespace(commit=AsyncMock())
        degraded = SimpleNamespace(
            id=uuid4(), status="running", attempt=1,
            model_metadata={"claim_recovery_mode": "degrade"},
        )
        service = ResearchService(Settings(claim_recovery_mode="strict"), redis=None)
        service._execute = AsyncMock()
        with patch("app.services.session_factory", return_value=_SessionContext(session)), \
             patch("app.services.claim_run", AsyncMock(return_value=degraded)), \
             patch("app.services.LeaseKeeper", _NoopLease):
            await service.run_queued(
                degraded.id, "Question", ResearchMode.fast,
                ResearchCacheMode.use_cached_answer, claim_recovery_mode="strict",
            )
        self.assertEqual(service._execute.await_args.kwargs["claim_recovery_mode"], "degrade")

        legacy = SimpleNamespace(id=uuid4(), status="running", attempt=1, model_metadata={})
        service = ResearchService(Settings(claim_recovery_mode="degrade"), redis=None)
        service._execute = AsyncMock()
        with patch("app.services.session_factory", return_value=_SessionContext(session)), \
             patch("app.services.claim_run", AsyncMock(return_value=legacy)), \
             patch("app.services.LeaseKeeper", _NoopLease):
            await service.run_queued(
                legacy.id, "Question", ResearchMode.fast,
                ResearchCacheMode.use_cached_answer,
            )
        self.assertEqual(service._execute.await_args.kwargs["claim_recovery_mode"], "strict")

    async def test_typed_zero_claim_failure_and_audit_are_persisted(self) -> None:
        now = datetime.now(timezone.utc)
        record = SimpleNamespace(
            id=uuid4(), question="Question", requested_by="tester", status="running",
            answer=None, error=None, model_metadata={
                "mode": "fast", "claim_recovery_mode": "degrade",
                "progress": {"stage": "Starting", "role": None, "model": None},
            }, created_at=now, completed_at=None, lease_owner=None, lease_expires_at=None,
        )
        session = SimpleNamespace(
            add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(), rollback=AsyncMock(),
            get=AsyncMock(return_value=record),
        )
        vector_service = SimpleNamespace(embedding_usage={}, retrieval_traces=[])
        removal = synthesis.RemovedField(
            "claims.1", "model_dropped", ("invalid quote",),
            "An invalid answer claim.", "S1",
        )
        failure = legacy_pipeline.NoValidClaimsRemain(
            {"claims_before": 1, "claims_after": 0, "removals": [removal.as_dict()]},
            (removal,),
        )

        class FailingOrchestrator:
            last_source_diagnostics = {}
            last_pubmed_inspector = []
            last_arxiv_inspector = []
            last_cache_diagnostics = {}
            last_mined_count = 0

            async def execute_full_study(self, *args, **kwargs):
                raise failure

        service = ResearchService(Settings(
            google_api_key="test", entrez_email="tester@example.com",
            claim_recovery_mode="degrade", cost_tracking_enabled=False,
        ), redis=None)
        service.cost_guard = None
        with patch("app.services.PostgresVectorService", return_value=vector_service), \
             patch("app.services.legacy_pipeline.ResearchAgent"), \
             patch("app.services.legacy_pipeline.Orchestrator", return_value=FailingOrchestrator()), \
             patch("app.services.legacy_pipeline.config.validate"), \
             patch("app.services.begin_trace", return_value="trace"), \
             patch("app.services.end_trace", return_value=[]), \
             patch("app.services.summarize_trace", return_value={"attempts": [], "estimated_cost_usd": None}):
            with self.assertRaises(legacy_pipeline.NoValidClaimsRemain):
                await service._execute(
                    session, record, "Question", ResearchMode.fast, False,
                    claim_recovery_mode="degrade",
                )
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.model_metadata["failure_kind"], "no_valid_claims_remain")
        persisted = record.model_metadata["claim_validation"]
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["recovery"]["claims_after"], 0)
        self.assertEqual(persisted["removals"][0]["reason"], "model_dropped")


class RecoveryProductionFlowTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, answer, repair) -> tuple[legacy_pipeline.ResearchAgent, AsyncMock]:
        documents = [
            Document(page_content="Alpha evidence supports the first result.",
                     metadata={"doc_id": "one", "title": "One"}),
            Document(page_content="Beta evidence supports the second result.",
                     metadata={"doc_id": "two", "title": "Two"}),
            Document(page_content="Gamma evidence supports the third result.",
                     metadata={"doc_id": "three", "title": "Three"}),
        ]
        vector_service = SimpleNamespace(retrieve_context=AsyncMock(return_value=documents))
        agent = legacy_pipeline.ResearchAgent(
            vector_service, max_synthesis_subquestions=1, claim_recovery_mode="degrade"
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        gateway = AsyncMock(side_effect=[answer, repair])
        return agent, gateway

    async def test_degrade_returns_valid_claims_with_a_visible_omission(self) -> None:
        answer = synthesis.StructuredEvidenceAnswerV2(claims=[
            _claim("The first result is supported by alpha evidence.", "S1", "Alpha evidence supports the first result"),
            _claim("The second result has an invalid citation quote.", "S2", "invented quotation"),
            _claim("The third result is supported by gamma evidence.", "S3", "Gamma evidence supports the third result"),
        ])
        repair = synthesis.V2RepairBatch(repairs=[synthesis.V2RepairItem(
            target="claims.2", action="drop", unresolved_reason="no supported replacement",
        )])
        agent, gateway = self._agent(answer, repair)
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            result = await agent.run_research("What do the studies report?")
        self.assertEqual(result["claim_validation"]["status"], "completed_with_removals")
        self.assertEqual([item["source_id"] for item in result["claim_validation"]["claims"]], ["S1", "S3"])
        self.assertIn("Some statements were omitted", result["answer"])
        self.assertEqual(gateway.await_count, 2)

    async def test_zero_survivors_raise_typed_failure_with_audit(self) -> None:
        answer = synthesis.StructuredEvidenceAnswerV2(claims=[
            _claim("This answer claim has an invalid citation quote.", "S1", "invented quotation"),
        ])
        repair = synthesis.V2RepairBatch(repairs=[synthesis.V2RepairItem(
            target="claims.1", action="drop", unresolved_reason="no supported replacement",
        )])
        agent, gateway = self._agent(answer, repair)
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            with self.assertRaises(legacy_pipeline.NoValidClaimsRemain) as raised:
                await agent.run_research("What do the studies report?")
        self.assertEqual(raised.exception.failure_kind, "no_valid_claims_remain")
        self.assertEqual(raised.exception.recovery_audit["claims_after"], 0)
        self.assertEqual(gateway.await_count, 2)

    async def test_generation_failure_is_not_converted_to_claim_recovery(self) -> None:
        vector_service = SimpleNamespace(retrieve_context=AsyncMock(return_value=[
            Document(page_content="Alpha evidence supports the first result.",
                     metadata={"doc_id": "one", "title": "One"}),
        ]))
        agent = legacy_pipeline.ResearchAgent(
            vector_service, max_synthesis_subquestions=1, claim_recovery_mode="degrade"
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        gateway = AsyncMock(side_effect=TimeoutError("provider timeout"))
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            with self.assertRaisesRegex(TimeoutError, "provider timeout"):
                await agent.run_research("What do the studies report?")
        gateway.assert_awaited_once()

    async def test_repair_failures_propagate_without_hidden_retries(self) -> None:
        answer = synthesis.StructuredEvidenceAnswerV2(claims=[
            _claim("This answer claim has an invalid citation quote.", "S1", "invented quotation"),
        ])
        failures = (
            TimeoutError("repair timeout"),
            CostBudgetExceeded("repair budget refused"),
            InvalidStructuredResponse("invalid repair output"),
            RuntimeError("repair provider failed"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                repair = synthesis.V2RepairBatch(repairs=[synthesis.V2RepairItem(
                    target="claims.1", action="drop", unresolved_reason="unused",
                )])
                agent, _ = self._agent(answer, repair)
                gateway = AsyncMock(side_effect=[answer, failure])
                with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
                    with self.assertRaises(type(failure)):
                        await agent.run_research("What do the studies report?")
                self.assertEqual(gateway.await_count, 2)

    async def test_strict_mode_keeps_post_repair_failure_message(self) -> None:
        answer = synthesis.StructuredEvidenceAnswerV2(claims=[
            _claim("This answer claim has an invalid citation quote.", "S1", "invented quotation"),
        ])
        repair = synthesis.V2RepairBatch(repairs=[synthesis.V2RepairItem(
            target="claims.1", action="replace",
            replacement=_claim("The replacement also has an invalid citation.", "S1", "still invented"),
        )])
        documents = [Document(page_content="Alpha evidence supports the first result.",
                              metadata={"doc_id": "one", "title": "One"})]
        agent = legacy_pipeline.ResearchAgent(
            SimpleNamespace(retrieve_context=AsyncMock(return_value=documents)),
            max_synthesis_subquestions=1,
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        gateway = AsyncMock(side_effect=[answer, repair])
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=gateway):
            with self.assertRaisesRegex(
                RuntimeError,
                "Structured synthesis still contained invalid evidence references after one repair attempt",
            ):
                await agent.run_research("What do the studies report?")
        self.assertEqual(gateway.await_count, 2)


if __name__ == "__main__":
    unittest.main()
