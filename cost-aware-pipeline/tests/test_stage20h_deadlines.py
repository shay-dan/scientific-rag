"""Synthetic offline probes; labels are regression expectations, not scientific gold."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from langchain_core.documents import Document
from scientific_rag.query_quality import extract_anchors, build_pubmed_query, concepts_covered
from scientific_rag.legacy_pipeline import MinerService, Orchestrator


class DeadlineRegressions(unittest.TestCase):
    def test_fake_clock_no_repair_when_reserve_unavailable(self):
        from scientific_rag.execution_budget import ExecutionBudget, ResearchDeadlineExceeded
        now = [0.0]
        budget = ExecutionBudget(100, clock=lambda:now[0])
        now[0] = 85
        with self.assertRaises(ResearchDeadlineExceeded):
            budget.require('repair', minimum_seconds=20)
        self.assertEqual(budget.remaining(), 15)

    def test_context_budget_isolated_and_restored(self):
        from scientific_rag.execution_budget import ExecutionBudget, current_budget, use_budget
        b = ExecutionBudget(50)
        with use_budget(b):
            self.assertIs(current_budget(), b)
        self.assertIsNone(current_budget())

    def test_cancellation_not_swallowed_or_retried(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, run_with_budget
        async def run():
            cancelled = []
            async def operation():
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(True)
            with use_budget(ExecutionBudget(20)):
                task = asyncio.create_task(run_with_budget('synthesis', operation, cap_seconds=10))
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(cancelled, [True])
        asyncio.run(run())


class GatewayBudgetRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_expired_budget_makes_zero_calls_and_zero_reservations(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, ResearchDeadlineExceeded
        from scientific_rag import llm_gateway
        now = [0.0]
        b = ExecutionBudget(10, clock=lambda:now[0])
        now[0] = 11
        with use_budget(b), patch.object(llm_gateway, 'create_chat_model') as model, patch.object(llm_gateway, 'shared_admit', new=AsyncMock()) as admit:
            with self.assertRaises(ResearchDeadlineExceeded):
                await llm_gateway.ainvoke('expert', 0, lambda x:x, 'synthetic')
        model.assert_not_called()
        admit.assert_not_awaited()

    async def test_actual_repair_path_refuses_new_call_and_retains_valid_fields(self):
        from types import SimpleNamespace
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, ResearchDeadlineExceeded
        from scientific_rag.legacy_pipeline import ResearchAgent
        from app.weak_model_synthesis import StructuredEvidenceAnswerV2, EvidenceClaim
        answer = StructuredEvidenceAnswerV2(claims=[
            EvidenceClaim(text='Aspirin was associated with mortality.', source_id='S1', evidence_quote='Aspirin was associated with mortality.'),
            EvidenceClaim(text='Aspirin also reduced bleeding.', source_id='S1', evidence_quote='An invented quotation.'),
        ])
        vector = SimpleNamespace(retrieve_context=AsyncMock(return_value=[Document(page_content='Aspirin was associated with mortality.', metadata={'doc_id':'one'})]))
        now = [0.0]
        budget = ExecutionBudget(100, clock=lambda:now[0])
        async def generate(*args, **kw):
            now[0] = 85
            return answer
        with patch('scientific_rag.legacy_pipeline.ainvoke_structured', new=AsyncMock(side_effect=generate)) as llm:
            agent = ResearchAgent(vector)
            agent.decompose_query_async = AsyncMock(return_value=['aspirin'])
            with use_budget(budget), self.assertRaises(ResearchDeadlineExceeded):
                await agent.run_research('Does aspirin reduce mortality and bleeding?')
        self.assertEqual(llm.await_count, 1)
        self.assertEqual(len(agent.last_validated_evidence['claims']), 1)
        self.assertEqual(agent.last_validated_evidence['sources'][0]['external_id'], 'one')
        self.assertEqual(agent.last_validated_evidence['sources'][0]['source_id'], 'S1')
        self.assertEqual(agent.last_validated_evidence['status'], 'structural_only_not_delivered')

    async def test_admission_precedes_coroutine_creation(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, run_with_budget, ResearchDeadlineExceeded
        now = [0.0]
        budget = ExecutionBudget(10, clock=lambda:now[0])
        now[0] = 9
        factory = AsyncMock()
        with use_budget(budget), self.assertRaises(ResearchDeadlineExceeded):
            await run_with_budget('repair', factory, cap_seconds=90, minimum_seconds=20)
        factory.assert_not_called()

    async def test_provider_timeout_with_remaining_budget_keeps_original_type(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, run_with_budget, ResearchDeadlineExceeded
        async def provider():
            raise TimeoutError('provider timed out')
        with use_budget(ExecutionBudget(100)):
            try:
                await run_with_budget('repair', provider, cap_seconds=90)
            except TimeoutError as error:
                self.assertNotIsInstance(error, ResearchDeadlineExceeded)
                self.assertEqual(str(error), 'provider timed out')
            else:
                self.fail('Expected timeout')

    async def test_local_stage_timeout_cancels_once(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, run_with_budget, ResearchDeadlineExceeded
        stopped = []
        async def provider():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)
        with use_budget(ExecutionBudget(100)):
            with self.assertRaises(ResearchDeadlineExceeded) as raised:
                await run_with_budget('repair', provider, cap_seconds=0.01)
        self.assertEqual(raised.exception.failure_kind, 'stage_deadline_expired')
        self.assertEqual(stopped, [True])

    async def test_child_contexts_keep_independent_budgets(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, current_budget
        async def child(seconds):
            b = ExecutionBudget(seconds)
            with use_budget(b):
                await asyncio.sleep(0)
                return current_budget() is b
        self.assertEqual(await asyncio.gather(child(10), child(20)), [True, True])
        self.assertIsNone(current_budget())

    async def test_gateway_cancellation_records_uncertain_usage_without_fallback(self):
        from scientific_rag import llm_gateway
        chain = AsyncMock()
        entered = asyncio.Event()
        async def hanging(*args, **kw):
            entered.set()
            await asyncio.Event().wait()
        chain.ainvoke.side_effect = hanging
        token = llm_gateway.begin_trace()
        try:
            with patch.object(llm_gateway, 'routing_mode', return_value='legacy'), patch.object(llm_gateway, '_configured_providers', return_value=['groq','openrouter']), patch.object(llm_gateway, 'create_chat_model', return_value=chain) as factory:
                task = asyncio.create_task(llm_gateway.ainvoke('expert', 0, lambda m:m, 'synthetic'))
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(factory.call_count, 1)
        finally:
            attempts = llm_gateway.end_trace(token)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['error_type'], 'CancelledError')
        self.assertIsNone(attempts[0]['input_tokens'])

    async def test_arxiv_pacing_expiration_prevents_transport(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget, ResearchDeadlineExceeded
        from scientific_rag.legacy_pipeline import _BoundedArxivSession, _ArxivSharedState
        now = [0.0]
        def pacing(*args):
            now[0] = 11
        with use_budget(ExecutionBudget(10, clock=lambda:now[0])), patch.object(_ArxivSharedState, 'wait_for_slot', side_effect=pacing), patch('requests.Session.request') as transport:
            with self.assertRaises(ResearchDeadlineExceeded):
                _BoundedArxivSession().request('GET','https://example.invalid')
        transport.assert_not_called()

    async def test_monotonic_phase_and_provider_wait_diagnostics(self):
        from scientific_rag.execution_budget import ExecutionBudget, use_budget
        from scientific_rag import llm_gateway
        now = [0.0]
        budget = ExecutionBudget(100, clock=lambda:now[0])
        with use_budget(budget):
            budget.set_stage('planning','planner')
            now[0] = 10
            llm_gateway._record(llm_gateway.ProviderAttempt('synthetic-call',0,'test','test','planner',2000,True,False,admission_wait_ms=500))
            budget.set_stage('retrieval')
            now[0] = 30
            report = budget.snapshot()
        self.assertEqual([s['duration_seconds'] for s in report['stage_spans']], [10,20])
        span = report['provider_spans'][0]
        self.assertEqual(span['stage'],'planning')
        self.assertEqual(span['approximate_started_offset_seconds'],8)
        self.assertEqual(span['admission_wait_ms'],500)
        self.assertFalse(span['usage_known'])
