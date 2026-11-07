import asyncio
import unittest

from langchain_core.documents import Document

from app.comparative_benchmark import RAGVariant
from app.real_comparison import StrategyBoundRetriever, execute_with_repository
from scientific_rag.domain import GatekeeperReview, QueryPlan, SubQuery


class FakeRepository:
    def __init__(self):
        self.calls = []
        self.retrieval_traces = []
        self.embedding_usage = {"estimated_cost_usd": 0.0, "calls": 0}

    async def retrieve_context(self, query, top_n, strategy):
        self.calls.append((query, top_n, strategy))
        return [Document(page_content="grounded evidence", metadata={"doc_id": "S1", "benchmark_context_id": "C1", "source": "pubmed"})]


class FakeAgent:
    def __init__(self, retriever, benchmark_subquestions, **kwargs):
        self.retriever = retriever
        self.subquestions = benchmark_subquestions

    async def run_research(self, question):
        docs = []
        for query in self.subquestions:
            docs.extend(await self.retriever.retrieve_context(query, top_n=5))
        return {"answer": "Grounded answer [S1]", "sub_questions": self.subquestions, "raw_docs_used": docs}


class FakeOrchestrator:
    mining_called = False

    def __init__(self, repository, agent, **kwargs):
        pass

    async def generate_plan(self, question):
        return QueryPlan(plan=[SubQuery(query="planned query", source_type="academic", domains=["medical"], external_sources=["pubmed"], reasoning="needed")])

    async def filter_plan(self, question, plan):
        return plan, [GatekeeperReview(relevance_score=1.0, should_skip=False, critique="aligned").model_dump()]

    @staticmethod
    def enforce_source_policy(plan, anchor_query=None):
        return plan

    async def run_mining_operation(self, *args, **kwargs):
        type(self).mining_called = True
        raise AssertionError("mining must never be called")


class RealComparisonTests(unittest.TestCase):
    def test_strategy_adapter_pins_real_variant(self):
        repository = FakeRepository()
        adapter = StrategyBoundRetriever(repository, "hybrid_basic")
        asyncio.run(adapter.retrieve_context("question", 5))
        self.assertEqual(repository.calls, [("question", 5, "hybrid_basic")])

    def test_non_agentic_uses_original_question_and_no_planner(self):
        repository = FakeRepository()
        result = asyncio.run(execute_with_repository(repository, "original question", RAGVariant.NAIVE, agent_factory=FakeAgent))
        self.assertEqual(repository.calls[0][0], "original question")
        self.assertEqual(repository.calls[0][2], "naive")
        self.assertEqual(result["selected_source_ids"], ["S1"])

    def test_agentic_runs_plan_and_gatekeeper_but_never_mines(self):
        FakeOrchestrator.mining_called = False
        repository = FakeRepository()
        result = asyncio.run(execute_with_repository(repository, "original question", RAGVariant.AGENTIC_FAST, agent_factory=FakeAgent, orchestrator_factory=FakeOrchestrator))
        self.assertEqual(repository.calls[0][0], "planned query")
        self.assertEqual(repository.calls[0][2], "enhanced")
        self.assertFalse(FakeOrchestrator.mining_called)
        self.assertFalse(result["planning"]["external_mining_permitted"])

    def test_unknown_llm_usage_never_becomes_zero_cost(self):
        repository = FakeRepository()
        result = asyncio.run(execute_with_repository(repository, "question", RAGVariant.HYBRID_ENHANCED, agent_factory=FakeAgent))
        self.assertIsNone(result["estimated_cost_usd"])
        self.assertFalse(result["cost_complete"])


if __name__ == "__main__":
    unittest.main()
