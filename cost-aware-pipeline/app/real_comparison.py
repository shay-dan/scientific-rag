"""Real frozen-corpus execution paths for the four RAG variants."""

from __future__ import annotations

import time
from typing import Any, Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from scientific_rag import legacy_pipeline
from scientific_rag.llm_gateway import begin_trace, end_trace, summarize_trace

from .comparative_benchmark import RAGVariant
from .config import Settings
from .cost_policy import ResearchBudget, budget_for
from .pgvector_retrieval import PostgresVectorService
from .schemas import ResearchMode


class StrategyBoundRetriever:
    """Expose the legacy retriever interface while pinning a real strategy."""

    def __init__(self, repository: PostgresVectorService, strategy: str) -> None:
        self.repository = repository
        self.strategy = strategy
        self.returned_documents: list[Any] = []

    async def retrieve_context(self, query: str, top_n: int = 10):
        documents = await self.repository.retrieve_context(query=query, top_n=top_n, strategy=self.strategy)
        self.returned_documents.extend(documents)
        return documents


class FrozenResearchAgent(legacy_pipeline.ResearchAgent):
    """Production ResearchAgent with benchmark-controlled retrieval queries."""

    def __init__(self, *args, benchmark_subquestions: list[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.benchmark_subquestions = benchmark_subquestions

    async def decompose_query_async(self, query: str) -> list[str]:
        # Planning is either intentionally disabled for a non-agentic baseline,
        # or already completed by the production Orchestrator + Gatekeeper.
        return self.benchmark_subquestions


def _strategy_for(variant: RAGVariant) -> str:
    return {
        RAGVariant.NAIVE: "naive",
        RAGVariant.HYBRID_BASIC: "hybrid_basic",
        RAGVariant.HYBRID_ENHANCED: "enhanced",
        RAGVariant.AGENTIC_FAST: "enhanced",
    }[variant]


def _source_identity(document: Any) -> str:
    metadata = document.metadata
    return str(metadata.get("doc_id") or metadata.get("url") or metadata.get("title") or "")


async def _agentic_queries(
    question: str,
    repository: PostgresVectorService,
    budget: ResearchBudget,
    *,
    orchestrator_factory: Callable[..., Any] = legacy_pipeline.Orchestrator,
) -> tuple[list[str], dict[str, Any]]:
    """Run only the real planning and gatekeeping portion of Orchestrator."""
    placeholder_agent = object()
    orchestrator = orchestrator_factory(
        repository,
        placeholder_agent,
        max_plan_steps=budget.max_plan_steps,
        papers_per_source=budget.papers_per_source,
    )
    raw_plan = await orchestrator.generate_plan(question)
    filtered_plan, gatekeeper_reviews = await orchestrator.filter_plan(question, raw_plan)
    filtered_plan = orchestrator.enforce_source_policy(filtered_plan, anchor_query=question)
    queries = [step.query for step in filtered_plan.plan][: budget.max_synthesis_subquestions]
    if not queries:
        raise RuntimeError("Agentic Fast planning produced no eligible frozen-corpus retrieval query")
    return queries, {
        "raw_plan": [step.model_dump() for step in raw_plan.plan],
        "filtered_plan": [step.model_dump() for step in filtered_plan.plan],
        "gatekeeper_reviews": gatekeeper_reviews,
        "external_mining_permitted": False,
    }


async def execute_with_repository(
    repository: PostgresVectorService,
    question: str,
    variant: RAGVariant,
    *,
    redis: Any | None = None,
    agent_factory: Callable[..., Any] = FrozenResearchAgent,
    orchestrator_factory: Callable[..., Any] = legacy_pipeline.Orchestrator,
) -> dict[str, Any]:
    """Execute one variant against an already frozen repository."""
    started = time.perf_counter()
    trace_token = begin_trace(redis)
    trace_closed = False
    planning: dict[str, Any] = {"external_mining_permitted": False}
    try:
        strategy = _strategy_for(variant)
        retriever = StrategyBoundRetriever(repository, strategy)
        if variant is RAGVariant.AGENTIC_FAST:
            budget = budget_for(ResearchMode.fast)
            subquestions, planning = await _agentic_queries(
                question,
                repository,
                budget,
                orchestrator_factory=orchestrator_factory,
            )
            retrieval_documents = budget.retrieval_documents
            max_context_documents = budget.max_context_documents
        else:
            subquestions = [question]
            retrieval_documents = 5
            max_context_documents = 5
        agent = agent_factory(
            retriever,
            benchmark_subquestions=subquestions,
            max_synthesis_subquestions=len(subquestions),
            retrieval_documents=retrieval_documents,
            max_context_documents=max_context_documents,
        )
        report = await agent.run_research(question)
        attempts = end_trace(trace_token)
        trace_closed = True
        llm = summarize_trace(attempts)
        documents = report.get("raw_docs_used", [])
        llm_cost = llm.get("estimated_cost_usd")
        embedding_cost = repository.embedding_usage.get("estimated_cost_usd")
        total_cost = (
            round(float(llm_cost) + float(embedding_cost), 8)
            if llm_cost is not None and embedding_cost is not None
            else None
        )
        return {
            "status": "completed",
            "variant": variant.value,
            "answer": report.get("answer"),
            "sub_questions": report.get("sub_questions", subquestions),
            "retrieval_strategy": strategy,
            "selected_source_ids": [_source_identity(document) for document in documents],
            "selected_context_ids": [document.metadata.get("benchmark_context_id") for document in documents],
            "source_count": len({_source_identity(document) for document in documents}),
            "context_count": len(documents),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm": llm,
            "embedding": dict(repository.embedding_usage),
            "estimated_cost_usd": total_cost,
            "cost_complete": total_cost is not None,
            "planning": planning,
            "retrieval_traces": list(repository.retrieval_traces),
        }
    finally:
        if not trace_closed:
            end_trace(trace_token)


async def execute_variant(
    session: AsyncSession,
    settings: Settings,
    snapshot_run_id: UUID,
    question: str,
    variant: RAGVariant,
    *,
    redis: Any | None = None,
) -> dict[str, Any]:
    """Create the real run-scoped repository and execute one comparison arm."""
    repository = PostgresVectorService(session, settings, snapshot_run_id)
    repository.restrict_final_retrieval_to_current_run()
    return await execute_with_repository(repository, question, variant, redis=redis)
