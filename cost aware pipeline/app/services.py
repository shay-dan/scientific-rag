"""Research execution and audit persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scientific_rag import legacy_pipeline

from .config import Settings
from .cost_policy import budget_for
from .models import ResearchRun, SourceDocument
from .pgvector_retrieval import PostgresVectorService
from .schemas import ResearchMode, ResearchResponse, SourceResponse

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    async def run(self, session: AsyncSession, question: str, mode: ResearchMode, generate_evaluation_dataset: bool) -> ResearchResponse:
        if generate_evaluation_dataset and not self.settings.allow_evaluation_dataset:
            raise ValueError("Evaluation-dataset generation is disabled to protect the API budget")
        if mode is ResearchMode.deep and not self.settings.enable_deep_research:
            raise ValueError("Deep research is disabled; set ENABLE_DEEP_RESEARCH=true to enable it")

        cached_record = (
            await session.execute(
                select(ResearchRun)
                .where(ResearchRun.question == question, ResearchRun.status == "completed")
                .order_by(ResearchRun.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if cached_record is not None:
            cached_sources = list((await session.execute(select(SourceDocument).where(SourceDocument.research_run_id == cached_record.id))).scalars())
            logger.info("research_cache_hit run_id=%s mode=%s", cached_record.id, mode.value)
            return to_response(cached_record, cached_sources, mode=mode, cached=True)

        budget = budget_for(mode)
        record = ResearchRun(question=question, status="running")
        session.add(record)
        await session.commit()
        await session.refresh(record)
        logger.info("research_started run_id=%s mode=%s budget=%s question_length=%s", record.id, mode.value, budget.as_dict(), len(question))

        try:
            async with self._lock:
                self.settings.validate_research_configuration()
                legacy_pipeline.config.validate()
                vector_service = PostgresVectorService(session, self.settings, record.id)
                research_agent = legacy_pipeline.ResearchAgent(
                    vector_service,
                    max_synthesis_subquestions=budget.max_synthesis_subquestions,
                    retrieval_documents=budget.retrieval_documents,
                    max_context_documents=budget.max_context_documents,
                )
                orchestrator = legacy_pipeline.Orchestrator(
                    vector_service,
                    research_agent,
                    max_plan_steps=budget.max_plan_steps,
                    papers_per_source=budget.papers_per_source,
                )
                pipeline_result = await asyncio.wait_for(
                    orchestrator.execute_full_study(
                        original_query=question,
                        clear_db=False,
                        generate_evaluation_dataset=generate_evaluation_dataset,
                    ),
                    timeout=self.settings.request_timeout_seconds,
                )
                report = pipeline_result["report"]
            record.status = "completed"
            record.answer = report["answer"]
            record.completed_at = datetime.now(timezone.utc)
            record.model_metadata = {
                "policy": "role-based Gemini policy",
                "mode": mode.value,
                "budget": budget.as_dict(),
                "estimated_max_generation_calls": budget.estimated_max_generation_calls,
                "cache_hit": False,
            }
            sources = [
                SourceResponse(
                    title=str(document.metadata.get("title", "Untitled source")),
                    source=str(document.metadata.get("source", "unknown")),
                    external_id=str(document.metadata.get("doc_id") or "") or None,
                    source_url=str(document.metadata.get("url") or "") or None,
                )
                for document in report.get("raw_docs_used", [])
            ]
            await session.commit()
            await session.refresh(record)
            logger.info("research_completed run_id=%s mode=%s source_count=%s estimated_max_generation_calls=%s", record.id, mode.value, len(sources), budget.estimated_max_generation_calls)
            return ResearchResponse(
                id=record.id,
                status=record.status,
                answer=record.answer,
                sources=sources,
                plan=list(pipeline_result.get("metadata", {}).get("plan_executed", [])),
                created_at=record.created_at,
                completed_at=record.completed_at,
                mode=mode,
            )
        except Exception as error:
            record.status = "failed"
            record.error = str(error)
            record.completed_at = datetime.now(timezone.utc)
            await session.commit()
            logger.exception("research_failed run_id=%s", record.id)
            raise


def to_response(record: ResearchRun, sources: list[SourceDocument] | None = None, mode: ResearchMode | None = None, cached: bool = False) -> ResearchResponse:
    return ResearchResponse(
        id=record.id,
        status=record.status,
        answer=record.answer,
        sources=[SourceResponse(title=s.title, source=s.source, external_id=s.external_id, source_url=s.source_url) for s in sources or []],
        created_at=record.created_at,
        completed_at=record.completed_at,
        mode=mode or ResearchMode(record.model_metadata.get("mode", "fast")),
        cached=cached,
    )
