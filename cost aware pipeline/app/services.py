"""Research execution, cache handling, and persisted UI progress."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scientific_rag import legacy_pipeline

from .config import Settings
from .cost_policy import ResearchBudget, budget_for
from .db import session_factory
from .models import ResearchRun, SourceDocument
from .pgvector_retrieval import PostgresVectorService
from .schemas import ResearchMode, ResearchResponse, SourceResponse

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    def _validate_request(self, mode: ResearchMode, generate_evaluation_dataset: bool) -> None:
        if generate_evaluation_dataset and not self.settings.allow_evaluation_dataset:
            raise ValueError("Evaluation-dataset generation is disabled to protect the API budget")
        if mode is ResearchMode.deep and not self.settings.enable_deep_research:
            raise ValueError("Deep research is disabled; set ENABLE_DEEP_RESEARCH=true to enable it")

    def _model_for_role(self, role: str | None) -> str | None:
        if role == "embedding":
            return self.settings.embedding_model
        if role is None:
            return None
        return legacy_pipeline.config.MODEL_POLICY.route_for(role).primary

    def _metadata(self, mode: ResearchMode, budget: ResearchBudget, stage: str, role: str | None = None) -> dict:
        return {
            "policy": "role-based Gemini policy",
            "mode": mode.value,
            "budget": budget.as_dict(),
            "estimated_max_generation_calls": budget.estimated_max_generation_calls,
            "cache_hit": False,
            "progress": {"stage": stage, "role": role, "model": self._model_for_role(role)},
        }

    async def _find_cached(self, session: AsyncSession, question: str) -> ResearchRun | None:
        return (
            await session.execute(
                select(ResearchRun)
                .where(ResearchRun.question == question, ResearchRun.status == "completed")
                .order_by(ResearchRun.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    async def _sources(self, session: AsyncSession, run_id: UUID) -> list[SourceDocument]:
        return list((await session.execute(select(SourceDocument).where(SourceDocument.research_run_id == run_id))).scalars())

    async def queue(self, session: AsyncSession, question: str, mode: ResearchMode, *, bypass_cache: bool = False, force_mining: bool = False) -> ResearchResponse:
        """Create an immediately visible job; its work continues in a separate session."""
        self._validate_request(mode, False)
        cached = await self._find_cached(session, question)
        if cached is not None and not bypass_cache:
            logger.info("research_cache_hit run_id=%s mode=%s", cached.id, mode.value)
            return to_response(cached, await self._sources(session, cached.id), mode=mode, cached=True)

        budget = budget_for(mode)
        metadata = self._metadata(mode, budget, "Queued for research")
        metadata["force_mining"] = force_mining
        record = ResearchRun(question=question, status="queued", model_metadata=metadata)
        session.add(record)
        await session.commit()
        await session.refresh(record)
        logger.info("research_queued run_id=%s mode=%s budget=%s", record.id, mode.value, budget.as_dict())
        asyncio.create_task(self._run_queued(record.id, question, mode, force_mining), name=f"research-{record.id}")
        return to_response(record, mode=mode)

    async def _run_queued(self, run_id: UUID, question: str, mode: ResearchMode, force_mining: bool) -> None:
        async with session_factory() as session:
            record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one()
            record.status = "running"
            await session.commit()
            try:
                await self._execute(session, record, question, mode, False, force_mining=force_mining)
            except Exception:
                # _execute persists the failure. Keep this task from producing an unhandled-task warning.
                logger.exception("queued_research_task_failed run_id=%s", run_id)

    async def run(self, session: AsyncSession, question: str, mode: ResearchMode, generate_evaluation_dataset: bool) -> ResearchResponse:
        """Synchronous API path retained for programmatic clients."""
        self._validate_request(mode, generate_evaluation_dataset)
        cached = await self._find_cached(session, question)
        if cached is not None:
            return to_response(cached, await self._sources(session, cached.id), mode=mode, cached=True)

        budget = budget_for(mode)
        record = ResearchRun(question=question, status="running", model_metadata=self._metadata(mode, budget, "Starting research"))
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return await self._execute(session, record, question, mode, generate_evaluation_dataset)

    async def _set_progress(self, session: AsyncSession, record: ResearchRun, stage: str, role: str | None = None) -> None:
        metadata = dict(record.model_metadata)
        metadata["progress"] = {"stage": stage, "role": role, "model": self._model_for_role(role)}
        record.model_metadata = metadata
        await session.commit()
        logger.info("research_progress run_id=%s stage=%s role=%s model=%s", record.id, stage, role, self._model_for_role(role))

    async def _execute(self, session: AsyncSession, record: ResearchRun, question: str, mode: ResearchMode, generate_evaluation_dataset: bool, *, force_mining: bool = False) -> ResearchResponse:
        budget = budget_for(mode)
        logger.info("research_started run_id=%s mode=%s question_length=%s", record.id, mode.value, len(question))
        try:
            async with self._lock:
                self.settings.validate_research_configuration()
                legacy_pipeline.config.validate()
                await self._set_progress(session, record, "Planning the research approach", "orchestrator")

                async def progress(stage: str, role: str | None = None) -> None:
                    await self._set_progress(session, record, stage, role)

                vector_service = PostgresVectorService(session, self.settings, record.id)
                research_agent = legacy_pipeline.ResearchAgent(
                    vector_service,
                    max_synthesis_subquestions=budget.max_synthesis_subquestions,
                    retrieval_documents=budget.retrieval_documents,
                    max_context_documents=budget.max_context_documents,
                    progress_callback=progress,
                )
                orchestrator = legacy_pipeline.Orchestrator(
                    vector_service,
                    research_agent,
                    max_plan_steps=budget.max_plan_steps,
                    papers_per_source=budget.papers_per_source,
                    progress_callback=progress,
                )
                pipeline_result = await asyncio.wait_for(
                    orchestrator.execute_full_study(question, clear_db=False, generate_evaluation_dataset=generate_evaluation_dataset, skip_database_cache=force_mining),
                    timeout=self.settings.request_timeout_seconds,
                )
                report = pipeline_result["report"]

            record.status = "completed"
            record.answer = report["answer"]
            record.completed_at = datetime.now(timezone.utc)
            metadata = self._metadata(mode, budget, "Research complete")
            metadata["progress"]["model"] = None
            record.model_metadata = metadata
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
            logger.info("research_completed run_id=%s mode=%s source_count=%s", record.id, mode.value, len(sources))
            return ResearchResponse(id=record.id, status=record.status, answer=record.answer, sources=sources, plan=list(pipeline_result.get("metadata", {}).get("plan_executed", [])), created_at=record.created_at, completed_at=record.completed_at, mode=mode)
        except Exception as error:
            record.status = "failed"
            record.error = str(error)
            record.completed_at = datetime.now(timezone.utc)
            metadata = dict(record.model_metadata)
            metadata["progress"] = {"stage": "Research failed", "role": None, "model": None}
            record.model_metadata = metadata
            await session.commit()
            logger.exception("research_failed run_id=%s", record.id)
            raise


def to_response(record: ResearchRun, sources: list[SourceDocument] | None = None, mode: ResearchMode | None = None, cached: bool = False) -> ResearchResponse:
    return ResearchResponse(id=record.id, status=record.status, answer=record.answer, sources=[SourceResponse(title=s.title, source=s.source, external_id=s.external_id, source_url=s.source_url) for s in sources or []], created_at=record.created_at, completed_at=record.completed_at, mode=mode or ResearchMode(record.model_metadata.get("mode", "fast")), cached=cached)
