"""Research execution, cache handling, and persisted UI progress."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from arq.connections import ArqRedis

from scientific_rag import legacy_pipeline

from .config import Settings
from .cost_policy import ResearchBudget, budget_for
from .db import session_factory
from .models import ResearchEvidence, ResearchRun, SourceDocument
from .pgvector_retrieval import PostgresVectorService
from .schemas import ResearchMode, ResearchResponse, SourceResponse

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(self, settings: Settings, redis: ArqRedis | None = None) -> None:
        self.settings = settings
        self.redis = redis
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

    async def _find_cached(self, session: AsyncSession, question: str, actor: str) -> ResearchRun | None:
        return (
            await session.execute(
                select(ResearchRun)
                .where(ResearchRun.question == question, ResearchRun.status == "completed", ResearchRun.requested_by == actor)
                .order_by(ResearchRun.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    async def _sources(self, session: AsyncSession, run_id: UUID) -> list[SourceDocument]:
        return list((await session.execute(select(SourceDocument).where(SourceDocument.research_run_id == run_id))).scalars())

    async def sources_for_run(self, session: AsyncSession, run_id: UUID) -> list[SourceResponse]:
        """Return the cited evidence sources, with a legacy fallback for older runs."""
        evidence = list((await session.execute(
            select(ResearchEvidence)
            .where(ResearchEvidence.research_run_id == run_id)
            .order_by(ResearchEvidence.ordinal)
        )).scalars())
        if not evidence:
            return [SourceResponse(title=source.title, source=source.source, external_id=source.external_id, source_url=source.source_url) for source in await self._sources(session, run_id)]
        sources: dict[str, SourceResponse] = {}
        for item in evidence:
            metadata = item.metadata_json
            citation_id = str(metadata.get("citation_id") or f"S{item.ordinal + 1}")
            sources.setdefault(citation_id, SourceResponse(
                citation_id=citation_id,
                title=str(metadata.get("title", "Untitled source")),
                source=str(metadata.get("source", "unknown")),
                external_id=metadata.get("external_id"),
                source_url=metadata.get("source_url"),
            ))
        return list(sources.values())

    async def queue(self, session: AsyncSession, question: str, mode: ResearchMode, *, actor: str, bypass_cache: bool = False, force_mining: bool = False) -> ResearchResponse:
        """Create an immediately visible job; its work continues in a separate session."""
        self._validate_request(mode, False)
        cached = await self._find_cached(session, question, actor)
        if cached is not None and not bypass_cache:
            logger.info("research_cache_hit run_id=%s mode=%s", cached.id, mode.value)
            return to_response(cached, await self.sources_for_run(session, cached.id), mode=mode, cached=True)

        budget = budget_for(mode)
        metadata = self._metadata(mode, budget, "Queued for research")
        metadata["force_mining"] = force_mining
        record = ResearchRun(question=question, requested_by=actor, status="queued", model_metadata=metadata)
        session.add(record)
        await session.commit()
        await session.refresh(record)
        logger.info("research_queued run_id=%s mode=%s budget=%s", record.id, mode.value, budget.as_dict())
        if self.redis is None:
            raise RuntimeError("Research queue is unavailable")
        await self.redis.enqueue_job("run_research_job", str(record.id), question, mode.value, force_mining)
        return to_response(record, mode=mode)

    async def run_queued(self, run_id: UUID, question: str, mode: ResearchMode, force_mining: bool) -> None:
        async with session_factory() as session:
            record = (await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))).scalar_one()
            record.status = "running"
            await session.commit()
            try:
                await self._execute(session, record, question, mode, False, force_mining=force_mining)
            except Exception:
                # _execute persists the failure. Keep this task from producing an unhandled-task warning.
                logger.exception("queued_research_task_failed run_id=%s", run_id)

    async def run(self, session: AsyncSession, question: str, mode: ResearchMode, generate_evaluation_dataset: bool, *, actor: str = "api") -> ResearchResponse:
        """Synchronous API path retained for programmatic clients."""
        self._validate_request(mode, generate_evaluation_dataset)
        cached = await self._find_cached(session, question, actor)
        if cached is not None:
            return to_response(cached, await self.sources_for_run(session, cached.id), mode=mode, cached=True)

        budget = budget_for(mode)
        record = ResearchRun(question=question, requested_by=actor, status="running", model_metadata=self._metadata(mode, budget, "Starting research"))
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
            evidence_documents = report.get("raw_docs_used", [])
            for ordinal, document in enumerate(evidence_documents):
                metadata = dict(document.metadata or {})
                session.add(ResearchEvidence(
                    research_run_id=record.id,
                    ordinal=ordinal,
                    content=str(document.page_content),
                    metadata_json={
                        "title": str(metadata.get("title", "Untitled source")),
                        "source": str(metadata.get("source", "unknown")),
                        "external_id": str(metadata.get("doc_id") or metadata.get("id") or "") or None,
                        "source_url": str(metadata.get("url") or "") or None,
                        "citation_id": str(metadata.get("citation_id") or "") or None,
                    },
                ))
            metadata = self._metadata(mode, budget, "Research complete")
            cache_diagnostics = dict(pipeline_result.get("metadata", {}).get("cache", {}))
            metadata["cache_hit"] = pipeline_result.get("status") == "retrieved_from_cache"
            metadata["cache"] = cache_diagnostics
            source_diagnostics = dict(pipeline_result.get("metadata", {}).get("source_diagnostics", {}))
            for document in evidence_documents:
                source = str(document.metadata.get("source", "unknown"))
                source_diagnostics.setdefault(source, {"requested": 0, "collected": 0, "errors": []})
                source_diagnostics[source]["used"] = source_diagnostics[source].get("used", 0) + 1
            metadata["source_diagnostics"] = source_diagnostics
            metadata["progress"]["model"] = None
            record.model_metadata = metadata
            sources = [
                SourceResponse(
                    citation_id=str(document.metadata.get("citation_id") or "") or None,
                    title=str(document.metadata.get("title", "Untitled source")),
                    source=str(document.metadata.get("source", "unknown")),
                    external_id=str(document.metadata.get("doc_id") or "") or None,
                    source_url=str(document.metadata.get("url") or "") or None,
                )
                for document in evidence_documents
            ]
            await session.commit()
            await session.refresh(record)
            logger.info("research_completed run_id=%s mode=%s source_count=%s", record.id, mode.value, len(sources))
            return ResearchResponse(id=record.id, status=record.status, answer=record.answer, sources=sources, plan=list(pipeline_result.get("metadata", {}).get("plan_executed", [])), created_at=record.created_at, completed_at=record.completed_at, mode=mode, source_diagnostics=source_diagnostics, cache_diagnostics=cache_diagnostics)
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


def to_response(record: ResearchRun, sources: list[SourceDocument | SourceResponse] | None = None, mode: ResearchMode | None = None, cached: bool = False) -> ResearchResponse:
    response_sources = [
        source if isinstance(source, SourceResponse) else SourceResponse(title=source.title, source=source.source, external_id=source.external_id, source_url=source.source_url)
        for source in sources or []
    ]
    return ResearchResponse(id=record.id, status=record.status, answer=record.answer, sources=response_sources, created_at=record.created_at, completed_at=record.completed_at, mode=mode or ResearchMode(record.model_metadata.get("mode", "fast")), cached=cached, source_diagnostics=record.model_metadata.get("source_diagnostics", {}), cache_diagnostics=record.model_metadata.get("cache", {}))
