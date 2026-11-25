"""Research execution, cache handling, and persisted UI progress."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import re
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from arq.connections import ArqRedis

from scientific_rag.execution_budget import ExecutionBudget, use_budget, ResearchDeadlineExceeded
from scientific_rag import legacy_pipeline
from scientific_rag.llm_gateway import begin_trace, end_trace, primary_route_label, summarize_trace

from .config import Settings
from .cost_policy import ResearchBudget, budget_for, estimated_generation_calls
from .cost_control import CostGuard
from .db import session_factory
from .job_lifecycle import (
    INTERRUPTED_ERROR,
    LEASE_UNVERIFIABLE_ERROR,
    LeaseKeeper,
    acquire_identity_lock,
    claim_run,
    enqueue_run,
    settle_terminated_cost,
    terminate_run,
    worker_identity,
)
from .models import AuditEvent, ResearchEvidence, ResearchRun, SourceDocument
from .pgvector_retrieval import PostgresVectorService
from .schemas import ResearchCacheMode, ResearchMode, ResearchResponse, SourceResponse

logger = logging.getLogger(__name__)


def _responses_for_evidence_documents(evidence_documents: list) -> list[SourceResponse]:
    """Collapse repeated parent/child passages into one cited paper."""
    sources_by_identity: dict[str, SourceResponse] = {}
    for document in evidence_documents:
        citation_id = str(document.metadata.get("citation_id") or "") or None
        source = SourceResponse(
            citation_id=citation_id,
            title=str(document.metadata.get("title", "Untitled source")),
            source=str(document.metadata.get("source", "unknown")),
            external_id=str(document.metadata.get("doc_id") or "") or None,
            source_url=str(document.metadata.get("url") or "") or None,
            study_design=str(document.metadata.get("study_design") or "") or None,
            excerpt=str(document.page_content)[:1200],
        )
        identity = citation_id or source.external_id or source.source_url or source.title.casefold()
        sources_by_identity.setdefault(identity, source)
    return list(sources_by_identity.values())
# v15 (R3): untrusted-content framing and injection guards. v14 answers were produced
# under the pre-R3 security contract and must not be reused as cache hits.
PIPELINE_VERSION = "postgres-hybrid-rrf-mmr-parent-child-structured-v15-stage20h-v2"
ENQUEUE_TIMEOUT_SECONDS = 10


class ResearchFencedOut(RuntimeError):
    """This execution no longer owns the run (cancelled or recovered); its results are discarded."""


def effective_pipeline_version(
    semantic_verification_mode: str,
    claim_recovery_mode: str = "strict",
    *,
    generation_prompt_version: str = "v15",
    quote_recovery_mode: str = "off",
) -> str:
    """Separate cache identities per verification mode; historical records keep their stored version."""
    if semantic_verification_mode == "off":
        version = PIPELINE_VERSION
    elif semantic_verification_mode in {"shadow", "enforce"}:
        from .claim_verification import semantic_verifier_identity
        identity = semantic_verifier_identity()
        version = f"{PIPELINE_VERSION}-semantic-{semantic_verification_mode}-v1-{identity['version']}-{identity['sha256'][:12]}"
    else:
        raise ValueError(f"Unsupported semantic verification mode: {semantic_verification_mode}")
    if claim_recovery_mode == "degrade":
        version = f"{version}-recovery-degrade-v1"
    elif claim_recovery_mode != "strict":
        raise ValueError(f"Unsupported claim recovery mode: {claim_recovery_mode}")
    if generation_prompt_version == "v16":
        version = f"{version}-genprompt-v16"
    elif generation_prompt_version != "v15":
        raise ValueError(f"Unsupported generation prompt version: {generation_prompt_version}")
    from .quote_recovery import validate_quote_recovery_mode
    validate_quote_recovery_mode(quote_recovery_mode, semantic_verification_mode)
    if quote_recovery_mode != "off":
        version = f"{version}-quote-recovery-{quote_recovery_mode}-v1"
    return version


def _timeline_phase(stage: str, role: str | None) -> str:
    text = stage.lower()
    if "collect" in text or "mining" in text:
        return "mining"
    if "index" in text or role == "embedding":
        return "indexing"
    if "synth" in text or "answer" in text or role == "expert":
        return "synthesis"
    if re.search(r"\bsearch(?:ing)?\b", text) or "retriev" in text or "database" in text or "evidence" in text:
        return "retrieval"
    return "planning"


def _close_active_timeline(metadata: dict, at: datetime | None = None) -> None:
    timeline = metadata.get("timeline", [])
    now = at or datetime.now(timezone.utc)
    for item in reversed(timeline):
        if item.get("completed_at") is None:
            item["completed_at"] = now.isoformat()
            started_at = datetime.fromisoformat(item["started_at"])
            item["duration_seconds"] = round(max((now - started_at).total_seconds(), 0), 2)
            break


class ResearchService:
    def __init__(self, settings: Settings, redis: ArqRedis | None = None) -> None:
        self.settings = settings
        self.redis = redis
        self._lock = asyncio.Lock()
        self.cost_guard = CostGuard(redis, settings)

    async def _reserve_cost(self, record: ResearchRun, mode: ResearchMode) -> None:
        guard = getattr(self, "cost_guard", None)
        if guard is None:
            return
        reservation = await guard.reserve_research(str(record.id), mode)
        metadata = dict(record.model_metadata)
        metadata["cost_reservation"] = reservation.as_dict()
        record.model_metadata = metadata

    async def _reconcile_cost(self, record: ResearchRun, metadata: dict) -> None:
        gateway_cost = metadata.get("llm_gateway", {}).get("estimated_cost_usd")
        embedding_cost = metadata.get("embedding_usage", {}).get("estimated_cost_usd")
        components = [value for value in (gateway_cost, embedding_cost) if value is not None]
        total = round(sum(float(value) for value in components), 8) if components else None
        complete = gateway_cost is not None and embedding_cost is not None
        metadata["cost_accounting"] = {
            "llm_estimated_cost_usd": gateway_cost,
            "embedding_estimated_cost_usd": embedding_cost,
            "estimated_cost_usd": total,
            "complete": complete,
            "reservation": dict(record.model_metadata.get("cost_reservation", {})),
        }
        guard = getattr(self, "cost_guard", None)
        # Never release a conservative reservation when any provider omitted
        # usage metadata; the known component is only a lower bound.
        if guard is not None and total is not None and complete:
            reconciled = await guard.reconcile_research(str(record.id), total)
            if reconciled is not None:
                metadata["cost_accounting"]["reservation"] = reconciled.as_dict()

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
        return primary_route_label(role)

    def _metadata(self, mode: ResearchMode, budget: ResearchBudget, stage: str, role: str | None = None, *,
                  semantic_verification_mode: str | None = None,
                  claim_recovery_mode: str | None = None,
                  claim_audit_mode: str | None = None,
                  generation_prompt_version: str | None = None,
                  quote_recovery_mode: str | None = None) -> dict:
        verification_mode = semantic_verification_mode or getattr(
            getattr(self, "settings", None), "semantic_claim_verification_mode", "off"
        )
        recovery_mode = claim_recovery_mode or getattr(
            getattr(self, "settings", None), "claim_recovery_mode", "strict"
        )
        audit_mode = claim_audit_mode or getattr(
            getattr(self, "settings", None), "claim_audit_mode", "off"
        )
        prompt_version = generation_prompt_version or getattr(
            getattr(self, "settings", None), "generation_prompt_version", "v15"
        )
        quote_mode = quote_recovery_mode or getattr(
            getattr(self, "settings", None), "quote_recovery_mode", "off"
        )
        metadata = {
            "policy": "provider-agnostic Groq -> OpenRouter -> Gemini gateway",
            "mode": mode.value,
            "budget": budget.as_dict(),
            "estimated_max_generation_calls": estimated_generation_calls(budget, verification_mode),
            "cache_hit": False,
            "retrieval_version": effective_pipeline_version(
                verification_mode, recovery_mode, generation_prompt_version=prompt_version,
                quote_recovery_mode=quote_mode,
            ),
            "semantic_claim_verification_mode": verification_mode,
            "progress": {"stage": stage, "role": role, "model": self._model_for_role(role)},
        }
        if verification_mode != "off":
            from .claim_verification import semantic_verifier_identity
            metadata["semantic_verifier_identity"] = semantic_verifier_identity()
        # Strict is represented by absence for backward-compatible metadata and
        # so historical queued jobs resolve safely to the production default.
        if recovery_mode != "strict":
            metadata["claim_recovery_mode"] = recovery_mode
        if audit_mode != "off":
            metadata["claim_audit_mode"] = audit_mode
        if prompt_version != "v15":
            metadata["generation_prompt_version"] = prompt_version
        if quote_mode != "off":
            metadata["quote_recovery_mode"] = quote_mode
        return metadata

    async def _find_cached(self, session: AsyncSession, question: str, actor: str, mode: ResearchMode,
                           semantic_verification_mode: str | None = None,
                           claim_recovery_mode: str | None = None,
                           generation_prompt_version: str | None = None) -> ResearchRun | None:
        verification_mode = semantic_verification_mode or getattr(
            getattr(self, "settings", None), "semantic_claim_verification_mode", "off"
        )
        recovery_mode = claim_recovery_mode or getattr(
            getattr(self, "settings", None), "claim_recovery_mode", "strict"
        )
        prompt_version = generation_prompt_version or getattr(
            getattr(self, "settings", None), "generation_prompt_version", "v15"
        )
        version = effective_pipeline_version(
            verification_mode, recovery_mode, generation_prompt_version=prompt_version,
            quote_recovery_mode=getattr(getattr(self, "settings", None), "quote_recovery_mode", "off"),
        )
        return (
            await session.execute(
                select(ResearchRun)
                .where(
                    ResearchRun.question == question,
                    ResearchRun.status == "completed",
                    ResearchRun.requested_by == actor,
                    ResearchRun.model_metadata["mode"].as_string() == mode.value,
                    ResearchRun.model_metadata["retrieval_version"].as_string() == version,
                )
                .order_by(ResearchRun.completed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    async def _find_in_flight(self, session: AsyncSession, question: str, actor: str, mode: ResearchMode,
                              version: str) -> ResearchRun | None:
        return (
            await session.execute(
                select(ResearchRun)
                .where(
                    ResearchRun.question == question,
                    ResearchRun.status.in_(("queued", "running")),
                    ResearchRun.requested_by == actor,
                    ResearchRun.model_metadata["mode"].as_string() == mode.value,
                    ResearchRun.model_metadata["retrieval_version"].as_string() == version,
                )
                .order_by(ResearchRun.created_at.desc())
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
                study_design=metadata.get("study_design"),
                excerpt=item.content[:1200],
            ))
        return list(sources.values())

    async def queue(self, session: AsyncSession, question: str, mode: ResearchMode, *, actor: str,
                    cache_mode: ResearchCacheMode = ResearchCacheMode.use_cached_answer,
                    bypass_cache: bool | None = None, force_mining: bool | None = None,
                    extra_metadata: dict | None = None) -> ResearchResponse:
        """Create an immediately visible job; its work continues in a separate session.

        The run and its cost reservation are committed before enqueueing. If the
        enqueue fails or the process stops in between, the run stays ``queued`` and
        the worker recovery sweep re-enqueues it with the same job id.
        """
        self._validate_request(mode, False)
        # Compatibility for existing internal callers while the public contract
        # migrates to one explicit three-state policy.
        if force_mining:
            cache_mode = ResearchCacheMode.refresh_sources
        elif bypass_cache:
            cache_mode = ResearchCacheMode.regenerate_answer
        if self.redis is None:
            raise RuntimeError("Research queue is unavailable")
        semantic_mode = getattr(
            getattr(self, "settings", None), "semantic_claim_verification_mode", "off"
        )
        recovery_mode = getattr(
            getattr(self, "settings", None), "claim_recovery_mode", "strict"
        )
        audit_mode = getattr(
            getattr(self, "settings", None), "claim_audit_mode", "off"
        )
        prompt_version = getattr(
            getattr(self, "settings", None), "generation_prompt_version", "v15"
        )
        version = effective_pipeline_version(
            semantic_mode, recovery_mode, generation_prompt_version=prompt_version,
            quote_recovery_mode=getattr(getattr(self, "settings", None), "quote_recovery_mode", "off"),
        )
        # Serialize identical submissions (same caller, mode, pipeline version, question).
        await acquire_identity_lock(session, actor, mode.value, version, question)
        cached = await self._find_cached(
            session, question, actor, mode, semantic_mode, recovery_mode,
            generation_prompt_version=prompt_version,
        )
        if cached is not None and cache_mode is ResearchCacheMode.use_cached_answer:
            logger.info("research_cache_hit run_id=%s mode=%s", cached.id, mode.value)
            return to_response(cached, await self.sources_for_run(session, cached.id), mode=mode, cached=True)
        if cache_mode is ResearchCacheMode.use_cached_answer:
            in_flight = await self._find_in_flight(session, question, actor, mode, version)
            if in_flight is not None:
                logger.info("research_in_flight_reused run_id=%s mode=%s", in_flight.id, mode.value)
                return to_response(in_flight, mode=mode)

        budget = budget_for(mode)
        metadata = self._metadata(
            mode, budget, "Queued for research",
            semantic_verification_mode=semantic_mode,
            claim_recovery_mode=recovery_mode,
            claim_audit_mode=audit_mode,
            generation_prompt_version=prompt_version,
        )
        metadata["cache_mode"] = cache_mode.value
        metadata["retrieval_version"] = version
        run_id = uuid4()
        metadata["queue_job_id"] = str(run_id)
        metadata.update(extra_metadata or {})
        record = ResearchRun(id=run_id, question=question, requested_by=actor, status="queued", model_metadata=metadata)
        session.add(record)
        await session.flush()
        try:
            await self._reserve_cost(record, mode)
        except Exception:
            await session.rollback()
            raise
        session.add(AuditEvent(actor=actor, action="research.started", target_type="research_run", target_id=str(record.id), details={"mode": mode.value}))
        try:
            await session.commit()
        except Exception:
            await self._compensate_failed_commit(session, record.id)
            raise
        logger.info("research_queued run_id=%s mode=%s budget=%s", record.id, mode.value, budget.as_dict())
        try:
            await asyncio.wait_for(enqueue_run(self.redis, record), timeout=ENQUEUE_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("research_enqueue_deferred run_id=%s; the recovery sweep will re-enqueue it", record.id, exc_info=True)
        return to_response(record, mode=mode)

    async def _compensate_failed_commit(self, session: AsyncSession, run_id: UUID) -> None:
        """Release the Redis reservation only when the run is confirmed not persisted.

        A commit error can be ambiguous (the commit may have succeeded before the
        connection failed). If persistence cannot be confirmed either way, the
        reservation is kept: over-counting is preferred to under-counting spend.
        """
        try:
            await session.rollback()
        except Exception:
            logger.warning("research_commit_rollback_failed run_id=%s", run_id, exc_info=True)
        try:
            async with session_factory() as check:
                persisted = await check.scalar(
                    select(func.count()).select_from(ResearchRun).where(ResearchRun.id == run_id)
                )
        except Exception:
            logger.warning("research_commit_outcome_unknown run_id=%s; reservation retained", run_id, exc_info=True)
            return
        if persisted:
            logger.warning("research_commit_error_but_persisted run_id=%s; reservation kept, recovery will enqueue", run_id)
            return
        guard = getattr(self, "cost_guard", None)
        if guard is not None:
            try:
                await guard.release_research(str(run_id))
            except Exception:
                logger.warning("research_reservation_release_failed run_id=%s", run_id, exc_info=True)

    async def run_queued(self, run_id: UUID, question: str, mode: ResearchMode, cache_mode: ResearchCacheMode,
                         semantic_verification_mode: str | None = None,
                         claim_recovery_mode: str | None = None,
                         claim_audit_mode: str | None = None,
                         generation_prompt_version: str | None = None) -> None:
        settings = self.settings
        async with session_factory() as session:
            record = await claim_run(session, run_id, owner=worker_identity(), lease_seconds=settings.research_lease_seconds)
            if record is None:
                # Duplicate delivery, redelivery after completion, or cancellation
                # before start: the run is not queued, so nothing is executed.
                logger.info("research_claim_skipped run_id=%s", run_id)
                return
            attempt = record.attempt
            # Persisted run intent is authoritative across worker restarts.
            # Historical runs without recovery metadata stay strict even if a
            # stale job argument or newer worker environment says degrade.
            stored_semantic_mode = str(
                (record.model_metadata or {}).get("semantic_claim_verification_mode")
                or semantic_verification_mode
                or "off"
            )
            stored_recovery_mode = str(
                (record.model_metadata or {}).get("claim_recovery_mode")
                or "strict"
            )
            stored_audit_mode = str(
                (record.model_metadata or {}).get("claim_audit_mode")
                or "off"
            )
            stored_prompt_version = str(
                (record.model_metadata or {}).get("generation_prompt_version")
                or "v15"
            )
            effective_pipeline_version(
                stored_semantic_mode,
                stored_recovery_mode,
                generation_prompt_version=stored_prompt_version,
                quote_recovery_mode=str((record.model_metadata or {}).get("quote_recovery_mode") or "off"),
            )
            lease = LeaseKeeper(
                session_factory, run_id, attempt,
                lease_seconds=settings.research_lease_seconds,
                heartbeat_seconds=settings.research_heartbeat_seconds,
            )
            try:
                async with lease:
                    await self._execute(
                        session, record, question, mode, False, cache_mode=cache_mode,
                        semantic_verification_mode=stored_semantic_mode,
                        claim_recovery_mode=stored_recovery_mode,
                        claim_audit_mode=stored_audit_mode,
                        generation_prompt_version=stored_prompt_version,
                    )
            except ResearchFencedOut:
                logger.warning("research_execution_fenced_out run_id=%s attempt=%s", run_id, attempt)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if lease.lost and task is not None:
                    # Our own heartbeat cancelled the work. "fenced": a newer owner
                    # (cancellation or recovery) already holds the run. "unverifiable":
                    # ownership could not be confirmed in time, so stop and record it
                    # with the same bounded, fenced cleanup (recovery covers failure).
                    task.uncancel()
                    logger.warning("research_execution_stopped_after_lease_loss run_id=%s attempt=%s reason=%s",
                                   run_id, attempt, lease.loss_reason)
                    if lease.loss_reason == "unverifiable":
                        await self._mark_interrupted(run_id, attempt, error=LEASE_UNVERIFIABLE_ERROR,
                                                     reason="lease_unverifiable")
                    return
                await self._mark_interrupted(run_id, attempt)
                raise
            except Exception:
                # _execute persists the failure. Keep this task from producing an unhandled-task warning.
                logger.exception("queued_research_task_failed run_id=%s", run_id)

    async def _mark_interrupted(self, run_id: UUID, attempt: int, *, error: str = INTERRUPTED_ERROR,
                                reason: str = "worker_interrupted") -> None:
        """Bounded, shielded terminal write after worker cancellation (shutdown or timeout).

        If this cannot finish (for example the process is killed), the lease expires
        and the recovery sweep applies the same transition.
        """
        async def cleanup() -> None:
            async with session_factory() as session:
                termination = await terminate_run(
                    session, run_id, status="failed", error=error, actor="system",
                    action="research.interrupted", expected_statuses=("running",), expected_attempt=attempt,
                    reason=reason,
                )
                if termination is not None:
                    await settle_terminated_cost(session, getattr(self, "cost_guard", None), termination)

        try:
            await asyncio.wait_for(asyncio.shield(cleanup()), timeout=15)
        except (Exception, asyncio.CancelledError):
            logger.warning("research_interrupt_cleanup_incomplete run_id=%s; recovery will finalize it", run_id, exc_info=True)

    async def run(self, session: AsyncSession, question: str, mode: ResearchMode, generate_evaluation_dataset: bool, *,
                  actor: str = "api", cache_mode: ResearchCacheMode = ResearchCacheMode.use_cached_answer,
                  fresh: bool = False) -> ResearchResponse:
        """Synchronous API path retained for programmatic clients."""
        self._validate_request(mode, generate_evaluation_dataset)
        if fresh:  # Deprecated API behavior remains a full source refresh.
            cache_mode = ResearchCacheMode.refresh_sources
        semantic_mode = getattr(
            getattr(self, "settings", None), "semantic_claim_verification_mode", "off"
        )
        recovery_mode = getattr(
            getattr(self, "settings", None), "claim_recovery_mode", "strict"
        )
        audit_mode = getattr(
            getattr(self, "settings", None), "claim_audit_mode", "off"
        )
        prompt_version = getattr(
            getattr(self, "settings", None), "generation_prompt_version", "v15"
        )
        cached = None if cache_mode is not ResearchCacheMode.use_cached_answer else await self._find_cached(
            session, question, actor, mode, semantic_mode, recovery_mode,
            generation_prompt_version=prompt_version,
        )
        if cached is not None:
            return to_response(cached, await self.sources_for_run(session, cached.id), mode=mode, cached=True)

        budget = budget_for(mode)
        initial_metadata = self._metadata(
            mode, budget, "Starting research",
            semantic_verification_mode=semantic_mode,
            claim_recovery_mode=recovery_mode,
            claim_audit_mode=audit_mode,
            generation_prompt_version=prompt_version,
        )
        initial_metadata["cache_mode"] = cache_mode.value
        record = ResearchRun(question=question, requested_by=actor, status="running", model_metadata=initial_metadata)
        session.add(record)
        if hasattr(session, "flush"):
            await session.flush()
        try:
            await self._reserve_cost(record, mode)
        except Exception:
            if hasattr(session, "rollback"):
                await session.rollback()
            raise
        session.add(AuditEvent(actor=actor, action="research.started", target_type="research_run", target_id=str(record.id), details={"mode": mode.value, "entrypoint": "api"}))
        await session.commit()
        return await self._execute(
            session, record, question, mode, generate_evaluation_dataset,
            cache_mode=cache_mode, semantic_verification_mode=semantic_mode,
            claim_recovery_mode=recovery_mode,
            claim_audit_mode=audit_mode,
            generation_prompt_version=prompt_version,
        )

    async def _set_progress(self, session: AsyncSession, record: ResearchRun, stage: str, role: str | None = None) -> None:
        metadata = dict(record.model_metadata)
        timeline = list(metadata.get("timeline", []))
        _close_active_timeline({"timeline": timeline})
        timeline.append({
            "phase": _timeline_phase(stage, role),
            "stage": stage,
            "role": role,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "duration_seconds": None,
        })
        metadata["timeline"] = timeline
        metadata["progress"] = {"stage": stage, "role": role, "model": self._model_for_role(role)}
        record.model_metadata = metadata
        await session.commit()
        logger.info("research_progress run_id=%s stage=%s role=%s model=%s", record.id, stage, role, self._model_for_role(role))

    async def _execute(self, session: AsyncSession, record: ResearchRun, question: str, mode: ResearchMode, generate_evaluation_dataset: bool, *,
                       cache_mode: ResearchCacheMode = ResearchCacheMode.use_cached_answer,
                       semantic_verification_mode: str | None = None,
                       claim_recovery_mode: str | None = None,
                       claim_audit_mode: str | None = None,
                       generation_prompt_version: str | None = None) -> ResearchResponse:
        budget = budget_for(mode)
        semantic_mode = str(
            semantic_verification_mode
            or (record.model_metadata or {}).get("semantic_claim_verification_mode")
            or "off"
        )
        recovery_mode = str(
            (record.model_metadata or {}).get("claim_recovery_mode")
            or "strict"
        )
        audit_mode = str(
            (record.model_metadata or {}).get("claim_audit_mode")
            or "off"
        )
        prompt_version = str(
            (record.model_metadata or {}).get("generation_prompt_version")
            or "v15"
        )
        # Historic queued/direct records without the mode remain off, independently of worker settings.
        quote_mode = str((record.model_metadata or {}).get("quote_recovery_mode") or "off")
        version = effective_pipeline_version(
            semantic_mode, recovery_mode, generation_prompt_version=prompt_version,
            quote_recovery_mode=quote_mode,
        )
        trace_token = begin_trace(self.redis)
        trace_closed = False
        orchestrator = None
        vector_service = None
        research_agent = None
        execution_budget = None
        # Fencing token (R2): the attempt this execution owns. A cancellation or
        # recovery increments it, after which every write from here fails.
        fence = getattr(record, "attempt", None)
        logger.info("research_started run_id=%s mode=%s question_length=%s", record.id, mode.value, len(question))
        try:
            async with self._lock:
                self.settings.validate_research_configuration()
                legacy_pipeline.config.validate()
                await self._set_progress(session, record, "Planning the research approach", "orchestrator")

                async def progress(stage: str, role: str | None = None) -> None:
                    if execution_budget is not None:
                        execution_budget.set_stage(stage, role)
                    await self._set_progress(session, record, stage, role)

                vector_service = PostgresVectorService(session, self.settings, record.id)
                research_agent = legacy_pipeline.ResearchAgent(
                    vector_service,
                    max_synthesis_subquestions=budget.max_synthesis_subquestions,
                    retrieval_documents=budget.retrieval_documents,
                    max_context_documents=budget.max_context_documents,
                    progress_callback=progress,
                    semantic_verification_mode=semantic_mode,
                    claim_recovery_mode=recovery_mode,
                    claim_audit_mode=audit_mode,
                    generation_prompt_version=prompt_version,
                    semantic_verification_timeout_seconds=120,
                    **({"quote_recovery_mode": quote_mode} if quote_mode != "off" else {}),
                )
                orchestrator = legacy_pipeline.Orchestrator(
                    vector_service,
                    research_agent,
                    max_plan_steps=budget.max_plan_steps,
                    papers_per_source=budget.papers_per_source,
                    progress_callback=progress,
                )
                timeout_seconds = (
                    self.settings.deep_research_timeout_seconds
                    if mode is ResearchMode.deep
                    else self.settings.request_timeout_seconds
                )
                try:
                    execution_budget = ExecutionBudget(timeout_seconds)
                    with use_budget(execution_budget):
                        pipeline_result = await asyncio.wait_for(
                            orchestrator.execute_full_study(question, clear_db=False, generate_evaluation_dataset=generate_evaluation_dataset, cache_mode=cache_mode.value),
                            timeout=timeout_seconds,
                        )
                except ResearchDeadlineExceeded:
                    raise
                except TimeoutError as error:
                    # Inner provider and stage timeouts are not proof that the
                    # outer research deadline expired.
                    if execution_budget is not None and execution_budget.remaining() > 0:
                        raise
                    stage = str((record.model_metadata or {}).get("progress", {}).get("stage") or "unknown stage")
                    raise ResearchDeadlineExceeded(stage, 0) from error
                report = pipeline_result["report"]

            record.status = str(pipeline_result.get("status") or "completed")
            record.answer = report["answer"]
            record.completed_at = datetime.now(timezone.utc)
            record.lease_owner = None
            record.lease_expires_at = None
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
                        "study_design": str(metadata.get("study_design") or "") or None,
                        "citation_id": str(metadata.get("citation_id") or "") or None,
                    },
                ))
            metadata = self._metadata(
                mode, budget, "Research complete",
                semantic_verification_mode=semantic_mode,
                claim_recovery_mode=recovery_mode,
                claim_audit_mode=audit_mode,
                generation_prompt_version=prompt_version,
                quote_recovery_mode=quote_mode,
            )
            metadata["retrieval_version"] = version
            if execution_budget is not None:
                metadata["execution_budget"] = execution_budget.snapshot()
            metadata["llm_gateway"] = summarize_trace(end_trace(trace_token))
            metadata["embedding_usage"] = dict(vector_service.embedding_usage)
            stored_metadata = dict(record.model_metadata)
            _close_active_timeline(stored_metadata, record.completed_at)
            metadata["timeline"] = stored_metadata.get("timeline", [])
            metadata["duration_seconds"] = round((record.completed_at - record.created_at).total_seconds(), 2)
            metadata["mined_count"] = int(pipeline_result.get("metadata", {}).get("mined_count", 0))
            metadata["reused_count"] = int(pipeline_result.get("metadata", {}).get("reused_count", 0))
            metadata["evidence_document_count"] = int(pipeline_result.get("metadata", {}).get("evidence_document_count", 0))
            trace_closed = True
            cache_diagnostics = dict(pipeline_result.get("metadata", {}).get("cache", {}))
            metadata["cache_hit"] = cache_diagnostics.get("decision") == "reuse_postgres_evidence"
            metadata["cache_mode"] = cache_mode.value
            metadata["source_sufficiency"] = dict(pipeline_result.get("metadata", {}).get("source_sufficiency", {}))
            metadata["cache"] = cache_diagnostics
            source_diagnostics = dict(pipeline_result.get("metadata", {}).get("source_diagnostics", {}))
            for document in evidence_documents:
                source = str(document.metadata.get("source", "unknown"))
                source_diagnostics.setdefault(source, {"requested": 0, "collected": 0, "errors": []})
                source_diagnostics[source]["used"] = source_diagnostics[source].get("used", 0) + 1
            metadata["source_diagnostics"] = source_diagnostics
            metadata["pubmed_inspector"] = list(pipeline_result.get("metadata", {}).get("pubmed_inspector", []))
            metadata["arxiv_inspector"] = list(pipeline_result.get("metadata", {}).get("arxiv_inspector", []))
            metadata["retrieval_inspector"] = list(vector_service.retrieval_traces)
            metadata["claim_validation"] = dict(report.get("claim_validation", {}))
            metadata["progress"]["model"] = None
            await self._reconcile_cost(record, metadata)
            record.model_metadata = metadata
            session.add(AuditEvent(actor=record.requested_by or "api", action="research.completed", target_type="research_run", target_id=str(record.id), details={
                "mode": mode.value,
                "providers": sorted({str(item.get("provider")) for item in metadata["llm_gateway"].get("attempts", []) if item.get("provider")}),
                "status": record.status,
            }))
            sources = _responses_for_evidence_documents(evidence_documents)
            await session.commit()
            await session.refresh(record)
            logger.info("research_completed run_id=%s mode=%s source_count=%s", record.id, mode.value, len(sources))
            return ResearchResponse(id=record.id, status=record.status, answer=record.answer, sources=sources, plan=list(pipeline_result.get("metadata", {}).get("plan_executed", [])), created_at=record.created_at, completed_at=record.completed_at, mode=mode, retrieval_version=version, source_diagnostics=source_diagnostics, cache_diagnostics=cache_diagnostics, llm_diagnostics=metadata["llm_gateway"], source_sufficiency=metadata["source_sufficiency"])
        except asyncio.CancelledError:
            if not trace_closed:
                end_trace(trace_token)
            raise
        except Exception as error:
            failed_run_id = record.id
            try:
                # A failed JSON/DB flush leaves the AsyncSession unusable until
                # rollback. Reload the run so the failure itself can be persisted.
                await session.rollback()
                recovered_record = await session.get(ResearchRun, failed_run_id, populate_existing=True)
                if recovered_record is not None:
                    record = recovered_record
            except Exception:
                logger.exception("research_failure_recovery_failed run_id=%s", failed_run_id)
            if fence is not None and (getattr(record, "attempt", fence) != fence or record.status != "running"):
                # Cancelled or recovered while executing: the newer owner already
                # wrote the terminal state and settled cost. Never overwrite it.
                if not trace_closed:
                    end_trace(trace_token)
                logger.warning("research_fenced_out run_id=%s attempt=%s current_status=%s", failed_run_id, fence, record.status)
                raise ResearchFencedOut(f"Research run {failed_run_id} is no longer owned by this execution") from error
            record.status = "failed"
            record.error = str(error) or type(error).__name__
            record.completed_at = datetime.now(timezone.utc)
            record.lease_owner = None
            record.lease_expires_at = None
            metadata = dict(record.model_metadata)
            if not trace_closed:
                metadata["llm_gateway"] = summarize_trace(end_trace(trace_token))
                trace_closed = True
            _close_active_timeline(metadata, record.completed_at)
            metadata["failure_stage"] = (metadata.get("progress") or {}).get("stage")
            if execution_budget is not None:
                metadata["execution_budget"] = execution_budget.snapshot()
            if isinstance(error, ResearchDeadlineExceeded):
                metadata["failure_kind"] = error.failure_kind
                metadata["deadline_stage"] = error.stage
            elif isinstance(error, TimeoutError):
                metadata["failure_kind"] = "provider_or_stage_timeout"
            else:
                metadata["failure_kind"] = getattr(error, "failure_kind", type(error).__name__)
            if research_agent is not None and getattr(research_agent, "last_validated_evidence", None):
                metadata["preserved_evidence"] = research_agent.last_validated_evidence
            if isinstance(error, legacy_pipeline.NoValidClaimsRemain):
                metadata["failure_kind"] = error.failure_kind
                metadata["claim_validation"] = {
                    "status": "failed",
                    "failure_kind": error.failure_kind,
                    "recovery": dict(error.recovery_audit),
                    "removals": [item.as_dict() for item in error.removals],
                }
            elif research_agent is not None and getattr(research_agent, "last_recovery_audit", None):
                metadata["claim_validation"] = {
                    "status": "failed_after_recovery",
                    "recovery": dict(research_agent.last_recovery_audit),
                    "removals": [
                        item.as_dict()
                        for item in getattr(research_agent, "last_recovery_removals", ())
                    ],
                    "downstream_failure_kind": type(error).__name__,
                }
            if (quote_mode != "off" and research_agent is not None
                    and getattr(research_agent, "last_quote_recovery_audit", None)):
                validation = dict(metadata.get("claim_validation") or {"status": "failed"})
                quote_failure = dict(research_agent.last_quote_recovery_audit)
                if quote_failure.get("semantic_verification_status") == "pending":
                    quote_failure["semantic_verification_status"] = "failed"
                validation["quote_recovery"] = quote_failure
                metadata["claim_validation"] = validation
            semantic_failure = getattr(research_agent, "last_semantic_verification_audit", None)
            if isinstance(semantic_failure, dict):
                if semantic_failure.get("status") == "error" or getattr(error, "failure_kind", None) == "no_semantically_supported_claims":
                    metadata["failure_kind"] = getattr(error, "failure_kind", "semantic_verification_error")
                metadata["claim_validation"] = {**metadata.get("claim_validation", {}), "status":"failed",
                    "semantic_verification":semantic_failure}
            metadata["progress"] = {"stage": "Research failed", "role": None, "model": None}
            if orchestrator is not None:
                metadata["source_diagnostics"] = getattr(orchestrator, "last_source_diagnostics", {})
                metadata["pubmed_inspector"] = getattr(orchestrator, "last_pubmed_inspector", [])
                metadata["arxiv_inspector"] = getattr(orchestrator, "last_arxiv_inspector", [])
                metadata["cache"] = getattr(orchestrator, "last_cache_diagnostics", metadata.get("cache", {}))
                metadata["mined_count"] = getattr(orchestrator, "last_mined_count", 0)
                metadata["reused_count"] = getattr(orchestrator, "last_reused_count", 0)
                metadata["evidence_document_count"] = getattr(orchestrator, "last_evidence_count", 0)
            if vector_service is not None:
                metadata["retrieval_inspector"] = list(vector_service.retrieval_traces)
                metadata["embedding_usage"] = dict(vector_service.embedding_usage)
            await self._reconcile_cost(record, metadata)
            record.model_metadata = metadata
            session.add(AuditEvent(actor=record.requested_by or "api", action="research.failed", target_type="research_run", target_id=str(record.id), details={"mode": mode.value, "error_type": type(error).__name__}))
            await session.commit()
            logger.exception("research_failed run_id=%s", record.id)
            raise


def to_response(record: ResearchRun, sources: list[SourceDocument | SourceResponse] | None = None, mode: ResearchMode | None = None, cached: bool = False) -> ResearchResponse:
    response_sources = [
        source if isinstance(source, SourceResponse) else SourceResponse(title=source.title, source=source.source, external_id=source.external_id, source_url=source.source_url, study_design=source.metadata_json.get("study_design"))
        for source in sources or []
    ]
    return ResearchResponse(id=record.id, status=record.status, answer=record.answer, sources=response_sources, created_at=record.created_at, completed_at=record.completed_at, mode=mode or ResearchMode(record.model_metadata.get("mode", "fast")), cached=cached, retrieval_version=record.model_metadata.get("retrieval_version"), source_diagnostics=record.model_metadata.get("source_diagnostics", {}), cache_diagnostics=record.model_metadata.get("cache", {}), llm_diagnostics=record.model_metadata.get("llm_gateway", {}), source_sufficiency=record.model_metadata.get("source_sufficiency", {}))
