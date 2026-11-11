"""Research-job lifecycle: claims, leases, fencing, cancellation, and recovery (roadmap R2).

Rules (see docs/roadmap.md, R2):

* A run is executed only after a worker *claims* it: ``queued -> running`` under a
  row lock, incrementing ``attempt`` and setting a lease. Duplicate or late
  deliveries find the run no longer ``queued`` and do nothing.
* ``attempt`` is the ORM version column of ``ResearchRun``. Cancellation and
  recovery increment it, so every later write from the previous owner fails with
  ``StaleDataError`` instead of overwriting the newer state.
* The executing worker renews its lease every ``research_heartbeat_seconds``. A run
  is treated as abandoned only when its lease has expired, never because of age
  alone. Lease acquisition, renewal, and expiry use the PostgreSQL clock, so
  application clock skew cannot expire a healthy lease. A worker that cannot
  confirm a renewal stops its own execution before its last confirmed lease could
  have expired (monotonic local deadline). Runs without a lease (pre-R2 rows and the synchronous API path) are
  recovered only after the hard execution timeout plus a grace period.
* Queued runs whose ARQ job is missing (enqueue failure, crash after commit, Redis
  data loss) are re-enqueued with the same job id. Nothing is executed twice:
  re-enqueueing a run that is already claimed is a no-op.
* Cost: a reservation is released only when the run provably never started paid
  work (still ``queued`` with ``attempt == 0``). Otherwise usage is unknown and the
  full reservation is retained. Finalization is idempotent.

Recovery does not re-execute interrupted work automatically; it terminates it with
a clear error and the user can retry. Exactly-once execution of external provider
calls is not guaranteed and not claimed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
import socket
import time
from typing import Any, Callable
from uuid import UUID

from redis.exceptions import WatchError
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from .models import AuditEvent, ResearchRun

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running")
LEASE_EXPIRED_ERROR = (
    "The research worker stopped renewing its lease (worker crash, restart, or lost "
    "database connection). Provider usage is unknown, so the cost reservation is retained. "
    "Retry the research to run it again."
)
UNLEASED_TIMEOUT_ERROR = (
    "The research run exceeded the maximum execution time without completing and has no "
    "active worker lease. Provider usage is unknown, so the cost reservation is retained."
)
INTERRUPTED_ERROR = (
    "The research worker was interrupted (shutdown or job timeout) before completion. "
    "Provider usage is unknown, so the cost reservation is retained. Retry the research to run it again."
)
LEASE_UNVERIFIABLE_ERROR = (
    "The research worker could not renew its lease within the lease period (database "
    "unreachable) and stopped execution. Provider usage is unknown, so the cost reservation "
    "is retained. Retry the research to run it again."
)
STRANDED_ERROR = (
    "The research job could not be delivered to a worker. No research was started and the "
    "cost reservation was released. Submit the research again."
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def worker_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"[:128]


def research_identity_lock_id(actor: str, mode: str, version: str, question: str) -> int:
    """Advisory-lock key for one research identity (sharing scope, mode, pipeline version, question)."""
    material = "\x1f".join(("research-identity-v1", actor, mode, version, question)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=True)


async def acquire_identity_lock(session: AsyncSession, actor: str, mode: str, version: str, question: str) -> None:
    """Serialize submissions of the same research identity until the transaction ends."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": research_identity_lock_id(actor, mode, version, question)},
    )


def _close_timeline(metadata: dict, at: datetime) -> None:
    for item in reversed(metadata.get("timeline", []) or []):
        if item.get("completed_at") is None:
            item["completed_at"] = at.isoformat()
            try:
                started = datetime.fromisoformat(item["started_at"])
                item["duration_seconds"] = round(max((at - started).total_seconds(), 0), 2)
            except (KeyError, TypeError, ValueError):
                pass
            break


async def _locked_run(session: AsyncSession, run_id: UUID) -> ResearchRun | None:
    return (await session.execute(
        select(ResearchRun)
        .where(ResearchRun.id == run_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )).scalar_one_or_none()


async def _database_now(session: AsyncSession) -> datetime:
    """The authoritative clock for lease decisions."""
    return await session.scalar(select(func.now()))


async def claim_run(session: AsyncSession, run_id: UUID, *, owner: str, lease_seconds: int) -> ResearchRun | None:
    """Atomically move ``queued -> running``. Returns the claimed record or ``None``."""
    record = await _locked_run(session, run_id)
    if record is None or record.status != "queued":
        await session.rollback()
        return None
    database_now = await _database_now(session)
    record.attempt = record.attempt + 1
    record.status = "running"
    record.lease_owner = owner
    record.lease_expires_at = database_now + timedelta(seconds=lease_seconds)
    metadata = dict(record.model_metadata or {})
    metadata["execution"] = {"attempt": record.attempt, "claimed_at": database_now.isoformat(), "worker": owner}
    record.model_metadata = metadata
    try:
        await session.commit()
    except StaleDataError:
        await session.rollback()
        return None
    return record


async def renew_lease(session_factory: Callable, run_id: UUID, attempt: int, lease_seconds: int) -> bool:
    """Extend the lease only if this attempt still owns the running run."""
    async with session_factory() as session:
        result = await session.execute(
            update(ResearchRun.__table__)
            .where(
                ResearchRun.__table__.c.id == run_id,
                ResearchRun.__table__.c.attempt == attempt,
                ResearchRun.__table__.c.status == "running",
            )
            .values(lease_expires_at=func.now() + timedelta(seconds=lease_seconds))
        )
        await session.commit()
        return result.rowcount == 1


class LeaseKeeper:
    """Background lease renewal for one claimed run.

    The owner task is cancelled when the lease is definitively lost (renewal
    matched no row: ``loss_reason == "fenced"``) or when no renewal has been
    confirmed for long enough that the last confirmed lease may expire
    (``loss_reason == "unverifiable"``). The local deadline uses a monotonic
    clock started *before* each successful renewal request, minus one heartbeat
    as a safety margin, so the worker stops before recovery may take over.
    """

    def __init__(self, session_factory: Callable, run_id: UUID, attempt: int, *, lease_seconds: float,
                 heartbeat_seconds: float, owner_task: asyncio.Task | None = None) -> None:
        self.session_factory = session_factory
        self.run_id = run_id
        self.attempt = attempt
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.margin_seconds = min(heartbeat_seconds, lease_seconds / 3)
        self.owner_task = owner_task or asyncio.current_task()
        self.lost = False
        self.loss_reason: str | None = None
        # The claim committed just before construction; its lease started no later than now.
        self.valid_until = time.monotonic() + lease_seconds - self.margin_seconds
        self._task: asyncio.Task | None = None

    def _lose(self, reason: str) -> None:
        self.lost = True
        self.loss_reason = reason
        logger.warning("research_lease_lost run_id=%s attempt=%s reason=%s", self.run_id, self.attempt, reason)
        if self.owner_task is not None and not self.owner_task.done():
            self.owner_task.cancel()

    async def _loop(self) -> None:
        while True:
            remaining = self.valid_until - time.monotonic()
            await asyncio.sleep(max(min(self.heartbeat_seconds, remaining), 0))
            attempt_started = time.monotonic()
            timeout = max(min(self.heartbeat_seconds, self.valid_until - attempt_started), 0.001)
            try:
                renewed = await asyncio.wait_for(
                    renew_lease(self.session_factory, self.run_id, self.attempt, self.lease_seconds),
                    timeout=timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("research_lease_renewal_error run_id=%s attempt=%s", self.run_id, self.attempt, exc_info=True)
                if time.monotonic() >= self.valid_until:
                    self._lose("unverifiable")
                    return
                continue
            if not renewed:
                self._lose("fenced")
                return
            self.valid_until = attempt_started + self.lease_seconds - self.margin_seconds

    async def __aenter__(self) -> "LeaseKeeper":
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


@dataclass(frozen=True)
class Termination:
    run_id: UUID
    previous_status: str
    previous_attempt: int
    status: str

    @property
    def never_started(self) -> bool:
        """True when no worker ever claimed the run, so no provider usage occurred."""
        return self.previous_status == "queued" and self.previous_attempt == 0


async def terminate_run(
    session: AsyncSession,
    run_id: UUID,
    *,
    status: str,
    error: str | None,
    actor: str,
    action: str,
    expected_statuses: tuple[str, ...] = ACTIVE_STATUSES,
    expected_attempt: int | None = None,
    require_expired_lease: bool = False,
    unleased_older_than_seconds: float | None = None,
    reason: str,
) -> Termination | None:
    """Fenced transition of an active run to a terminal state. ``None`` if preconditions no longer hold."""
    record = await _locked_run(session, run_id)
    if record is None or record.status not in expected_statuses:
        await session.rollback()
        return None
    if expected_attempt is not None and record.attempt != expected_attempt:
        await session.rollback()
        return None
    if require_expired_lease or unleased_older_than_seconds is not None:
        database_now = await _database_now(session)
        if require_expired_lease and (record.lease_expires_at is None or record.lease_expires_at >= database_now):
            await session.rollback()
            return None
        if unleased_older_than_seconds is not None and (
            record.lease_expires_at is not None
            or record.created_at >= database_now - timedelta(seconds=unleased_older_than_seconds)
        ):
            await session.rollback()
            return None
    previous_status, previous_attempt = record.status, record.attempt
    now = utcnow()
    record.attempt = previous_attempt + 1
    record.status = status
    record.completed_at = now
    record.lease_owner = None
    record.lease_expires_at = None
    if error:
        record.error = error
    metadata = dict(record.model_metadata or {})
    _close_timeline(metadata, now)
    if status == "failed":
        metadata.setdefault("failure_stage", (metadata.get("progress") or {}).get("stage"))
    metadata["progress"] = {
        "stage": "Research cancelled" if status == "cancelled" else "Research failed",
        "role": None,
        "model": None,
    }
    metadata["termination"] = {
        "reason": reason,
        "previous_status": previous_status,
        "previous_attempt": previous_attempt,
        "at": now.isoformat(),
    }
    record.model_metadata = metadata
    session.add(AuditEvent(
        actor=actor, action=action, target_type="research_run", target_id=str(run_id),
        details={"previous_status": previous_status, "reason": reason, "mode": metadata.get("mode", "fast")},
    ))
    try:
        await session.commit()
    except StaleDataError:
        await session.rollback()
        return None
    return Termination(run_id, previous_status, previous_attempt, status)


async def settle_terminated_cost(session: AsyncSession, cost_guard, termination: Termination) -> dict | None:
    """Release a never-started reservation; retain it when usage is unknown. Records the outcome."""
    if cost_guard is None:
        return None
    try:
        if termination.never_started:
            reservation = await cost_guard.release_research(str(termination.run_id))
        else:
            reservation = await cost_guard.retain_research(str(termination.run_id))
    except Exception:
        logger.warning("research_cost_settlement_failed run_id=%s", termination.run_id, exc_info=True)
        return None
    outcome = reservation.as_dict() if reservation is not None else {"status": "missing"}
    record = await session.get(ResearchRun, termination.run_id, populate_existing=True)
    if record is not None:
        metadata = dict(record.model_metadata or {})
        accounting = dict(metadata.get("cost_accounting") or {})
        accounting["reservation"] = outcome
        if termination.never_started:
            # No worker ever claimed the run, so no provider call was made.
            accounting.update({"estimated_cost_usd": 0.0, "complete": True})
        else:
            # Usage after the claim is unknown; the retained reservation is the upper bound.
            accounting["complete"] = False
        metadata["cost_accounting"] = accounting
        record.model_metadata = metadata
        try:
            await session.commit()
        except StaleDataError:
            await session.rollback()
    return outcome


async def cancel_run(session: AsyncSession, run_id: UUID, *, actor: str, cost_guard) -> Termination | None:
    """User cancellation: persist first (fencing any worker), then settle cost."""
    termination = await terminate_run(
        session, run_id, status="cancelled", error=None, actor=actor,
        action="research.cancel_requested", reason="user_cancelled",
    )
    if termination is not None:
        await settle_terminated_cost(session, cost_guard, termination)
    return termination


def _queue_name(redis) -> str:
    from arq.constants import default_queue_name

    return getattr(redis, "default_queue_name", None) or default_queue_name


async def _arq_job_active(redis, job_id: str) -> bool:
    """True when a worker can still pick the job (queued) or is running it (in progress)."""
    from arq.constants import in_progress_key_prefix

    if await redis.zscore(_queue_name(redis), job_id) is not None:
        return True
    return bool(await redis.exists(in_progress_key_prefix + job_id))


async def _clear_orphaned_job(redis, job_id: str) -> bool:
    """Delete a job hash that no worker can ever pick up (no queue entry, not in progress).

    ARQ's ``enqueue_job`` refuses a job id whose hash exists, so an orphaned hash
    would block re-enqueueing until it expires. The check and delete run under
    WATCH, so a concurrent enqueue or pickup aborts the delete.
    """
    from arq.constants import in_progress_key_prefix, job_key_prefix

    job_key, in_progress_key, queue = job_key_prefix + job_id, in_progress_key_prefix + job_id, _queue_name(redis)
    async with redis.pipeline(transaction=True) as pipe:
        try:
            await pipe.watch(job_key, in_progress_key, queue)
            if (await pipe.zscore(queue, job_id) is not None or await pipe.exists(in_progress_key)
                    or not await pipe.exists(job_key)):
                await pipe.reset()
                return False
            pipe.multi()
            pipe.delete(job_key)
            await pipe.execute()
            return True
        except WatchError:
            return False


async def _arq_result_only(redis, job_id: str) -> bool:
    """True when ARQ holds a finished result for this id and no live job."""
    from arq.constants import job_key_prefix, result_key_prefix

    if not await redis.exists(result_key_prefix + job_id) or await _arq_job_active(redis, job_id):
        return False
    return not await redis.exists(job_key_prefix + job_id)


def _enqueue_arguments(record: ResearchRun) -> tuple:
    metadata = record.model_metadata or {}
    return (
        "run_research_job", str(record.id), record.question, str(metadata.get("mode", "fast")),
        str(metadata.get("cache_mode", "use_cached_answer")),
        str(metadata.get("semantic_claim_verification_mode") or "off"),
    )


async def enqueue_run(redis, record: ResearchRun):
    """Enqueue with the run id as ARQ job id; returns ``None`` if ARQ already holds that job or its result."""
    return await redis.enqueue_job(*_enqueue_arguments(record), _job_id=str(record.id))


@dataclass
class RecoveryReport:
    expired: list[str]
    unleased: list[str]
    requeued: list[str]
    stranded: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {"expired": self.expired, "unleased": self.unleased, "requeued": self.requeued, "stranded": self.stranded}


async def recover_research_runs(session_factory: Callable, redis, cost_guard, settings) -> RecoveryReport:
    """One idempotent recovery sweep. Safe to run concurrently from several workers.

    All time comparisons use the PostgreSQL clock.
    """
    report = RecoveryReport([], [], [], [])
    hard_limit = max(settings.request_timeout_seconds, settings.deep_research_timeout_seconds) + 900
    async with session_factory() as session:
        database_now = await _database_now(session)
        rows = (await session.execute(
            select(ResearchRun.id, ResearchRun.status, ResearchRun.attempt, ResearchRun.lease_expires_at, ResearchRun.created_at)
            .where(ResearchRun.status.in_(ACTIVE_STATUSES))
        )).all()

    for run_id, status, attempt, lease_expires_at, created_at in rows:
        try:
            if status == "running" and lease_expires_at is not None and lease_expires_at < database_now:
                async with session_factory() as session:
                    termination = await terminate_run(
                        session, run_id, status="failed", error=LEASE_EXPIRED_ERROR, actor="system",
                        action="research.recovered", expected_statuses=("running",), expected_attempt=attempt,
                        require_expired_lease=True, reason="lease_expired",
                    )
                    if termination is not None:
                        await settle_terminated_cost(session, cost_guard, termination)
                        report.expired.append(str(run_id))
            elif (status == "running" and lease_expires_at is None
                  and created_at < database_now - timedelta(seconds=hard_limit)):
                async with session_factory() as session:
                    termination = await terminate_run(
                        session, run_id, status="failed", error=UNLEASED_TIMEOUT_ERROR, actor="system",
                        action="research.recovered", expected_statuses=("running",), expected_attempt=attempt,
                        unleased_older_than_seconds=hard_limit, reason="unleased_timeout",
                    )
                    if termination is not None:
                        await settle_terminated_cost(session, cost_guard, termination)
                        report.unleased.append(str(run_id))
            elif (status == "queued" and attempt == 0 and redis is not None
                  and created_at < database_now - timedelta(seconds=settings.research_requeue_grace_seconds)):
                if await _arq_job_active(redis, str(run_id)):
                    continue
                async with session_factory() as session:
                    record = await session.get(ResearchRun, run_id)
                    if record is None or record.status != "queued" or record.attempt != 0:
                        continue
                    if await _clear_orphaned_job(redis, str(run_id)):
                        logger.warning("research_orphaned_job_cleared run_id=%s", run_id)
                    job = await enqueue_run(redis, record)
                    if job is not None:
                        session.add(AuditEvent(actor="system", action="research.requeued", target_type="research_run",
                                               target_id=str(run_id), details={"reason": "queue_job_missing"}))
                        await session.commit()
                        report.requeued.append(str(run_id))
                        continue
                    if not await _arq_result_only(redis, str(run_id)):
                        # A concurrent sweep enqueued it first; nothing to do.
                        continue
                    # ARQ already has a result for this job id, yet the run was never
                    # claimed: it can never start. Fail it and release the reservation.
                    termination = await terminate_run(
                        session, run_id, status="failed", error=STRANDED_ERROR, actor="system",
                        action="research.recovered", expected_statuses=("queued",), expected_attempt=0,
                        reason="stranded_job",
                    )
                    if termination is not None:
                        await settle_terminated_cost(session, cost_guard, termination)
                        report.stranded.append(str(run_id))
        except Exception:
            logger.warning("research_recovery_item_failed run_id=%s", run_id, exc_info=True)
    if any(report.as_dict().values()):
        logger.info("research_recovery_sweep %s", report.as_dict())
    return report


async def recovery_loop(session_factory: Callable, redis, cost_guard, settings) -> None:
    """Run recovery sweeps for the life of a worker process."""
    while True:
        try:
            await recover_research_runs(session_factory, redis, cost_guard, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("research_recovery_sweep_failed", exc_info=True)
        await asyncio.sleep(settings.research_recovery_interval_seconds)
