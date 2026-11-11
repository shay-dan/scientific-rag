"""R2 failure-injection tests against real PostgreSQL and Redis.

No LLM, embedding, or literature provider is called: research execution is
replaced by deterministic fakes, while claims, leases, fencing, recovery,
deduplication, ARQ enqueueing, and cost reservations use the real services.
Requires RUN_INTEGRATION_TESTS=1, DATABASE_URL_TEST (migrated to head), and
REDIS_URL_TEST.
"""
import asyncio
from datetime import timedelta
import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.constants import default_queue_name, in_progress_key_prefix, job_key_prefix, result_key_prefix
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm.exc import StaleDataError

from app.config import Settings
from app.cost_control import CostGuard
from app.job_lifecycle import (
    LeaseKeeper,
    cancel_run,
    claim_run,
    recover_research_runs,
    renew_lease,
    utcnow,
)
from app.models import AuditEvent, ResearchRun
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import ResearchService

RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"


def _async_url(url: str) -> str:
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


@unittest.skipUnless(RUN and os.getenv("DATABASE_URL_TEST"), "Set RUN_INTEGRATION_TESTS=1, DATABASE_URL_TEST and REDIS_URL_TEST")
class JobLifecycleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(_async_url(os.environ["DATABASE_URL_TEST"]), pool_size=10)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.redis = await create_pool(RedisSettings.from_dsn(os.getenv("REDIS_URL_TEST", "redis://localhost:6379/0")))
        self.actor = f"r2-test-{uuid4().hex[:12]}"
        self.namespace = f"test:r2:cost:{uuid4().hex}"
        self.settings = Settings(
            cost_tracking_enabled=True, cost_enforce_hard_limit=True,
            cost_daily_budget_usd=100.0, cost_monthly_budget_usd=1000.0,
            cost_fast_reservation_usd=0.05, cost_standard_reservation_usd=0.20,
            research_lease_seconds=3, research_heartbeat_seconds=0.2, research_requeue_grace_seconds=0,
        )
        self.guard = CostGuard(self.redis, self.settings, namespace=self.namespace)
        self.service = ResearchService(self.settings, self.redis)
        self.service.cost_guard = self.guard
        self.run_ids: list[UUID] = []
        self.factory_patch = patch("app.services.session_factory", self.sessions)
        self.factory_patch.start()

    async def asyncTearDown(self):
        self.factory_patch.stop()
        async with self.sessions() as session:
            ids = list((await session.execute(select(ResearchRun.id).where(ResearchRun.requested_by == self.actor))).scalars())
            ids += self.run_ids
            if ids:
                await session.execute(delete(AuditEvent).where(AuditEvent.target_id.in_([str(i) for i in ids])))
                await session.execute(delete(ResearchRun).where(ResearchRun.id.in_(ids)))
            await session.commit()
        for run_id in ids:
            await self.redis.delete(job_key_prefix + str(run_id), in_progress_key_prefix + str(run_id), result_key_prefix + str(run_id))
            await self.redis.zrem(default_queue_name, str(run_id))
        keys = [key async for key in self.redis.scan_iter(match=f"{self.namespace}:*")]
        if keys:
            await self.redis.delete(*keys)
        await self.redis.aclose()
        await self.engine.dispose()

    # ----- helpers -------------------------------------------------------
    async def _queued_run(self, *, question: str | None = None, mode: str = "fast", reserve: bool = True,
                          created_offset_seconds: int = 0) -> UUID:
        run_id = uuid4()
        metadata = {"mode": mode, "cache_mode": "use_cached_answer", "retrieval_version": "test",
                    "semantic_claim_verification_mode": "off", "queue_job_id": str(run_id)}
        async with self.sessions() as session:
            session.add(ResearchRun(id=run_id, question=question or f"Question {run_id}", requested_by=self.actor,
                                    status="queued", model_metadata=metadata))
            await session.commit()
            if created_offset_seconds:
                await session.execute(update(ResearchRun).where(ResearchRun.id == run_id)
                                      .values(created_at=func.now() - timedelta(seconds=created_offset_seconds)))
                await session.commit()
        if reserve:
            await self.guard.reserve_research(str(run_id), ResearchMode(mode))
        self.run_ids.append(run_id)
        return run_id

    async def _claim(self, run_id: UUID, lease_seconds: float = 3):
        async with self.sessions() as session:
            return await claim_run(session, run_id, owner="test-worker", lease_seconds=lease_seconds)

    async def _get(self, run_id: UUID) -> ResearchRun:
        async with self.sessions() as session:
            return await session.get(ResearchRun, run_id)

    async def _expire_lease(self, run_id: UUID) -> None:
        async with self.sessions() as session:
            await session.execute(update(ResearchRun.__table__).where(ResearchRun.__table__.c.id == run_id)
                                  .values(lease_expires_at=func.now() - timedelta(seconds=30)))
            await session.commit()

    async def _daily_spend(self) -> int:
        total = 0
        async for key in self.redis.scan_iter(match=f"{self.namespace}:daily:*"):
            total += int(await self.redis.get(key) or 0)
        return total

    async def _reservation_status(self, run_id: UUID) -> str | None:
        value = await self.redis.hget(f"{self.namespace}:reservation:research:{run_id}", "status")
        return value.decode() if value else None

    async def _audit_count(self, run_id: UUID, action: str) -> int:
        async with self.sessions() as session:
            return await session.scalar(select(func.count()).select_from(AuditEvent)
                                        .where(AuditEvent.target_id == str(run_id), AuditEvent.action == action))

    # ----- 1 & 2: enqueue failure windows ----------------------------------
    async def test_01_crash_after_commit_before_enqueue_is_reenqueued(self):
        run_id = await self._queued_run(created_offset_seconds=120)  # committed, never enqueued
        self.assertFalse(await self.redis.exists(job_key_prefix + str(run_id)))
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertIn(str(run_id), report.requeued)
        self.assertTrue(await self.redis.exists(job_key_prefix + str(run_id)))
        second = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), second.requeued)  # job present: no duplicate enqueue
        self.assertEqual((await self._get(run_id)).status, "queued")

    async def test_02_queue_unavailable_during_enqueue_keeps_recoverable_run(self):
        with patch("app.services.enqueue_run", AsyncMock(side_effect=ConnectionError("redis down"))):
            response = await self.service.queue(self._session(), "Queue outage question", ResearchMode.fast, actor=self.actor)
        await self._close_session()
        self.assertEqual(response.status, "queued")
        record = await self._get(response.id)
        self.assertEqual((record.status, record.attempt), ("queued", 0))
        self.assertEqual(await self._reservation_status(response.id), "reserved")
        async with self.sessions() as session:
            await session.execute(update(ResearchRun).where(ResearchRun.id == response.id)
                                  .values(created_at=func.now() - timedelta(seconds=120)))
            await session.commit()
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertIn(str(response.id), report.requeued)

    async def test_02b_queue_unavailable_before_commit_persists_nothing(self):
        broken = ArqRedis(host="127.0.0.1", port=1, socket_connect_timeout=1)
        service = ResearchService(self.settings, broken)
        try:
            with self.assertRaises(Exception):
                await service.queue(self._session(), "Unreachable queue question", ResearchMode.fast, actor=self.actor)
        finally:
            await self._close_session(rollback=True)
            await broken.aclose()
        async with self.sessions() as session:
            count = await session.scalar(select(func.count()).select_from(ResearchRun).where(
                ResearchRun.requested_by == self.actor, ResearchRun.question == "Unreachable queue question"))
        self.assertEqual(count, 0)

    # ----- 3: duplicate delivery -------------------------------------------
    async def test_03_concurrent_duplicate_claims_start_one_execution(self):
        run_id = await self._queued_run()
        claims = await asyncio.gather(*(self._claim(run_id) for _ in range(5)))
        winners = [claim for claim in claims if claim is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual((await self._get(run_id)).attempt, 1)

    async def test_03b_duplicate_job_delivery_executes_once(self):
        run_id = await self._queued_run()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def fake_execute(session, record, *args, **kwargs):
            calls.append(record.attempt)
            started.set()
            await release.wait()
            record.status = "completed"
            await session.commit()

        self.service._execute = fake_execute
        first = asyncio.create_task(self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer))
        await asyncio.wait_for(started.wait(), 5)
        await self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
        release.set()
        await asyncio.wait_for(first, 5)
        self.assertEqual(calls, [1])
        self.assertEqual((await self._get(run_id)).status, "completed")

    # ----- 4 & 10: crash, lease expiry, late completion ---------------------
    async def test_04_worker_crash_is_recovered_after_lease_expiry_not_before(self):
        run_id = await self._queued_run()
        await self._claim(run_id, lease_seconds=60)
        early = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), early.expired)  # live lease: long jobs are protected
        await self._expire_lease(run_id)  # the crashed worker stopped renewing
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertIn(str(run_id), report.expired)
        record = await self._get(run_id)
        self.assertEqual((record.status, record.attempt), ("failed", 2))
        self.assertIn("lease", record.error)
        self.assertEqual(await self._reservation_status(run_id), "retained_unknown_usage")
        # ARQ redelivers the crashed job: it must not execute again.
        self.service._execute = AsyncMock()
        await self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
        self.service._execute.assert_not_awaited()

    async def test_10_late_worker_cannot_overwrite_recovered_run(self):
        run_id = await self._queued_run()
        async with self.sessions() as late_session:
            record = await claim_run(late_session, run_id, owner="late-worker", lease_seconds=60)
            await self._expire_lease(run_id)
            await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
            self.assertFalse(await renew_lease(self.sessions, run_id, 1, 60))
            record.status = "completed"
            record.answer = "late answer"
            with self.assertRaises(StaleDataError):
                await late_session.commit()
            await late_session.rollback()
        final = await self._get(run_id)
        self.assertEqual((final.status, final.answer), ("failed", None))

    # ----- 5 & 6: cancellation -------------------------------------------------
    async def test_05_cancellation_during_execution_stops_worker_and_retains_cost(self):
        run_id = await self._queued_run()
        started = asyncio.Event()
        finished_normally = []

        async def fake_execute(session, record, *args, **kwargs):
            started.set()
            await asyncio.sleep(10)  # provider calls in progress
            finished_normally.append(True)

        self.service._execute = fake_execute
        worker = asyncio.create_task(self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer))
        await asyncio.wait_for(started.wait(), 5)
        async with self.sessions() as session:
            termination = await cancel_run(session, run_id, actor=self.actor, cost_guard=self.guard)
        self.assertEqual(termination.previous_status, "running")
        await asyncio.wait_for(worker, 5)  # lease renewal fails -> worker stops itself
        self.assertEqual(finished_normally, [])
        record = await self._get(run_id)
        self.assertEqual((record.status, record.attempt), ("cancelled", 2))
        self.assertEqual(await self._reservation_status(run_id), "retained_unknown_usage")

    async def test_05b_worker_interruption_is_persisted_with_fencing(self):
        run_id = await self._queued_run()
        started = asyncio.Event()

        async def fake_execute(session, record, *args, **kwargs):
            started.set()
            await asyncio.sleep(10)

        self.service._execute = fake_execute
        worker = asyncio.create_task(self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer))
        await asyncio.wait_for(started.wait(), 5)
        worker.cancel()  # ARQ shutdown or job timeout
        with self.assertRaises(asyncio.CancelledError):
            await worker
        record = await self._get(run_id)
        self.assertEqual((record.status, record.attempt), ("failed", 2))
        self.assertIn("interrupted", record.error)
        self.assertEqual(await self._reservation_status(run_id), "retained_unknown_usage")

    async def test_06_cancellation_before_execution_releases_cost_and_blocks_start(self):
        before = await self._daily_spend()
        run_id = await self._queued_run()
        self.assertGreater(await self._daily_spend(), before)
        async with self.sessions() as session:
            termination = await cancel_run(session, run_id, actor=self.actor, cost_guard=self.guard)
        self.assertTrue(termination.never_started)
        self.assertEqual(await self._reservation_status(run_id), "released")
        self.assertEqual(await self._daily_spend(), before)
        self.service._execute = AsyncMock()
        await self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
        self.service._execute.assert_not_awaited()
        async with self.sessions() as session:
            self.assertIsNone(await cancel_run(session, run_id, actor=self.actor, cost_guard=self.guard))

    # ----- 7: redelivery after completion --------------------------------------
    async def test_07_redelivery_of_completed_job_is_ignored(self):
        run_id = await self._queued_run()

        async def fake_execute(session, record, *args, **kwargs):
            record.status, record.answer = "completed", "answer"
            await session.commit()

        self.service._execute = fake_execute
        await self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
        self.service._execute = AsyncMock()
        await self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
        self.service._execute.assert_not_awaited()
        record = await self._get(run_id)
        self.assertEqual((record.status, record.answer, record.attempt), ("completed", "answer", 1))

    # ----- 8 & 9: identity and deduplication --------------------------------
    def _session(self):
        self._open = self.sessions()
        return self._open

    async def _close_session(self, rollback: bool = False):
        if rollback:
            await self._open.rollback()
        await self._open.close()

    async def test_08_concurrent_identical_submissions_share_one_run(self):
        async def submit():
            async with self.sessions() as session:
                return await self.service.queue(session, "Identical concurrent question", ResearchMode.fast, actor=self.actor)

        with patch("app.services.enqueue_run", AsyncMock()):
            responses = await asyncio.gather(*(submit() for _ in range(4)))
        self.assertEqual(len({response.id for response in responses}), 1)
        async with self.sessions() as session:
            count = await session.scalar(select(func.count()).select_from(ResearchRun).where(
                ResearchRun.requested_by == self.actor, ResearchRun.question == "Identical concurrent question"))
        self.assertEqual(count, 1)
        self.assertEqual(await self._daily_spend(), 50_000)  # one fast reservation

    async def test_09_same_question_in_different_modes_creates_distinct_runs(self):
        async def submit(mode):
            async with self.sessions() as session:
                return await self.service.queue(session, "Mode identity question", mode, actor=self.actor)

        with patch("app.services.enqueue_run", AsyncMock()):
            fast, standard = await asyncio.gather(submit(ResearchMode.fast), submit(ResearchMode.standard))
            fast_again = await submit(ResearchMode.fast)
        self.assertNotEqual(fast.id, standard.id)
        self.assertEqual(fast.id, fast_again.id)
        self.assertEqual((fast.mode, standard.mode), (ResearchMode.fast, ResearchMode.standard))

    # ----- 11: Redis data loss -------------------------------------------------
    async def test_11_redis_queue_and_reservation_loss_is_recovered_conservatively(self):
        run_id = await self._queued_run(created_offset_seconds=120)
        async with self.sessions() as session:
            record = await session.get(ResearchRun, run_id)
            from app.job_lifecycle import enqueue_run
            self.assertIsNotNone(await enqueue_run(self.redis, record))
        # Simulate Redis restarting without persisted data for this run.
        await self.redis.delete(job_key_prefix + str(run_id), f"{self.namespace}:reservation:research:{run_id}")
        await self.redis.zrem(default_queue_name, str(run_id))
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertIn(str(run_id), report.requeued)
        await self._claim(run_id)
        await self._expire_lease(run_id)
        await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        record = await self._get(run_id)
        self.assertEqual(record.status, "failed")
        accounting = record.model_metadata["cost_accounting"]
        self.assertEqual(accounting["reservation"], {"status": "missing"})
        self.assertFalse(accounting["complete"])  # missing usage is not zero spend

    async def test_11b_stranded_job_with_existing_result_is_failed_and_released(self):
        run_id = await self._queued_run(created_offset_seconds=120)
        await self.redis.set(result_key_prefix + str(run_id), b"x", ex=60)  # ARQ result exists, never claimed
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertIn(str(run_id), report.stranded)
        record = await self._get(run_id)
        self.assertEqual(record.status, "failed")
        self.assertEqual(await self._reservation_status(run_id), "released")

    # ----- 12: budget reconciliation --------------------------------------------
    async def test_12_reconciled_spend_is_not_undone_by_later_release_or_retain(self):
        run_id = await self._queued_run(mode="standard")
        await self.guard.reconcile_research(str(run_id), 0.03)
        self.assertEqual(await self._daily_spend(), 30_000)
        await self.guard.release_research(str(run_id))
        await self.guard.retain_research(str(run_id))
        await self.guard.reconcile_research(str(run_id), 0.01)
        self.assertEqual(await self._daily_spend(), 30_000)
        self.assertEqual(await self._reservation_status(run_id), "reconciled")

    async def test_12b_failed_run_with_unknown_usage_keeps_reservation(self):
        run_id = await self._queued_run(mode="standard")
        await self._claim(run_id)
        await self._expire_lease(run_id)
        await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertEqual(await self._daily_spend(), 200_000)
        self.assertEqual(await self._reservation_status(run_id), "retained_unknown_usage")

    # ----- 13: concurrent recovery ----------------------------------------------
    async def test_13_concurrent_recovery_sweeps_terminate_once(self):
        run_id = await self._queued_run()
        await self._claim(run_id)
        await self._expire_lease(run_id)
        reports = await asyncio.gather(*(
            recover_research_runs(self.sessions, self.redis, self.guard, self.settings) for _ in range(4)
        ))
        self.assertEqual(sum(str(run_id) in report.expired for report in reports), 1)
        record = await self._get(run_id)
        self.assertEqual((record.status, record.attempt), ("failed", 2))
        self.assertEqual(await self._audit_count(run_id, "research.recovered"), 1)
        self.assertEqual(await self._daily_spend(), 50_000)

    async def test_13b_concurrent_requeue_enqueues_one_job(self):
        run_id = await self._queued_run(created_offset_seconds=120)
        reports = await asyncio.gather(*(
            recover_research_runs(self.sessions, self.redis, self.guard, self.settings) for _ in range(4)
        ))
        self.assertLessEqual(sum(str(run_id) in report.requeued for report in reports), 1)
        self.assertEqual(await self.redis.zscore(default_queue_name, str(run_id)) is not None, True)
        self.assertEqual(await self._audit_count(run_id, "research.requeued"), 1)

    # ----- 14: successful behavior preserved ----------------------------------------
    async def test_14_successful_run_completes_and_reconciles(self):
        run_id = await self._queued_run(mode="standard")

        async def fake_execute(session, record, *args, **kwargs):
            await asyncio.sleep(0.5)  # several heartbeats renew the lease
            metadata = dict(record.model_metadata)
            metadata["llm_gateway"] = {"estimated_cost_usd": 0.02}
            metadata["embedding_usage"] = {"estimated_cost_usd": 0.001}
            await self.service._reconcile_cost(record, metadata)
            record.model_metadata = metadata
            record.status, record.answer = "completed", "A supported answer"
            record.lease_owner = record.lease_expires_at = None
            await session.commit()

        self.service._execute = fake_execute
        await self.service.run_queued(run_id, "q", ResearchMode.standard, ResearchCacheMode.use_cached_answer)
        record = await self._get(run_id)
        self.assertEqual((record.status, record.answer, record.attempt), ("completed", "A supported answer", 1))
        self.assertIsNone(record.lease_expires_at)
        self.assertEqual(await self._reservation_status(run_id), "reconciled")
        self.assertEqual(await self._daily_spend(), 21_000)
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), report.expired + report.requeued + report.stranded + report.unleased)

    async def test_14b_live_lease_is_renewed_during_long_execution(self):
        run_id = await self._queued_run()
        record = await self._claim(run_id, lease_seconds=1)
        async with LeaseKeeper(self.sessions, run_id, record.attempt, lease_seconds=1, heartbeat_seconds=0.2):
            await asyncio.sleep(1.5)  # longer than the lease itself
            report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), report.expired)
        self.assertEqual((await self._get(run_id)).status, "running")

    # ----- PR #8 review: database clock is authoritative --------------------------
    async def test_15_worker_clock_behind_does_not_cause_false_recovery(self):
        run_id = await self._queued_run()
        skewed = utcnow() - timedelta(minutes=10)
        with patch("app.job_lifecycle.utcnow", return_value=skewed):
            await self._claim(run_id, lease_seconds=60)
            self.assertTrue(await renew_lease(self.sessions, run_id, 1, 60))
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), report.expired)
        self.assertEqual((await self._get(run_id)).status, "running")

    async def test_15b_recovery_worker_clock_ahead_does_not_expire_live_lease(self):
        run_id = await self._queued_run()
        await self._claim(run_id, lease_seconds=60)
        with patch("app.job_lifecycle.utcnow", return_value=utcnow() + timedelta(minutes=10)):
            report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), report.expired)
        self.assertEqual((await self._get(run_id)).status, "running")

    async def test_15c_lease_expiry_is_measured_on_the_database_clock(self):
        run_id = await self._queued_run()
        await self._claim(run_id, lease_seconds=60)
        async with self.sessions() as session:
            remaining = await session.scalar(
                select(func.extract("epoch", ResearchRun.lease_expires_at - func.now())).where(ResearchRun.id == run_id))
        self.assertTrue(50 <= float(remaining) <= 61, remaining)

    # ----- PR #8 review: bounded execution without lease renewal -------------------
    async def test_16_database_unavailable_for_lease_window_stops_paid_execution(self):
        settings = Settings(**{**self.settings.model_dump(), "research_lease_seconds": 1.0, "research_heartbeat_seconds": 0.2})
        service = ResearchService(settings, self.redis)
        service.cost_guard = self.guard
        run_id = await self._queued_run()
        started = asyncio.Event()
        finished_normally = []

        async def fake_execute(session, record, *args, **kwargs):
            started.set()
            await asyncio.sleep(10)
            finished_normally.append(True)

        service._execute = fake_execute
        with patch("app.job_lifecycle.renew_lease", AsyncMock(side_effect=ConnectionError("database unavailable"))):
            await asyncio.wait_for(
                service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer), 5)
        self.assertEqual(finished_normally, [])
        record = await self._get(run_id)
        self.assertEqual((record.status, record.attempt), ("failed", 2))
        self.assertEqual(await self._reservation_status(run_id), "retained_unknown_usage")

    # ----- PR #8 review: orphaned ARQ job hash ----------------------------------------
    async def test_17_orphaned_job_hash_without_queue_entry_is_requeued(self):
        run_id = await self._queued_run(created_offset_seconds=120)
        async with self.sessions() as session:
            from app.job_lifecycle import enqueue_run
            self.assertIsNotNone(await enqueue_run(self.redis, await session.get(ResearchRun, run_id)))
        await self.redis.zrem(default_queue_name, str(run_id))  # job hash survives, queue entry lost
        self.assertTrue(await self.redis.exists(job_key_prefix + str(run_id)))
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertIn(str(run_id), report.requeued)
        self.assertIsNotNone(await self.redis.zscore(default_queue_name, str(run_id)))

    async def test_17b_job_being_executed_is_not_requeued(self):
        run_id = await self._queued_run(created_offset_seconds=120)
        async with self.sessions() as session:
            from app.job_lifecycle import enqueue_run
            await enqueue_run(self.redis, await session.get(ResearchRun, run_id))
        await self.redis.zrem(default_queue_name, str(run_id))
        await self.redis.set(in_progress_key_prefix + str(run_id), b"1", ex=60)  # a worker holds it
        report = await recover_research_runs(self.sessions, self.redis, self.guard, self.settings)
        self.assertNotIn(str(run_id), report.requeued)
        self.assertIsNone(await self.redis.zscore(default_queue_name, str(run_id)))

    # ----- PR #8 review: PostgreSQL commit failure after Redis reservation -----------
    async def test_18_commit_failure_after_reservation_releases_it(self):
        async with self.sessions() as session:
            with patch.object(session, "commit", AsyncMock(side_effect=ConnectionError("commit lost"))), \
                    patch("app.services.enqueue_run", AsyncMock()):
                with self.assertRaises(ConnectionError):
                    await self.service.queue(session, "Commit failure question", ResearchMode.fast, actor=self.actor)
        async with self.sessions() as session:
            count = await session.scalar(select(func.count()).select_from(ResearchRun).where(
                ResearchRun.requested_by == self.actor, ResearchRun.question == "Commit failure question"))
        self.assertEqual(count, 0)
        self.assertEqual(await self._daily_spend(), 0)

    async def test_18b_ambiguous_commit_that_succeeded_keeps_reservation(self):
        async with self.sessions() as session:
            real_commit = session.commit

            async def commit_then_fail():
                await real_commit()
                raise ConnectionError("acknowledgement lost")

            with patch.object(session, "commit", commit_then_fail), patch("app.services.enqueue_run", AsyncMock()):
                with self.assertRaises(ConnectionError):
                    await self.service.queue(session, "Ambiguous commit question", ResearchMode.fast, actor=self.actor)
        async with self.sessions() as session:
            record = (await session.execute(select(ResearchRun).where(
                ResearchRun.requested_by == self.actor, ResearchRun.question == "Ambiguous commit question"))).scalar_one()
        self.assertEqual(await self._reservation_status(record.id), "reserved")
        self.assertEqual(await self._daily_spend(), 50_000)



if __name__ == "__main__":
    unittest.main()
