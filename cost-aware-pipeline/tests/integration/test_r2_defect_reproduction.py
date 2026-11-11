"""R2 defect reproductions, written against interfaces that predate R2.

On the pre-R2 implementation these tests fail; they remain as regression tests.
Real PostgreSQL and Redis; research execution is a deterministic fake.
"""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import RedisSettings
from arq.constants import default_queue_name, job_key_prefix
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.cost_control import CostGuard
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
class R2DefectReproductionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(_async_url(os.environ["DATABASE_URL_TEST"]))
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.redis = await create_pool(RedisSettings.from_dsn(os.getenv("REDIS_URL_TEST", "redis://localhost:6379/0")))
        self.actor = f"r2-repro-{uuid4().hex[:12]}"
        self.namespace = f"test:r2repro:cost:{uuid4().hex}"
        # Unknown keys are ignored by Settings on code that predates them.
        self.settings = Settings(cost_daily_budget_usd=100.0, cost_monthly_budget_usd=1000.0,
                                 research_lease_seconds=3, research_heartbeat_seconds=0.2)
        self.service = ResearchService(self.settings, self.redis)
        self.service.cost_guard = CostGuard(self.redis, self.settings, namespace=self.namespace)
        self.patcher = patch("app.services.session_factory", self.sessions)
        self.patcher.start()

    async def asyncTearDown(self):
        self.patcher.stop()
        async with self.sessions() as session:
            ids = list((await session.execute(select(ResearchRun.id).where(ResearchRun.requested_by == self.actor))).scalars())
            if ids:
                await session.execute(delete(AuditEvent).where(AuditEvent.target_id.in_([str(i) for i in ids])))
                await session.execute(delete(ResearchRun).where(ResearchRun.id.in_(ids)))
            await session.commit()
        for run_id in ids:
            await self.redis.delete(job_key_prefix + str(run_id))
            await self.redis.zrem(default_queue_name, str(run_id))
        keys = [key async for key in self.redis.scan_iter(match=f"{self.namespace}:*")]
        if keys:
            await self.redis.delete(*keys)
        await self.redis.aclose()
        await self.engine.dispose()

    async def _queued_run(self) -> UUID:
        run_id = uuid4()
        async with self.sessions() as session:
            session.add(ResearchRun(id=run_id, question=f"Repro {run_id}", requested_by=self.actor, status="queued",
                                    model_metadata={"mode": "fast", "semantic_claim_verification_mode": "off"}))
            await session.commit()
        return run_id

    async def _status(self, run_id: UUID) -> str:
        async with self.sessions() as session:
            return (await session.get(ResearchRun, run_id)).status

    async def test_enqueue_failure_after_commit_does_not_lose_the_submission(self):
        with patch.object(self.redis, "enqueue_job", AsyncMock(side_effect=ConnectionError("redis unavailable"))):
            async with self.sessions() as session:
                response = await self.service.queue(session, "Repro enqueue failure", ResearchMode.fast, actor=self.actor)
        self.assertEqual(response.status, "queued")

    async def test_duplicate_delivery_executes_only_once(self):
        run_id = await self._queued_run()
        calls = []

        async def fake_execute(session, record, *args, **kwargs):
            calls.append(record.id)
            await asyncio.sleep(0.3)
            record.status = "completed"
            await session.commit()

        self.service._execute = fake_execute
        await asyncio.gather(*(self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
                               for _ in range(2)))
        self.assertEqual(len(calls), 1)

    async def test_cancelled_before_start_is_not_executed(self):
        run_id = await self._queued_run()
        async with self.sessions() as session:
            await session.execute(update(ResearchRun.__table__).where(ResearchRun.__table__.c.id == run_id).values(status="cancelled"))
            await session.commit()
        self.service._execute = AsyncMock()
        await self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer)
        self.service._execute.assert_not_awaited()
        self.assertEqual(await self._status(run_id), "cancelled")

    async def test_worker_does_not_overwrite_cancellation_made_while_running(self):
        run_id = await self._queued_run()
        started = asyncio.Event()

        async def fake_execute(session, record, *args, **kwargs):
            started.set()
            await asyncio.sleep(1.0)
            record.status = "completed"
            await session.commit()

        self.service._execute = fake_execute
        worker = asyncio.create_task(self.service.run_queued(run_id, "q", ResearchMode.fast, ResearchCacheMode.use_cached_answer))
        await asyncio.wait_for(started.wait(), 5)
        async with self.sessions() as session:
            await session.execute(update(ResearchRun.__table__).where(ResearchRun.__table__.c.id == run_id).values(status="cancelled"))
            await session.commit()
        try:
            await asyncio.wait_for(worker, 5)
        except Exception:
            pass
        self.assertEqual(await self._status(run_id), "cancelled")

    async def test_release_after_reconciliation_keeps_recorded_spend(self):
        guard = self.service.cost_guard
        await guard.reserve_research("repro-run", ResearchMode.standard)
        await guard.reconcile_research("repro-run", 0.03)
        await guard.release_research("repro-run")
        daily = [key async for key in self.redis.scan_iter(match=f"{self.namespace}:daily:*")]
        self.assertEqual(int(await self.redis.get(daily[0])), 30_000)

    async def test_concurrent_identical_submissions_create_one_run(self):
        async def submit():
            async with self.sessions() as session:
                return await self.service.queue(session, "Repro identical submission", ResearchMode.fast, actor=self.actor)

        responses = await asyncio.gather(*(submit() for _ in range(3)))
        self.assertEqual(len({response.id for response in responses}), 1)
        async with self.sessions() as session:
            count = await session.scalar(select(func.count()).select_from(ResearchRun).where(
                ResearchRun.requested_by == self.actor, ResearchRun.question == "Repro identical submission"))
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
