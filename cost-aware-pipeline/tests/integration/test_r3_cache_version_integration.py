"""R3 cache-version separation against real PostgreSQL and Redis.

A completed answer produced under the pre-R3 security contract must not be reused
for new research in off, shadow, or enforce mode. Historical records stay intact,
and in-flight deduplication and cost reservation keep working under the new
version. No LLM or literature provider is called (enqueueing is stubbed).
Requires RUN_INTEGRATION_TESTS=1, DATABASE_URL_TEST (migrated) and REDIS_URL_TEST.
"""
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.cost_control import CostGuard
from app.models import AuditEvent, ResearchRun
from app.schemas import ResearchMode
from app.services import ResearchService, effective_pipeline_version

RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"
PRE_R3_VERSIONS = {
    "off": "postgres-hybrid-rrf-mmr-parent-child-structured-v14",
    "shadow": "postgres-hybrid-rrf-mmr-parent-child-structured-v14-semantic-shadow-v1",
    "enforce": "postgres-hybrid-rrf-mmr-parent-child-structured-v14-semantic-enforce-v1",
}


def _async_url(url: str) -> str:
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


@unittest.skipUnless(RUN and os.getenv("DATABASE_URL_TEST"), "Set RUN_INTEGRATION_TESTS=1, DATABASE_URL_TEST and REDIS_URL_TEST")
class R3CacheVersionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(_async_url(os.environ["DATABASE_URL_TEST"]), pool_size=5)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.redis = await create_pool(RedisSettings.from_dsn(os.getenv("REDIS_URL_TEST", "redis://localhost:6379/0")))
        self.actor = f"r3-test-{uuid4().hex[:12]}"
        self.namespace = f"test:r3:cost:{uuid4().hex}"
        self.factory_patch = patch("app.services.session_factory", self.sessions)
        self.factory_patch.start()

    async def asyncTearDown(self):
        self.factory_patch.stop()
        async with self.sessions() as session:
            ids = list((await session.execute(select(ResearchRun.id).where(ResearchRun.requested_by == self.actor))).scalars())
            if ids:
                await session.execute(delete(AuditEvent).where(AuditEvent.target_id.in_([str(i) for i in ids])))
                await session.execute(delete(ResearchRun).where(ResearchRun.id.in_(ids)))
            await session.commit()
        keys = [key async for key in self.redis.scan_iter(match=f"{self.namespace}:*")]
        if keys:
            await self.redis.delete(*keys)
        await self.redis.aclose()
        await self.engine.dispose()

    def _service(self, mode: str) -> tuple[ResearchService, CostGuard]:
        settings = Settings(
            semantic_claim_verification_mode=mode,
            cost_tracking_enabled=True, cost_enforce_hard_limit=True,
            cost_daily_budget_usd=100.0, cost_monthly_budget_usd=1000.0,
            cost_fast_reservation_usd=0.05, cost_standard_reservation_usd=0.20,
        )
        guard = CostGuard(self.redis, settings, namespace=f"{self.namespace}:{mode}")
        service = ResearchService(settings, self.redis)
        service.cost_guard = guard
        return service, guard

    async def _daily_spend(self, mode: str) -> int:
        total = 0
        async for key in self.redis.scan_iter(match=f"{self.namespace}:{mode}:daily:*"):
            total += int(await self.redis.get(key) or 0)
        return total

    async def test_pre_r3_completed_answers_are_not_reused_in_any_mode(self):
        for mode, old_version in PRE_R3_VERSIONS.items():
            with self.subTest(mode=mode):
                question = f"R3 cache separation question ({mode})"
                old_id = uuid4()
                async with self.sessions() as session:
                    session.add(ResearchRun(
                        id=old_id, question=question, requested_by=self.actor, status="completed",
                        answer="Pre-R3 answer.", completed_at=datetime.now(timezone.utc),
                        model_metadata={"mode": "fast", "retrieval_version": old_version,
                                        "semantic_claim_verification_mode": mode},
                    ))
                    await session.commit()

                service, _guard = self._service(mode)
                with patch("app.services.enqueue_run", AsyncMock()):
                    async with self.sessions() as session:
                        first = await service.queue(session, question, ResearchMode.fast, actor=self.actor)
                    async with self.sessions() as session:
                        second = await service.queue(session, question, ResearchMode.fast, actor=self.actor)

                self.assertNotEqual(first.id, old_id)          # no reuse of the pre-R3 answer
                self.assertFalse(first.cached)
                self.assertEqual(second.id, first.id)          # in-flight deduplication still applies
                self.assertEqual(await self._daily_spend(mode), 50_000)  # one fast reservation
                async with self.sessions() as session:
                    new = await session.get(ResearchRun, first.id)
                    old = await session.get(ResearchRun, old_id)
                    count = await session.scalar(select(func.count()).select_from(ResearchRun).where(
                        ResearchRun.requested_by == self.actor, ResearchRun.question == question))
                self.assertEqual(new.model_metadata["retrieval_version"], effective_pipeline_version(mode))
                self.assertEqual(count, 2)
                self.assertEqual(old.status, "completed")       # historical record preserved
                self.assertEqual(old.answer, "Pre-R3 answer.")
                self.assertEqual(old.model_metadata["retrieval_version"], old_version)

    async def test_completed_answer_under_current_version_is_still_reused(self):
        for mode in PRE_R3_VERSIONS:
            with self.subTest(mode=mode):
                question = f"R3 current-version reuse question ({mode})"
                current_id = uuid4()
                async with self.sessions() as session:
                    session.add(ResearchRun(
                        id=current_id, question=question, requested_by=self.actor, status="completed",
                        answer="Current answer.", completed_at=datetime.now(timezone.utc),
                        model_metadata={"mode": "fast", "retrieval_version": effective_pipeline_version(mode),
                                        "semantic_claim_verification_mode": mode},
                    ))
                    await session.commit()
                service, _guard = self._service(mode)
                with patch("app.services.enqueue_run", AsyncMock()) as enqueue:
                    async with self.sessions() as session:
                        response = await service.queue(session, question, ResearchMode.fast, actor=self.actor)
                self.assertEqual(response.id, current_id)
                self.assertTrue(response.cached)
                enqueue.assert_not_awaited()
                self.assertEqual(await self._daily_spend(mode), 0)


if __name__ == "__main__":
    unittest.main()
