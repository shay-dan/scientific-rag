"""R2 unit tests for job-lifecycle rules that do not need a database.

Concurrency, fencing, and recovery against real PostgreSQL and Redis are in
tests/integration/test_job_lifecycle_integration.py.
"""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.config import Settings
from app.job_lifecycle import LeaseKeeper, Termination, research_identity_lock_id, settle_terminated_cost
from app.schemas import ResearchCacheMode, ResearchMode
from app.services import ResearchService


class IdentityLockTests(unittest.TestCase):
    def test_identity_includes_actor_mode_version_and_question(self) -> None:
        base = research_identity_lock_id("svc", "fast", "v14", "Does X improve Y?")
        self.assertEqual(base, research_identity_lock_id("svc", "fast", "v14", "Does X improve Y?"))
        for variant in (
            ("svc", "standard", "v14", "Does X improve Y?"),
            ("other", "fast", "v14", "Does X improve Y?"),
            ("svc", "fast", "v14-semantic-shadow-v1", "Does X improve Y?"),
            ("svc", "fast", "v14", "Does X improve Z?"),
        ):
            with self.subTest(variant=variant):
                self.assertNotEqual(base, research_identity_lock_id(*variant))

    def test_fields_cannot_be_shifted_into_each_other(self) -> None:
        self.assertNotEqual(research_identity_lock_id("ab", "c", "v", "q"), research_identity_lock_id("a", "bc", "v", "q"))


class TerminationTests(unittest.TestCase):
    def test_only_unclaimed_queued_runs_count_as_never_started(self) -> None:
        run_id = uuid4()
        self.assertTrue(Termination(run_id, "queued", 0, "cancelled").never_started)
        self.assertFalse(Termination(run_id, "running", 1, "cancelled").never_started)
        self.assertFalse(Termination(run_id, "running", 0, "failed").never_started)  # synchronous API run
        self.assertFalse(Termination(run_id, "queued", 1, "failed").never_started)


class _FakeSession:
    def __init__(self, record):
        self.record = record
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def get(self, model, run_id, populate_existing=False):
        return self.record


class SettleCostTests(unittest.IsolatedAsyncioTestCase):
    async def test_never_started_run_releases_reservation(self) -> None:
        guard = SimpleNamespace(
            release_research=AsyncMock(return_value=SimpleNamespace(as_dict=lambda: {"amount_usd": 0.0, "status": "released"})),
            retain_research=AsyncMock(),
        )
        record = SimpleNamespace(model_metadata={})
        outcome = await settle_terminated_cost(_FakeSession(record), guard, Termination(uuid4(), "queued", 0, "cancelled"))
        guard.release_research.assert_awaited_once()
        guard.retain_research.assert_not_awaited()
        self.assertEqual(outcome["status"], "released")
        self.assertTrue(record.model_metadata["cost_accounting"]["complete"])

    async def test_started_run_retains_reservation_as_unknown_usage(self) -> None:
        guard = SimpleNamespace(
            release_research=AsyncMock(),
            retain_research=AsyncMock(return_value=SimpleNamespace(as_dict=lambda: {"amount_usd": 0.2, "status": "retained_unknown_usage"})),
        )
        record = SimpleNamespace(model_metadata={})
        await settle_terminated_cost(_FakeSession(record), guard, Termination(uuid4(), "running", 1, "cancelled"))
        guard.retain_research.assert_awaited_once()
        guard.release_research.assert_not_awaited()
        self.assertFalse(record.model_metadata["cost_accounting"]["complete"])

    async def test_missing_reservation_is_recorded_not_assumed_zero(self) -> None:
        guard = SimpleNamespace(release_research=AsyncMock(), retain_research=AsyncMock(return_value=None))
        record = SimpleNamespace(model_metadata={})
        outcome = await settle_terminated_cost(_FakeSession(record), guard, Termination(uuid4(), "running", 1, "failed"))
        self.assertEqual(outcome, {"status": "missing"})
        self.assertFalse(record.model_metadata["cost_accounting"]["complete"])


class LeaseKeeperTests(unittest.IsolatedAsyncioTestCase):
    async def test_lost_lease_cancels_owner_task(self) -> None:
        async def owner():
            async with LeaseKeeper(None, uuid4(), 1, lease_seconds=1, heartbeat_seconds=0.01) as lease:
                await asyncio.sleep(5)
            return lease

        with patch("app.job_lifecycle.renew_lease", AsyncMock(return_value=False)):
            task = asyncio.create_task(owner())
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)

    async def test_transient_renewal_errors_do_not_cancel(self) -> None:
        renewals = AsyncMock(side_effect=[ConnectionError("db blip"), True, True, True, True, True])
        with patch("app.job_lifecycle.renew_lease", renewals):
            async with LeaseKeeper(None, uuid4(), 1, lease_seconds=1, heartbeat_seconds=0.01) as lease:
                await asyncio.sleep(0.08)
        self.assertFalse(lease.lost)
        self.assertGreaterEqual(renewals.await_count, 2)

class LeaseDeadlineTests(unittest.IsolatedAsyncioTestCase):
    """PR #8 review: a worker that cannot renew must stop within the lease window."""

    async def test_prolonged_renewal_failure_cancels_within_lease_window(self) -> None:
        loop = asyncio.get_running_loop()
        holder = {}

        async def owner():
            async with LeaseKeeper(None, uuid4(), 1, lease_seconds=0.3, heartbeat_seconds=0.05) as lease:
                holder["lease"] = lease
                await asyncio.sleep(2)

        started = loop.time()
        with patch("app.job_lifecycle.renew_lease", AsyncMock(side_effect=ConnectionError("db down"))):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(owner())), 1.5)
        self.assertLess(loop.time() - started, 0.6)
        self.assertTrue(holder["lease"].lost)
        self.assertEqual(holder["lease"].loss_reason, "unverifiable")

    async def test_connectivity_returning_before_deadline_keeps_execution(self) -> None:
        renewals = AsyncMock(side_effect=[ConnectionError("blip")] * 3 + [True] * 50)
        with patch("app.job_lifecycle.renew_lease", renewals):
            async with LeaseKeeper(None, uuid4(), 1, lease_seconds=0.5, heartbeat_seconds=0.05) as lease:
                await asyncio.sleep(0.7)  # longer than the lease; renewals succeeded meanwhile
        self.assertFalse(lease.lost)

    async def test_fenced_renewal_reports_definitive_loss(self) -> None:
        holder = {}

        async def owner():
            async with LeaseKeeper(None, uuid4(), 1, lease_seconds=5, heartbeat_seconds=0.02) as lease:
                holder["lease"] = lease
                await asyncio.sleep(2)

        with patch("app.job_lifecycle.renew_lease", AsyncMock(return_value=False)):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(owner(), 1)
        self.assertEqual(holder["lease"].loss_reason, "fenced")



class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc_info):
        return False


class _NoopLease:
    def __init__(self, *args, **kwargs):
        self.lost = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


class RunQueuedClaimTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, service, claimed):
        with patch("app.services.session_factory", return_value=_AsyncContext(SimpleNamespace())), \
                patch("app.services.claim_run", AsyncMock(return_value=claimed)), \
                patch("app.services.LeaseKeeper", _NoopLease):
            await service.run_queued(uuid4(), "Question", ResearchMode.fast, ResearchCacheMode.use_cached_answer)

    async def test_unclaimable_run_is_not_executed(self) -> None:
        # Duplicate delivery, redelivery of a completed job, or cancellation before start.
        service = ResearchService(Settings(), redis=None)
        service._execute = AsyncMock()
        await self._run(service, None)
        service._execute.assert_not_awaited()

    async def test_claimed_run_is_executed_once(self) -> None:
        service = ResearchService(Settings(), redis=None)
        service._execute = AsyncMock()
        await self._run(service, SimpleNamespace(id=uuid4(), attempt=1, model_metadata={}))
        service._execute.assert_awaited_once()

    async def test_worker_cancellation_marks_interrupted_and_reraises(self) -> None:
        service = ResearchService(Settings(), redis=None)
        service._execute = AsyncMock(side_effect=asyncio.CancelledError())
        service._mark_interrupted = AsyncMock()
        with self.assertRaises(asyncio.CancelledError):
            await self._run(service, SimpleNamespace(id=uuid4(), attempt=3, model_metadata={}))
        self.assertEqual(service._mark_interrupted.await_args.args[1], 3)


class QueueWithoutRedisTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_queue_fails_before_anything_is_persisted(self) -> None:
        service = ResearchService(Settings(), redis=None)
        session = SimpleNamespace(add=lambda *_: self.fail("nothing may be persisted"), execute=AsyncMock())
        with self.assertRaisesRegex(RuntimeError, "queue is unavailable"):
            await service.queue(session, "A research question", ResearchMode.fast, actor="alice")
        session.execute.assert_not_awaited()


class LeaseSettingsTests(unittest.TestCase):
    def test_lease_must_cover_several_heartbeats(self) -> None:
        settings = Settings(google_api_key="k", entrez_email="e@example.com",
                            research_lease_seconds=60, research_heartbeat_seconds=30)
        with self.assertRaisesRegex(ValueError, "RESEARCH_LEASE_SECONDS"):
            settings.validate_research_configuration()


if __name__ == "__main__":
    unittest.main()
