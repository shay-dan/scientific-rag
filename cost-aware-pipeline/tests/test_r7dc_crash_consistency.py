"""R7D-C final review of PR #25 at a6551f1: crash-consistency fault injection.

1. Lost-checkpoint accounting: settled usage (cost above the reservation, or retries) written
   to the checkpoint after the pre-transmission ledger mark must never be understated by
   ``ledger-close-lost``.
2. A failed ledger-mark write after the reserved attempt was persisted (including real lock
   contention) must send nothing and leave reconcilable, non-lost accounting.
3. ``accounting_summary`` must not report from a stale or alternate checkpoint.

Zero provider calls: scripted gateways, synthetic traces, real private files.
"""
import os
import shutil
from unittest.mock import patch

from app.benchmark import program_ledger as programs
from app.benchmark.incremental_generation_evaluation import AtomicCheckpoint, CheckpointError
from tests.test_r7dc_generation_reliability import ScriptedProvider, ledger_module
from tests.test_r7dc_review_corrections import MutatingProvider, _Corrections, _cost, _retry


class SimulatedCrash(BaseException):
    """Process death at an injected point."""


class _Faults(_Corrections):
    def inject_mark(self, *, at: int, effect):
        """Replace the ``at``-th ledger mark write (1-based) with ``effect(real, *args)``."""
        real = programs.record_checkpoint_mark
        count = {"n": 0}

        def wrapper(*args, **kwargs):
            count["n"] += 1
            if count["n"] == at:
                return effect(real, *args, **kwargs)
            return real(*args, **kwargs)
        return patch.object(programs, "record_checkpoint_mark", wrapper)


class LostCheckpointAccountingTests(_Faults):
    async def lose_after_settlement(self, mutation):
        """Mark 1: checkpoint created; mark 2: before transmission; mark 3: the next refresh."""
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = MutatingProvider(["ok"], {1: mutation})
        path = self.directory / "A.json"

        def crash(real, *args, **kwargs):
            raise SimulatedCrash()
        with self.inject_mark(at=3, effect=crash), self.assertRaises(SimulatedCrash):
            await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization, pairs=2)
        observed = AtomicCheckpoint(path).load()["execution"]["budget"]
        path.unlink()                                                     # the checkpoint becomes unavailable
        return ledger_store, authorization, observed, provider

    async def test_lost_checkpoint_never_understates_settled_consumption(self):
        for label, mutation in (("cost above reservation", _cost(0.05)), ("unexpected retry", _retry)):
            with self.subTest(label=label):
                self.setUp()
                ledger_store, authorization, observed, provider = await self.lose_after_settlement(mutation)
                self.assertEqual(1, len(provider.requests))
                try:
                    closed = ledger_module().close_lost_campaign(ledger_store, authorization["authorization_id"])
                except ledger_module().ProgramLedgerError as error:
                    self.assertIn("cannot be established", str(error))   # ambiguous: refused, campaign stays open
                    entry = ledger_store.load()["campaigns"][0]
                    self.assertEqual("open", entry["state"])
                    with self.assertRaisesRegex(ledger_module().ProgramLedgerError, "already open"):
                        await self.run_v2(ScriptedProvider(), self.checkpoint("next.json"), ledger_store,
                                          config=__import__("tests.test_r7dc_generation_reliability",
                                                            fromlist=["v2_config"]).v2_config(max_output_tokens=3000),
                                          authorization=self.authorization(ledger_store, authorization_id="next"))
                    continue
                consumed = closed["consumed"]
                self.assertGreaterEqual(consumed["provider_calls_charged"], observed["provider_calls_charged"])
                self.assertGreaterEqual(consumed["budget_charged_usd"] + 1e-12, observed["budget_charged_usd"])

    async def test_settled_consumption_is_marked_immediately_after_durable_settlement(self):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = MutatingProvider(["ok"], {1: _cost(0.05)})
        path = self.directory / "A.json"
        checkpoint = await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization,
                                       pairs=2)
        self.assertEqual("reservation_exceeded", checkpoint["execution"]["stop_code"])
        path.unlink()
        closed = ledger_module().close_lost_campaign(ledger_store, authorization["authorization_id"])
        self.assertAlmostEqual(0.05, closed["consumed"]["cost_complete_usd"], places=8)
        self.assertEqual(1, closed["consumed"]["provider_calls_charged"])

    async def test_a_lost_checkpoint_with_a_request_in_flight_is_refused(self):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        path = self.directory / "A.json"

        def mark_then_crash(real, *args, **kwargs):
            real(*args, **kwargs)                                         # in-flight mark persisted
            raise SimulatedCrash()                                        # dies before settlement
        provider = ScriptedProvider()
        with self.inject_mark(at=2, effect=mark_then_crash), self.assertRaises(SimulatedCrash):
            await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        self.assertEqual([], provider.requests)
        path.unlink()
        with self.assertRaisesRegex(ledger_module().ProgramLedgerError, "cannot be established"):
            ledger_module().close_lost_campaign(ledger_store, authorization["authorization_id"])


class PreTransmissionPersistenceTests(_Faults):
    async def check_failed_mark(self, effect):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        store = self.checkpoint("A.json")
        with self.inject_mark(at=2, effect=effect):                      # the mark before transmission
            checkpoint = await self.run_v2(provider, store, ledger_store, authorization=authorization)
        self.assertEqual([], provider.requests)                          # zero provider calls
        self.assertEqual("ledger_mark_failed", checkpoint["execution"]["stop_code"])
        saved = store.load()
        budget = saved["execution"]["budget"]
        self.assertEqual((0, 0.0, 0.0), (budget["provider_calls_charged"], budget["unsettled_reserved_usd"],
                                         budget["budget_charged_usd"]))   # released: provably never sent
        self.assertIsNone(saved["active_attempt"])
        with self.assertRaisesRegex(CheckpointError, r"stopped \(ledger_mark_failed\)"):
            await self.run_v2(provider, store, ledger_store, authorization=authorization)
        self.assertEqual([], provider.requests)                          # no automatic retransmission
        closed = ledger_module().close_campaign(ledger_store, store)    # reconciles and closes safely
        self.assertEqual(0, closed["consumed"]["provider_calls_charged"])

    async def test_a_failed_mark_write_sends_nothing_and_keeps_accounting_reconcilable(self):
        def fail(real, *args, **kwargs):
            raise OSError("disk full")
        await self.check_failed_mark(fail)

    async def test_ordinary_ledger_lock_contention_sends_nothing(self):
        ledger_store = programs.ProgramLedgerStore(self.directory / "r7d-program-ledger.private.json")

        def contended(real, *args, **kwargs):
            with ledger_store.locked():                                  # another process holds the ledger
                return real(*args, **kwargs)
        await self.check_failed_mark(contended)

    async def test_a_reservation_whose_release_cannot_be_persisted_stays_unresolved(self):
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        store = self.checkpoint("A.json")
        real_write = AtomicCheckpoint.write
        state = {"armed": False}

        def fail(real, *args, **kwargs):
            state["armed"] = True
            raise OSError("ledger unavailable")

        def write(self_store, value):
            if state["armed"]:
                raise OSError("checkpoint unavailable")
            return real_write(self_store, value)
        provider = ScriptedProvider()
        with self.inject_mark(at=2, effect=fail), patch.object(AtomicCheckpoint, "write", write), \
                self.assertRaises(OSError):
            await self.run_v2(provider, store, ledger_store, authorization=authorization)
        self.assertEqual([], provider.requests)
        saved = store.load()                                              # last durable state: reserved
        self.assertIsNotNone(saved["active_attempt"])
        self.assertEqual(1, saved["execution"]["budget"]["provider_calls_charged"])
        with self.assertRaisesRegex(CheckpointError, "unresolved"):
            await self.run_v2(provider, store, ledger_store, authorization=authorization)
        closed = ledger_module().close_campaign(ledger_store, store)
        self.assertTrue(closed["unresolved_attempt"])
        self.assertGreater(closed["consumed"]["uncertain_reserved_usd"], 0.0)


class SummaryIntegrityTests(_Faults):
    async def test_a_stale_or_alternate_checkpoint_cannot_drive_the_summary(self):
        module = ledger_module()
        ledger_store = self.initialized_ledger()
        authorization = self.authorization(ledger_store)
        provider = ScriptedProvider()
        path = self.directory / "A.json"
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        old = self.directory / "A-old.json"
        shutil.copyfile(path, old)
        os.chmod(old, 0o600)
        await self.run_v2(provider, AtomicCheckpoint(path), ledger_store, authorization=authorization)
        stale = AtomicCheckpoint(old).load()
        for arguments in ({}, {"checkpoint_path": old}, {"checkpoint_path": path}):
            with self.subTest(arguments=list(arguments)), self.assertRaises(module.ProgramLedgerError):
                module.accounting_summary(ledger_store.load(), stale, **arguments)
        current = AtomicCheckpoint(path).load()
        summary = module.accounting_summary(ledger_store.load(), current, checkpoint_path=path)
        self.assertEqual(("checkpoint", 4), (summary["current_experiment"]["basis"],
                                             summary["cumulative"]["provider_calls_charged"]))
        from_ledger = module.accounting_summary(ledger_store.load())
        self.assertEqual(("ledger_mark", 4), (from_ledger["current_experiment"]["basis"],
                                              from_ledger["cumulative"]["provider_calls_charged"]))
