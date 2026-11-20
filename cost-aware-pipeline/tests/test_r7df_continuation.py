"""R7D-F: offline successor and partial-report regressions; no real provider."""
import copy
import json
import unittest

from app.benchmark import generation_evaluation as gen
from app.benchmark import program_ledger as programs
from app.benchmark.incremental_generation_evaluation import (
    run_incremental_evaluation, verify_checkpoint,
)
from app.benchmark.generation_continuation import (
    build_continuation_contract, reporting_records,
)
from tests import test_r7dc_generation_reliability as rel


class SuccessorTests(rel._CampaignHarness):
    async def prepare(self):
        self.ledger = self.initialized_ledger()
        self.old = self.checkpoint("old.json")
        self.new = self.checkpoint("new.json")
        self.old_config = rel.v2_config()
        self.new_config = dict(self.old_config, code_sha="b" * 40)
        self.key = gen.create_blind_key(self.manifest)
        baseline = programs.program_baseline(self.ledger.load())
        self.old_auth = rel.v2_authorization(self.manifest, self.old_config,
            baseline_sha256=baseline["sha256"], authorization_id="predecessor")
        self.old_auth["blind_key_sha256"] = gen.canonical_sha256(self.key)
        self.old_plan = gen.dry_run(self.manifest, self.old_config)
        self.old_cp = await self.run_v2(rel.ScriptedProvider(["ok", "ok", "json_validate"]),
            self.old, self.ledger, pairs=2, config=self.old_config, authorization=self.old_auth)
        self.assertEqual("provider_structured_generation_failure", self.old_cp["execution"]["stop_code"])
        self.old_bytes = self.old.path.read_bytes()
        programs.close_campaign(self.ledger, self.old)
        baseline = programs.program_baseline(self.ledger.load())
        self.auth = rel.v2_authorization(self.manifest, self.new_config,
            baseline_sha256=baseline["sha256"], authorization_id="successor")
        self.auth["blind_key_sha256"] = self.old_auth["blind_key_sha256"]
        self.auth["continuation"] = self.contract()

    def contract(self, **overrides):
        values = dict(manifest=self.manifest, config=self.new_config, key=self.key,
            program_ledger=self.ledger, predecessor_store=self.old,
            predecessor_config=self.old_config, predecessor_authorization=self.old_auth,
            predecessor_plan=self.old_plan)
        values.update(overrides)
        return build_continuation_contract(**values)

    async def run_successor(self, provider, pairs=1, **changes):
        values = dict(authorization=self.auth, checkpoint_store=self.new, pairs=pairs,
            gateway=provider.gateway, provenance_resolver=lambda: {
                "sha": self.new_config["code_sha"], "source": "test"},
            clock=lambda: 1000.0, program_ledger=self.ledger, blind_key=self.key,
            predecessor=(self.old, self.old_config, self.old_auth, self.old_plan))
        values.update(changes)
        patches = rel._trace_patches(provider)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return await run_incremental_evaluation(self.manifest, self.new_config, **values)

    async def test_successor_executes_only_untouched_arms_and_accounts_once(self):
        await self.prepare()
        provider = rel.ScriptedProvider()
        result = await self.run_successor(provider)
        self.assertEqual(1, len(provider.requests))
        self.assertEqual(self.old_cp["records"][0], result["records"][0])
        self.assertEqual(self.old_cp["records"][1]["arms"][0], result["records"][1]["arms"][0])
        self.assertEqual("interrupted", result["records"][1]["status"])
        self.assertEqual(1, len(result["execution"]["calls"]))
        self.assertEqual("v15", result["execution"]["calls"][0]["prompt_version"])
        self.assertEqual(4, result["execution"]["budget"]["provider_calls_charged"])
        result = await self.run_successor(provider)
        self.assertEqual(3, len(provider.requests))
        self.assertEqual(6, result["execution"]["budget"]["provider_calls_charged"])
        self.assertEqual(self.old_bytes, self.old.path.read_bytes())
        programs.close_campaign(self.ledger, self.new)
        self.assertEqual(6, programs.program_baseline(self.ledger.load())["budget"]["provider_calls_charged"])
        self.assertEqual(self.old_bytes, self.old.path.read_bytes())

    async def test_missing_predecessor_or_changed_authorization_refused(self):
        await self.prepare()
        for changes in ({"predecessor": None},
                        {"authorization": dict(self.auth, continuation={})},
                        {"blind_key": gen.create_blind_key(self.manifest)}):
            provider = rel.ScriptedProvider()
            with self.assertRaises((ValueError, PermissionError)):
                await self.run_successor(provider, **changes)
            self.assertFalse(provider.requests)
        self.assertFalse(self.new.exists())

    async def test_config_schema_plan_and_blind_drift_refused(self):
        await self.prepare()
        for changes in ({"config": dict(self.new_config, max_output_tokens=3000)},
                        {"key": gen.create_blind_key(self.manifest)},
                        {"predecessor_plan": dict(self.old_plan, schemas={})}):
            with self.assertRaises((ValueError, PermissionError)):
                self.contract(**changes)

    async def test_stale_and_unresolved_predecessor_refused(self):
        await self.prepare()
        original = self.old.path.read_bytes()
        for mutations in ({"active_attempt": {"phase": "initial"}}, {"active_arm": {}}, {}):
            cp = copy.deepcopy(self.old_cp)
            cp.update(mutations)
            self.old.write(cp)
            with self.assertRaises((ValueError, PermissionError)):
                self.contract()
            self.old.path.write_bytes(original)

    async def test_inherited_result_edit_and_replayed_call_refused(self):
        await self.prepare()
        cp = await self.run_successor(rel.ScriptedProvider())
        for mode in ("arm", "call"):
            bad = copy.deepcopy(cp)
            if mode == "arm":
                bad["records"][0]["arms"][0]["status"] = "failed"
            else:
                call = bad["execution"]["calls"][0]
                call.update(scenario_id=self.old_cp["execution"]["calls"][0]["scenario_id"],
                            prompt_version=self.old_cp["execution"]["calls"][0]["prompt_version"])
            with self.assertRaises((ValueError, PermissionError)):
                verify_checkpoint(bad, self.manifest, self.new_config, self.auth)

    async def test_predecessor_inputs_without_continuation_refused(self):
        await self.prepare()
        provider = rel.ScriptedProvider()
        auth = dict(self.auth)
        del auth["continuation"]
        with self.assertRaises(PermissionError):
            await self.run_successor(provider, authorization=auth)
        self.assertFalse(provider.requests)
        self.assertFalse(self.new.exists())

    async def test_partial_export_preserves_denominator_and_excludes_interrupted(self):
        await self.prepare()
        records = reporting_records(self.manifest, self.old_config, self.old_cp)
        packet, _ = gen.build_blind_scoring_packet(self.manifest, records, key=self.key,
            commitment=self.old_auth["blind_key_sha256"])
        self.assertEqual(20, len(packet["cases"]))
        self.assertEqual(1, sum(row["scorable"] for row in packet["cases"]))
        self.assertEqual(18, sum(all(a["status"] == "not_executed" for a in row["arms"].values())
                                 for row in packet["cases"]))
        report = gen.build_public_report(records, scores={}, manifest_sha256=self.manifest["manifest_sha256"],
                                         config_sha256=gen.canonical_sha256(self.old_config))
        self.assertEqual(20, len(report["scenarios"]))
        self.assertEqual(19, report["excluded_from_pairing"]["pairs"])
        self.assertEqual(self.old_bytes, self.old.path.read_bytes())
        self.assertNotIn(self.key["nonce"], json.dumps(packet))
        with self.assertRaises(ValueError):
            reporting_records(self.manifest, self.old_config,
                              dict(self.old_cp, active_arm={}))
        with self.assertRaises(ValueError):
            reporting_records(self.manifest, self.old_config,
                              dict(self.old_cp, execution=dict(self.old_cp["execution"], stop_code="rate_limit")))


if __name__ == "__main__":
    unittest.main()
