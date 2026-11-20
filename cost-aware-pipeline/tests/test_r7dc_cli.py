"""R7D-C operator CLI: refusals, program-ledger commands and the cumulative dry-run report.

Offline: no provider client is constructed and no credential is read. Private files live in
a temporary directory outside the repository.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import test_r7d_generation_evaluation as base
from tests import test_r7dc_generation_reliability as reliability

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generation_evaluation_r7d.py"


def _cli():
    spec = importlib.util.spec_from_file_location("generation_evaluation_r7d_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.private = Path(self.temporary.name) / "private"
        self.private.mkdir(mode=0o700)
        self.cli = _cli()

    def write(self, name, value):
        path = self.private / name
        path.write_text(json.dumps(value))
        os.chmod(path, 0o600)
        return path

    def run_cli(self, *arguments):
        output = io.StringIO()
        with patch.object(sys, "argv", ["generation_evaluation_r7d.py", *map(str, arguments)]), redirect_stdout(output):
            code = self.cli.main()
        return code, output.getvalue()

    def test_paid_run_is_refused(self):
        manifest, config = self.write("m.json", base._manifest()), self.write("c.json", base._config())
        with self.assertRaisesRegex(SystemExit, "superseded by incremental-run"):
            self.run_cli("paid-run", "--manifest", manifest, "--config", config,
                         "--private-records", self.private / "records.json")

    def test_incremental_run_requires_a_v2_config_and_the_program_ledger(self):
        manifest = self.write("m.json", base._manifest())
        for config, ledger in ((base._config(), True), (reliability.v2_config(), False)):
            with self.subTest(schema=config["schema"], ledger=ledger):
                arguments = ["incremental-run", "--manifest", manifest, "--config", self.write("c.json", config),
                             "--authorization", self.write("a.json", {}), "--pairs", 1,
                             "--private-checkpoint", self.private / "checkpoint.json"]
                if ledger:
                    arguments += ["--program-ledger", self.private / "ledger.json"]
                with self.assertRaisesRegex(SystemExit, "v2 config and --program-ledger"):
                    self.run_cli(*arguments)
                self.assertFalse((self.private / "checkpoint.json").exists())

    def test_ledger_commands_and_cumulative_dry_run(self):
        ledger = self.private / "ledger.json"
        code, output = self.run_cli("ledger-init", "--ledger", ledger, "--program-id", "r7d-b-program",
                                    "--max-calls", 80, "--max-cost-usd", 1.5)
        self.assertEqual(0, code)
        self.assertEqual(80, json.loads(output)["baseline"]["remaining"]["calls"])
        history = self.private / "history.private.json"
        reliability._historical_checkpoint(history, authorization_id="history", calls=[
            reliability._call(complete=False, reserved=0.001) for _ in range(4)])
        code, output = self.run_cli("ledger-import", "--ledger", ledger, "--checkpoint", history)
        self.assertEqual(4, json.loads(output)["baseline"]["budget"]["provider_calls_charged"])
        source = self.private / "diagnostic.jsonl"
        source.write_text("{}\n")
        self.run_cli("ledger-record-diagnostic", "--ledger", ledger, "--source", source, "--label", "discovery",
                     "--requests", 20, "--estimated-cost-usd", 0.01, "--uncertain-reserved-usd", 0.002)
        code, output = self.run_cli("ledger-summary", "--ledger", ledger)
        summary = json.loads(output)
        self.assertEqual((76, 20), (summary["accounting"]["cumulative"]["remaining_calls"],
                                    summary["accounting"]["diagnostics"]["requests"]))
        plan_path = self.private / "plan.json"
        code, output = self.run_cli("dry-run", "--manifest", self.write("m.json", base._manifest()),
                                    "--config", self.write("c.json", reliability.v2_config()), "--out", plan_path,
                                    "--program-ledger", ledger)
        plan = json.loads(plan_path.read_text())
        self.assertEqual(0, plan["provider_calls_made"])
        self.assertEqual(76, plan["program"]["remaining"]["calls"])
        self.assertTrue(any("remaining program request allowance" in item for item in plan["program"]["warnings"]))
        self.assertEqual([], plan["blockers"])

    def test_blind_init_exclusive_and_private(self):
        manifest = self.write("m.json", base._manifest())
        path = self.private / "key.json"
        code, out = self.run_cli("blind-init", "--manifest", manifest, "--private-key", path)
        key = json.loads(path.read_text())
        self.assertEqual(0, code)
        self.assertEqual(0o600, path.stat().st_mode & 0o777)
        self.assertEqual(self.cli.canonical_sha256(key), json.loads(out)["blind_key_sha256"])
        self.assertNotIn(key["nonce"], out)
        self.assertNotIn("v15", out)
        with self.assertRaises(SystemExit):
            self.run_cli("blind-init", "--manifest", manifest, "--private-key", path)
        self.assertEqual(key, json.loads(path.read_text()))

    def test_incremental_run_refuses_missing_or_changed_allocation_before_runner(self):
        manifest = base._manifest()
        mpath = self.write("m.json", manifest)
        config = reliability.v2_config()
        key = self.cli.create_blind_key(manifest)
        auth = {"blind_key_sha256": self.cli.canonical_sha256(key)}
        args = ["incremental-run", "--manifest", mpath, "--config", self.write("c.json", config),
                "--authorization", self.write("a.json", auth), "--pairs", 1,
                "--private-checkpoint", self.private / "checkpoint.json",
                "--program-ledger", self.private / "ledger.json"]
        with patch.object(self.cli, "run_incremental_evaluation") as runner:
            with self.assertRaisesRegex(SystemExit, "precommitted"):
                self.run_cli(*args)
            with self.assertRaisesRegex(SystemExit, "invalid blind key"):
                self.run_cli(*args, "--private-key", self.write("bad.json", self.cli.create_blind_key(manifest)))
            runner.assert_not_called()
        self.assertFalse((self.private / "checkpoint.json").exists())

    def test_freeze_scores_bound_to_exact_packet_and_bounded_values(self):
        packet = {"schema": "scientific-rag-generation-eval-blind-scoring/v1",
                  "cases": [{"scenario_id": "D001", "scorable": True},
                            {"scenario_id": "D002", "scorable": False}]}
        scores = {"D001": {"winner": "A", "metrics": {"coverage": {"A": 2, "B": 1}}}}
        ppath, spath = self.write("packet.json", packet), self.write("scores.json", scores)
        frozen = self.private / "frozen.json"
        _, out = self.run_cli("freeze-scores", "--private-packet", ppath, "--scores", spath,
                              "--frozen-scores", frozen)
        value = json.loads(frozen.read_text())
        self.assertEqual(self.cli.canonical_sha256(value), json.loads(out)["frozen_scores_sha256"])
        self.assertEqual(self.cli.canonical_sha256(packet), value["packet_sha256"])
        with self.assertRaises(SystemExit):
            self.run_cli("freeze-scores", "--private-packet", ppath, "--scores", spath,
                         "--frozen-scores", frozen)
        for invalid in ({}, {"D002": scores["D001"]},
                        {"D001": {"winner": "A", "metrics": {"coverage": {"A": 3, "B": 1}}}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.cli._score_freeze(packet, invalid)
        changed = dict(packet, cases=packet["cases"][:1])
        self.assertNotEqual(value, self.cli._score_freeze(changed, scores))

    def test_ledger_files_must_stay_outside_the_repository(self):
        inside = SCRIPT.parents[1] / "r7d-ledger.private.json"
        with self.assertRaisesRegex(SystemExit, "outside the public repository"):
            self.run_cli("ledger-init", "--ledger", inside, "--program-id", "p", "--max-calls", 80,
                         "--max-cost-usd", 1.5)
        self.assertFalse(inside.exists())


if __name__ == "__main__":
    unittest.main()
