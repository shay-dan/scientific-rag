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
        self.assertEqual((76, 20), (summary["accounting"]["program"]["remaining_calls"],
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

    def test_ledger_files_must_stay_outside_the_repository(self):
        inside = SCRIPT.parents[1] / "r7d-ledger.private.json"
        with self.assertRaisesRegex(SystemExit, "outside the public repository"):
            self.run_cli("ledger-init", "--ledger", inside, "--program-id", "p", "--max-calls", 80,
                         "--max-cost-usd", 1.5)
        self.assertFalse(inside.exists())


if __name__ == "__main__":
    unittest.main()
