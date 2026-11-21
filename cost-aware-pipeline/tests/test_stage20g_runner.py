"""Stage 20G operator command refusals (offline; nothing reaches a provider or database)."""
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark import e2e_v2

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage20_e2e", ROOT / "scripts" / "stage20_e2e.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

MANIFEST = {"schema": "scientific-rag-e2e-development/v1", "cases": [
    {"case_id": f"E00{i}", "question": f"Synthetic placeholder question number {i} for an offline test?"}
    for i in range(1, 6)]}
MANIFEST_SHA = e2e_v2.canonical_sha256(MANIFEST)


def call(*argv):
    output = io.StringIO()
    with redirect_stdout(output):
        code = runner.main(list(argv))
    return code, output.getvalue()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "questions.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        (self.root / "other.json").write_text("{}", encoding="utf-8")
        self.manifest = str(self.root / "questions.json")

    def test_locate_manifest_matches_only_the_frozen_hash(self):
        code, output = call("locate-manifest", "--dir", str(self.root))
        self.assertEqual((2, []), (code, json.loads(output)["matches"]))
        with patch.object(e2e_v2, "EXPECTED_MANIFEST_SHA256", MANIFEST_SHA):
            code, output = call("locate-manifest", "--dir", str(self.root))
        self.assertEqual((0, ["questions.json"]), (code, json.loads(output)["matches"]))

    def test_prepare_refuses_a_different_manifest_and_binds_the_fingerprint(self):
        with self.assertRaises(SystemExit):
            call("prepare", "--manifest", self.manifest, "--code-sha", "a" * 40)
        with patch.object(e2e_v2, "EXPECTED_MANIFEST_SHA256", MANIFEST_SHA):
            code, output = call("prepare", "--manifest", self.manifest, "--code-sha", "a" * 40)
            fallback = json.loads(output)
            _, output = call("prepare", "--manifest", self.manifest, "--code-sha", "a" * 40, "--no-fallback")
            groq_only = json.loads(output)
        self.assertEqual(0, code)
        self.assertEqual(0, fallback["provider_calls"])
        self.assertAlmostEqual(0.81632, fallback["maximum_modeled_reservation_usd"])
        self.assertNotEqual(fallback["fingerprint"], groq_only["fingerprint"])
        self.assertEqual(fallback["fingerprint"], e2e_v2.fingerprint(
            e2e_v2.protocol(fallback=True), manifest_sha256=MANIFEST_SHA, code_sha="a" * 40))

    def test_run_refusals_happen_before_any_side_effect(self):
        results = self.root / "results"
        results.mkdir()
        with patch.object(e2e_v2, "EXPECTED_MANIFEST_SHA256", MANIFEST_SHA), \
                patch.dict(os.environ, {"HTTPS_PROXY": ""}):
            good = e2e_v2.fingerprint(e2e_v2.protocol(fallback=True), manifest_sha256=MANIFEST_SHA, code_sha="a" * 40)
            base = ["run", "--manifest", self.manifest, "--code-sha", "a" * 40, "--results", str(results)]
            with self.assertRaisesRegex(SystemExit, "fingerprint"):
                call(*base, "--authorization", "0" * 64)
            with self.assertRaisesRegex(SystemExit, "fingerprint"):
                call(*base, "--authorization", good, "--no-fallback")
            with patch("app.evaluation_report.resolve_git_provenance",
                       return_value={"sha": "b" * 40, "source": "build_arg"}), \
                    self.assertRaisesRegex(SystemExit, "provenance"):
                call(*base, "--authorization", good)
            (results / "journal.jsonl").write_text("{}\n")
            with patch("app.evaluation_report.resolve_git_provenance",
                       return_value={"sha": "a" * 40, "source": "build_arg"}), \
                    self.assertRaisesRegex(SystemExit, "new, empty"):
                call(*base, "--authorization", good)
            with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy:3128"}), \
                    self.assertRaisesRegex(RuntimeError, "proxy"):
                call(*base, "--authorization", good)
        self.assertEqual(["journal.jsonl"], sorted(path.name for path in results.iterdir()))

    def test_report_never_overwrites(self):
        results, out = self.root / "results", self.root / "out"
        results.mkdir()
        (results / "summary.json").write_text(json.dumps({"fingerprint": "f" * 64, "completed": 0,
                                                          "reserved_usd": 0.0}))
        (results / "review-private.json").write_text("{}")
        self.assertEqual(0, call("report", "--results", str(results), "--out", str(out))[0])
        with self.assertRaises(SystemExit):
            call("report", "--results", str(results), "--out", str(out))


if __name__ == "__main__":
    unittest.main()
