"""R4 review reproductions: usage-completeness and git-SHA provenance (pre-fix behavior).

Offline and deterministic; no provider is contacted.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

project_root = Path(__file__).resolve().parents[1]
repo_root = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.evaluation import EvaluationUsageTracker
from app.evaluation_report import build_evaluation_report

MODEL = "gemini-3.6-flash"


def _google(usage: bool):
    return SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=1000, candidates_token_count=200, thoughts_token_count=0) if usage else None)


def _judge(tracker: EvaluationUsageTracker) -> dict:
    return tracker.summary(judge_model=MODEL, embedding_model="gemini-embedding-001")["judge"]


class UsageCompletenessReproductionTests(unittest.TestCase):
    def test_successful_response_without_usage_is_not_confirmed_zero(self) -> None:
        tracker = EvaluationUsageTracker()
        tracker.begin_judge_call()
        tracker.record_google_response(_google(False))
        judge = _judge(tracker)
        self.assertFalse(judge["usage_actual"], "missing usage reported as actual usage")
        self.assertIsNone(judge["cost_usd"], "missing usage reported as a known cost")

    def test_partial_usage_is_a_lower_bound_not_a_total(self) -> None:
        tracker = EvaluationUsageTracker()
        for usage in (True, False):
            tracker.begin_judge_call()
            tracker.record_google_response(_google(usage))
        judge = _judge(tracker)
        self.assertFalse(judge["usage_actual"])
        self.assertEqual(judge.get("cost_status"), "partial_lower_bound")

    def test_failed_call_makes_spending_uncertain(self) -> None:
        tracker = EvaluationUsageTracker()
        tracker.begin_judge_call()
        tracker.fail_judge_call()
        judge = _judge(tracker)
        self.assertFalse(judge["usage_actual"])
        self.assertIsNone(judge["cost_usd"])

    def test_complete_usage_is_known(self) -> None:
        tracker = EvaluationUsageTracker()
        tracker.begin_judge_call()
        tracker.record_google_response(_google(True))
        self.assertEqual(_judge(tracker).get("cost_status"), "complete")

    def test_no_calls_is_explicit(self) -> None:
        self.assertEqual(_judge(EvaluationUsageTracker()).get("cost_status"), "no_calls")


class GitShaProvenanceReproductionTests(unittest.TestCase):
    @unittest.skipUnless((repo_root / ".github" / "workflows").is_dir(),
                         "workflow files are not in the test image; CI verifies the built image SHA directly")
    def test_ci_builds_pass_the_source_sha(self) -> None:
        for workflow in ("ci.yml", "security.yml"):
            text = (repo_root / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
            with self.subTest(workflow=workflow):
                self.assertIn("--build-arg GIT_SHA=", text)

    def test_compose_builds_pass_the_source_sha(self) -> None:
        compose = (project_root / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("GIT_SHA: ${GIT_SHA", compose)

    def test_unknown_sha_is_warned_about(self) -> None:
        report = build_evaluation_report({"git_sha": "unknown"}, [])
        self.assertTrue(any("code attribution" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
