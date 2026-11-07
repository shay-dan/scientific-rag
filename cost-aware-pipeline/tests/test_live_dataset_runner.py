import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LiveDatasetRunnerTests(unittest.TestCase):
    def test_runner_forces_fresh_sources_and_persists_each_case(self) -> None:
        source = (ROOT / "scripts" / "run_live_dataset.py").read_text(encoding="utf-8")
        self.assertIn('"cache_mode": "refresh_sources"', source)
        self.assertIn('parser.add_argument("--case-ids"', source)
        self.assertIn("Unknown or ineligible case IDs", source)
        self.assertIn("args.output.write_text", source)
        self.assertIn("combined_conservative_reservation_usd", source)

    def test_saved_run_scorer_accepts_silver_reference_context_lists(self) -> None:
        source = (ROOT / "scripts" / "score_saved_run.py").read_text(encoding="utf-8")
        self.assertIn("isinstance(reference_context, list)", source)
        self.assertIn('item.get("excerpt", "")', source)
        self.assertIn('benchmark.get("dataset_name")', source)

    def test_live_dataset_scorer_only_judges_completed_research(self) -> None:
        source = (ROOT / "scripts" / "score_live_dataset.py").read_text(encoding="utf-8")
        self.assertIn('row.get("status") == "completed"', source)
        self.assertIn("research_completion_rate", source)
        self.assertIn("judge_reservation_usd", source)


if __name__ == "__main__":
    unittest.main()
