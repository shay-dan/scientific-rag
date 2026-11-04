import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.comparative_benchmark import (
    RAGVariant,
    estimate_comparative_plan,
    recall_at_k,
    validate_comparative_dataset,
)


class ComparativeBenchmarkTests(unittest.TestCase):
    def test_current_bank_is_reported_as_insufficient_for_requested_100(self):
        payload = json.loads((PROJECT_ROOT / "evaluation" / "benchmark-fast-v2.json").read_text(encoding="utf-8"))
        errors = validate_comparative_dataset(payload, required_cases=100)
        self.assertTrue(any("expected exactly 100" in error for error in errors))
        self.assertFalse(any("invent" in error.lower() for error in errors))

    def test_cost_plan_is_provider_free_and_contains_all_variants(self):
        plan = estimate_comparative_plan(case_count=100, repetitions=3)
        self.assertEqual(plan["total_variant_runs"], 1200)
        self.assertEqual(set(plan["variants"]), {item.value for item in RAGVariant})
        self.assertEqual(plan["estimate_type"], "conservative_upper_bound")
        self.assertGreater(plan["estimated_total_cost_usd"], 0)

    def test_no_judge_estimate_removes_judge_calls(self):
        plan = estimate_comparative_plan(case_count=3, repetitions=1, judge=False)
        self.assertFalse(plan["judge_enabled"])
        self.assertTrue(all(row["upper_bound_calls"]["judge_calls"] == 0 for row in plan["variants"].values()))

    def test_recall_at_k_uses_verified_source_ids(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["b", "c"], k=2), 0.5)
        self.assertIsNone(recall_at_k(["a"], [], k=5))

    def test_valid_case_requires_claims_and_provenance_url(self):
        case = {
            "case_id": "X01",
            "cohort": "pubmed",
            "question": "A valid question?",
            "reference_answer": "A valid answer.",
            "reference_context": "Verified: https://pubmed.ncbi.nlm.nih.gov/12345/",
            "reference_claims": ["A claim"],
        }
        payload = {"name": "test", "cases": [case]}
        self.assertEqual(validate_comparative_dataset(payload, required_cases=1), [])


if __name__ == "__main__":
    unittest.main()

