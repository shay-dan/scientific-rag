import json
from pathlib import Path
import unittest

from app.schemas import EvaluationDatasetInput
from scripts.run_fast_benchmark_matrix import QUALITY_THRESHOLDS, distribution, select_matrix_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrozenFastBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((PROJECT_ROOT / "evaluation" / "benchmark-fast-v2.json").read_text(encoding="utf-8"))

    def test_frozen_bank_has_exactly_ten_cases_per_source_cohort(self):
        counts = {cohort: 0 for cohort in ("pubmed", "arxiv", "mixed")}
        for case in self.payload["cases"]:
            counts[case["cohort"]] += 1
        self.assertEqual(counts, {"pubmed": 10, "arxiv": 10, "mixed": 10})

    def test_active_matrix_has_three_three_four_cases(self):
        selected = select_matrix_cases(self.payload)
        counts = {cohort: 0 for cohort in ("pubmed", "arxiv", "mixed")}
        for case in selected["cases"]:
            counts[case["cohort"]] += 1
        self.assertEqual(len(selected["cases"]), 10)
        self.assertEqual(counts, {"pubmed": 3, "arxiv": 3, "mixed": 4})

    def test_every_case_has_verified_provenance_and_atomic_claims(self):
        for case in self.payload["cases"]:
            self.assertIn("https://", case["reference_context"])
            self.assertGreaterEqual(len(case["reference_claims"]), 4)
            self.assertTrue(case["case_id"])

    def test_mixed_cases_have_pubmed_and_arxiv_provenance(self):
        for case in self.payload["cases"]:
            if case["cohort"] != "mixed":
                continue
            self.assertIn("pubmed.ncbi.nlm.nih.gov", case["reference_context"], case["case_id"])
            self.assertIn("arxiv.org/abs/", case["reference_context"], case["case_id"])

    def test_evaluation_worker_forces_independent_research(self):
        worker_source = (PROJECT_ROOT / "app" / "worker.py").read_text(encoding="utf-8")
        self.assertIn('cache_mode=ResearchCacheMode.regenerate_answer,', worker_source)

    def test_dataset_matches_the_public_evaluation_contract(self):
        model = EvaluationDatasetInput.model_validate({"name": self.payload["name"], "cases": self.payload["cases"]})
        self.assertEqual(len(model.cases), 30)

    def test_distribution_reports_requested_statistics(self):
        result = distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["stddev_population"], 0.8165)

    def test_regression_thresholds_are_frozen(self):
        self.assertEqual(QUALITY_THRESHOLDS, {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.65,
            "context_recall": 0.65,
            "success_rate": 0.95,
        })


if __name__ == "__main__":
    unittest.main()
