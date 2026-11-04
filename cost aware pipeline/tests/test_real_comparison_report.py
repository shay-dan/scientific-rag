import unittest

from scripts.generate_real_comparison_report import _distribution, aggregate


class RealComparisonReportTests(unittest.TestCase):
    def test_distribution_includes_variance_and_p95(self):
        result = _distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean"], 2.5)
        self.assertGreater(result["stddev"], 0)
        self.assertEqual(result["p95"], 4.0)

    def test_aggregate_does_not_claim_complete_cost_when_usage_is_missing(self):
        row = {
            "status": "completed", "recall_at_5": 1.0, "latency_ms": 1000,
            "source_count": 2, "estimated_cost_usd": None,
            "llm": {"provider_calls": 1, "fallback_count": 0},
            "judge": {name: 1.0 for name in (
                "answer_correctness", "answer_relevancy", "citation_correctness",
                "citation_completeness", "overall_quality",
            )},
        }
        result = aggregate([row])
        self.assertFalse(result["research_cost"]["complete"])
        self.assertEqual(result["research_cost"]["unknown_result_count"], 1)


if __name__ == "__main__":
    unittest.main()
