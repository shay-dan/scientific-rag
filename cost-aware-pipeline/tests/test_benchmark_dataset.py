import json
from pathlib import Path
import unittest

from app.benchmark_dataset import (
    add_explicit_claim_support,
    lexical_claim_support,
    validate_pilot_dataset,
)


class BenchmarkDatasetTests(unittest.TestCase):
    def _payload(self):
        return {
            "cases": [
                {
                    "id": "pm-01",
                    "source_group": "pubmed",
                    "question": "What does treatment improve?",
                    "reference_answer": "Treatment improves the measured clinical outcome.",
                    "reference_claims": ["Treatment improves the measured clinical outcome."],
                    "reference_context": [
                        {
                            "source_id": "PMID:1",
                            "title": "Trial",
                            "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                            "excerpt": "The treatment improves the measured clinical outcome in adults.",
                        }
                    ],
                    "validation_score": 0.9,
                }
            ]
        }

    def test_repair_adds_stable_context_and_claim_mapping(self):
        repaired = add_explicit_claim_support(self._payload())
        case = repaired["cases"][0]
        self.assertEqual(case["reference_context"][0]["context_id"], "pm-01-ctx-01")
        self.assertEqual(case["claim_support"][0]["context_id"], "pm-01-ctx-01")
        self.assertGreaterEqual(case["claim_support"][0]["lexical_support"], 0.20)

    def test_validator_rejects_missing_mapping(self):
        errors = validate_pilot_dataset(self._payload(), expected_cases=1)
        self.assertTrue(any("map every claim" in error for error in errors))

    def test_lexical_score_is_not_implicitly_perfect(self):
        score = lexical_claim_support("CRISPR improves cancer therapy", "CRISPR edits bacterial genes")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_committed_pilot_is_structurally_valid(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "evaluation" / "pilot-silver-v1.json").read_text(encoding="utf-8"))
        repaired = add_explicit_claim_support(payload)
        self.assertEqual(validate_pilot_dataset(repaired), [])

    def test_semantically_audited_v3_is_structurally_valid(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "evaluation" / "pilot-silver-v3.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_pilot_dataset(payload), [])


if __name__ == "__main__":
    unittest.main()
