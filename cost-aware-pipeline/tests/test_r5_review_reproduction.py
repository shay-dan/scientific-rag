"""R5 review regressions (owner review of PR #12), written before the fix.

1. predictions_hash must not depend on record order, while duplicates stay detected
   and genuinely different predictions hash differently.
2. Scientific claim coverage counts only claims.N targets; supported limitations are
   reported in a separate all-supported-target coverage metric.
Hand-calculated against tests/fixtures/verifier_eval (see test_verifier_evaluation).
"""
import copy
import json
from pathlib import Path
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.verifier_evaluation import evaluate, render_json, render_markdown

FIXTURES = project_root / "tests" / "fixtures" / "verifier_eval"
DATASET = json.loads((FIXTURES / "dataset.json").read_text(encoding="utf-8"))
MIXED = json.loads((FIXTURES / "predictions_mixed.json").read_text(encoding="utf-8"))
COMPLETE = json.loads((FIXTURES / "predictions_complete.json").read_text(encoding="utf-8"))
GIT = {"sha": "a" * 40, "source": "explicit"}


def _report(predictions: dict) -> dict:
    return evaluate(copy.deepcopy(DATASET), copy.deepcopy(predictions), git_provenance=GIT)


class PredictionHashOrderTests(unittest.TestCase):
    def test_record_order_does_not_change_the_report_or_hash(self) -> None:
        reordered = {**MIXED, "predictions": list(reversed(MIXED["predictions"]))}
        original, shuffled = _report(MIXED), _report(reordered)
        self.assertEqual(original["provenance"]["predictions_hash"], shuffled["provenance"]["predictions_hash"])
        self.assertEqual(render_json(original), render_json(shuffled))
        self.assertEqual(shuffled["population"]["outcomes"]["duplicate_assessment"], 1)  # still detected

    def test_changed_or_removed_predictions_change_the_hash(self) -> None:
        base = _report(COMPLETE)["provenance"]["predictions_hash"]
        changed = copy.deepcopy(COMPLETE)
        changed["predictions"][0]["verdict"] = "contradicted"
        self.assertNotEqual(base, _report(changed)["provenance"]["predictions_hash"])
        dropped_duplicate = copy.deepcopy(MIXED)
        dropped_duplicate["predictions"] = [record for index, record in enumerate(MIXED["predictions"])
                                            if not (record.get("item_id") == "A4-c2" and record.get("verdict") == "contradicted")]
        self.assertNotEqual(_report(MIXED)["provenance"]["predictions_hash"],
                            _report(dropped_duplicate)["provenance"]["predictions_hash"])


class ClaimCoverageTests(unittest.TestCase):
    def test_claim_coverage_excludes_limitations(self) -> None:
        # Complete predictions: delivered expected-supported targets are A1-c1, A1-l1, A2-c3.
        # Claims only: expected-supported claims = 5 (A1-l1 is a limitation); delivered = A1-c1, A2-c3.
        simulation = _report(COMPLETE)["enforcement_simulation"]
        self.assertEqual((simulation["claim_coverage"]["numerator"], simulation["claim_coverage"]["denominator"]), (2, 5))
        self.assertIn("supported_target_coverage", simulation)
        target = simulation["supported_target_coverage"]
        self.assertEqual((target["numerator"], target["denominator"]), (3, 6))
        self.assertIn("supported_target_coverage", render_markdown(_report(COMPLETE)))

    def test_limitation_only_retention_is_not_claim_coverage(self) -> None:
        # Only the limitation is predicted supported in A1; A1 then has no supported claim.
        predictions = copy.deepcopy(COMPLETE)
        for record in predictions["predictions"]:
            if record["item_id"] in {"A1-c1", "A2-c3"}:
                record["verdict"] = "insufficient_evidence"
        simulation = _report(predictions)["enforcement_simulation"]
        self.assertEqual(simulation["claim_coverage"]["numerator"], 0)
        self.assertEqual(simulation["supported_target_coverage"]["numerator"], 0)  # A1 fails: no supported claim


if __name__ == "__main__":
    unittest.main()
