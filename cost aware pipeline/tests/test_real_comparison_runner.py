import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("real_runner", ROOT / "scripts" / "run_real_frozen_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RealComparisonRunnerTests(unittest.TestCase):
    def test_explicit_case_ids_preserve_requested_cross_cohort_order(self):
        payload = {"cases": [{"id": "pm-01"}, {"id": "pm-02"}, {"id": "ax-01"}, {"id": "mx-01"}]}
        selected = MODULE.select_cases(payload, case_ids="pm-01,ax-01,mx-01")
        self.assertEqual([item["id"] for item in selected], ["pm-01", "ax-01", "mx-01"])

    def test_unknown_explicit_case_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.select_cases({"cases": [{"id": "pm-01"}]}, case_ids="missing")


if __name__ == "__main__":
    unittest.main()
