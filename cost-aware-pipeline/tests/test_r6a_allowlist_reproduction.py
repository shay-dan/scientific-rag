"""R6A review: holdout output must not echo report-supplied schema strings or dictionary keys.

Written before the fix. Adversarial reports inject holdout question/claim content into
an unknown schema value and into nested dictionary keys; none of it may be emitted,
while legitimate R5 verifier and R6A retrieval aggregates are preserved.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.benchmark.adapters import load_benchmark
from app.benchmark.bridges import to_verifier_dataset
from app.benchmark.holdout import aggregate_only
from app.benchmark.retrieval import build_evidence_pack, evaluate_retrieval
from app.benchmark.splits import filter_split
from app.verifier_evaluation import evaluate as evaluate_verifier
from app.verifier_evaluation import structural_only_baseline, validate_dataset

FIXTURE = project_root / "tests" / "fixtures" / "benchmark_r6a"
BENCHMARK, _ = load_benchmark(FIXTURE / "manifest.json")
GIT = {"sha": "a" * 40, "source": "explicit"}
QUESTION_KEY = "Does_drug_A_lower_systolic_blood_pressure"   # identifier-shaped holdout content
CLAIM_KEY = "drug_A_reduced_systolic_bp_by_8_mmHg"
LABEL_KEY = "C1:supported"


def _verifier_report() -> dict:
    dataset = to_verifier_dataset(BENCHMARK, split="development")
    return evaluate_verifier(dataset, structural_only_baseline(validate_dataset(dataset)), git_provenance=GIT)


def _retrieval_report() -> dict:
    pack = build_evidence_pack(BENCHMARK, split="development")
    run = json.loads((FIXTURE / "retrieval_run.json").read_text(encoding="utf-8"))
    return evaluate_retrieval(filter_split(BENCHMARK, "development"), pack, run, ks=(1, 3))


class AllowlistKeyAndSchemaTests(unittest.TestCase):
    def assertClean(self, output: dict) -> None:
        text = json.dumps(output)
        for secret in (QUESTION_KEY, CLAIM_KEY, LABEL_KEY, "Does drug A", "reduced systolic"):
            self.assertNotIn(secret, text)

    def test_unknown_schema_is_never_echoed(self) -> None:
        for schema in ("Does drug A lower systolic blood pressure?", QUESTION_KEY, None, 7):
            with self.subTest(schema=schema):
                output = aggregate_only({"schema": schema, "population": {"items": 1}})
                self.assertEqual(output, {"schema": "unknown", "note": "unknown report schema"})
        self.assertEqual(aggregate_only("not a report"), {"schema": "unknown", "note": "unknown report schema"})

    def test_injected_keys_in_verifier_report_are_dropped(self) -> None:
        report = _verifier_report()
        report["population"]["outcomes"][QUESTION_KEY] = 1
        report["coverage"][CLAIM_KEY] = {"numerator": 1, "denominator": 1, "rate": 1.0}
        report["confusion_matrix"]["counts"][LABEL_KEY] = {"supported": 1}
        report["confusion_matrix"]["counts"]["supported"][CLAIM_KEY] = 1
        report["classes"][CLAIM_KEY] = {"true_positives": 1}
        report["safety"][QUESTION_KEY] = 1
        report["by_error_category"][CLAIM_KEY] = {"items": 1, "assessed": 1}
        report["by_evidence_type"][QUESTION_KEY] = {"items": 1}
        report["provenance"]["dataset"]["label_provenance"][CLAIM_KEY] = 1
        report["enforcement_simulation"]["totals"][CLAIM_KEY] = 1
        report["enforcement_simulation"][QUESTION_KEY] = {"numerator": 1, "denominator": 1}
        output = aggregate_only(report)
        self.assertClean(output)
        self.assertEqual(output["population"]["items"], 4)            # legitimate aggregates preserved
        self.assertEqual(output["summary"]["macro_f1"], report["summary"]["macro_f1"])
        self.assertEqual(output["classes"]["supported"], report["classes"]["supported"])
        self.assertIn("critical_contradiction_acceptance", output["safety"])
        self.assertIn("numerical", output["by_error_category"])
        self.assertIn("claim_coverage", output["enforcement_simulation"])

    def test_injected_keys_in_retrieval_report_are_dropped(self) -> None:
        report = _retrieval_report()
        report["aggregates"][QUESTION_KEY] = copy.deepcopy(report["aggregates"]["completed"])
        report["aggregates"]["completed"]["by_k"][CLAIM_KEY] = {"paper_recall": {"n": 1, "mean": 1.0}}
        report["aggregates"]["completed"]["by_k"]["3"][QUESTION_KEY] = {"n": 1, "mean": 1.0}
        report["population"]["result_status"][LABEL_KEY] = 1
        output = aggregate_only(report)
        self.assertClean(output)
        self.assertEqual(output["aggregates"]["completed"]["by_k"]["3"]["paper_ndcg"]["mean"], 0.482)
        self.assertEqual(set(output["aggregates"]), {"completed", "all_questions"})
        self.assertEqual(output["population"]["result_status"], {"ok": 3, "missing": 1, "failed": 0, "duplicate": 0})


if __name__ == "__main__":
    unittest.main()
