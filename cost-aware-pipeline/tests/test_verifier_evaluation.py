"""R5 offline verifier-evaluation harness tests (standard library only; no model calls).

Expected values are hand-calculated from the fixture tables below, not produced by
running the harness. Fixture: tests/fixtures/verifier_eval/dataset.json (10 items,
4 answers; synthetic, unreviewed).

    item   expected  notes                       mixed predictions        complete predictions
    A1-c1  S                                     S                        S
    A1-c2  S         essential                   I                        I
    A1-l1  S         limitation                  invalid ("maybe")        S
    A2-c1  C         critical, numerical         S                        S
    A2-c2  I         causal, quote not verbatim  S                        S
    A2-c3  S                                     (missing)                S
    A3-c1  S         essential, full_text        provider_failure         I
    A3-c2  I         population, full_text       I                        I
    A4-c1  C         critical, negation          C                        C
    A4-c2  S                                     duplicate (S and C)      C
"""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.verifier_evaluation import (
    PREDICTIONS_SCHEMA,
    VerifierEvaluationError,
    evaluate,
    lexical_numeric_baseline,
    predictions_from_semantic_audits,
    render_errors_csv,
    render_json,
    render_markdown,
    resolve_predictions,
    structural_only_baseline,
    validate_dataset,
)

FIXTURES = project_root / "tests" / "fixtures" / "verifier_eval"
GIT = {"sha": "a" * 40, "source": "explicit"}


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


DATASET = _load("dataset.json")
MIXED = _load("predictions_mixed.json")
COMPLETE = _load("predictions_complete.json")


def _report(predictions: dict, dataset: dict | None = None) -> dict:
    return evaluate(copy.deepcopy(dataset or DATASET), copy.deepcopy(predictions), git_provenance=GIT)


class MixedPredictionTests(unittest.TestCase):
    """Hand-calculated: 6 assessed, 1 missing, 1 invalid, 1 duplicate, 1 provider failure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _report(MIXED)

    def test_population_keeps_every_item(self) -> None:
        population = self.report["population"]
        self.assertEqual(population["items"], 10)
        self.assertEqual(population["outcomes"], {"assessed": 6, "missing": 1, "invalid_output": 1,
                                                  "duplicate_assessment": 1, "provider_failure": 1})
        self.assertEqual(population["unknown_item_predictions"], 1)   # "ghost"
        self.assertEqual(population["unattributed_predictions"], 1)   # record without item_id

    def test_confusion_matrix_has_a_no_valid_prediction_column(self) -> None:
        counts = self.report["confusion_matrix"]["counts"]
        self.assertEqual(counts["supported"], {"supported": 1, "contradicted": 0, "insufficient_evidence": 1,
                                               "no_valid_prediction": 4})
        self.assertEqual(counts["contradicted"], {"supported": 1, "contradicted": 1, "insufficient_evidence": 0,
                                                  "no_valid_prediction": 0})
        self.assertEqual(counts["insufficient_evidence"], {"supported": 1, "contradicted": 0, "insufficient_evidence": 1,
                                                           "no_valid_prediction": 0})

    def test_per_class_metrics(self) -> None:
        classes = self.report["classes"]
        # supported: TP 1; predicted 3 (A1-c1, A2-c1, A2-c2); assessed support 2; all support 6.
        self.assertEqual((classes["supported"]["precision"], classes["supported"]["recall"], classes["supported"]["f1"]),
                         (0.3333, 0.5, 0.4))
        self.assertEqual((classes["supported"]["recall_all_items"], classes["supported"]["f1_all_items"]), (0.1667, 0.2222))
        # contradicted: TP 1; predicted 1; support 2.  F1 = 2*1*0.5/1.5
        self.assertEqual((classes["contradicted"]["precision"], classes["contradicted"]["recall"],
                          classes["contradicted"]["f1"]), (1.0, 0.5, 0.6667))
        # insufficient_evidence: TP 1; predicted 2 (A1-c2, A3-c2); support 2.
        self.assertEqual((classes["insufficient_evidence"]["precision"], classes["insufficient_evidence"]["recall"],
                          classes["insufficient_evidence"]["f1"]), (0.5, 0.5, 0.5))

    def test_summary_metrics(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["accuracy_assessed"]["numerator"], 3)
        self.assertEqual(summary["accuracy_assessed"]["denominator"], 6)
        self.assertEqual(summary["macro_f1"], 0.5222)              # (0.4 + 0.6667 + 0.5) / 3
        self.assertEqual(summary["balanced_accuracy"], 0.5)        # (0.5 + 0.5 + 0.5) / 3
        self.assertEqual(summary["macro_f1_all_items"], 0.463)     # (0.2222 + 0.6667 + 0.5) / 3
        self.assertEqual(summary["balanced_accuracy_all_items"], 0.3889)  # (1/6 + 1/2 + 1/2) / 3

    def test_safety_rates(self) -> None:
        safety = self.report["safety"]
        # Non-supported with a valid prediction: A2-c1, A2-c2, A3-c2, A4-c1; accepted: A2-c1, A2-c2.
        self.assertEqual((safety["unsupported_false_acceptance"]["numerator"],
                          safety["unsupported_false_acceptance"]["denominator"]), (2, 4))
        self.assertEqual(safety["unsupported_false_acceptance"]["ci95_wilson"], [0.1500, 0.8500])
        # Critical contradictions assessed: A2-c1 (accepted), A4-c1.
        self.assertEqual((safety["critical_contradiction_acceptance"]["numerator"],
                          safety["critical_contradiction_acceptance"]["denominator"]), (1, 2))
        # Supported with a valid prediction: A1-c1, A1-c2 (rejected).
        self.assertEqual((safety["supported_false_rejection"]["numerator"],
                          safety["supported_false_rejection"]["denominator"]), (1, 2))
        self.assertEqual(safety["supported_without_valid_prediction"], 4)
        self.assertEqual(safety["non_supported_without_valid_prediction"], 0)

    def test_coverage_rates(self) -> None:
        coverage = self.report["coverage"]
        self.assertEqual(coverage["assessed_rate"]["rate"], 0.6)
        self.assertEqual(coverage["missing_assessment_rate"]["rate"], 0.1)
        self.assertEqual(coverage["invalid_output_rate"]["numerator"], 2)  # invalid label + duplicate
        self.assertEqual(coverage["duplicate_assessment_rate"]["numerator"], 1)
        self.assertEqual(coverage["provider_failure_rate"]["numerator"], 1)
        self.assertEqual(coverage["missing_assessment_rate"]["ci95_wilson"], [0.0179, 0.4042])

    def test_error_categories(self) -> None:
        categories = self.report["by_error_category"]
        self.assertEqual(categories["numerical"]["predicted_supported"]["numerator"], 1)
        self.assertEqual(categories["causal"]["accuracy"]["numerator"], 0)
        self.assertEqual(categories["population"]["accuracy"]["numerator"], 1)
        self.assertEqual(categories["negation"]["accuracy"]["numerator"], 1)
        self.assertEqual((categories["none"]["items"], categories["none"]["assessed"],
                          categories["none"]["accuracy"]["numerator"]), (6, 2, 1))

    def test_partial_batch_fails_every_answer_closed(self) -> None:
        simulation = self.report["enforcement_simulation"]
        # A1 (invalid limitation), A2 (missing claim), A3 (provider failure), A4 (duplicate).
        self.assertEqual(simulation["outcomes"], {"complete_answer": 0, "partial_answer": 0,
                                                  "no_supported_claims": 0, "failed_closed": 4})
        self.assertEqual(simulation["totals"]["targets_lost_to_fail_closed"], 10)
        self.assertEqual(simulation["totals"]["supported_lost_to_fail_closed"], 6)
        self.assertEqual(simulation["totals"]["essential_lost"], 2)        # A1-c2, A3-c1
        self.assertEqual(simulation["answer_coverage"]["rate"], 0.0)
        self.assertEqual(simulation["answers_losing_oracle_support"]["numerator"], 4)

    def test_cost_is_copied_with_its_status(self) -> None:
        self.assertEqual(self.report["cost"]["cost_status"], "partial_lower_bound")
        self.assertIsNone(self.report["cost"]["cost_usd"])


class CompletePredictionTests(unittest.TestCase):
    """Hand-calculated: all 10 items assessed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _report(COMPLETE)

    def test_classification(self) -> None:
        classes, summary = self.report["classes"], self.report["summary"]
        # supported: TP 3 (A1-c1, A1-l1, A2-c3); predicted 5; support 6.  F1 = 2*.6*.5/1.1
        self.assertEqual((classes["supported"]["precision"], classes["supported"]["recall"], classes["supported"]["f1"]),
                         (0.6, 0.5, 0.5455))
        # contradicted: TP 1; predicted 2 (A4-c1, A4-c2).   insufficient: TP 1; predicted 3.
        self.assertEqual(classes["contradicted"]["f1"], 0.5)
        self.assertEqual((classes["insufficient_evidence"]["precision"], classes["insufficient_evidence"]["f1"]),
                         (0.3333, 0.4))
        self.assertEqual(summary["macro_f1"], 0.4818)              # (0.5455 + 0.5 + 0.4) / 3
        self.assertEqual(summary["balanced_accuracy"], 0.5)
        self.assertEqual(summary["macro_f1"], summary["macro_f1_all_items"])  # nothing missing
        self.assertEqual(self.report["safety"]["supported_false_rejection"]["numerator"], 3)

    def test_enforcement_simulation(self) -> None:
        simulation = self.report["enforcement_simulation"]
        self.assertEqual(simulation["outcomes"], {"complete_answer": 1, "partial_answer": 1,
                                                  "no_supported_claims": 2, "failed_closed": 0})
        self.assertEqual(simulation["oracle_outcomes"], {"complete_answer": 1, "partial_answer": 3,
                                                         "no_supported_claims": 0, "failed_closed": 0})
        self.assertEqual(simulation["totals"], {
            "supported_retained": 3,             # A1-c1, A1-l1, A2-c3
            "supported_claims_retained": 2,      # A1-c1, A2-c3
            "supported_limitations_retained": 1,  # A1-l1
            "unsupported_retained": 2,           # A2-c1, A2-c2
            "critical_contradictions_retained": 1,  # A2-c1
            "supported_removed": 3,              # A1-c2, A3-c1, A4-c2
            "supported_lost_to_fail_closed": 0,
            "essential_lost": 2,                 # A1-c2, A3-c1
            "claims_removed": 5,                 # A1: c2; A3: c1, c2; A4: c1, c2
            "limitations_removed": 0,
            "targets_lost_to_fail_closed": 0,
        })
        # Claims only: 5 expected-supported claims (A1-l1 is a limitation), 2 delivered.
        self.assertEqual((simulation["claim_coverage"]["numerator"], simulation["claim_coverage"]["denominator"]), (2, 5))
        self.assertEqual((simulation["supported_target_coverage"]["numerator"],
                          simulation["supported_target_coverage"]["denominator"]), (3, 6))
        self.assertEqual(simulation["answer_coverage"]["rate"], 0.5)
        self.assertEqual(simulation["answers_without_usable_claims"]["numerator"], 2)
        self.assertEqual(simulation["incomplete_answer_rate"]["numerator"], 1)
        self.assertEqual(simulation["answers_losing_oracle_support"]["numerator"], 2)  # A3, A4
        self.assertEqual((simulation["unsupported_retention"]["numerator"],
                          simulation["unsupported_retention"]["denominator"]), (2, 4))
        per_answer = {row["answer_id"]: row for row in simulation["per_answer"]}
        self.assertEqual(per_answer["A1"]["retained_targets"], ["claims.1", "limitations.1"])
        self.assertEqual(per_answer["A1"]["removed_targets"], ["claims.2"])


class BaselineTests(unittest.TestCase):
    def test_structural_only_baseline_accepts_every_verbatim_quote(self) -> None:
        report = _report(structural_only_baseline(validate_dataset(DATASET)))
        classes, safety = report["classes"], report["safety"]
        # Everything supported except A2-c2 (quote not verbatim -> insufficient_evidence).
        self.assertEqual(report["confusion_matrix"]["counts"]["contradicted"]["supported"], 2)
        self.assertEqual((classes["supported"]["precision"], classes["supported"]["recall"], classes["supported"]["f1"]),
                         (0.6667, 1.0, 0.8))
        self.assertIsNone(classes["contradicted"]["precision"])   # never predicted: 0/0
        self.assertEqual(classes["contradicted"]["f1"], 0.0)       # has expected items, no true positive
        self.assertEqual(classes["insufficient_evidence"]["f1"], 0.6667)
        self.assertEqual(report["summary"]["macro_f1"], 0.4889)   # (0.8 + 0 + 0.6667) / 3
        self.assertEqual(safety["unsupported_false_acceptance"]["rate"], 0.75)       # 3 / 4
        self.assertEqual(safety["critical_contradiction_acceptance"]["rate"], 1.0)   # 2 / 2
        self.assertEqual(safety["supported_false_rejection"]["rate"], 0.0)
        simulation = report["enforcement_simulation"]
        self.assertEqual(simulation["outcomes"]["complete_answer"], 3)
        self.assertEqual(simulation["outcomes"]["partial_answer"], 1)
        self.assertEqual(simulation["totals"]["unsupported_retained"], 3)
        self.assertTrue(any("uncalibrated baseline" in warning for warning in report["warnings"]))

    def test_lexical_numeric_baseline_rules(self) -> None:
        verdicts = {record["item_id"]: record["verdict"]
                    for record in lexical_numeric_baseline(validate_dataset(DATASET))["predictions"]}
        self.assertEqual(verdicts["A2-c1"], "insufficient_evidence")  # 18 absent from evidence
        self.assertEqual(verdicts["A2-c2"], "insufficient_evidence")  # quote not verbatim
        self.assertEqual(verdicts["A4-c1"], "contradicted")           # negation only in the quote
        self.assertEqual(verdicts["A4-c2"], "supported")
        self.assertEqual(verdicts["A3-c2"], "insufficient_evidence")  # 80 absent from evidence
        self.assertEqual(verdicts["A1-c1"], "supported")
        causal = copy.deepcopy(DATASET)
        causal["items"] = [{**causal["items"][5], "item_id": "causal", "claim_text": "Statin use prevents dementia in older adults."}]
        self.assertEqual(lexical_numeric_baseline(validate_dataset(causal))["predictions"][0]["verdict"],
                         "insufficient_evidence")


class EdgeCaseTests(unittest.TestCase):
    def test_empty_dataset_reports_without_division_errors(self) -> None:
        empty = {**DATASET, "items": []}
        report = _report({"schema": PREDICTIONS_SCHEMA, "predictor": {"name": "x", "kind": "model"}, "predictions": []}, empty)
        self.assertEqual(report["population"]["items"], 0)
        self.assertIsNone(report["coverage"]["assessed_rate"]["rate"])
        self.assertIsNone(report["coverage"]["assessed_rate"]["ci95_wilson"])
        self.assertIsNone(report["summary"]["macro_f1"])
        self.assertEqual(report["enforcement_simulation"]["answers"], 0)
        self.assertTrue(any("empty" in warning for warning in report["warnings"]))
        render_markdown(report)

    def test_zero_denominators_are_none_not_zero(self) -> None:
        only_supported = {**DATASET, "items": [item for item in DATASET["items"] if item["expected_verdict"] == "supported"]}
        report = _report(COMPLETE, only_supported)
        self.assertIsNone(report["safety"]["unsupported_false_acceptance"]["rate"])
        self.assertIsNone(report["safety"]["critical_contradiction_acceptance"]["rate"])
        self.assertEqual(report["summary"]["classes_without_expected_items"], ["contradicted", "insufficient_evidence"])
        self.assertEqual(report["summary"]["classes_in_macro"], ["supported"])
        self.assertEqual(report["summary"]["balanced_accuracy"], 0.5)  # supported recall 3/6 only

    def test_duplicate_item_ids_and_targets_are_rejected(self) -> None:
        duplicate_id = {**DATASET, "items": [DATASET["items"][0], DATASET["items"][0]]}
        with self.assertRaisesRegex(VerifierEvaluationError, "duplicate item_id"):
            validate_dataset(duplicate_id)
        duplicate_target = {**DATASET, "items": [DATASET["items"][0], {**DATASET["items"][1], "target": "claims.1"}]}
        with self.assertRaisesRegex(VerifierEvaluationError, "duplicate target"):
            validate_dataset(duplicate_target)

    def test_malformed_records_and_contract_violations(self) -> None:
        resolved, integrity = resolve_predictions(validate_dataset(DATASET), {
            "schema": PREDICTIONS_SCHEMA, "predictions": [
                "not-a-dict", {"item_id": "A1-c1", "status": "assessed"},
                {"item_id": "A1-c2", "status": "assessed", "verdict": "SUPPORTED"},
                {"item_id": "A1-l1", "status": "weird", "verdict": "supported"}]})
        self.assertEqual(integrity["unattributed_predictions"], 1)
        self.assertEqual({resolved[i]["outcome"] for i in ("A1-c1", "A1-c2", "A1-l1")}, {"invalid_output"})
        with self.assertRaises(VerifierEvaluationError):
            resolve_predictions(validate_dataset(DATASET), {"schema": "other", "predictions": []})
        for bad in ({**DATASET, "schema": "x"}, {**DATASET, "label_scheme": "pubmedqa"},
                    {**DATASET, "items": [{**DATASET["items"][0], "expected_verdict": "maybe"}]},
                    {**DATASET, "items": [{**DATASET["items"][0], "target": "answer.1"}]},
                    {**DATASET, "items": [{**DATASET["items"][0], "label_provenance": "human_reviewed"}]}):
            with self.subTest(bad=str(bad)[:60]):
                with self.assertRaises(VerifierEvaluationError):
                    validate_dataset(bad)


class ExternalAdapterContractTests(unittest.TestCase):
    def test_foreign_label_schemes_need_a_declared_reviewed_mapping(self) -> None:
        adapted = {**DATASET, "source_label_scheme": "pubmedqa_yes_no_maybe"}
        with self.assertRaisesRegex(VerifierEvaluationError, "never translated implicitly"):
            validate_dataset(adapted)
        declared = {**adapted, "label_mapping": {"mapping_version": "m1", "reviewed_by": "reviewer-a",
                                                 "rules": "per-item human judgement; no automatic yes/no/maybe mapping"}}
        report = _report(COMPLETE, declared)
        self.assertEqual(report["provenance"]["dataset"]["source_label_scheme"], "pubmedqa_yes_no_maybe")
        self.assertEqual(report["provenance"]["dataset"]["label_mapping_version"], "m1")

    def test_synthetic_labels_are_never_reported_as_gold(self) -> None:
        report = _report(COMPLETE)
        self.assertEqual(report["provenance"]["dataset"]["label_provenance"], {"synthetic": 10})
        self.assertTrue(any("not evidence of verifier accuracy" in warning for warning in report["warnings"]))

    def test_shadow_audits_import_as_predictions(self) -> None:
        audits = {
            "A1": {"status": "completed", "assessments": [
                {"original_target_id": "claims.1", "verdict": "supported", "reason": "ok"},
                {"original_target_id": "claims.2", "verdict": "insufficient_evidence", "reason": "no"},
                {"original_target_id": "limitations.1", "verdict": "supported", "reason": "ok"}]},
            "A3": {"mode": "shadow", "status": "error", "error_type": "SemanticVerificationBudgetError"},
        }
        predictions = predictions_from_semantic_audits(validate_dataset(DATASET), audits,
                                                       {"name": "shadow-run", "provider": "gemini", "model": "m"})
        report = _report(predictions)
        self.assertEqual(report["population"]["outcomes"]["assessed"], 3)
        self.assertEqual(report["population"]["outcomes"]["provider_failure"], 2)
        self.assertEqual(report["population"]["outcomes"]["missing"], 5)   # A2, A4 had no audit
        self.assertEqual(report["provenance"]["predictor"]["kind"], "model")


class ReproducibilityTests(unittest.TestCase):
    def test_reports_are_identical_across_runs_and_input_order(self) -> None:
        shuffled_dataset, shuffled_predictions = copy.deepcopy(DATASET), copy.deepcopy(COMPLETE)
        random.Random(5).shuffle(shuffled_dataset["items"])
        first = render_json(_report(COMPLETE))
        self.assertEqual(first, render_json(_report(COMPLETE)))
        self.assertEqual(first, render_json(_report(shuffled_predictions, shuffled_dataset)))
        self.assertEqual(render_markdown(_report(MIXED)), render_markdown(_report(MIXED)))
        self.assertEqual(render_errors_csv(DATASET, MIXED), render_errors_csv(shuffled_dataset, MIXED))

    def test_errors_csv_lists_misclassified_and_unassessed_items(self) -> None:
        rows = render_errors_csv(DATASET, MIXED).splitlines()[1:]
        self.assertEqual([row.split(",")[0] for row in rows],
                         ["A1-c2", "A1-l1", "A2-c1", "A2-c2", "A2-c3", "A3-c1", "A4-c2"])

    def test_cli_regenerates_identical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for index in range(2):
                out_dir = Path(directory) / f"out{index}"
                subprocess.run([sys.executable, str(project_root / "scripts" / "evaluate_verifier.py"),
                                "--dataset", str(FIXTURES / "dataset.json"),
                                "--predictions", str(FIXTURES / "predictions_mixed.json"),
                                "--out-dir", str(out_dir), "--git-sha", "b" * 40],
                               check=True, capture_output=True, text=True)
                outputs.append({name: (out_dir / name).read_text(encoding="utf-8")
                                for name in ("report.json", "report.md", "errors.csv")})
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(json.loads(outputs[0]["report.json"])["population"]["items"], 10)
            baseline_dir = Path(directory) / "baseline"
            subprocess.run([sys.executable, str(project_root / "scripts" / "evaluate_verifier.py"),
                            "--dataset", str(FIXTURES / "dataset.json"), "--baseline", "structural_only",
                            "--out-dir", str(baseline_dir), "--git-sha", "b" * 40],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((baseline_dir / "report.json").read_text())["provenance"]["predictor"]["name"],
                             "structural_only")


class IsolationTests(unittest.TestCase):
    def test_production_code_does_not_import_the_harness(self) -> None:
        for path in [*(project_root / "app").glob("*.py"), *(project_root / "src" / "scientific_rag").glob("*.py")]:
            if path.name == "verifier_evaluation.py":
                continue
            with self.subTest(module=path.name):
                self.assertNotIn("verifier_evaluation", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
