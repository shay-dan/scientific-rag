"""R6B review reproductions: owner-side reference identity, annotation integrity, label
mappings, review status, and whole-answer metrics (PR #14 review at 8209d33).

Synthetic only (archives built at run time by ``make_handoff``). The exported claims are
fc-...01 (lexical baseline: supported) and fc-...02 (lexical baseline: contradicted).
"""
import copy
import json
import unittest

from app.benchmark import owner_scoring
from app.benchmark.handoff import import_handoff
from app.benchmark.owner_scoring import score_export
from tests.test_benchmark_r6b import TempCase, _id, default_files, make_handoff

FC1, FC2 = _id("fc", 1), _id("fc", 2)
BENCHMARK_HASH = "b" * 64   # owner's full (labelled) benchmark hash: never the label-free handoff hash

NATIVE = {"label_scheme": "verifier_verdict_v1"}
SCIFACT_MAPPING = {
    "mapping_id": "scifact-claims", "source_label_scheme": "scifact_v1", "target_scheme": "verifier_verdict_v1",
    "mapping_version": "2026-11-01", "reviewed_by": "owner-reviewer", "review_status": "reviewed",
    "rules": {"SUPPORT": "supported", "CONTRADICT": "contradicted", "NOINFO": "insufficient_evidence"},
}


def claim_ref(input_id, verdict, population="A", **extra):
    return {"input_id": input_id, "task": "claim_verification", "population": population,
            "label_provenance": "human_authored", "reviewer_status": "unreviewed",
            "reference": {**NATIVE, "verdict": verdict}, **extra}


class ReviewCase(TempCase):
    def setUp(self):
        super().setUp()
        self.export = self.export_for()

    def references(self, records, *, mappings=(), answers=None, export=None, **overrides):
        dataset = (export or self.export)["run"]["dataset"]
        refs = {"schema": owner_scoring.REFERENCE_SCHEMA, "dataset_version": dataset["dataset_version"],
                "handoff": {"dataset_hash": dataset["dataset_hash"], "manifest_sha256": dataset["manifest_sha256"]},
                "benchmark": {"benchmark_hash": BENCHMARK_HASH},
                "label_mappings": [copy.deepcopy(m) for m in mappings], "records": records}
        if answers is not None:
            refs["answers"] = answers
        refs.update(overrides)
        return refs


class ReferenceIdentityTests(ReviewCase):
    def test_references_from_another_handoff_with_the_same_version_are_refused(self):
        files = default_files()
        files["eval_inputs/open_discovery_questions.jsonl"][1]["question"] = "A different handoff question?"
        other = self.dir / "other"
        other.mkdir()
        _, _, other_report = import_handoff(make_handoff(other, files))
        self.assertEqual(other_report["handoff"]["dataset_version"], self.export["run"]["dataset"]["dataset_version"])
        refs = self.references([claim_ref(FC1, "supported")])
        refs["handoff"] = {"dataset_hash": other_report["handoff"]["dataset_hash"],
                           "manifest_sha256": other_report["handoff"]["manifest_sha256"]}
        with self.assertRaisesRegex(ValueError, "handoff"):
            score_export(self.export, refs)

    def test_unbound_v1_references_are_refused(self):
        refs = self.references([claim_ref(FC1, "supported")], schema="scientific-rag-benchmark-references/v1")
        with self.assertRaisesRegex(ValueError, "v1"):
            score_export(self.export, refs)

    def test_handoff_binding_is_required(self):
        refs = self.references([claim_ref(FC1, "supported")])
        del refs["handoff"]
        with self.assertRaisesRegex(ValueError, "handoff"):
            score_export(self.export, refs)

    def test_benchmark_hash_is_kept_distinct_from_the_handoff_hash(self):
        refs = self.references([claim_ref(FC1, "supported")])
        refs["benchmark"] = {"benchmark_hash": self.export["run"]["dataset"]["dataset_hash"]}
        with self.assertRaisesRegex(ValueError, "benchmark"):
            score_export(self.export, refs)
        scores = score_export(self.export, self.references([claim_ref(FC1, "supported")]))
        self.assertEqual(scores["benchmark_hash"], BENCHMARK_HASH)
        self.assertEqual(scores["handoff"]["dataset_hash"], self.export["run"]["dataset"]["dataset_hash"])


class AnnotationIntegrityTests(ReviewCase):
    def test_duplicate_references_are_refused_not_overwritten(self):
        refs = self.references([claim_ref(FC1, "supported"), claim_ref(FC1, "supported")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_export(self.export, refs)

    def test_conflicting_labels_for_one_prediction_are_refused(self):
        refs = self.references([claim_ref(FC1, "supported"), claim_ref(FC1, "contradicted", population="B")])
        with self.assertRaisesRegex(ValueError, "duplicate|conflict"):
            score_export(self.export, refs)

    def test_owner_key_and_input_id_resolving_to_one_prediction_conflict(self):
        refs = self.references([claim_ref(FC1, "supported"), claim_ref("owner-key-1", "contradicted")])
        with self.assertRaisesRegex(ValueError, "conflict"):
            score_export(self.export, refs, id_map={FC1: "owner-key-1"})

    def test_id_map_must_be_one_to_one(self):
        refs = self.references([claim_ref("owner-key-1", "supported")])
        with self.assertRaisesRegex(ValueError, "id_map"):
            score_export(self.export, refs, id_map={FC1: "owner-key-1", FC2: "owner-key-1"})

    def test_review_fields_are_required_not_defaulted(self):
        record = claim_ref(FC1, "supported")
        del record["reviewer_status"]
        with self.assertRaisesRegex(ValueError, "reviewer_status"):
            score_export(self.export, self.references([record]))

    def test_references_without_a_prediction_count_as_missing(self):
        refs = self.references([claim_ref(FC1, "supported"), claim_ref(_id("fc", 7), "contradicted")])
        report = score_export(self.export, refs)
        claims = report["by_population"]["A"]["tasks"]["claim_verification"]
        self.assertEqual(report["references_without_prediction"], 1)
        self.assertEqual(claims["status_counts"].get("missing"), 1)


class LabelMappingTests(ReviewCase):
    def _source_ref(self, input_id, label, mapping_id="scifact-claims"):
        return {"input_id": input_id, "task": "claim_verification", "population": "A",
                "label_provenance": "human_authored", "reviewer_status": "unreviewed",
                "reference": {"source_label_scheme": "scifact_v1", "source_label": label,
                              "label_mapping": mapping_id}}

    def test_source_labels_without_a_declared_mapping_are_refused(self):
        with self.assertRaisesRegex(ValueError, "mapping"):
            score_export(self.export, self.references([self._source_ref(FC1, "SUPPORT")]))

    def test_unreviewed_or_unversioned_mappings_are_refused(self):
        for field, value in (("review_status", "unreviewed"), ("reviewed_by", ""), ("mapping_version", None),
                             ("target_scheme", "pubmedqa_decision")):
            mapping = dict(SCIFACT_MAPPING, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mapping"):
                score_export(self.export, self.references([self._source_ref(FC1, "SUPPORT")], mappings=[mapping]))

    def test_mapping_must_cover_the_label_and_map_to_verifier_verdicts(self):
        bad_rule = dict(SCIFACT_MAPPING, rules={"SUPPORT": "yes"})
        with self.assertRaisesRegex(ValueError, "mapping"):
            score_export(self.export, self.references([self._source_ref(FC1, "SUPPORT")], mappings=[bad_rule]))
        with self.assertRaisesRegex(ValueError, "mapping"):
            score_export(self.export, self.references([self._source_ref(FC1, "MAYBE")], mappings=[SCIFACT_MAPPING]))

    def test_native_verdict_needs_the_verifier_scheme_and_cannot_mix_with_source_labels(self):
        record = claim_ref(FC1, "supported")
        record["reference"] = {"verdict": "supported", "source_label_scheme": "pubmedqa"}
        with self.assertRaisesRegex(ValueError, "mapping|label_scheme"):
            score_export(self.export, self.references([record]))

    def test_reviewed_mapping_is_applied_and_recorded(self):
        refs = self.references([self._source_ref(FC1, "SUPPORT"), self._source_ref(FC2, "SUPPORT")],
                               mappings=[SCIFACT_MAPPING])
        claims = score_export(self.export, refs)["by_population"]["A"]["tasks"]["claim_verification"]
        source = claims["by_label_source"]["scifact-claims"]
        self.assertEqual(source["label_mapping"]["mapping_version"], "2026-11-01")
        r5 = source["r5_report"]
        self.assertEqual(r5["provenance"]["dataset"]["label_mapping_version"], "2026-11-01")
        self.assertEqual(r5["confusion_matrix"]["counts"]["supported"]["supported"], 1)
        self.assertEqual(r5["confusion_matrix"]["counts"]["supported"]["contradicted"], 1)

    def test_expert_source_annotations_are_not_marked_independently_reviewed(self):
        report = score_export(self.export, self.references([claim_ref(FC1, "supported")]))
        population = report["by_population"]["A"]
        self.assertEqual(population["independently_human_reviewed"], {"records": 0, "of": 1})
        r5 = population["tasks"]["claim_verification"]["by_label_source"]["native"]["r5_report"]
        self.assertEqual(r5["provenance"]["dataset"]["reviewer_status"], {"unreviewed": 1})
        self.assertTrue(any("human-reviewed" in warning for warning in r5["warnings"]))

    def test_human_reviewed_requires_reviewed_status(self):
        record = claim_ref(FC1, "supported", label_provenance="human_reviewed")
        with self.assertRaisesRegex(ValueError, "human_reviewed"):
            score_export(self.export, self.references([record]))
        record["reviewer_status"] = "reviewed"
        population = score_export(self.export, self.references([record]))["by_population"]["A"]
        self.assertEqual(population["independently_human_reviewed"], {"records": 1, "of": 1})


class WholeAnswerMetricTests(ReviewCase):
    def test_two_claims_of_one_answer_are_not_scored_as_two_answers(self):
        """Claim-only references carry no answer grouping: whole-answer metrics are unavailable."""
        refs = self.references([claim_ref(FC1, "supported"), claim_ref(FC2, "supported")])
        claims = score_export(self.export, refs)["by_population"]["A"]["tasks"]["claim_verification"]
        r5 = claims["r5_report"] if "r5_report" in claims else claims["by_label_source"]["native"]["r5_report"]
        simulation = r5["enforcement_simulation"]
        self.assertNotEqual(simulation.get("answers"), 2, "two claims were simulated as two independent answers")
        self.assertEqual(simulation["status"], "unavailable")
        self.assertIn("answer grouping", simulation["reason"])
        for key in ("answers", "answer_coverage", "outcomes", "claim_coverage"):
            self.assertNotIn(key, simulation)
        self.assertIsNone(r5["provenance"]["dataset"]["answers"])
        # claim-level metrics stay
        self.assertEqual(r5["confusion_matrix"]["counts"]["supported"]["supported"], 1)
        self.assertIn("unsupported_false_acceptance", r5["safety"])
        self.assertIn("assessed_rate", r5["coverage"])
        self.assertIn("f1", r5["classes"]["supported"])

    def test_declared_answer_grouping_enables_the_real_simulation(self):
        answers = {"answer-1": {"targets": ["claims.1", "claims.2"]}}
        refs = self.references([
            claim_ref(FC1, "supported", answer={"answer_id": "answer-1", "target": "claims.1", "essential": True}),
            claim_ref(FC2, "supported", answer={"answer_id": "answer-1", "target": "claims.2"}),
        ], answers=answers)
        r5 = score_export(self.export, refs)["by_population"]["A"]["tasks"]["claim_verification"]["by_label_source"]["native"]["r5_report"]
        simulation = r5["enforcement_simulation"]
        self.assertEqual(simulation["answers"], 1)
        self.assertEqual(simulation["outcomes"]["partial_answer"], 1)     # fc2 predicted contradicted -> removed
        self.assertEqual(simulation["totals"]["supported_removed"], 1)
        self.assertEqual(r5["provenance"]["dataset"]["answers"], 1)

    def test_invalid_answer_grouping_is_refused(self):
        good = {"answer-1": {"targets": ["claims.1", "claims.2"]}}
        cases = {
            "undeclared answer": ([("answer-1", "claims.1"), ("answer-2", "claims.1")], good),
            "target missing from the references": ([("answer-1", "claims.1")], good),
            "duplicate target": ([("answer-1", "claims.1"), ("answer-1", "claims.1")],
                                 {"answer-1": {"targets": ["claims.1"]}}),
            "invalid target": ([("answer-1", "claim-one"), ("answer-1", "claims.2")],
                               {"answer-1": {"targets": ["claim-one", "claims.2"]}}),
            "no claim target": ([("answer-1", "limitations.1"), ("answer-1", "limitations.2")],
                                {"answer-1": {"targets": ["limitations.1", "limitations.2"]}}),
        }
        for name, (groups, answers) in cases.items():
            records = [claim_ref(input_id, "supported", answer={"answer_id": a, "target": t})
                       for input_id, (a, t) in zip((FC1, FC2), groups)]
            with self.subTest(name), self.assertRaisesRegex(ValueError, "answer"):
                score_export(self.export, self.references(records, answers=answers))

    def test_partial_grouping_is_refused(self):
        refs = self.references([
            claim_ref(FC1, "supported", answer={"answer_id": "answer-1", "target": "claims.1"}),
            claim_ref(FC2, "supported"),
        ], answers={"answer-1": {"targets": ["claims.1"]}})
        with self.assertRaisesRegex(ValueError, "answer"):
            score_export(self.export, refs)


if __name__ == "__main__":
    unittest.main()
