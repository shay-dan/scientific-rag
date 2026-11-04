import re
import unittest

from langchain_core.documents import Document

from app.claim_verification import (
    ClaimVerificationError,
    build_verification_inputs,
    validate_assessment_batch,
)
from app.weak_model_synthesis import (
    EvidenceClaim,
    StructuredEvidenceAnswerV2,
    build_evidence_blocks,
)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


class EvidenceMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        # These fixtures intentionally include semantic contradictions for later
        # integration tests. Stage 1A maps evidence only; it does not judge them.
        self.documents = [
            Document(
                page_content=(
                    "The validation cohort used historical data. "
                    "In held-out validation, the reported AUC was 0.841."
                ),
                metadata={"doc_id": "study-a", "title": "Historical validation study"},
            ),
            Document(
                page_content=(
                    "The investigation was retrospective and included 120 patients. "
                    "Age-stratified enrollment was not reported."
                ),
                metadata={"doc_id": "study-a", "title": "Historical validation study"},
            ),
            Document(
                page_content=(
                    "A separate prospective study included 80 adults and reported "
                    "clinical follow-up."
                ),
                metadata={"doc_id": "study-b", "title": "Prospective clinical study"},
            ),
        ]
        self.blocks = build_evidence_blocks(self.documents)

        self.incorrect_auc_claim = EvidenceClaim(
            text="The held-out validation achieved an AUC of 0.93.",
            source_id="S1",
            evidence_quote="the reported AUC was 0.841",
        )
        self.study_design_confusion_claim = EvidenceClaim(
            text="The investigation used a prospective study design.",
            source_id="S1",
            evidence_quote="The investigation was retrospective",
        )
        self.population_mismatch_claim = EvidenceClaim(
            text="The study included 120 adults in its analysis.",
            source_id="S1",
            evidence_quote="included 120 patients",
        )

    def test_maps_one_valid_claim(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="The held-out validation reported an AUC of 0.841.",
            source_id="S1",
            evidence_quote="the reported AUC was 0.841",
        )])

        mapped = build_verification_inputs(answer, self.blocks)

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].target, "claims.1")
        self.assertEqual(mapped[0].source_id, "S1")
        self.assertEqual(mapped[0].matched_source_title, "Historical validation study")
        self.assertIn("AUC was 0.841", mapped[0].matched_passage_context)

    def test_maps_multiple_valid_claims(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[
            self.incorrect_auc_claim,
            self.study_design_confusion_claim,
            self.population_mismatch_claim,
        ])

        mapped = build_verification_inputs(answer, self.blocks)

        self.assertEqual([item.target for item in mapped], ["claims.1", "claims.2", "claims.3"])
        self.assertEqual([item.source_id for item in mapped], ["S1", "S1", "S1"])

    def test_selects_the_matching_excerpt_when_one_source_has_multiple_excerpts(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[self.study_design_confusion_claim])

        mapped = build_verification_inputs(answer, self.blocks)

        self.assertIn("retrospective", mapped[0].matched_passage_context)
        self.assertNotIn("AUC was 0.841", mapped[0].matched_passage_context)

    def test_rejects_unknown_source_id(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="The study reported an AUC of 0.841.",
            source_id="S9",
            evidence_quote="the reported AUC was 0.841",
        )])

        with self.assertRaisesRegex(ClaimVerificationError, "unknown source_id S9"):
            build_verification_inputs(answer, self.blocks)

    def test_rejects_quote_that_belongs_to_a_different_source(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="The prospective study included 80 adults.",
            source_id="S1",
            evidence_quote="prospective study included 80 adults",
        )])

        with self.assertRaisesRegex(ClaimVerificationError, "not a verbatim substring of S1"):
            build_verification_inputs(answer, self.blocks)

    def test_rejects_missing_quotation(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="The study reported an external validation result.",
            source_id="S1",
            evidence_quote="quotation absent from every passage",
        )])

        with self.assertRaisesRegex(ClaimVerificationError, "not a verbatim substring of S1"):
            build_verification_inputs(answer, self.blocks)

    def test_matches_whitespace_normalized_quotation(self) -> None:
        documents = [Document(
            page_content="Held-out validation reported that AUC   was\n0.841 in the cohort.",
            metadata={"doc_id": "study-c", "title": "Whitespace study"},
        )]
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="Held-out validation reported an AUC of 0.841.",
            source_id="S1",
            evidence_quote="AUC was 0.841",
        )])

        mapped = build_verification_inputs(answer, build_evidence_blocks(documents))

        self.assertIn(_normalise(answer.claims[0].evidence_quote), _normalise(mapped[0].matched_passage_context))

    def test_claim_and_limitation_targets_are_numbered_independently(self) -> None:
        answer = StructuredEvidenceAnswerV2(
            claims=[self.incorrect_auc_claim, self.population_mismatch_claim],
            limitations=[self.study_design_confusion_claim],
        )

        mapped = build_verification_inputs(answer, self.blocks)

        self.assertEqual(
            [item.target for item in mapped],
            ["claims.1", "claims.2", "limitations.1"],
        )

    def test_long_passage_context_keeps_quote_near_the_end(self) -> None:
        quote = "The final analysis reported an AUC of 0.841."
        document = Document(
            page_content=("background evidence " * 400) + quote + " Closing sentence.",
            metadata={"doc_id": "long-study", "title": "Long study"},
        )
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="The final analysis reported an AUC of 0.841.",
            source_id="S1",
            evidence_quote=quote,
        )])

        mapped = build_verification_inputs(answer, build_evidence_blocks([document]))

        self.assertLessEqual(len(mapped[0].matched_passage_context), 2400)
        self.assertIn(quote, mapped[0].matched_passage_context)

    def test_preserves_source_identity_from_the_matched_block(self) -> None:
        answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="The prospective study included 80 adults.",
            source_id="S2",
            evidence_quote="prospective study included 80 adults",
        )])

        mapped = build_verification_inputs(answer, self.blocks)

        self.assertEqual(mapped[0].source_id, "S2")
        self.assertEqual(mapped[0].matched_source_title, "Prospective clinical study")


class AssessmentContractTests(unittest.TestCase):
    def test_all_supported_assessments_are_returned_in_target_order(self) -> None:
        result = validate_assessment_batch({"assessments": [
            {"target": "limitations.1", "verdict": "supported", "reason": "The passage directly states the limitation."},
            {"target": "claims.1", "verdict": "supported", "reason": "The passage directly states the result."},
        ]}, ["limitations.1", "claims.1"])

        self.assertEqual([item.target for item in result], ["claims.1", "limitations.1"])
        self.assertTrue(all(item.verdict.value == "supported" for item in result))

    def test_mixed_verdicts_are_preserved(self) -> None:
        result = validate_assessment_batch({"assessments": [
            {"target": "claims.1", "verdict": "contradicted", "reason": "The reported AUC is 0.841, not 0.93."},
            {"target": "claims.2", "verdict": "insufficient_evidence", "reason": "The passage does not identify adults."},
            {"target": "limitations.1", "verdict": "supported", "reason": "The passage calls the study retrospective."},
        ]}, ["claims.1", "claims.2", "limitations.1"])

        self.assertEqual(
            [item.verdict.value for item in result],
            ["contradicted", "insufficient_evidence", "supported"],
        )

    def test_duplicate_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "duplicate targets"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.1", "verdict": "supported", "reason": "First assessment."},
                {"target": "claims.1", "verdict": "supported", "reason": "Duplicate assessment."},
            ]}, ["claims.1"])

    def test_missing_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "missing assessments.*claims.2"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.1", "verdict": "supported", "reason": "Direct support."},
            ]}, ["claims.1", "claims.2"])

    def test_unexpected_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "unknown assessments.*claims.2"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.1", "verdict": "supported", "reason": "Direct support."},
                {"target": "claims.2", "verdict": "supported", "reason": "Not requested."},
            ]}, ["claims.1"])

    def test_invalid_verdict_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "invalid assessment batch"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.1", "verdict": "probably_supported", "reason": "Invalid enum."},
            ]}, ["claims.1"])

    def test_empty_response_is_rejected_when_targets_are_expected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "missing assessments.*claims.1"):
            validate_assessment_batch({"assessments": []}, ["claims.1"])

    def test_empty_expected_targets_accepts_only_an_empty_response(self) -> None:
        self.assertEqual(validate_assessment_batch({"assessments": []}, []), ())
        with self.assertRaisesRegex(ClaimVerificationError, "unknown assessments.*claims.1"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.1", "verdict": "supported", "reason": "Unexpected."},
            ]}, [])

    def test_incorrect_assessment_target_format_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "invalid assessment batch"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.0", "verdict": "supported", "reason": "Invalid target."},
            ]}, ["claims.1"])

    def test_incorrect_expected_target_format_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "invalid expected target"):
            validate_assessment_batch({"assessments": []}, ["claims.01"])

    def test_blank_reason_is_rejected(self) -> None:
        with self.assertRaisesRegex(ClaimVerificationError, "invalid assessment batch"):
            validate_assessment_batch({"assessments": [
                {"target": "claims.1", "verdict": "supported", "reason": "   "},
            ]}, ["claims.1"])


if __name__ == "__main__":
    unittest.main()
