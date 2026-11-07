import unittest

from langchain_core.documents import Document

from app.weak_model_synthesis import (
    ClaimRepairBatch,
    ClaimRepairItem,
    CoverageAddition,
    CoverageCompletion,
    EvidenceClaim,
    StructuredEvidenceAnswer,
    StructuredEvidenceAnswerV2,
    build_evidence_blocks,
    claim_validation_errors,
    merge_local_repairs,
    merge_coverage_v2,
    render_answer,
    validate_answer,
    validate_answer_v2,
)


class WeakModelSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(page_content="Treatment reduced symptoms in the studied population.",
                     metadata={"doc_id": "P1", "title": "Study one"}),
            Document(page_content="The experiment used a retrospective test set.",
                     metadata={"doc_id": "P2", "title": "Study two"}),
        ]
        self.blocks = build_evidence_blocks(self.documents)

    def test_stable_labels_are_assigned_per_source(self):
        repeated = self.documents + [Document(page_content="Another passage", metadata={"doc_id": "P1"})]
        self.assertEqual([row.source_id for row in build_evidence_blocks(repeated)], ["S1", "S2", "S1"])

    def test_existing_production_citation_labels_are_preserved(self):
        documents = [Document(
            page_content="Quoted evidence from the selected passage.",
            metadata={"doc_id": "P9", "title": "Study", "citation_id": "S7"},
        )]
        self.assertEqual(build_evidence_blocks(documents)[0].source_id, "S7")

    def test_valid_quote_and_source_pass(self):
        answer = StructuredEvidenceAnswer(claims=[EvidenceClaim(
            text="The treatment reduced symptoms in the studied population.",
            source_id="S1", evidence_quote="reduced symptoms in the studied population",
        )])
        self.assertTrue(validate_answer(answer, self.blocks).valid)

    def test_unknown_source_and_invented_quote_fail(self):
        answer = StructuredEvidenceAnswer(claims=[
            EvidenceClaim(text="A sufficiently long unsupported claim.", source_id="S9", evidence_quote="invented evidence"),
            EvidenceClaim(text="Another sufficiently long unsupported claim.", source_id="S1", evidence_quote="not in source"),
        ])
        result = validate_answer(answer, self.blocks)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 2)

    def test_local_repair_preserves_valid_claim_and_replaces_only_invalid_claim(self):
        original = StructuredEvidenceAnswer(claims=[
            EvidenceClaim(text="Treatment reduced symptoms in the studied population.", source_id="S1",
                          evidence_quote="reduced symptoms in the studied population"),
            EvidenceClaim(text="The experiment used a retrospective test set.", source_id="S2",
                          evidence_quote="invented quote"),
        ])
        self.assertEqual(set(claim_validation_errors(original, self.blocks)), {2})
        batch = ClaimRepairBatch(repairs=[ClaimRepairItem(
            original_claim_index=2,
            action="replace",
            replacement=EvidenceClaim(
                text="The experiment used a retrospective test set.", source_id="S2",
                evidence_quote="retrospective test set",
            ),
        )])
        merged, audit, validation = merge_local_repairs(original, batch, self.blocks)
        self.assertTrue(validation.valid)
        self.assertEqual(merged.claims[0], original.claims[0])
        self.assertEqual(audit.preserved_claim_indices, (1,))
        self.assertEqual(audit.repaired_claim_indices, (2,))
        self.assertEqual(audit.claims_before, audit.claims_after)

    def test_local_drop_becomes_explicit_unanswered_part(self):
        original = StructuredEvidenceAnswer(claims=[EvidenceClaim(
            text="A sufficiently long unsupported scientific claim.", source_id="S1",
            evidence_quote="invented quote",
        )])
        batch = ClaimRepairBatch(repairs=[ClaimRepairItem(
            original_claim_index=1, action="drop", unresolved_reason="The supplied evidence does not support this claim."
        )])
        with self.assertRaisesRegex(ValueError, "cannot remove every claim"):
            merge_local_repairs(original, batch, self.blocks)

    def test_rendering_is_deterministic(self):
        answer = StructuredEvidenceAnswer(
            claims=[EvidenceClaim(text="The study used a retrospective test set.", source_id="S2",
                                  evidence_quote="retrospective test set")],
            limitations=["This does not establish clinical benefit."],
            unanswered_parts=["Prospective outcomes were not reported."],
        )
        rendered = render_answer(answer)
        self.assertIn("[S2]", rendered)
        self.assertIn("**Limitations**", rendered)
        self.assertIn("**Not established by the retrieved evidence**", rendered)

    def test_v2_requires_citations_for_limitations(self):
        answer = StructuredEvidenceAnswerV2(
            claims=[EvidenceClaim(text="Treatment reduced symptoms in the studied population.", source_id="S1",
                                  evidence_quote="reduced symptoms in the studied population")],
            limitations=[EvidenceClaim(text="The experiment used a retrospective test set.", source_id="S2",
                                       evidence_quote="retrospective test set")],
        )
        self.assertEqual(validate_answer_v2(answer, self.blocks), {})

    def test_coverage_merge_preserves_existing_and_adds_valid_limitation(self):
        original = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="Treatment reduced symptoms in the studied population.", source_id="S1",
            evidence_quote="reduced symptoms in the studied population",
        )])
        completion = CoverageCompletion(additions=[CoverageAddition(
            kind="limitation",
            claim=EvidenceClaim(text="The experiment used a retrospective test set.", source_id="S2",
                                evidence_quote="retrospective test set"),
        )])
        merged, audit, errors = merge_coverage_v2(original, completion, self.blocks)
        self.assertEqual(errors, {})
        self.assertEqual(merged.claims[0], original.claims[0])
        self.assertEqual(len(merged.limitations), 1)
        self.assertEqual(audit["added_limitations"], 1)

    def test_coverage_merge_rejects_invalid_addition_without_losing_valid_answer(self):
        original = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(
            text="Treatment reduced symptoms in the studied population.", source_id="S1",
            evidence_quote="reduced symptoms in the studied population",
        )])
        completion = CoverageCompletion(additions=[CoverageAddition(
            kind="claim",
            claim=EvidenceClaim(
                text="An unsupported but syntactically valid scientific claim.",
                source_id="S1", evidence_quote="invented evidence quote",
            ),
        )])
        merged, audit, errors = merge_coverage_v2(original, completion, self.blocks)
        self.assertEqual(errors, {})
        self.assertEqual(merged.claims, original.claims)
        self.assertEqual(len(audit["rejected_additions"]), 1)

    def test_claim_text_rejects_serialized_object_syntax(self):
        with self.assertRaisesRegex(ValueError, "plain factual sentence"):
            EvidenceClaim(
                text="EvidenceClaim(source_id='S1', evidence_quote='quoted evidence')",
                source_id="S1", evidence_quote="quoted evidence",
            )


if __name__ == "__main__":
    unittest.main()
