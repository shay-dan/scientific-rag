import unittest
import json
from pathlib import Path

from app.silver_dataset_validation import (
    CaseJudgment,
    ClaimJudgment,
    SourceDocument,
    best_evidence_excerpt,
    excerpt_verification,
    normalize_judgments,
    normalize_scientific_typography,
    rebuild_case,
    resolve_source_id,
    validate_rebuilt_cases,
)


class SilverDatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.document = SourceDocument(
            source_id="PMID:1",
            title="A useful study",
            abstract="The intervention reduced risk by 20% in adults. Adverse events were uncommon.",
            url="https://example.test/1",
        )
        self.case = {
            "id": "case-1",
            "question": "Does it work?",
            "reference_answer": "Old generated answer.",
            "reference_claims": ["The intervention reduced risk by 20%.", "It cured every patient.", "Adverse events were uncommon."],
            "reference_context": [{"source_id": "PMID:1", "excerpt": "The intervention reduced risk by 20% in adults."}],
        }

    def test_verifies_normalized_verbatim_excerpt(self):
        result = excerpt_verification("The intervention reduced risk by 20% in adults.", self.document)
        self.assertEqual("verbatim_verified", result["status"])

    def test_evidence_excerpt_comes_from_original_document(self):
        excerpt = best_evidence_excerpt("The intervention reduced risk by 20%.", self.document)
        self.assertIn("reduced risk by 20%", excerpt)
        self.assertIn(excerpt, self.document.abstract)

    def test_missing_judgment_becomes_insufficient(self):
        judgment = CaseJudgment(judgments=[
            ClaimJudgment(claim_index=1, decision="supported", confidence=0.9, source_ids=["PMID:1"], rationale="Directly stated.")
        ])
        normalized = normalize_judgments(self.case, judgment)
        self.assertEqual(3, len(normalized))
        self.assertEqual("insufficient", normalized[1]["decision"])
        self.assertFalse(normalized[1]["judge_returned"])

    def test_source_id_prefix_omission_is_resolved_without_fuzzy_guessing(self):
        self.assertEqual("PMID:1", resolve_source_id("1", {"PMID:1": self.document}))
        self.assertIsNone(resolve_source_id("2", {"PMID:1": self.document}))

    def test_scientific_typography_is_repaired_without_rewriting_meaning(self):
        value = normalize_scientific_typography("Genomewide CRISPRCas9 screens improve Tcell fitness in CART studies.")
        self.assertEqual("genome-wide CRISPR-Cas9 screens improve T cell fitness in CAR-T studies.", value)
        self.assertEqual("CAR-T cell", normalize_scientific_typography("CAR\u2011T\u00a0cell"))

    def test_version_three_has_twelve_eligible_cases(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "evaluation" / "pilot-silver-v3.json").read_text(encoding="utf-8"))
        self.assertEqual("pilot-silver-v3", payload["dataset_name"])
        self.assertEqual(12, len(payload["cases"]))
        self.assertTrue(all(case["eligible_for_benchmark"] for case in payload["cases"]))
        for case in payload["cases"]:
            self.assertGreaterEqual(len(case["reference_claims"]), 3)
            self.assertEqual(len(case["reference_claims"]), len(case["reference_context"]))

    def test_rebuilt_case_excerpts_are_contiguous_and_hash_verified(self):
        judgments = [
            {"claim_index": index, "decision": "supported", "confidence": 0.95, "source_ids": ["PMID:1"], "rationale": "Direct."}
            for index in range(1, 4)
        ]
        rebuilt = rebuild_case(self.case, judgments, {"PMID:1": self.document})
        self.assertEqual([], validate_rebuilt_cases([rebuilt], {"PMID:1": self.document}))

    def test_rebuild_removes_unsupported_claims_without_model_rewrite(self):
        judgments = [
            {"claim_index": 1, "decision": "supported", "confidence": 0.95, "source_ids": ["PMID:1"], "rationale": "Direct."},
            {"claim_index": 2, "decision": "contradicted", "confidence": 0.99, "source_ids": ["PMID:1"], "rationale": "Not supported."},
            {"claim_index": 3, "decision": "supported", "confidence": 0.95, "source_ids": ["PMID:1"], "rationale": "Direct."},
        ]
        repaired = rebuild_case(self.case, judgments, {"PMID:1": self.document}, min_supported_claims=2)
        self.assertTrue(repaired["eligible_for_benchmark"])
        self.assertNotIn("cured every patient", repaired["reference_answer"])
        self.assertEqual(2, len(repaired["reference_claims"]))
        self.assertEqual("Old generated answer.", repaired["reference_answer_original"])


if __name__ == "__main__":
    unittest.main()
