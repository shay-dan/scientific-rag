import unittest

from langchain_core.documents import Document

from app.claim_checks import audit_answer_claims
from app.weak_model_synthesis import (
    EvidenceClaim,
    StructuredEvidenceAnswerV2,
    build_evidence_blocks,
)


def _claim(text: str, quote: str, source_id: str = "S1") -> EvidenceClaim:
    return EvidenceClaim(text=text, source_id=source_id, evidence_quote=quote)


def _audit(text: str, quote: str) -> dict:
    answer = StructuredEvidenceAnswerV2(claims=[_claim(text, quote)])
    blocks = build_evidence_blocks([
        Document(page_content=quote, metadata={"doc_id": "one", "title": "One"}),
    ])
    return audit_answer_claims(answer, blocks)["findings"][0]


class NumericClaimAuditTests(unittest.TestCase):
    def test_unrecognized_or_misaligned_measurement_context_is_unverifiable(self) -> None:
        cases = (
            ("Age was 45 years.", "Age was 45 months."),
            ("Among 120 patients, response improved.", "Response improved at 12 weeks."),
            ("The phase 3 trial enrolled patients.", "450 stable patients were enrolled."),
            (
                "COVID-19 vaccination reduced hospitalization.",
                "Hospitalization decreased among 2059 adults.",
            ),
            (
                "IL-6 signaling was reduced.",
                "Signaling decreased among 200 participants.",
            ),
            ("The dose was 12 mg/day.", "The dose was 12 mg/week."),
        )
        for text, quote in cases:
            with self.subTest(text=text, quote=quote):
                self.assertEqual(_audit(text, quote)["status"], "unverifiable")

    def test_unicode_and_implicit_signs_are_conservative(self) -> None:
        cases = (
            ("Change was -5 mg.", "Change was −5 mg.", "consistent", None),
            ("Change was -5 mg.", "Change was –5 mg.", "unverifiable", "ambiguous_sign"),
            ("Change was +5 mg.", "Change was -5 mg.", "inconsistent", "sign_mismatch"),
            (
                "Decreased by 5 mg.", "Change was -5 mg.",
                "unverifiable", "implicit_sign_direction",
            ),
            (
                "Change was +5 mg.", "Change was 5 mg.",
                "unverifiable", "implicit_sign_direction",
            ),
        )
        for text, quote, status, reason in cases:
            with self.subTest(text=text, quote=quote):
                finding = _audit(text, quote)
                self.assertEqual(finding["status"], status)
                if reason is not None:
                    self.assertEqual(finding["reason_code"], reason)

    def test_rounding_direction_and_ties_are_conservative(self) -> None:
        cases = (
            (
                "The result was 12.0 mg.", "The result was 12.05 mg.",
                "unverifiable", "ambiguous_rounding_tie",
            ),
            (
                "The change was -2.4 mg.", "The change was -2.45 mg.",
                "unverifiable", "ambiguous_rounding_tie",
            ),
            (
                "The result was 12.04 mg.", "The result was 12.0 mg.",
                "unverifiable", "claim_precision_exceeds_evidence",
            ),
            (
                "The result was 12.3 mg.", "The result was 12.34 mg.",
                "consistent", "rounded_numeric_match",
            ),
        )
        for text, quote, status, reason in cases:
            with self.subTest(text=text, quote=quote):
                finding = _audit(text, quote)
                self.assertEqual(finding["status"], status)
                self.assertEqual(finding["reason_code"], reason)

    def test_lancet_middle_dot_decimal_is_not_split_into_integers(self) -> None:
        finding = _audit(
            "The risk ratio was 0.74.",
            "The risk ratio was 0·74.",
        )
        self.assertEqual(finding["claim_values"][0]["normalized"], "0.74")
        self.assertEqual(finding["evidence_values"][0]["normalized"], "0.74")
        self.assertEqual(finding["status"], "consistent")

    def test_middle_dot_confidence_interval_endpoints_are_decimal_values(self) -> None:
        finding = _audit(
            "The confidence interval was 0.67-0.83.",
            "The confidence interval was 0·67-0·83.",
        )
        self.assertEqual(finding["claim_values"][0]["normalized"], "0.67..0.83")
        self.assertEqual(finding["evidence_values"][0]["normalized"], "0.67..0.83")
        self.assertEqual(finding["status"], "consistent")

    def test_ambiguous_numeric_formats_do_not_create_partial_values(self) -> None:
        cases = (
            ("The dose was 0,5 mg.", "The dose was 0.5 mg."),
            ("The result was 12,34 mg.", "The result was 1234 mg."),
            ("The concentration was 1.2e-3 mg.", "The concentration was 0.0012 mg."),
        )
        for text, quote in cases:
            with self.subTest(text=text, quote=quote):
                finding = _audit(text, quote)
                self.assertEqual(finding["status"], "unverifiable")
                self.assertEqual(finding["reason_code"], "ambiguous_numeric_format")
                self.assertEqual(finding["claim_values"], [])

    def test_valid_thousands_separator_is_preserved(self) -> None:
        finding = _audit("The dose was 1,000 mg.", "The dose was 1000 mg.")
        self.assertEqual(finding["status"], "consistent")
        self.assertEqual(finding["claim_values"][0]["normalized"], "1000")

    def test_unsupported_long_decimal_precision_is_unverifiable(self) -> None:
        finding = _audit(
            "The result was 123456789012345678901234567890.1 mg.",
            "The result was 123456789012345678901234567890.12 mg.",
        )
        self.assertEqual(finding["status"], "unverifiable")
        self.assertEqual(finding["reason_code"], "numeric_precision_unsupported")

    def test_reordered_separate_quantities_are_not_positionally_mismatched(self) -> None:
        finding = _audit(
            "Mortality was 12% and survival was 88%.",
            "Survival was 88% and mortality was 12%.",
        )
        self.assertEqual(finding["status"], "unverifiable")
        self.assertEqual(finding["reason_code"], "ambiguous_numeric_alignment")

    def test_equal_sequences_with_different_outcomes_are_not_semantic_agreement(self) -> None:
        finding = _audit(
            "Mortality was 12% and survival was 88%.",
            "Response was 12% and withdrawal was 88%.",
        )
        self.assertEqual(finding["status"], "unverifiable")
        self.assertEqual(finding["reason_code"], "ambiguous_numeric_alignment")

    def test_unit_parser_does_not_treat_ordinary_words_as_units(self) -> None:
        cases = (
            ("The cohort had 12 participants.", ["participant"]),
            ("The cohort had 12 patients.", ["patient"]),
            ("The estimate was between 0.1 and 0.2.", [None, None]),
            ("The dose was 12 mg.", ["mg"]),
            ("The mass was 12 g.", ["g"]),
            ("The concentration was 12 mg/dL.", ["mg/dl"]),
        )
        for text, expected_units in cases:
            with self.subTest(text=text):
                finding = _audit(text, text)
                self.assertEqual(
                    [value["unit"] for value in finding["claim_values"]],
                    expected_units,
                )

    def test_table_driven_numeric_relationships(self) -> None:
        cases = (
            (
                "exact integer", "The dose was 12 mg.", "The dose was 12 mg.",
                "consistent", "exact_numeric_match",
            ),
            (
                "redundant zeros", "The dose was 12.0 mg.", "The dose was 12.00 mg.",
                "consistent", "reformatted_numeric_match",
            ),
            (
                "same-form rounding", "The mean was 12.3 mg.", "The mean was 12.34 mg.",
                "consistent", "rounded_numeric_match",
            ),
            (
                "negative formatting", "The change was -5.0 mg.", "The change was -5 mg.",
                "consistent", "reformatted_numeric_match",
            ),
            (
                "sign mismatch", "The change was -5 mg.", "The change was 5 mg.",
                "unverifiable", "implicit_sign_direction",
            ),
            (
                "interval formatting", "The interval was 10.0-20.0 mg.",
                "The interval was 10 to 20 mg.", "consistent", "reformatted_numeric_match",
            ),
            (
                "interval endpoint mismatch", "The interval was 10-20 mg.",
                "The interval was 10-21 mg.", "inconsistent", "numeric_mismatch",
            ),
            (
                "percentage formatting", "The response rate was 12.0%.",
                "The response rate was 12 percent.", "consistent", "reformatted_numeric_match",
            ),
            (
                "decimal versus percentage", "The response rate was 0.12.",
                "The response rate was 12%.", "unverifiable", "incomparable_numeric_forms",
            ),
            (
                "fraction versus percentage", "The response rate was 3/10.",
                "The response rate was 30%.", "unverifiable", "incomparable_numeric_forms",
            ),
            (
                "ratio versus percentage", "The response ratio was 3:10.",
                "The response rate was 30%.", "unverifiable", "incomparable_numeric_forms",
            ),
            (
                "separate numerator denominator", "The response rate was 30%.",
                "There were 3 responses among 10 participants.",
                "unverifiable", "incomparable_numeric_forms",
            ),
            (
                "incompatible units", "The dose was 5 mg.", "The dose was 5 g.",
                "unverifiable", "incompatible_units",
            ),
            (
                "comparable unit expression", "The concentration was 5 mg/dL.",
                "The concentration was 5.0 MG / DL.",
                "consistent", "reformatted_numeric_match",
            ),
            (
                "missing unit context", "The dose was 5 mg.", "The reported value was 5.",
                "unverifiable", "missing_unit_context",
            ),
            (
                "missing numeric quote", "The dose was 5 mg.",
                "The dose was reported in the article.",
                "unverifiable", "missing_numeric_evidence",
            ),
            (
                "numeric mismatch", "The dose was 6 mg.", "The dose was 5 mg.",
                "inconsistent", "numeric_mismatch",
            ),
            (
                "different quantities", "Participants received 5 mg at week 4.",
                "Participants received 5 mg at week 8.",
                "unverifiable", "ambiguous_numeric_alignment",
            ),
            (
                "unrelated passage number", "The dose was 12 mg.",
                "A 2020 study reported a dose of 12 mg.",
                "unverifiable", "ambiguous_numeric_alignment",
            ),
            (
                "ambiguous alternatives", "The dose was 5 mg.",
                "The dose was either 5 mg or 6 mg.",
                "unverifiable", "ambiguous_numeric_alignment",
            ),
        )
        for name, text, quote, status, reason in cases:
            with self.subTest(name=name):
                finding = _audit(text, quote)
                self.assertEqual(finding["status"], status)
                self.assertEqual(finding["reason_code"], reason)
                self.assertEqual(finding["category"], "quotation_numeric_consistency")

    def test_non_numeric_claim_is_not_applicable(self) -> None:
        finding = _audit(
            "The study reported improved symptoms.",
            "The study reported improved symptoms.",
        )
        self.assertEqual(finding["status"], "not_applicable")
        self.assertEqual(finding["reason_code"], "no_numeric_claim")

    def test_report_contains_only_normalized_numeric_diagnostics(self) -> None:
        finding = _audit("The dose was 12.0 mg.", "Confidential passage: dose 12.00 mg.")
        serialized = str(finding)
        self.assertNotIn("Confidential passage", serialized)
        self.assertEqual(finding["claim_values"][0]["normalized"], "12")
        self.assertEqual(finding["evidence_values"][0]["normalized"], "12")
        self.assertNotIn("supported", serialized.casefold())

    def test_multiple_claims_and_limitations_keep_original_targets(self) -> None:
        blocks = build_evidence_blocks([
            Document(page_content="Dose 5 mg. Follow-up was short.",
                     metadata={"doc_id": "one", "title": "One"}),
        ])
        answer = StructuredEvidenceAnswerV2(
            claims=[_claim("The dose was 5 mg.", "Dose 5 mg")],
            limitations=[_claim("Follow-up was short.", "Follow-up was short")],
        )
        report = audit_answer_claims(answer, blocks)
        self.assertEqual([item["target"] for item in report["findings"]], [
            "claims.1", "limitations.1",
        ])
        self.assertEqual(report["summary"], {
            "consistent": 1, "inconsistent": 0, "unverifiable": 0, "not_applicable": 1,
        })

    def test_structurally_invalid_quote_is_rejected_before_audit(self) -> None:
        blocks = build_evidence_blocks([
            Document(page_content="The dose was 5 mg.",
                     metadata={"doc_id": "one", "title": "One"}),
        ])
        answer = StructuredEvidenceAnswerV2(claims=[
            _claim("The dose was 5 mg.", "invented quote"),
        ])
        with self.assertRaisesRegex(ValueError, "structurally validated"):
            audit_answer_claims(answer, blocks)


if __name__ == "__main__":
    unittest.main()
