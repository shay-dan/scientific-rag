"""R7E regressions: no real provider requests, no private benchmark material."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from app.claim_verification import (
    ClaimAssessment, ClaimVerificationBatch, ClaimVerificationError,
    NoSemanticallySupportedClaimsError,
)
from app.config import Settings
from app.cost_policy import budget_for
from app.quote_recovery import recover_answer_quotes, recover_repair_quotes
from app.schemas import ResearchMode
from app.services import PIPELINE_VERSION, ResearchService, effective_pipeline_version
from app.weak_model_synthesis import (
    EvidenceBlock, EvidenceClaim, StructuredEvidenceAnswerV2, V2RepairBatch,
    V2RepairItem, merge_v2_repairs, validate_answer_v2,
)
from scientific_rag.legacy_pipeline import ResearchAgent


def block(text, source="S1"):
    return EvidenceBlock(source_id=source, title="Test source", excerpt=text,
                         document=Document(page_content=text))


def claim(quote, source="S1", text="The study reported a measured result."):
    return EvidenceClaim(text=text, source_id=source, evidence_quote=quote)


def answer(quote, source="S1"):
    return StructuredEvidenceAnswerV2(
        claims=[claim(quote, source)], unanswered_parts=["Other results were not reported."]
    )


class QuoteRecoveryTests(unittest.TestCase):
    def test_hyphen_recovery_exact_and_non_mutating(self):
        for original, source in (
            ("The long\u2011term result was measured.", "The long-term result was measured."),
            ("The long-term result was measured.", "The long\u2011term result was measured."),
            ("The 100\u2011g sample was measured.", "The 100-g sample was measured."),
        ):
            with self.subTest(original=original):
                old = answer(original)
                old.limitations = [claim("Study design is not reported.", text="The study design is not reported.")]
                evidence = [block(source + " Study design is not reported.")]
                snapshot = old.model_dump()
                self.assertTrue(validate_answer_v2(old, evidence))
                new, audit, errors = recover_answer_quotes(old, evidence, "unicode")
                self.assertFalse(errors)
                self.assertEqual(new.claims[0].evidence_quote, source)
                self.assertIn(new.claims[0].evidence_quote, evidence[0].excerpt)
                self.assertEqual(new.claims[0].text, old.claims[0].text)
                self.assertEqual(new.claims[0].source_id, old.claims[0].source_id)
                self.assertEqual(new.limitations, old.limitations)
                self.assertEqual(new.unanswered_parts, old.unanswered_parts)
                self.assertEqual(old.model_dump(), snapshot)
                self.assertEqual(audit[0]["status"], "recovered")
                self.assertTrue(audit[0]["exact_source_match"])
                self.assertNotIn(source, str(audit))
                again, repeated, remaining = recover_answer_quotes(new, evidence, "unicode")
                self.assertEqual(again, new)
                self.assertEqual(repeated, [])
                self.assertFalse(remaining)

    def test_unicode_recovery_of_limitation(self):
        evidence = [block("The result was measured. Long-term outcomes were unknown.")]
        old = StructuredEvidenceAnswerV2(
            claims=[claim("The result was measured.")],
            limitations=[claim("Long\u2011term outcomes were unknown.",
                               text="Long-term outcomes were unknown.")],
        )
        new, audit, remaining = recover_answer_quotes(old, evidence, "unicode")
        self.assertFalse(remaining)
        self.assertEqual(new.claims, old.claims)
        self.assertEqual(new.limitations[0].evidence_quote, "Long-term outcomes were unknown.")
        self.assertEqual(new.limitations[0].text, old.limitations[0].text)
        self.assertEqual(audit[0]["target"], "limitations.1")

    def test_prohibited_scientific_equivalences(self):
        pairs = [
            ("The value was \u22125.0 units.", "The value was -5.0 units."),
            ("The value was \u20115.0 units.", "The value was -5.0 units."),
            ("The value was 1e\u20113 units.", "The value was 1e-3 units."),
            ("The interval was 3\u20115 mg.", "The interval was 3-5 mg."),
            ("The area was 4 cm2.", "The area was 4 cm\u00b2."),
            ("The molecule was CO2.", "The molecule was CO\u2082."),
            ("The coefficient was \u03b1.", "The coefficient was a."),
            ("The result was p < 0.05.", "The result was p \u2264 0.05."),
            ("The value was 4.5 mg.", "The value was 45 mg."),
            ("The value was +5 units.", "The value was -5 units."),
            ("The ABC1 identifier was observed.", "The ABC2 identifier was observed."),
        ]
        for generated, source in pairs:
            with self.subTest(generated=generated, source=source):
                old = answer(generated)
                new, audit, remaining = recover_answer_quotes(old, [block(source)], "unicode_ellipsis")
                self.assertEqual(new, old)
                self.assertTrue(remaining)
                self.assertFalse(any(e["status"] == "recovered" for e in audit))

    def test_ambiguity_in_same_or_multiple_passages(self):
        quote = "The long\u2011term result was measured."
        exact = "The long-term result was measured."
        for evidence in ([block(exact + " " + exact)], [block(exact), block(exact)]):
            with self.subTest(evidence_count=len(evidence)):
                new, audit, remaining = recover_answer_quotes(answer(quote), evidence, "unicode")
                self.assertTrue(remaining)
                self.assertEqual(new.claims[0].evidence_quote, quote)
                self.assertEqual(audit[0]["status"], "ambiguous")
                self.assertEqual(audit[0]["match_count_capped_at_two"], 2)

    def test_must_not_switch_sources(self):
        quote = "The long\u2011term result was measured."
        exact = "The long-term result was measured."
        old = answer(quote)
        _, _, remaining = recover_answer_quotes(old, [block("Different evidence."), block(exact, "S2")], "unicode")
        self.assertTrue(remaining)
        fixed, _, remaining = recover_answer_quotes(old, [block(exact), block(exact, "S2")], "unicode")
        self.assertFalse(remaining)
        self.assertEqual(fixed.claims[0].source_id, "S1")

    def test_unicode_does_not_change_current_casefold_contract(self):
        old = answer("The Study reported a measured result.")
        new, audit, remaining = recover_answer_quotes(old, [block("the study reported a measured result.")], "unicode")
        self.assertIs(old, new)
        self.assertFalse(remaining)
        self.assertEqual(audit, [])

    def test_unique_ellipsis_restores_negation_and_subgroup(self):
        left = "The adjusted analysis"
        right = "the measured outcome was significant."
        exact = left + " in patients over 60 did not establish that " + right
        for mark in ("...", "\u2026"):
            with self.subTest(mark=mark):
                old = answer(left + " " + mark + " " + right)
                _, _, initial = recover_answer_quotes(old, [block(exact)], "unicode")
                self.assertTrue(initial)
                fixed, audit, remaining = recover_answer_quotes(old, [block(exact)], "unicode_ellipsis")
                self.assertFalse(remaining)
                self.assertEqual(fixed.claims[0].evidence_quote, exact)
                self.assertIn("did not establish", fixed.claims[0].evidence_quote)
                self.assertEqual(fixed.claims[0].text, old.claims[0].text)
                self.assertEqual(audit[-1]["method"], "ellipsis")
                self.assertEqual(audit[-1]["status"], "recovered")

    def test_ellipsis_multiple_right_endpoints_and_left_anchors(self):
        left, right = "The adjusted analysis", "the measured outcome was significant."
        quote = left + " ... " + right
        for exact in (left + " found " + right + " Also " + right,
                      left + " and " + left + " found " + right):
            with self.subTest(exact=exact):
                fixed, audit, remaining = recover_answer_quotes(answer(quote), [block(exact)], "unicode_ellipsis")
                self.assertTrue(remaining)
                self.assertEqual(fixed.claims[0].evidence_quote, quote)
                self.assertEqual(audit[-1]["status"], "ambiguous")

    def test_ellipsis_bounded_and_not_cross_passage(self):
        left, right = "The adjusted analysis", "the measured outcome was significant."
        quote = left + " ... " + right
        tests = [
            (quote, [block(left + " " + "x" * 161 + " " + right)]),
            (quote, [block(left + " evidence"), block("additional " + right)]),
            ("Short ... " + right, [block("Short evidence " + right)]),
            (left + " ... middle ... " + right, [block(left + " evidence middle evidence " + right)]),
            ("A" * 300 + " ... " + "B" * 300, [block("A" * 300 + " gap " + "B" * 300)]),
        ]
        for generated, evidence in tests:
            with self.subTest(generated=generated[:30]):
                old = answer(generated)
                new, _, remaining = recover_answer_quotes(old, evidence, "unicode_ellipsis")
                self.assertEqual(new, old)
                self.assertTrue(remaining)

    def test_off_and_unrelated_errors_remain_unchanged(self):
        quotation = "The long\u2011term result was measured."
        evidence = [block("The long-term result was measured.")]
        old = answer(quotation)
        new, audit, remaining = recover_answer_quotes(old, evidence, "off")
        self.assertIs(new, old)
        self.assertEqual(audit, [])
        self.assertEqual(remaining, validate_answer_v2(old, evidence))
        unknown = answer(quotation, "S9")
        new, audit, remaining = recover_answer_quotes(unknown, evidence, "unicode")
        self.assertEqual(new, unknown)
        self.assertFalse(audit)
        self.assertTrue(remaining)

    def test_repair_replacements_only_and_strict_merge(self):
        original = answer("Unsupported quotation not present.")
        evidence = [block("The long-term result was measured.")]
        batch = V2RepairBatch(repairs=[V2RepairItem(
            target="claims.1", action="replace",
            replacement=claim("The long\u2011term result was measured."),
        )])
        snapshot = batch.model_dump()
        corrected, audit = recover_repair_quotes(batch, evidence, "unicode", {"claims.1"})
        result, _, remaining = merge_v2_repairs(original, corrected, evidence)
        self.assertFalse(remaining)
        self.assertEqual(result.claims[0].evidence_quote, evidence[0].excerpt)
        self.assertEqual(corrected.repairs[0].target, batch.repairs[0].target)
        self.assertEqual(batch.model_dump(), snapshot)
        self.assertEqual(audit[0]["status"], "recovered")
        ignored, audit = recover_repair_quotes(batch, evidence, "unicode", {"limitations.1"})
        self.assertEqual(ignored, batch)
        self.assertFalse(audit)

    def test_recovered_field_does_not_hide_other_invalid_fields(self):
        original = StructuredEvidenceAnswerV2(claims=[
            claim("The long\u2011term result was measured."),
            claim("A quotation that does not exist.", text="The second result was measured."),
        ])
        new, _, remaining = recover_answer_quotes(original, [block("The long-term result was measured.")], "unicode")
        self.assertEqual(set(remaining), {"claims.2"})
        self.assertEqual(remaining, validate_answer_v2(new, [block("The long-term result was measured.")]))


class QuoteRecoveryConfigurationTests(unittest.TestCase):
    def test_defaults_and_mode_guard(self):
        self.assertEqual(Settings().quote_recovery_mode, "off")
        self.assertEqual(effective_pipeline_version("off"), PIPELINE_VERSION)
        for mode in ("unicode", "unicode_ellipsis"):
            for semantic in ("off", "shadow"):
                with self.subTest(mode=mode, semantic=semantic):
                    with self.assertRaises(ValueError):
                        Settings(quote_recovery_mode=mode, semantic_claim_verification_mode=semantic)
                    with self.assertRaises(ValueError):
                        effective_pipeline_version(semantic, quote_recovery_mode=mode)
                    with self.assertRaises(ValueError):
                        ResearchAgent(SimpleNamespace(), quote_recovery_mode=mode,
                                      semantic_verification_mode=semantic)

    def test_cache_identity_and_persisted_metadata(self):
        versions = [effective_pipeline_version(
            "enforce", "degrade", generation_prompt_version="v16", quote_recovery_mode=mode
        ) for mode in ("off", "unicode", "unicode_ellipsis")]
        self.assertEqual(len(set(versions)), 3)
        self.assertTrue(versions[0].endswith("-genprompt-v16"))
        service = ResearchService(Settings(
            semantic_claim_verification_mode="enforce", claim_recovery_mode="degrade",
            generation_prompt_version="v16", quote_recovery_mode="unicode",
        ), redis=None)
        meta = service._metadata(ResearchMode.fast, budget_for(ResearchMode.fast), "Starting")
        self.assertEqual(meta["quote_recovery_mode"], "unicode")
        self.assertEqual(meta["retrieval_version"], versions[1])
        historical = service._metadata(ResearchMode.fast, budget_for(ResearchMode.fast), "Historical",
                                       quote_recovery_mode="off")
        self.assertNotIn("quote_recovery_mode", historical)
        self.assertEqual(historical["retrieval_version"], versions[0])

    def test_unknown_mode_refused(self):
        with self.assertRaises(ValueError):
            effective_pipeline_version("enforce", quote_recovery_mode="fuzzy")
        with self.assertRaises(ValueError):
            Settings(quote_recovery_mode="fuzzy")


class QuoteRecoveryProductionFlowTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, mode="unicode", semantic="enforce", recovery="strict"):
        docs = [
            Document(page_content="The long-term result was measured.",
                     metadata={"doc_id": "paper-one", "title": "Study one", "source": "pubmed"}),
            Document(page_content="A separate technical outcome was measured.",
                     metadata={"doc_id": "paper-two", "title": "Study two", "source": "pubmed"}),
        ]
        agent = ResearchAgent(
            SimpleNamespace(retrieve_context=AsyncMock(return_value=docs)),
            max_synthesis_subquestions=1,
            semantic_verification_mode=semantic, claim_recovery_mode=recovery,
            quote_recovery_mode=mode, generation_prompt_version="v16",
        )
        agent.decompose_query_async = AsyncMock(return_value=["focused query"])
        return agent

    def _generated(self, quote="The long\u2011term result was measured."):
        return StructuredEvidenceAnswerV2(claims=[
            claim(quote, text="The long-term clinical result was measured."),
            claim("A separate technical outcome was measured.", "S2",
                  text="A separate technical outcome was measured."),
        ])

    def _verdicts(self, first="supported", second="supported"):
        return ClaimVerificationBatch(assessments=[
            ClaimAssessment(target="claims.1", verdict=first, reason="Assessment of first claim."),
            ClaimAssessment(target="claims.2", verdict=second, reason="Assessment of second claim."),
        ])

    async def test_recovered_initial_avoids_repair_but_not_enforcement(self):
        agent = self._agent()
        generator = AsyncMock(return_value=self._generated())
        verifier = AsyncMock(return_value=self._verdicts("insufficient_evidence"))
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", generator), \
             patch("app.claim_verification.ainvoke_structured", verifier):
            report = await agent.run_research("What did the studies report?")
        generator.assert_awaited_once()
        verifier.assert_awaited_once()
        self.assertNotIn("The long-term clinical result", report["answer"])
        self.assertIn("A separate technical outcome", report["answer"])
        self.assertIn("Some statements were omitted", report["answer"])
        audit = report["claim_validation"]["quote_recovery"]
        self.assertTrue(audit["repair_avoided"])
        self.assertEqual(audit["remaining_structural_errors"], 0)
        self.assertEqual(audit["semantic_removed_fields"], 1)
        self.assertEqual(audit["semantic_verification_status"], "completed")
        self.assertFalse(report["claim_validation"]["remaining_errors"])

    async def test_repair_replacement_uses_one_repair_and_verifies(self):
        agent = self._agent()
        batch = V2RepairBatch(repairs=[V2RepairItem(
            target="claims.1", action="replace",
            replacement=claim("The long\u2011term result was measured.",
                              text="The long-term clinical result was measured."),
        )])
        generator = AsyncMock(side_effect=[self._generated("An invented quotation."), batch])
        verifier = AsyncMock(return_value=self._verdicts())
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", generator), \
             patch("app.claim_verification.ainvoke_structured", verifier):
            report = await agent.run_research("What did the studies report?")
        self.assertEqual(generator.await_count, 2)
        verifier.assert_awaited_once()
        audit = report["claim_validation"]["quote_recovery"]
        self.assertFalse(audit["repair_avoided"])
        self.assertEqual(audit["repair_attempts"][0]["status"], "recovered")
        self.assertEqual(audit["remaining_structural_errors"], 0)

    async def test_off_keeps_original_repair_path(self):
        agent = self._agent(mode="off", semantic="off")
        batch = V2RepairBatch(repairs=[V2RepairItem(
            target="claims.1", action="replace",
            replacement=claim("The long-term result was measured.",
                              text="The long-term clinical result was measured."),
        )])
        generator = AsyncMock(side_effect=[self._generated(), batch])
        verifier = AsyncMock()
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", generator), \
             patch("app.claim_verification.ainvoke_structured", verifier):
            report = await agent.run_research("What did the studies report?")
        self.assertEqual(generator.await_count, 2)
        verifier.assert_not_awaited()
        self.assertNotIn("quote_recovery", report["claim_validation"])

    async def test_semantic_rejection_of_all_claims_fails_closed(self):
        agent = self._agent()
        generator = AsyncMock(return_value=self._generated())
        verifier = AsyncMock(return_value=self._verdicts("contradicted", "insufficient_evidence"))
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", generator), \
             patch("app.claim_verification.ainvoke_structured", verifier):
            with self.assertRaises(NoSemanticallySupportedClaimsError):
                await agent.run_research("What did the studies report?")
        generator.assert_awaited_once()
        verifier.assert_awaited_once()

    async def test_verifier_failure_does_not_trigger_unverified_answer_or_extra_generation(self):
        agent = self._agent()
        generator = AsyncMock(return_value=self._generated())
        verifier = AsyncMock(side_effect=RuntimeError("provider failure"))
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", generator), \
             patch("app.claim_verification.ainvoke_structured", verifier):
            with self.assertRaises(ClaimVerificationError):
                await agent.run_research("What did the studies report?")
        generator.assert_awaited_once()
        self.assertGreaterEqual(verifier.await_count, 1)


if __name__ == "__main__":
    unittest.main()
