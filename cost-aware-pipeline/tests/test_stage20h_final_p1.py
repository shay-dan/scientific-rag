"""Pre-fix, independently specified synthetic controls; not scientific gold labels.

Questions/expected decisions were fixed before implementation. No live discovery,
screening, generation or verification is used by these tests.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.documents import Document
from scientific_rag.domain import QuestionEntity
from scientific_rag.query_quality import extract_anchors, topical_documents, build_pubmed_query
from scientific_rag.evidence_adequacy import assess_adequacy
from scientific_rag.legacy_pipeline import MinerService, Orchestrator, ResearchAgent


def document(text, key="study", **metadata):
    return Document(page_content=text, metadata={"doc_id": key, **metadata})


# Independent unseen questions (distinct from the supplied review examples).
ENTITY_CASES = (
    ("Is lacosamide effective for Dravet syndrome in adolescents?",
     {"lacosamide": "intervention", "dravet syndrome": "disease", "adolescents": "population"}),
    ("Compare brivaracetam versus perampanel for pain in adolescents with Dravet syndrome.",
     {"brivaracetam": "intervention", "perampanel": "intervention", "adolescents": "population", "dravet syndrome": "disease"}),
    ("Does aspirin reduce pain in pregnant women with migraine?",
     {"aspirin": "intervention", "pregnant women": "population", "migraine": "disease"}),
    ("Is rimegepant better than ubrogepant for pain in adults with migraine?",
     {"rimegepant": "intervention", "ubrogepant": "intervention", "adults": "population", "migraine": "disease"}),
    ("Compare aerogel and mineral wool for energy efficiency in buildings.",
     {"aerogel": "intervention", "mineral wool": "intervention", "buildings": "subject"}),
)


class EssentialEntities(unittest.TestCase):
    def test_unseen_entities_and_roles_without_planner(self):
        for question, expected in ENTITY_CASES:
            with self.subTest(question=question):
                anchors = extract_anchors(question)
                self.assertTrue(set(expected).issubset(anchors.required), anchors)
                self.assertEqual({k: dict(anchors.entity_roles).get(k) for k in expected}, expected)
                self.assertNotIn("pain", anchors.required)

    def test_review_examples_remain_regressions(self):
        for question, expected in (
            ("Is warfarin or apixaban safer for stroke prevention in atrial fibrillation?", {"warfarin", "apixaban", "atrial fibrillation"}),
            ("Does colchicine reduce cardiovascular events in patients with gout?", {"colchicine", "gout"}),
            ("Does tirzepatide reduce weight in adolescents with obesity?", {"tirzepatide", "adolescents", "obesity"}),
            ("Does fecal microbiota transplantation help recurrent Clostridioides difficile infection?", {"fecal microbiota transplantation", "recurrent clostridioides difficile infection"}),
        ):
            with self.subTest(question=question):
                self.assertTrue(expected.issubset(extract_anchors(question).required))

    def test_grounded_hints_are_not_truncated_or_lost_to_earlier_lexicon(self):
        q = "Does vitamin D supplementation reduce pain in pregnant women with migraine?"
        hints = [QuestionEntity(text=t, role=r) for t, r in (
            ("vitamin D supplementation", "intervention"), ("pregnant women", "population"), ("migraine", "disease"))]
        anchors = extract_anchors(q, entity_hints=hints)
        self.assertTrue({h.text.lower() for h in hints}.issubset(anchors.required))
        self.assertFalse(extract_anchors(q, entity_hints=[QuestionEntity(text="insulin", role="intervention")]).required.count("insulin"))

    def test_wrong_topic_and_wrong_population_are_not_admitted(self):
        q = "Does aspirin reduce pain in pregnant women with migraine?"
        a = extract_anchors(q)
        for text in ("Aspirin reduced pain in pregnant women with arthritis.",
                     "Aspirin reduced pain in men with migraine.", "Ibuprofen reduced pain in pregnant women with migraine."):
            with self.subTest(text=text):
                self.assertFalse(topical_documents([document(text)], a))
        self.assertTrue(topical_documents([document("Aspirin reduced pain in pregnant women with migraine.")], a))

    def test_generic_or_unresolved_subject_is_not_complete(self):
        for q in ("Which approach is better?", "Can a newly described treatment improve wellbeing?"):
            with self.subTest(question=q):
                a = extract_anchors(q)
                self.assertTrue(a.unresolved_subject, a)
                self.assertFalse(Orchestrator._source_sufficiency([document(q, relevance_label="uncertain")], a, q)["sufficient"])

    def test_parenthetical_abbreviation_is_query_grounded(self):
        q = "Does transcranial alternating current stimulation (tACS) reduce pain in adults?"
        a = extract_anchors(q, entity_hints=[QuestionEntity(text="transcranial alternating current stimulation", role="intervention")])
        self.assertTrue(topical_documents([document("In adults, tACS reduced pain.")], a))
        self.assertFalse(topical_documents([document("In adults, tDCS reduced pain.")], a))
        self.assertIn("tacs", build_pubmed_query(a).lower())


class ScientificParaphrases(unittest.TestCase):
    def test_safe_paraphrases_and_negative_controls(self):
        cases = (
            ("Does vitamin D supplementation reduce pain in adolescents with migraine?",
             "Supplemental vitamin D reduced pain in adolescent patients with migraine.",
             "Supplemental vitamin C reduced pain in adolescent patients with migraine."),
            ("Does fecal microbiota transplantation help recurrent Clostridioides difficile infection?",
             "Fecal microbiota transplantation reduced recurrent C. difficile infection.",
             "Fecal microbiota transplantation reduced recurrent urinary tract infection."),
            ("Is lacosamide effective for Dravet syndrome in adolescents?",
             "Lacosamide was studied in adolescent patients diagnosed with Dravet syndrome.",
             "Levetiracetam was studied in adolescent patients diagnosed with Dravet syndrome."),
        )
        for q, positive, negative in cases:
            with self.subTest(question=q):
                a = extract_anchors(q)
                self.assertTrue(topical_documents([document(positive)], a), a)
                self.assertFalse(topical_documents([document(negative)], a))

    def test_grounded_population_phrase_is_not_an_exact_prose_requirement(self):
        q = "Does lacosamide reduce pain in adolescents with Dravet syndrome?"
        a = extract_anchors(q, entity_hints=[QuestionEntity(text="adolescents with Dravet syndrome", role="population")])
        self.assertTrue(topical_documents([document("Dravet syndrome was diagnosed in adolescent patients; lacosamide reduced pain.")], a))
        self.assertFalse(topical_documents([document("Dravet syndrome was diagnosed in adults; lacosamide reduced pain.")], a))


class ComparativeCoverage(unittest.TestCase):
    question = "Compare rimegepant versus ubrogepant for pain in adults with migraine."

    def assess(self, *texts):
        return assess_adequacy(self.question, [document(t, str(i)) for i, t in enumerate(texts)])

    def test_each_single_arm_is_partial(self):
        for arm in ("Rimegepant", "Ubrogepant"):
            with self.subTest(arm=arm):
                a = self.assess(f"{arm} reduced pain in adults with migraine.")
                self.assertNotEqual(a["status"], "candidate_coverage_complete")
                self.assertIn("pain", a["unanswered_dimensions"])

    def test_separate_studies_are_indirect_not_a_direct_comparison(self):
        texts = ["Rimegepant reduced pain in adults with migraine.", "Ubrogepant reduced pain in adults with migraine."]
        a = self.assess(*texts)
        self.assertEqual(a.get("comparison", {}).get("status"), "indirect_candidates")
        self.assertNotEqual(a["status"], "candidate_coverage_complete")
        self.assertIn("pain", a["unanswered_dimensions"])
        # Indirect sources remain useful for qualified synthesis, not cache admission.
        docs = [document(t, str(i)) for i, t in enumerate(texts)]
        self.assertEqual(len(topical_documents(docs, extract_anchors(self.question))), 2)
        preflight = Orchestrator._source_sufficiency(docs, extract_anchors(self.question), self.question)
        self.assertFalse(preflight["cache_eligible"])
        self.assertFalse(preflight["sufficient"])
        self.assertTrue(preflight["qualified_synthesis_eligible"])
        self.assertEqual(preflight.get("comparison", {}).get("status"), "indirect_candidates")

    def test_same_document_without_comparison_relation_is_not_direct(self):
        a = self.assess("Rimegepant reduced pain in adults with migraine. Ubrogepant is also used in adults with migraine.")
        self.assertNotEqual(a["status"], "candidate_coverage_complete")

    def test_direct_comparison_retains_every_arm_and_trace(self):
        a = self.assess("In adults with migraine, rimegepant reduced pain compared with ubrogepant.")
        self.assertEqual(a.get("comparison", {}).get("status"), "direct_candidate")
        self.assertEqual(a["status"], "candidate_coverage_complete")
        self.assertFalse(a["full_sufficiency_established"])
        self.assertEqual(a["dimensions"][0]["passages"][0]["source_id"], "0")

    def test_three_arm_comparison_cannot_drop_last_arm(self):
        q = "Compare rimegepant versus ubrogepant versus zavegepant for pain in adults with migraine."
        a = assess_adequacy(q, [document("In adults with migraine, rimegepant reduced pain compared with ubrogepant.")])
        self.assertIn("zavegepant", extract_anchors(q).required)
        self.assertNotEqual(a["status"], "candidate_coverage_complete")

    def test_or_alternatives_are_useful_but_do_not_cover_missing_arm(self):
        q = "Can lacosamide or perampanel reduce pain in adults with migraine?"
        docs = [document("Lacosamide reduced pain in adults with migraine.")]
        self.assertTrue(topical_documents(docs, extract_anchors(q)))
        a = assess_adequacy(q, docs)
        self.assertNotEqual(a["status"], "candidate_coverage_complete")
        self.assertNotEqual(a.get("comparison", {}).get("status"), "direct_candidate")


class SourceFailures(unittest.IsolatedAsyncioTestCase):
    async def test_screening_failure_is_not_external_outage_for_either_source(self):
        for source in ("pubmed", "arxiv"):
            with self.subTest(source=source):
                def fetch(*args, **kwargs):
                    kwargs["diagnostics"].update(returned_count=1, candidates=[{"pmid": "offline", "arxiv_id": "offline"}])
                    key = "PMID:offline" if source == "pubmed" else "arxiv:offline"
                    return [document("Aspirin was studied in adults. " * 5, key, title="Aspirin trial", abstract="Aspirin was studied in adults. " * 5)]
                trace = {}
                with patch.object(MinerService, "mine_" + source, side_effect=fetch), patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=RuntimeError("offline screening failure"))):
                    docs = await MinerService.mine("aspirin in adults", source, diagnostics=trace)
                self.assertFalse(docs)
                self.assertEqual(trace.get("retry_ladder_stopped"), "relevance_screening_failed")
                self.assertFalse(trace.get("external_unavailable"))
                self.assertEqual(Orchestrator._trace_outcome(trace), "relevance_screening_failed")
                self.assertFalse(trace["repair_budget_exhausted"])

    def test_empty_rejected_outage_deadline_and_partial_remain_distinct(self):
        for trace, expected in (
            ({"returned_count": 0}, "no_candidates"),
            ({"returned_count": 2, "candidates": [{"decision": "rejected_irrelevant"}]}, "all_candidates_rejected"),
            ({"external_unavailable": True}, "external_source_unavailable"),
            ({"indexed_count": 1, "external_unavailable": True}, "accepted_sources"),
        ):
            with self.subTest(trace=trace):
                self.assertEqual(Orchestrator._trace_outcome(trace), expected)
        self.assertEqual(Orchestrator._evidence_outcome({"pubmed": {"attempt_outcomes": ["source_deadline_expired"]}}, {"sufficient": False}), "retrieval_deadline_expired")


class AnswerCoverageContracts(unittest.IsolatedAsyncioTestCase):
    """Additional integration contracts, separate from the pre-fix holdout."""
    async def test_supported_one_arm_quote_cannot_mark_comparison_complete(self):
        from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2
        from app.claim_verification import ClaimAssessment, ClaimVerificationBatch
        q = ComparativeCoverage.question
        passage = "Rimegepant reduced pain in adults with migraine."
        docs = [document(passage), document("Ubrogepant reduced pain in adults with migraine.", "other")]
        structured = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(text="Rimegepant reduced pain.", source_id="S1", evidence_quote=passage)])
        verdict = ClaimVerificationBatch(assessments=[ClaimAssessment(target="claims.1", verdict="supported", reason="Synthetic fixed verdict for the one-arm claim only.")])
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=structured)), patch("scientific_rag.legacy_pipeline.verify_claims", new=AsyncMock(return_value=verdict)) as verifier:
            agent = ResearchAgent(SimpleNamespace(retrieve_context=AsyncMock(return_value=docs)), semantic_verification_mode="enforce")
            agent.decompose_query_async = AsyncMock(return_value=[q])
            result = await agent.run_research(q)
        verifier.assert_awaited_once()
        validation = result["claim_validation"]
        self.assertIn("ubrogepant", validation["answer_completeness"]["missing_entities"])
        self.assertNotEqual(validation["answer_completeness"]["status"], "candidate_coverage_complete")
        self.assertTrue(any("direct comparison" in part for part in validation["unanswered_parts"]))
        self.assertTrue(validation["semantic_verification"]["filtering_applied"])
        self.assertEqual(validation["claims"][0]["validation"], "source_id_and_verbatim_quote_verified")

    def test_irrelevant_grounded_hint_cannot_replace_missing_intervention(self):
        q = "Does lacosamide reduce pain in adolescents with Dravet syndrome?"
        hints = [QuestionEntity(text="pain", role="intervention"), QuestionEntity(text="insulin", role="intervention")]
        a = extract_anchors(q, entity_hints=hints)
        self.assertIn("lacosamide", a.required)
        self.assertNotIn("pain", a.required)
        self.assertNotIn("insulin", a.required)

    def test_nonmedical_three_arm_comparison_and_alternative_spelling(self):
        q = "Compare aerogel versus mineral wool versus cellulose for energy efficiency in buildings."
        positive = document("In buildings, aerogel achieved higher energy efficiency compared with mineral wool and cellulose.")
        negative = document("In buildings, aerogel achieved higher energy efficiency compared with mineral wool.")
        self.assertEqual(assess_adequacy(q, [positive])["comparison"]["status"], "direct_candidate")
        self.assertNotEqual(assess_adequacy(q, [negative])["comparison"]["status"], "direct_candidate")

    async def test_partial_comparison_can_synthesize_but_wrong_disease_cannot(self):
        from scientific_rag.domain import QueryPlan, SubQuery
        q = ComparativeCoverage.question
        plan = QueryPlan(plan=[SubQuery(query=q, source_type="academic", domains=["medical"], external_sources=["pubmed"], reasoning="offline")])
        for condition, allowed in (("migraine", True), ("arthritis", False)):
            with self.subTest(condition=condition):
                doc = document(f"Rimegepant reduced pain in adults with {condition}.")
                o = Orchestrator(SimpleNamespace(restrict_final_retrieval_to_current_run=lambda: None), None)
                o.generate_plan = AsyncMock(return_value=plan)
                o.filter_plan = AsyncMock(return_value=(plan, []))
                o.ask_db = AsyncMock(return_value=([], {}))
                o.run_mining_operation = AsyncMock(return_value=([doc], 1, {}, [], []))
                o.populate_knowledge_base = AsyncMock()
                o.synthesize_final_report = AsyncMock(return_value={"answer": "qualified partial answer", "raw_docs_used": [doc]})
                result = await o.execute_full_study(q)
                self.assertEqual(o.synthesize_final_report.await_count, int(allowed))
                preflight = result["metadata"]["source_sufficiency"]
                self.assertFalse(preflight["sufficient"])
                self.assertIn("ubrogepant", preflight["missing_concepts"])
                self.assertFalse(preflight["cache_eligible"])


if __name__ == "__main__":
    unittest.main()
