"""Offline comparison-wording regressions with grounded planner entities."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.documents import Document
from scientific_rag.domain import QuestionEntity, QueryPlan, SubQuery
from scientific_rag.query_quality import extract_anchors
from scientific_rag.evidence_adequacy import assess_adequacy
from scientific_rag.legacy_pipeline import Orchestrator, ResearchAgent
from app.weak_model_synthesis import EvidenceClaim, StructuredEvidenceAnswerV2
from app.claim_verification import ClaimAssessment, ClaimVerificationBatch


THERAPY = "cognitive behavioural therapy"
ARMS = {THERAPY, "zolpidem"}
ENTITIES = [QuestionEntity(text=THERAPY, role="intervention"),
            QuestionEntity(text="insomnia", role="disease"),
            QuestionEntity(text="zolpidem", role="intervention")]


def question(wording):
    return f"Does cognitive behavioural therapy improve insomnia {wording} zolpidem?"


def studies():
    return [Document(page_content="Cognitive behavioural therapy reduced insomnia symptoms in a study.", metadata={"doc_id": "therapy"}),
            Document(page_content="Zolpidem reduced insomnia symptoms in a separate study.", metadata={"doc_id": "drug"})]


class ComparisonWording(unittest.TestCase):
    def test_compared_with_and_to_retain_comparison_intent(self):
        for wording in ("compared with", "compared to"):
            a = extract_anchors(question(wording), entity_hints=ENTITIES)
            for name, actual, expected in (
                ("required", set(a.required) & ARMS, ARMS),
                ("arms", set(a.comparison), ARMS),
                ("intent", a.intent, ("comparison",)),
            ):
                with self.subTest(wording=wording, field=name):
                    self.assertEqual(actual, expected)

    def test_separate_one_arm_studies_are_indirect_and_not_sufficient(self):
        for wording in ("compared with", "compared to"):
            q = question(wording)
            a = extract_anchors(q, entity_hints=ENTITIES)
            docs = studies()
            preflight = Orchestrator._source_sufficiency(docs, a, q)
            with self.subTest(wording=wording, field="sufficient"):
                self.assertFalse(preflight["sufficient"])
            with self.subTest(wording=wording, field="indirect"):
                self.assertEqual(assess_adequacy(q, docs, anchors=a)["comparison"]["status"], "indirect_candidates")
            with self.subTest(wording=wording, field="cache"):
                self.assertFalse(preflight["cache_eligible"])

    def test_existing_positive_comparison_wordings(self):
        for q in (question("versus"),
                  "Is cognitive behavioural therapy better than zolpidem for insomnia?",
                  "Compare cognitive behavioural therapy and zolpidem for insomnia.",
                  "Comparison of cognitive behavioural therapy and zolpidem for insomnia."):
            with self.subTest(question=q):
                a = extract_anchors(q, entity_hints=ENTITIES)
                self.assertEqual(set(a.comparison), ARMS)
                self.assertEqual(a.intent, ("comparison",))

    def test_alternatives_and_single_intervention_are_unchanged(self):
        a = extract_anchors("Does cognitive behavioural therapy or zolpidem improve insomnia?", entity_hints=ENTITIES)
        self.assertEqual(set(a.comparison), ARMS)
        self.assertEqual(a.intent, ("alternatives",))
        a = extract_anchors("Does cognitive behavioural therapy improve insomnia?", entity_hints=ENTITIES)
        self.assertEqual(a.comparison, ())
        self.assertEqual(a.intent, ())
        self.assertNotIn("zolpidem", a.required)

    def test_direct_comparative_result_remains_eligible(self):
        for wording in ("compared with", "compared to", "versus"):
            with self.subTest(wording=wording):
                q = f"Does cognitive behavioural therapy reduce pain in insomnia {wording} zolpidem?"
                a = extract_anchors(q, entity_hints=ENTITIES)
                docs = [Document(page_content="Cognitive behavioural therapy reduced pain in insomnia compared with zolpidem.", metadata={"doc_id": "comparison"})]
                self.assertTrue(Orchestrator._source_sufficiency(docs, a, q)["sufficient"])
                self.assertEqual(assess_adequacy(q, docs, anchors=a)["comparison"]["status"], "direct_candidate")


class SelectedAnswerWording(unittest.IsolatedAsyncioTestCase):
    async def test_selected_answer_discloses_missing_direct_comparison(self):
        for wording in ("compared with", "compared to"):
            with self.subTest(wording=wording):
                q = question(wording)
                docs = studies()
                plan = QueryPlan(entities=ENTITIES, plan=[SubQuery(query=q, source_type="academic", domains=["medical"], external_sources=["pubmed"], reasoning="offline comparison regression")])
                vector = SimpleNamespace(retrieve_context=AsyncMock(return_value=docs), restrict_final_retrieval_to_current_run=lambda: None)
                agent = ResearchAgent(vector, semantic_verification_mode="enforce")
                agent.decompose_query_async = AsyncMock(return_value=[q])
                o = Orchestrator(vector, agent)
                o.generate_plan = AsyncMock(return_value=plan)
                o.filter_plan = AsyncMock(return_value=(plan, []))
                o.ask_db = AsyncMock(return_value=([], {}))
                o.run_mining_operation = AsyncMock(return_value=(docs, 2, {}, [], []))
                o.populate_knowledge_base = AsyncMock()
                answer = StructuredEvidenceAnswerV2(claims=[EvidenceClaim(text=doc.page_content, source_id=f"S{i}", evidence_quote=doc.page_content) for i, doc in enumerate(docs, 1)])
                verdicts = ClaimVerificationBatch(assessments=[ClaimAssessment(target=f"claims.{i}", verdict="supported", reason="Fixed synthetic verdict for this individual one-arm claim.") for i in (1, 2)])
                with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=answer)), patch("scientific_rag.legacy_pipeline.verify_claims", new=AsyncMock(return_value=verdicts)) as verifier:
                    result = await o.execute_full_study(q)
                verifier.assert_awaited_once()
                report = result["report"]
                self.assertIn("direct comparison", report["answer"].lower())
                self.assertIn("not established", report["answer"].lower())
                self.assertEqual(report["claim_validation"]["answer_completeness"]["comparison"]["status"], "indirect_candidates")
                self.assertTrue(report["claim_validation"]["semantic_verification"]["filtering_applied"])


if __name__ == "__main__":
    unittest.main()
