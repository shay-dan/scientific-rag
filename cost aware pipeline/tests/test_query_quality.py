import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scientific_rag.domain import QueryRepairProposal
from scientific_rag.legacy_pipeline import MinerService, Orchestrator
from scientific_rag.query_quality import QueryAnchors, build_arxiv_query, build_pubmed_query, extract_anchors, query_is_valid


class QueryQualityTests(unittest.TestCase):
    def test_robotic_manipulation_keeps_application_and_rl_as_required_anchors(self):
        anchors = extract_anchors(
            "How are offline and online reinforcement learning approaches utilized and enhanced for robotic manipulation tasks?"
        )
        self.assertEqual(anchors.required[:2], ("robotic manipulation", "reinforcement learning"))
        query = build_arxiv_query(anchors)
        self.assertIn('all:"robotic manipulation"', query)
        self.assertIn('all:"reinforcement learning"', query)

    def test_crispr_cancer_anchors_survive_both_provider_builders(self):
        anchors = extract_anchors("What are current uses of CRISPR in cancer research?", "recent studies demonstrate uses")
        self.assertEqual(anchors.required, ("cancer", "crispr"))
        pubmed = build_pubmed_query(anchors)
        arxiv = build_arxiv_query(anchors)
        self.assertTrue(query_is_valid(pubmed, anchors))
        self.assertTrue(query_is_valid(arxiv, anchors))
        self.assertIn("crispr", pubmed.lower())
        self.assertIn("cancer", arxiv.lower())

    def test_surface_code_is_preserved_as_a_phrase(self):
        anchors = extract_anchors("What are current approaches to quantum error correction using surface codes?")
        query = build_arxiv_query(anchors, broad=True)
        self.assertIn('all:"surface code"', query)
        self.assertIn('all:"quantum error correction"', query)

    def test_antimicrobial_resistance_keeps_subject_and_setting_not_intent_words(self):
        anchors = extract_anchors(
            "What are the main causes and prevention strategies for antimicrobial resistance in hospitals and healthcare settings?"
        )
        self.assertEqual(anchors.required, ("antimicrobial resistance", "healthcare setting"))
        self.assertNotIn("causes", anchors.required)
        self.assertNotIn("prevention", anchors.required)
        self.assertNotIn("strategies", anchors.required)
        query = build_pubmed_query(anchors, broad=True).lower()
        self.assertIn("antimicrobial resistance", query)
        self.assertIn("hospital", query)

    def test_hyphenated_spelling_does_not_create_a_duplicate_anchor(self):
        surface = extract_anchors("How do rotated surface-code approaches improve quantum error correction?")
        lung = extract_anchors("How is AI used in lung-cancer imaging?")
        alphafold = extract_anchors("What are the strengths of AlphaFold-style structure prediction for drug discovery?")
        self.assertNotIn("surface-code", surface.required)
        self.assertNotIn("lung-cancer", lung.required)
        self.assertNotIn("alphafold-style", alphafold.required)
        self.assertIn("surface code", surface.required)
        self.assertIn("lung cancer", lung.required)
        self.assertIn("alphafold", alphafold.required)
        self.assertIn("drug discovery", alphafold.required)

    def test_source_sufficiency_requires_every_topic_anchor(self):
        anchors = QueryAnchors(("crispr", "cancer"), ())
        bacterial = [Document(page_content="CRISPR editing in bacteria", metadata={"doc_id": "1", "title": "Bacterial editing"})]
        cancer = [Document(page_content="CRISPR editing in cancer immunotherapy", metadata={"doc_id": "2", "title": "Cancer"})]
        self.assertFalse(Orchestrator._source_sufficiency(bacterial, anchors)["sufficient"])
        self.assertTrue(Orchestrator._source_sufficiency(cancer, anchors)["sufficient"])

    def test_llm_repair_is_only_called_after_two_empty_deterministic_attempts(self):
        async def verify():
            attempts = []

            def empty_pubmed(query, limit, **kwargs):
                attempts.append(kwargs["search_term_override"])
                kwargs["diagnostics"].update({"returned_count": 0, "screened_count": 0, "indexed_count": 0, "candidates": [], "errors": []})
                return []

            repair = QueryRepairProposal(
                required_concepts=["crispr", "cancer"], optional_concepts=["therapy"],
                excluded_concepts=["bacteria"], rationale="Restore the requested intervention and disease.",
            )
            with patch.object(MinerService, "mine_pubmed", side_effect=empty_pubmed), \
                 patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=repair)) as llm:
                trace = {}
                await MinerService.mine("generic planner wording", "pubmed", 2,
                                        anchor_query="CRISPR applications in cancer", diagnostics=trace)
            self.assertEqual(len(attempts), 3)
            llm.assert_awaited_once()
            self.assertEqual(trace["attempts"][2]["strategy"], "llm_repair")

        asyncio.run(verify())

    def test_external_arxiv_failure_stops_query_ladder_and_skips_llm_repair(self):
        async def verify():
            def unavailable(_query, _limit, **kwargs):
                kwargs["diagnostics"].update({
                    "returned_count": 0, "screened_count": 0, "indexed_count": 0,
                    "candidates": [], "errors": ["HTTP 429"], "external_unavailable": True,
                })
                return []

            with patch.object(MinerService, "mine_arxiv", side_effect=unavailable) as miner, \
                 patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock()) as llm:
                trace = {}
                docs = await MinerService.mine(
                    "surface code decoders", "arxiv", 2,
                    anchor_query="Quantum error correction using surface codes", diagnostics=trace,
                )
            self.assertEqual(docs, [])
            self.assertEqual(miner.call_count, 1)
            llm.assert_not_awaited()
            self.assertEqual(trace["retry_ladder_stopped"], "external_source_unavailable")
            self.assertTrue(trace["external_unavailable"])

        asyncio.run(verify())

    def test_source_outcome_distinguishes_transport_rejection_and_anchor_gap(self):
        self.assertEqual(
            Orchestrator._trace_outcome({"external_unavailable": True, "indexed_count": 0}),
            "external_source_unavailable",
        )
        self.assertEqual(
            Orchestrator._trace_outcome({"returned_count": 3, "indexed_count": 0, "candidates": [{}]}),
            "all_candidates_rejected",
        )
        outcome = Orchestrator._evidence_outcome(
            {"pubmed": {"attempt_outcomes": ["accepted_sources"]}},
            {"sufficient": False, "unique_sources": 1, "missing_concepts": ["healthcare setting"]},
        )
        self.assertEqual(outcome, "insufficient_anchor_coverage")


if __name__ == "__main__":
    unittest.main()
