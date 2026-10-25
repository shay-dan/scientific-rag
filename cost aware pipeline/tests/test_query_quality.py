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


if __name__ == "__main__":
    unittest.main()
