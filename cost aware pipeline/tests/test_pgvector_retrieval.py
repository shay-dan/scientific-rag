from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
import asyncio

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from langchain_core.documents import Document

from app.pgvector_retrieval import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PostgresVectorService, RetrievalCandidate, maximal_marginal_relevance, parent_context_key, reciprocal_rank_fusion, source_fingerprint
from scientific_rag.domain import QueryPlan, SubQuery
from scientific_rag.legacy_pipeline import MinerService, Orchestrator


def candidate(identifier: uuid.UUID) -> RetrievalCandidate:
    return RetrievalCandidate(chunk=SimpleNamespace(id=identifier), source=SimpleNamespace())


def embedded_candidate(identifier: uuid.UUID, embedding: list[float]) -> RetrievalCandidate:
    return RetrievalCandidate(chunk=SimpleNamespace(id=identifier, embedding=embedding), source=SimpleNamespace())


class PgvectorRetrievalTests(unittest.TestCase):
    def test_pubmed_query_is_bounded_and_keeps_surface_code_terms(self) -> None:
        query = (
            "How do surface-code decoding algorithms map stabilizer syndrome "
            "measurements to Pauli error chains using MWPM and Union-Find?"
        )
        keywords = MinerService._get_keywords(query)
        self.assertEqual(keywords, ["surface", "code", "decoding", "algorithms", "stabilizer"])
        self.assertEqual(
            MinerService._format_pubmed_query(keywords),
            "surface[tiab] AND code[tiab] AND decoding[tiab] AND algorithms[tiab] AND stabilizer[tiab]",
        )

    def test_pubmed_section_selector_prioritizes_readable_evidence_and_caps_volume(self) -> None:
        readable = "This section contains readable scientific evidence. " * 20
        sections = [
            {"section": "References", "text": readable},
            {"section": "Methods", "text": readable},
            {"section": "Introduction", "text": readable},
            {"section": "Results", "text": readable},
            {"section": "Discussion", "text": readable},
            {"section": "Conclusions", "text": readable},
            {"section": "Equations", "text": "\\usepackage{amsmath} $$ x = y $$ " * 30},
        ]
        selected = MinerService.select_pmc_sections(sections)
        self.assertEqual([item["section"] for item in selected], ["Conclusions", "Results", "Discussion"])
        self.assertLessEqual(len(selected), MinerService.MAX_PMC_SECTIONS_PER_PAPER)

    def test_final_retrieval_scope_can_be_restricted_to_the_current_run(self) -> None:
        service = object.__new__(PostgresVectorService)
        service.restrict_to_current_run = False
        service.restrict_final_retrieval_to_current_run()
        self.assertTrue(service.restrict_to_current_run)

    def test_source_policy_routes_mixed_research_to_both_and_physics_to_arxiv(self) -> None:
        plan = QueryPlan(plan=[
            SubQuery(query="AI imaging models for lung cancer prognosis", source_type="academic", domains=["medical", "technical"], reasoning="test"),
            SubQuery(query="CRISPR adverse effects in clinical cancer trials", source_type="academic", domains=["medical"], reasoning="test"),
            SubQuery(query="surface code decoder algorithms for quantum error correction", source_type="academic", domains=["physics", "technical"], reasoning="test"),
        ])
        selected = Orchestrator.enforce_source_policy(plan)
        self.assertEqual(selected.plan[0].external_sources, ["pubmed", "arxiv"])
        self.assertEqual(selected.plan[1].external_sources, ["pubmed"])
        self.assertEqual(selected.plan[2].external_sources, ["arxiv"])

    def test_source_mining_starts_pubmed_and_arxiv_concurrently(self) -> None:
        async def verify() -> None:
            started = []
            both_started = asyncio.Event()

            async def mine(query, source, limit):
                started.append(source)
                if len(started) == 2:
                    both_started.set()
                await both_started.wait()
                return [Document(page_content=source, metadata={"source": source})]

            orchestrator = object.__new__(Orchestrator)
            orchestrator.progress_callback = None
            orchestrator.papers_per_source = 1
            plan = QueryPlan(plan=[SubQuery(query="technical question", source_type="academic", domains=["technical"], external_sources=["pubmed", "arxiv"], reasoning="test")])
            with patch("scientific_rag.legacy_pipeline.MinerService.mine", side_effect=mine):
                docs, count, diagnostics = await orchestrator.run_mining_operation(plan)
            self.assertEqual(started, ["pubmed", "arxiv"])
            self.assertEqual(count, 2)
            self.assertEqual({doc.page_content for doc in docs}, {"pubmed", "arxiv"})
            self.assertEqual(diagnostics["arxiv"]["collected"], 1)

        asyncio.run(verify())

    def test_arxiv_miner_keeps_citable_relevant_abstracts_without_full_text_download(self) -> None:
        class Result:
            def __init__(self, title, summary, entry_id):
                self.title, self.summary, self.entry_id = title, summary, entry_id

        class Client:
            search_query = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def results(self, search):
                Client.search_query = search.query
                return iter([
                    Result("AI-Enabled Lung Cancer Prognosis", "Machine learning prognosis methods.", "https://arxiv.org/abs/2402.09476v1"),
                    Result("CRISPR screens in cancer", "CRISPR methods identify cancer therapy targets.", "https://arxiv.org/abs/2501.00001v2"),
                ])

        with patch("scientific_rag.legacy_pipeline.arxiv.Client", Client):
            docs = MinerService.mine_arxiv("CRISPR cancer research", limit=1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["doc_id"], "arxiv:2501.00001v2")
        self.assertEqual(docs[0].metadata["source_url"], "https://arxiv.org/abs/2501.00001v2")
        self.assertEqual(docs[0].metadata["type"], "abstract")
        self.assertEqual(Client.search_query, "CRISPR cancer research")

    def test_arxiv_miner_requires_surface_code_phrase_when_query_requests_it(self) -> None:
        class Result:
            def __init__(self, title, summary, entry_id):
                self.title, self.summary, self.entry_id = title, summary, entry_id

        class Client:
            def __init__(self, **kwargs):
                pass

            def results(self, search):
                return iter([
                    Result("Continuous-time quantum error correction", "A general quantum error correction chapter.", "https://arxiv.org/abs/1311.2485v2"),
                    Result("Thermodynamic recycling in quantum processors", "Quantum error correction can reduce heat dissipation.", "https://arxiv.org/abs/2501.00003v1"),
                    Result("Surface-code decoders", "Surface code decoding improves fault-tolerant quantum error correction.", "https://arxiv.org/abs/2501.00002v1"),
                ])

        with patch("scientific_rag.legacy_pipeline.arxiv.Client", Client):
            docs = MinerService.mine_arxiv("surface code quantum error correction", limit=2)
        self.assertEqual([doc.metadata["title"] for doc in docs], ["Surface-code decoders"])

    def test_parent_and_child_chunking_matches_the_original_design(self) -> None:
        self.assertEqual((PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP), (2_500, 500))
        self.assertEqual((CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP), (500, 75))

    def test_mmr_prefers_a_diverse_second_semantic_chunk(self) -> None:
        first = embedded_candidate(uuid.uuid4(), [1.0, 0.0])
        near_duplicate = embedded_candidate(uuid.uuid4(), [0.99, 0.01])
        diverse = embedded_candidate(uuid.uuid4(), [0.5, 0.866])
        selected = maximal_marginal_relevance([first, near_duplicate, diverse], [1.0, 0.0], limit=2, lambda_mult=0.5)
        self.assertEqual([item.chunk.id for item in selected], [first.chunk.id, diverse.chunk.id])

    def test_mmr_rejects_invalid_diversity_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "lambda_mult"):
            maximal_marginal_relevance([], [1.0], limit=1, lambda_mult=1.1)

    def test_parent_context_key_groups_children_under_the_same_parent(self) -> None:
        parent_id = uuid.uuid4()
        child_one = RetrievalCandidate(chunk=SimpleNamespace(id=uuid.uuid4(), parent_document_id=parent_id), source=SimpleNamespace())
        child_two = RetrievalCandidate(chunk=SimpleNamespace(id=uuid.uuid4(), parent_document_id=parent_id), source=SimpleNamespace())
        legacy_child = RetrievalCandidate(chunk=SimpleNamespace(id=uuid.uuid4(), parent_document_id=None), source=SimpleNamespace())
        self.assertEqual(parent_context_key(child_one), parent_context_key(child_two))
        self.assertEqual(parent_context_key(legacy_child), legacy_child.chunk.id)

    def test_source_fingerprint_prefers_provider_document_id(self) -> None:
        document = Document(page_content="First version", metadata={"doc_id": "12345", "source": "pubmed"})
        self.assertEqual(source_fingerprint(document), "external:12345")

    def test_source_fingerprint_falls_back_to_content_identity(self) -> None:
        document = Document(page_content="Same paper text", metadata={"title": "Paper", "source": "pubmed"})
        self.assertEqual(source_fingerprint(document), source_fingerprint(document))
        changed = Document(page_content="Changed paper text", metadata={"title": "Paper", "source": "pubmed"})
        self.assertNotEqual(source_fingerprint(document), source_fingerprint(changed))

    def test_source_segments_with_the_same_document_id_share_one_source_identity(self) -> None:
        abstract = Document(page_content="Abstract", metadata={"doc_id": "PMID:42", "type": "abstract"})
        full_text_section = Document(page_content="Methods section", metadata={"doc_id": "PMID:42", "type": "full_text"})
        self.assertEqual(source_fingerprint(abstract), source_fingerprint(full_text_section))

    def test_reciprocal_rank_fusion_rewards_results_found_by_both_searches(self) -> None:
        shared = candidate(uuid.uuid4())
        vector_only = candidate(uuid.uuid4())
        text_only = candidate(uuid.uuid4())
        fused = reciprocal_rank_fusion([vector_only, shared], [shared, text_only], limit=3)
        self.assertEqual(fused[0].chunk.id, shared.chunk.id)

    def test_reciprocal_rank_fusion_respects_limit(self) -> None:
        results = [candidate(uuid.uuid4()) for _ in range(4)]
        fused = reciprocal_rank_fusion(results, [], limit=2)
        self.assertEqual(len(fused), 2)
