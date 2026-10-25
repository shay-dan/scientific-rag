from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import time
import requests

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from langchain_core.documents import Document

from app.pgvector_retrieval import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PostgresVectorService, RetrievalCandidate, maximal_marginal_relevance, parent_context_key, reciprocal_rank_fusion, source_fingerprint
from scientific_rag.domain import ArxivRelevanceAssessment, PubMedRelevanceAssessment, QueryPlan, SubQuery
from scientific_rag.legacy_pipeline import MinerService, Orchestrator, ResearchAgent, _BoundedArxivSession, _bounded_entrez_urlopen, _entrez_request_context


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
        self.assertEqual(keywords, ["surface-code", "decoding", "stabilizer", "syndrome", "measurements", "pauli"])
        self.assertEqual(
            MinerService._format_pubmed_query(keywords),
            '"surface-code"[tiab] AND decoding[tiab] AND stabilizer[tiab] AND syndrome[tiab] AND measurements[tiab] AND pauli[tiab]',
        )

    def test_pubmed_query_preserves_original_topic_anchors_over_generic_planner_words(self) -> None:
        original = "What are current uses of CRISPR in cancer treatment?"
        planner_query = "Recent preclinical and clinical studies demonstrate therapeutic applications"
        keywords = MinerService._get_keywords(planner_query, original)
        self.assertEqual(keywords[:3], ["crispr", "cancer", "treatment"])
        self.assertNotIn("recent", keywords)
        self.assertNotIn("studies", keywords)
        self.assertIn("crispr[tiab]", MinerService._format_pubmed_query(keywords))
        self.assertIn("cancer[tiab]", MinerService._format_pubmed_query(keywords))
        anchor_terms = MinerService._get_keywords(original)[:4]
        detail_terms = [term for term in MinerService._get_keywords(planner_query) if term not in anchor_terms][:2]
        formatted = MinerService._format_pubmed_query([*anchor_terms, *detail_terms], len(anchor_terms))
        self.assertIn("crispr[tiab] AND cancer[tiab]", formatted)
        self.assertIn("(preclinical[tiab] OR clinical[tiab])", formatted)

    def test_pubmed_original_anchors_keep_mammography_and_breast_cancer_terms(self) -> None:
        original = (
            "How are machine-learning algorithms used for breast cancer detection in mammography, "
            "and what evidence supports their diagnostic performance?"
        )
        anchors = MinerService._pubmed_anchor_terms(original)
        self.assertEqual(anchors, ["machine-learning", "breast cancer", "mammography"])
        formatted = MinerService._format_pubmed_query(anchors, len(anchors))
        for term in ('"machine learning"[tiab]', '"deep learning"[tiab]', '"breast cancer"[tiab]', "mammography[tiab]", "tomosynthesis[tiab]"):
            self.assertIn(term, formatted)
        self.assertNotIn("diagnosis[tiab]", formatted)
        self.assertNotIn("detection[tiab]", formatted)

    def test_pubmed_query_keeps_hyphenated_biomedical_terms_as_phrases(self) -> None:
        keywords = MinerService._get_keywords("CRISPR-Cas9 gene editing in cancer")
        self.assertEqual(keywords[:2], ["crispr-cas9", "gene"])
        self.assertIn('"crispr-cas9"[tiab]', MinerService._format_pubmed_query(keywords))

    def test_pubmed_car_t_query_keeps_intervention_and_lymphoma_anchors(self) -> None:
        question = (
            "What clinical evidence supports CAR-T therapy for relapsed or refractory "
            "B-cell lymphoma, and what are its major toxicities?"
        )
        trace = {}
        with patch("scientific_rag.legacy_pipeline.Entrez.esearch", return_value=MagicMock()) as search, \
             patch("scientific_rag.legacy_pipeline.Entrez.read", return_value={"IdList": []}):
            MinerService.mine_pubmed("clinical trial efficacy of CAR-T", anchor_query=question, diagnostics=trace)
        self.assertEqual(trace["keywords"][:2], ["car-t", "b-cell lymphoma"])
        self.assertEqual(trace["anchor_count"], 2)
        filtered_term = search.call_args_list[0].kwargs["term"]
        self.assertIn('"CAR-T"[tiab]', filtered_term)
        self.assertIn('"B-cell lymphoma"[tiab]', filtered_term)
        self.assertNotIn("relapsed[tiab]", filtered_term)
        self.assertIn('"Clinical Trial"[pt]', filtered_term)
        self.assertIn('"Systematic Review"[pt]', filtered_term)
        self.assertNotIn("supports[tiab]", filtered_term)
        self.assertEqual(search.call_args_list[0].kwargs["retmax"], 20)
        self.assertEqual(trace["accepted_limit"], 5)
        self.assertEqual(search.call_count, 2)
        self.assertEqual(trace["search_fallback"], "no_publication_type_results")
        self.assertNotIn('"Clinical Trial"[pt]', trace["fallback_search_term"])

    def test_pubmed_clinical_outcomes_question_uses_study_filter_and_wider_pool(self) -> None:
        question = "What are the clinical outcomes and toxicities of CAR-T therapy in B-cell lymphoma?"
        trace = {}
        with patch("scientific_rag.legacy_pipeline.Entrez.esearch", return_value=MagicMock()) as search, \
             patch("scientific_rag.legacy_pipeline.Entrez.read", return_value={"IdList": []}):
            MinerService.mine_pubmed("recent CAR-T outcomes", limit=2, anchor_query=question, diagnostics=trace)
        filtered = search.call_args_list[0].kwargs
        self.assertIn('"Clinical Trial"[pt]', filtered["term"])
        self.assertIn('"Observational Study"[pt]', filtered["term"])
        self.assertNotIn("outcomes[tiab]", filtered["term"])
        self.assertNotIn("toxicities[tiab]", filtered["term"])
        self.assertEqual(filtered["retmax"], 12)
        self.assertEqual(trace["accepted_limit"], 2)

    def test_pubmed_clinical_candidate_pool_uses_abstracts_before_full_text(self) -> None:
        question = "What clinical evidence supports CAR-T therapy for relapsed B-cell lymphoma?"
        article = {"PubmedArticle": [{"MedlineCitation": {"Article": {
            "ArticleTitle": "A clinical trial of CAR-T in B-cell lymphoma",
            "Abstract": {"AbstractText": ["CAR-T outcomes in relapsed B-cell lymphoma."]},
        }}}]}
        trace = {}
        with patch("scientific_rag.legacy_pipeline.Entrez.esearch", return_value=MagicMock()) as search, \
             patch("scientific_rag.legacy_pipeline.Entrez.efetch", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.read", side_effect=[{"IdList": ["123"]}, article]), \
             patch.object(MinerService, "get_pmcid_from_pmid") as full_text:
            docs = MinerService.mine_pubmed("clinical CAR-T outcomes", limit=2, anchor_query=question, diagnostics=trace)
        self.assertEqual(search.call_args.kwargs["retmax"], 12)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["type"], "abstract")
        full_text.assert_not_called()

    def test_pubmed_study_design_uses_publication_type_before_title(self) -> None:
        classify = MinerService._pubmed_study_design
        self.assertEqual(classify(["Journal Article", "Clinical Trial, Phase II"], "Results"),
                         ("primary_clinical_study", "pubmed_publication_type"))
        self.assertEqual(classify(["Review", "Systematic Review"], "CAR-T review"),
                         ("systematic_review", "pubmed_publication_type"))
        self.assertEqual(classify(["Review"], "CAR-T trial results"),
                         ("narrative_review", "pubmed_publication_type"))
        self.assertEqual(classify(["Observational Study"], "CAR-T lymphoma outcomes"),
                         ("primary_clinical_study", "pubmed_publication_type"))
        self.assertEqual(classify([], "CAR-T for lymphoma: a meta-analysis"),
                         ("systematic_review", "title"))
        self.assertEqual(classify([], "CAR-T in lymphoma"), ("unknown", "not_classified"))

    def test_pubmed_to_pmc_mapper_uses_direct_link_not_cited_references(self) -> None:
        direct = {"LinkName": "pubmed_pmc", "Link": [{"Id": "123"}]}
        cited = {"LinkName": "pubmed_pmc_refs", "Link": [{"Id": "999"}]}
        with patch("scientific_rag.legacy_pipeline.Entrez.elink", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.read", return_value=[{"LinkSetDb": [cited, direct]}]):
            self.assertEqual(MinerService.get_pmcid_from_pmid("42"), "PMC123")
        with patch("scientific_rag.legacy_pipeline.Entrez.elink", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.read", return_value=[{"LinkSetDb": [cited]}]):
            self.assertIsNone(MinerService.get_pmcid_from_pmid("42"))

    def test_pmc_full_text_rejects_a_different_article_pmid(self) -> None:
        xml = "<article><front><article-meta><article-id pub-id-type='pmid'>999</article-id></article-meta></front><body><p>Wrong paper.</p></body></article>"
        handle = MagicMock()
        handle.read.return_value = xml
        with patch("scientific_rag.legacy_pipeline.Entrez.efetch", return_value=handle):
            with self.assertRaisesRegex(ValueError, "does not match requested PMID 42"):
                MinerService.fetch_pmc_full_text("PMC999", expected_pmid="42")

    def test_pubmed_miner_keeps_abstract_when_pmc_identity_check_fails(self) -> None:
        article = {"PubmedArticle": [{"MedlineCitation": {"Article": {
            "ArticleTitle": "CRISPR in cancer research",
            "Abstract": {"AbstractText": ["CRISPR screens in cancer research."]},
        }}}]}
        trace = {}
        with patch("scientific_rag.legacy_pipeline.Entrez.esearch", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.efetch", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.read", side_effect=[{"IdList": ["42"]}, article]), \
             patch.object(MinerService, "get_pmcid_from_pmid", return_value="PMC999"), \
             patch.object(MinerService, "fetch_pmc_full_text", side_effect=ValueError("wrong PMID")):
            docs = MinerService.mine_pubmed("CRISPR in cancer research", limit=1, diagnostics=trace)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["type"], "abstract")
        self.assertIn("ValueError", trace["candidates"][0]["pmc_fetch_error"])

    def test_pubmed_miner_sends_all_original_mammography_anchors_to_entrez(self) -> None:
        article = {"PubmedArticle": [{"MedlineCitation": {"Article": {
            "ArticleTitle": "Machine learning for breast cancer detection in mammography",
            "Abstract": {"AbstractText": ["A study of machine-learning detection in breast cancer mammography." ]},
        }}}]}
        with patch("scientific_rag.legacy_pipeline.Entrez.esearch", return_value=MagicMock()) as search, \
             patch("scientific_rag.legacy_pipeline.Entrez.efetch", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.read", side_effect=[{"IdList": ["456"]}, article]), \
             patch.object(MinerService, "get_pmcid_from_pmid", return_value=None):
            MinerService.mine_pubmed(
                "machine-learning approaches for breast cancer diagnosis",
                limit=1,
                anchor_query=("How are machine-learning algorithms used for breast cancer detection in mammography, "
                              "and what evidence supports their diagnostic performance?"),
            )
        search_term = search.call_args.kwargs["term"]
        for term in ('"machine learning"[tiab]', '"breast cancer"[tiab]', "mammography[tiab]", "tomosynthesis[tiab]"):
            self.assertIn(term, search_term)
        self.assertNotIn("algorithms[tiab]", search_term)
        self.assertNotIn("detection[tiab]", search_term)
        self.assertEqual(search.call_args.kwargs["retmax"], 6)

    def test_pubmed_miner_records_returned_candidates_without_hard_relevance_rejection(self) -> None:
        article = {"PubmedArticle": [{"MedlineCitation": {"Article": {
            "ArticleTitle": "CRISPR-Cas9 in cancer therapy",
            "Abstract": {"AbstractText": ["A relevant study of CRISPR approaches in cancer treatment."]},
        }}}]}
        trace = {}
        with patch("scientific_rag.legacy_pipeline.Entrez.esearch", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.efetch", return_value=MagicMock()), \
             patch("scientific_rag.legacy_pipeline.Entrez.read", side_effect=[{"IdList": ["123"]}, article]), \
             patch.object(MinerService, "get_pmcid_from_pmid", return_value=None):
            docs = MinerService.mine_pubmed(
                "recent studies demonstrate therapeutic applications",
                limit=2,
                anchor_query="What are current uses of CRISPR in cancer treatment?",
                diagnostics=trace,
            )
        self.assertEqual(len(docs), 1)
        self.assertEqual(trace["returned_count"], 1)
        self.assertEqual(trace["indexed_count"], 1)
        self.assertEqual(trace["candidates"][0]["decision"], "retained_from_pubmed_results")
        self.assertIn("crispr", trace["candidates"][0]["matched_query_terms"])
        self.assertIn("cancer", trace["candidates"][0]["matched_query_terms"])

    def test_pubmed_relevance_screen_rejects_clear_mismatch_before_indexing_all_paper_chunks(self) -> None:
        abstract = Document(page_content="Abstract about CRISPR-based tumor therapy and cancer models.", metadata={
            "doc_id": "PMID:1", "title": "CRISPR in cancer therapy", "source": "pubmed", "type": "abstract"})
        full_text = Document(page_content="Detailed cancer therapy evidence.", metadata={
            "doc_id": "PMID:1", "title": "CRISPR in cancer therapy", "source": "pubmed", "type": "full_text"})
        mismatch = Document(page_content="Embryonic genome editing raises developmental ethics concerns.", metadata={
            "doc_id": "PMID:2", "title": "CRISPR editing in human embryos", "source": "pubmed", "type": "abstract"})
        bacteria = Document(page_content="CRISPR-Cas systems provide adaptive immunity against phages in bacteria.", metadata={
            "doc_id": "PMID:5", "title": "CRISPR immunity in bacteria", "source": "pubmed", "type": "abstract"})
        trace = {"candidates": [
            {"pmid": "1", "decision": "retained_from_pubmed_results"},
            {"pmid": "2", "decision": "retained_from_pubmed_results"},
            {"pmid": "5", "decision": "retained_from_pubmed_results"},
        ]}

        async def review(_role, _temperature, prompt, _schema):
            if "human embryos" in prompt:
                return PubMedRelevanceAssessment(decision="irrelevant", confidence=0.96, reason="The abstract concerns embryo editing, not cancer research.", intervention_match="mismatch", population_match="mismatch", evidence_role="unrelated")
            if "bacteria" in prompt:
                return PubMedRelevanceAssessment(decision="irrelevant", confidence=0.98, reason="The abstract concerns bacterial immunity, not cancer research.", intervention_match="mismatch", population_match="mismatch", evidence_role="unrelated")
            return PubMedRelevanceAssessment(decision="relevant", confidence=0.94, reason="The abstract directly concerns cancer models and therapy.", intervention_match="match", population_match="match", evidence_role="direct")

        documents = [abstract, full_text, mismatch, bacteria]
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=review)):
            kept = asyncio.run(MinerService._screen_pubmed_documents(
                documents, research_question="What are current CRISPR uses in cancer research?", diagnostics=trace,
            ))

        self.assertEqual(kept, [abstract, full_text])
        self.assertEqual(trace["candidates"][0]["decision"], "retained_relevant")
        self.assertEqual(trace["candidates"][1]["decision"], "rejected_intervention_mismatch")
        self.assertEqual(trace["candidates"][2]["decision"], "rejected_intervention_mismatch")
        self.assertIn("embryo editing", trace["candidates"][1]["decision_reason"])
        self.assertIn("bacterial immunity", trace["candidates"][2]["decision_reason"])
        self.assertEqual(trace["screened_count"], 3)
        self.assertEqual(trace["relevance_rejected_count"], 2)
        self.assertEqual(trace["indexed_count"], 2)
        self.assertEqual(abstract.metadata["relevance_label"], "relevant")

    def test_pubmed_car_t_screen_excludes_car_nk_other_disease_and_other_therapy(self) -> None:
        question = "What clinical evidence supports CAR-T therapy for relapsed or refractory B-cell lymphoma, and what are its major toxicities?"
        titles = [
            "CD19 CAR-T cells in relapsed B-cell lymphoma: a phase 2 trial",
            "Induced pluripotent stem-cell-derived CD19-directed chimeric antigen receptor natural killer cells in B-cell lymphoma: a phase 1 trial",
            "BCMA CAR-T cells in relapsed multiple myeloma",
            "Glofitamab for relapsed B-cell lymphoma",
        ]
        docs = [Document(page_content=f"Abstract describing the study intervention and outcomes for {title}.", metadata={
            "doc_id": f"PMID:{index}", "title": title, "source": "pubmed", "type": "abstract",
        }) for index, title in enumerate(titles, start=1)]
        trace = {"candidates": [{"pmid": str(index)} for index in range(1, 5)]}

        async def review(_role, _temperature, prompt, _schema):
            if "multiple myeloma" in prompt:
                return PubMedRelevanceAssessment(decision="relevant", confidence=0.97, reason="CAR-T trial, but in myeloma.", intervention_match="match", population_match="mismatch", evidence_role="unrelated")
            if "Glofitamab for" in prompt:
                return PubMedRelevanceAssessment(decision="relevant", confidence=0.95, reason="Lymphoma trial of another treatment.", intervention_match="mismatch", population_match="match", evidence_role="unrelated")
            return PubMedRelevanceAssessment(decision="relevant", confidence=0.96, reason="The trial evaluates CAR-T in B-cell lymphoma.", intervention_match="match", population_match="match", evidence_role="direct")

        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=review)) as reviewer:
            kept = asyncio.run(MinerService._screen_pubmed_documents(docs, research_question=question, diagnostics=trace))
        self.assertEqual(kept, [docs[0]])
        self.assertEqual(reviewer.await_count, 3)  # The explicit CAR-NK title is rejected before the model call.
        self.assertEqual([item["decision"] for item in trace["candidates"]], [
            "retained_relevant", "rejected_intervention_mismatch", "rejected_population_mismatch", "rejected_intervention_mismatch",
        ])
        self.assertIn("CAR-NK", trace["candidates"][1]["decision_reason"])

    def test_pubmed_background_only_cannot_support_requested_clinical_outcome(self) -> None:
        question = "What clinical evidence supports CAR-T therapy for B-cell lymphoma?"
        document = Document(page_content="A review of engineered immune cells, without CAR-T trial outcomes.", metadata={
            "doc_id": "PMID:10", "title": "Engineered immune cells for cancer", "source": "pubmed", "type": "abstract",
        })
        trace = {"candidates": [{"pmid": "10"}]}
        review = PubMedRelevanceAssessment(decision="uncertain", confidence=0.88, reason="Related background but no direct trial.", intervention_match="unclear", population_match="match", evidence_role="background")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=review)):
            kept = asyncio.run(MinerService._screen_pubmed_documents([document], research_question=question, diagnostics=trace))
        self.assertEqual(kept, [])
        self.assertEqual(trace["candidates"][0]["decision"], "rejected_background_only")

    def test_pubmed_review_cannot_be_promoted_to_primary_by_llm(self) -> None:
        question = "What clinical evidence supports CAR-T therapy for B-cell lymphoma?"
        docs = [Document(page_content="CAR-T efficacy and toxicity outcomes in B-cell lymphoma.", metadata={
            "doc_id": f"PMID:{index}", "title": f"CAR-T paper {index}", "source": "pubmed", "type": "abstract",
            "study_design": design,
        }) for index, design in [(1, "narrative_review"), (2, "systematic_review"), (3, "primary_clinical_study")]]
        trace = {"candidates": [{"pmid": str(index), "study_design": design} for index, design in [
            (1, "narrative_review"), (2, "systematic_review"), (3, "primary_clinical_study")]]}
        review = PubMedRelevanceAssessment(decision="relevant", confidence=0.93, reason="Directly addresses the question.",
                                           intervention_match="match", population_match="match", evidence_role="direct")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=review)):
            kept = asyncio.run(MinerService._screen_pubmed_documents(docs, research_question=question, diagnostics=trace))
        self.assertEqual(kept, docs[1:])
        self.assertEqual(trace["candidates"][0]["decision"], "rejected_background_only")
        self.assertEqual([doc.metadata["study_design"] for doc in kept], ["systematic_review", "primary_clinical_study"])

    def test_pubmed_clinical_budget_prefers_direct_trial_after_earlier_candidates(self) -> None:
        question = "What clinical evidence supports CAR-T therapy for B-cell lymphoma?"
        docs = [Document(page_content="CAR-T trial outcomes in B-cell lymphoma.", metadata={
            "doc_id": f"PMID:{index}", "title": f"CAR-T lymphoma trial {index}", "source": "pubmed", "type": "abstract",
        }) for index in range(1, 4)]
        trace = {"accepted_limit": 1, "candidates": [{"pmid": str(index)} for index in range(1, 4)]}

        async def review(_role, _temperature, prompt, _schema):
            confidence = 0.95 if "trial 3" in prompt else 0.8
            return PubMedRelevanceAssessment(decision="relevant", confidence=confidence, reason="Direct trial.",
                                             intervention_match="match", population_match="match", evidence_role="direct")

        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=review)):
            kept = asyncio.run(MinerService._screen_pubmed_documents(docs, research_question=question, diagnostics=trace))
        self.assertEqual(kept, [docs[2]])
        self.assertEqual([item["decision"] for item in trace["candidates"]], [
            "not_indexed_evidence_budget", "not_indexed_evidence_budget", "retained_relevant",
        ])
        self.assertEqual(trace["indexed_count"], 1)

    def test_pubmed_clinical_budget_prefers_primary_study_over_systematic_review(self) -> None:
        question = "What clinical evidence supports CAR-T therapy for B-cell lymphoma?"
        designs = ["systematic_review", "primary_clinical_study"]
        docs = [Document(page_content="CAR-T outcomes in B-cell lymphoma.", metadata={
            "doc_id": f"PMID:{index}", "title": f"CAR-T lymphoma evidence {index}",
            "source": "pubmed", "type": "abstract", "study_design": design,
        }) for index, design in enumerate(designs, start=1)]
        trace = {"accepted_limit": 1, "candidates": [
            {"pmid": str(index), "study_design": design} for index, design in enumerate(designs, start=1)
        ]}
        review = PubMedRelevanceAssessment(decision="relevant", confidence=0.9, reason="Applicable clinical evidence.",
                                           intervention_match="match", population_match="match", evidence_role="direct")
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=review)):
            kept = asyncio.run(MinerService._screen_pubmed_documents(docs, research_question=question, diagnostics=trace))
        self.assertEqual(kept, [docs[1]])
        self.assertEqual(trace["candidates"][0]["decision"], "not_indexed_evidence_budget")

    def test_pubmed_relevance_screen_retains_ambiguous_candidates_for_recall(self) -> None:
        document = Document(page_content="Broad CRISPR applications across biomedicine.", metadata={
            "doc_id": "PMID:3", "title": "CRISPR Advancements for Human Health", "source": "pubmed", "type": "abstract"})
        trace = {"candidates": [{"pmid": "3"}]}
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(return_value=
            PubMedRelevanceAssessment(decision="uncertain", confidence=0.51, reason="Cancer is mentioned only as one broad application.", intervention_match="unclear", population_match="unclear", evidence_role="background"))):
            kept = asyncio.run(MinerService._screen_pubmed_documents(
                [document], research_question="CRISPR applications in cancer", diagnostics=trace,
            ))
        self.assertEqual(kept, [document])
        self.assertEqual(trace["candidates"][0]["decision"], "retained_uncertain")

    def test_pubmed_relevance_screen_does_not_index_when_judge_fails(self) -> None:
        document = Document(page_content="Abstract text", metadata={
            "doc_id": "PMID:4", "title": "Candidate paper", "source": "pubmed", "type": "abstract"})
        trace = {"candidates": [{"pmid": "4"}]}
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=TimeoutError("judge timed out"))), \
             patch("scientific_rag.legacy_pipeline.logger.warning"):
            kept = asyncio.run(MinerService._screen_pubmed_documents(
                [document], research_question="research question", diagnostics=trace,
            ))
        self.assertEqual(kept, [])
        self.assertEqual(trace["candidates"][0]["decision"], "screening_failed_not_indexed")
        self.assertEqual(trace["indexed_count"], 0)

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

    def test_medical_anchor_forces_pubmed_even_if_planner_marks_step_technical(self) -> None:
        plan = QueryPlan(plan=[SubQuery(
            query="Recent machine-learning approaches to red light therapy",
            source_type="academic", domains=["medical", "technical"], reasoning="test",
        )])
        Orchestrator.enforce_source_policy(plan, anchor_query="What are risks of red light therapy?")
        self.assertEqual(plan.plan[0].external_sources, ["pubmed"])

    def test_explicit_medical_machine_learning_question_routes_to_both_corpora(self) -> None:
        plan = QueryPlan(plan=[SubQuery(
            query="Machine learning for mammography-based breast cancer diagnosis",
            source_type="academic", domains=["medical", "technical"], reasoning="test",
        )])
        Orchestrator.enforce_source_policy(plan, anchor_query="How are machine-learning methods used for breast cancer diagnosis?")
        self.assertEqual(plan.plan[0].external_sources, ["pubmed", "arxiv"])

    def test_medical_alphafold_question_routes_to_both_corpora(self) -> None:
        plan = QueryPlan(plan=[SubQuery(
            query="AlphaFold structure prediction for drug discovery",
            source_type="academic", domains=["medical", "technical"], reasoning="test",
        )])
        Orchestrator.enforce_source_policy(
            plan,
            anchor_query="What are the strengths and limitations of AlphaFold-style structure prediction for drug discovery?",
        )
        self.assertEqual(plan.plan[0].external_sources, ["pubmed", "arxiv"])

    def test_no_fresh_sources_aborts_without_synthesis(self) -> None:
        async def verify() -> None:
            class VectorService:
                restricted = False
                def restrict_final_retrieval_to_current_run(self):
                    self.restricted = True

            orchestrator = object.__new__(Orchestrator)
            orchestrator.vs = VectorService()
            orchestrator.progress_callback = None
            orchestrator.generate_plan = AsyncMock(return_value=QueryPlan(plan=[SubQuery(
                query="specific academic query", source_type="academic", domains=["medical"], reasoning="test",
            )]))
            orchestrator.filter_plan = AsyncMock(return_value=(QueryPlan(plan=[SubQuery(
                query="specific academic query", source_type="academic", domains=["medical"], reasoning="test",
            )]), []))
            orchestrator.ask_db = AsyncMock(return_value=(None, {"decision": "reuse_existing"}))
            orchestrator.run_mining_operation = AsyncMock(return_value=([], 0, {"pubmed": {"requested": 1, "collected": 0}}, [], []))
            orchestrator.synthesize_final_report = AsyncMock()
            result = await orchestrator.execute_full_study("medical research question", cache_mode="refresh_sources")
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertTrue(orchestrator.vs.restricted)
            orchestrator.ask_db.assert_not_awaited()
            orchestrator.synthesize_final_report.assert_not_awaited()

        asyncio.run(verify())

    def test_no_direct_clinical_source_returns_an_explicit_evidence_gap(self) -> None:
        async def verify() -> None:
            class VectorService:
                def restrict_final_retrieval_to_current_run(self):
                    pass

            orchestrator = object.__new__(Orchestrator)
            orchestrator.vs = VectorService()
            orchestrator.progress_callback = None
            plan = QueryPlan(plan=[SubQuery(
                query="CAR-T trials in B-cell lymphoma", source_type="academic", domains=["medical"], reasoning="Direct evidence",
            )])
            orchestrator.generate_plan = AsyncMock(return_value=plan)
            orchestrator.filter_plan = AsyncMock(return_value=(plan, []))
            orchestrator.run_mining_operation = AsyncMock(return_value=(
                [], 0, {"pubmed": {"requested": 1, "collected": 0, "errors": []}},
                [{"candidates": [{"decision": "rejected_intervention_mismatch"}]}], [],
            ))
            orchestrator.synthesize_final_report = AsyncMock()
            result = await orchestrator.execute_full_study("What clinical evidence supports CAR-T in B-cell lymphoma?", cache_mode="refresh_sources")
            self.assertIn("did not find direct evidence", result["report"]["answer"])
            self.assertEqual(result["report"]["raw_docs_used"], [])
            orchestrator.synthesize_final_report.assert_not_awaited()

        asyncio.run(verify())

    def test_planning_failure_does_not_downgrade_to_internal_knowledge(self) -> None:
        async def verify() -> None:
            orchestrator = object.__new__(Orchestrator)
            orchestrator.progress_callback = None
            orchestrator.parser = MagicMock()
            with patch("scientific_rag.legacy_pipeline.ainvoke_with_fallback", new=AsyncMock(side_effect=TimeoutError("provider unavailable"))):
                with self.assertRaisesRegex(RuntimeError, "Research planning failed"):
                    await orchestrator.generate_plan("What is the evidence?")

        asyncio.run(verify())

    def test_gatekeeper_does_not_drop_only_topic_aligned_query_as_redundant(self) -> None:
        async def verify() -> None:
            orchestrator = object.__new__(Orchestrator)
            orchestrator.progress_callback = None
            orchestrator.review_parser = MagicMock()
            plan = QueryPlan(plan=[SubQuery(
                query="Current approaches to quantum error correction using surface codes",
                source_type="academic", domains=["physics"], reasoning="Search the requested topic.",
            )])
            rejection = {
                "relevance_score": 0.2,
                "should_skip": True,
                "critique": "The sub-query is a near-identical re-phrasing of the original question, so it is redundant.",
            }
            with patch(
                "scientific_rag.legacy_pipeline.ainvoke_with_fallback",
                new=AsyncMock(return_value=rejection),
            ):
                filtered, reviews = await orchestrator.filter_plan(
                    "What are current approaches to quantum error correction using surface codes?", plan,
                )
            self.assertEqual(filtered.plan, plan.plan)
            self.assertEqual(reviews, [rejection])

        asyncio.run(verify())

    def test_gatekeeper_still_rejects_single_off_topic_query(self) -> None:
        async def verify() -> None:
            orchestrator = object.__new__(Orchestrator)
            orchestrator.progress_callback = None
            orchestrator.review_parser = MagicMock()
            plan = QueryPlan(plan=[SubQuery(
                query="How to grow tomatoes indoors",
                source_type="academic", domains=["general"], reasoning="Unrelated.",
            )])
            rejection = {"relevance_score": 0.1, "should_skip": True, "critique": "Irrelevant to the original question."}
            with patch(
                "scientific_rag.legacy_pipeline.ainvoke_with_fallback",
                new=AsyncMock(return_value=rejection),
            ):
                with self.assertRaisesRegex(RuntimeError, "no safe query remains"):
                    await orchestrator.filter_plan("surface-code quantum error correction", plan)

        asyncio.run(verify())

    def test_empty_current_run_retrieval_does_not_call_expert(self) -> None:
        async def verify() -> None:
            class EmptyVectorService:
                async def retrieve_context(self, query, top_n):
                    return []

            agent = object.__new__(ResearchAgent)
            agent.vector_service = EmptyVectorService()
            agent.max_synthesis_subquestions = 1
            agent.retrieval_documents = 1
            agent.max_context_documents = 1
            agent.progress_callback = None
            agent.decompose_query_async = AsyncMock(return_value=["topic evidence"])
            with patch("scientific_rag.legacy_pipeline.ainvoke_with_fallback", new=AsyncMock()) as invoke:
                with self.assertRaisesRegex(RuntimeError, "No evidence from this research run"):
                    await agent.run_research("research question")
            invoke.assert_not_awaited()

        asyncio.run(verify())

    def test_synthesis_marks_full_text_prior_work_and_arxiv_preprint(self) -> None:
        async def verify() -> None:
            class VectorService:
                async def retrieve_context(self, query, top_n):
                    return [
                        Document(page_content="Ueda et al. reported an AUC of 0.93 in a prior study.", metadata={
                            "doc_id": "PMID:40608200", "title": "Japanese mammography study", "source": "pubmed",
                            "type": "full_text", "section": "Discussion", "study_design": "primary_clinical_study",
                        }),
                        Document(page_content="A validation case pF1 of 0.56 was reported on RSNA images.", metadata={
                            "doc_id": "arxiv:2307.11274", "title": "Screening Mammography Breast Cancer Detection",
                            "source": "arxiv",
                        }),
                    ]

            agent = ResearchAgent(VectorService(), max_synthesis_subquestions=1)
            agent.decompose_query_async = AsyncMock(return_value=["mammography AI performance"])
            with patch("scientific_rag.legacy_pipeline.ainvoke_with_fallback", new=AsyncMock(return_value="Limited evidence [S1, S2].")) as invoke:
                report = await agent.run_research("How does AI perform in mammography?")
            context = invoke.await_args.args[3]["context"]
            self.assertIn("full-text section Discussion", context)
            self.assertIn("OTHER studies", context)
            self.assertIn("arXiv preprint", context)
            self.assertEqual(report["raw_docs_used"][0].metadata["citation_id"], "S1")
            self.assertEqual(report["raw_docs_used"][1].metadata["citation_id"], "S2")

        asyncio.run(verify())

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

    def test_current_surface_code_research_overrides_internal_planner_label(self) -> None:
        plan = QueryPlan(plan=[SubQuery(
            query="Fundamental principles and current surface-code decoders",
            source_type="internal_knowledge", domains=["physics"], reasoning="background",
        )])
        selected = Orchestrator.enforce_source_policy(
            plan, anchor_query="What are current approaches to quantum error correction using surface codes?",
        )
        self.assertEqual(selected.plan[0].source_type, "academic")
        self.assertEqual(selected.plan[0].external_sources, ["arxiv"])

    def test_general_feynman_learning_question_stays_internal(self) -> None:
        plan = QueryPlan(plan=[SubQuery(
            query="Explain the Feynman learning technique",
            source_type="internal_knowledge", domains=["education"], reasoning="definition",
        )])
        selected = Orchestrator.enforce_source_policy(plan, anchor_query="What is the Feynman learning technique?")
        self.assertEqual(selected.plan[0].source_type, "internal_knowledge")
        self.assertEqual(selected.plan[0].external_sources, [])

    def test_source_mining_starts_pubmed_and_arxiv_concurrently(self) -> None:
        async def verify() -> None:
            started = []
            both_started = asyncio.Event()

            async def mine(query, source, limit, **kwargs):
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
                docs, count, diagnostics, pubmed_traces, arxiv_traces = await orchestrator.run_mining_operation(plan)
            self.assertEqual(started, ["pubmed", "arxiv"])
            self.assertEqual(count, 2)
            self.assertEqual({doc.page_content for doc in docs}, {"pubmed", "arxiv"})
            self.assertEqual(diagnostics["arxiv"]["collected"], 1)
            self.assertEqual(len(pubmed_traces), 1)
            self.assertEqual(len(arxiv_traces), 1)

        asyncio.run(verify())

    def test_source_timeout_does_not_discard_other_source_or_hide_reason(self) -> None:
        async def verify() -> None:
            async def mine(query, source, limit, **kwargs):
                if source == "pubmed":
                    await asyncio.sleep(1)
                return [Document(page_content=source, metadata={"source": source})]

            orchestrator = object.__new__(Orchestrator)
            orchestrator.progress_callback = None
            orchestrator.papers_per_source = 1
            plan = QueryPlan(plan=[SubQuery(query="technical question", source_type="academic", domains=["technical"], external_sources=["pubmed", "arxiv"], reasoning="test")])
            with patch.dict(os.environ, {"MINING_SOURCE_TIMEOUT_SECONDS": "0.01"}), \
                 patch("scientific_rag.legacy_pipeline.MinerService.mine", side_effect=mine):
                docs, count, diagnostics, pubmed_traces, _ = await orchestrator.run_mining_operation(plan)
            self.assertEqual(count, 1)
            self.assertEqual(docs[0].metadata["source"], "arxiv")
            self.assertIn("exceeded", diagnostics["pubmed"]["errors"][0])
            self.assertTrue(pubmed_traces[0]["errors"])

        asyncio.run(verify())

    def test_entrez_transport_passes_timeout_to_each_network_request(self) -> None:
        with patch.dict(os.environ, {"ENTREZ_REQUEST_TIMEOUT_SECONDS": "4"}), \
             patch("scientific_rag.legacy_pipeline._entrez_urlopen", return_value="handle") as opener:
            self.assertEqual(_bounded_entrez_urlopen("request"), "handle")
        opener.assert_called_once_with("request", timeout=4.0)

    def test_entrez_transport_respects_thread_local_fetch_deadline(self) -> None:
        _entrez_request_context.deadline = time.monotonic() - 1
        try:
            with patch("scientific_rag.legacy_pipeline._entrez_urlopen") as opener:
                with self.assertRaisesRegex(TimeoutError, "budget exhausted"):
                    _bounded_entrez_urlopen("request")
            opener.assert_not_called()
        finally:
            del _entrez_request_context.deadline

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

        trace = {}
        with patch("scientific_rag.legacy_pipeline.arxiv.Client", Client):
            docs = MinerService.mine_arxiv("CRISPR cancer research", limit=1, diagnostics=trace)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["doc_id"], "arxiv:2501.00001v2")
        self.assertEqual(docs[0].metadata["source_url"], "https://arxiv.org/abs/2501.00001v2")
        self.assertEqual(docs[0].metadata["type"], "abstract")
        self.assertEqual(Client.search_query, "all:cancer AND all:crispr")
        self.assertEqual(trace["returned_count"], 2)
        self.assertEqual(trace["candidates"][0]["decision"], "rejected_lexical_overlap")
        self.assertEqual(trace["candidates"][1]["decision"], "awaiting_llm_relevance_screen")
        self.assertIsInstance(trace["candidates"][1]["specific_terms"], list)

    def test_arxiv_query_preserves_neural_operator_and_pde_phrases(self) -> None:
        query = MinerService._format_arxiv_query(
            "What recent benchmark results exist?",
            "What neural operator architectures solve partial differential equations?",
        )
        self.assertEqual(
            query,
            'all:"neural operator" AND all:"partial differential equation"',
        )

    def test_arxiv_query_preserves_surface_code_phrase(self) -> None:
        query = MinerService._format_arxiv_query(
            "recent decoder implementations",
            "What are current surface-code approaches to quantum error correction?",
        )
        self.assertEqual(
            query,
            'all:"surface code" AND all:"quantum error correction"',
        )

    def test_arxiv_session_sets_transport_timeout(self) -> None:
        session = _BoundedArxivSession()
        with patch.object(_BoundedArxivSession, "delay_seconds", 0), patch.object(
            requests.Session, "request", return_value=object(),
        ) as send:
            session.get("https://export.arxiv.org/api/query")
        self.assertEqual(send.call_args.kwargs["timeout"], (5.0, 25.0))
        session.close()

    def test_arxiv_transport_timeout_is_bounded_per_retry_ladder_step(self) -> None:
        trace = {}
        with patch.object(_BoundedArxivSession, "delay_seconds", 0), patch.object(
            requests.Session, "request", side_effect=requests.exceptions.ReadTimeout("arXiv timed out"),
        ) as send:
            docs = asyncio.run(MinerService.mine(
                "surface code quantum error correction", source="arxiv", limit=1, diagnostics=trace,
            ))
        self.assertEqual(docs, [])
        # Three query-ladder steps, each with two bounded transport attempts.
        self.assertEqual(send.call_count, 6, trace["errors"])
        self.assertEqual(len(trace["attempts"]), 3)
        self.assertIn("ReadTimeout", trace["errors"][0])

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

        trace = {}
        with patch("scientific_rag.legacy_pipeline.arxiv.Client", Client):
            docs = MinerService.mine_arxiv("surface code quantum error correction", limit=2, diagnostics=trace)
        self.assertEqual([doc.metadata["title"] for doc in docs], ["Surface-code decoders"])
        self.assertIn("rejected_missing_specific_anchor", [item["decision"] for item in trace["candidates"]])

    def test_arxiv_screen_rejects_clear_mismatch_but_keeps_uncertain(self) -> None:
        relevant = Document(page_content="A detailed evaluation of neural decoders for rotated surface codes. " * 3, metadata={
            "doc_id": "arxiv:2401.00001v1", "title": "Neural decoders for rotated surface codes", "source": "arxiv"})
        uncertain = Document(page_content="An overview of quantum computing architectures with limited discussion of error correction. " * 3, metadata={
            "doc_id": "arxiv:2401.00002v1", "title": "Quantum computing architectures", "source": "arxiv"})
        irrelevant = Document(page_content="A paper about image recognition and object detection benchmarks. " * 3, metadata={
            "doc_id": "arxiv:2401.00003v1", "title": "Image recognition benchmarks", "source": "arxiv"})
        trace = {"candidates": [
            {"arxiv_id": "2401.00001v1"}, {"arxiv_id": "2401.00002v1"}, {"arxiv_id": "2401.00003v1"},
        ]}

        async def review(_role, _temperature, _prompt, _schema):
            if _schema is not ArxivRelevanceAssessment:
                raise AssertionError("Expected the arXiv relevance schema")
            if "Image recognition" in _prompt:
                return ArxivRelevanceAssessment(decision="irrelevant", confidence=0.97, reason="The abstract is about image benchmarks, not quantum error correction.")
            if "Quantum computing architectures" in _prompt:
                return ArxivRelevanceAssessment(decision="irrelevant", confidence=0.51, reason="The abstract only briefly mentions error correction.")
            return ArxivRelevanceAssessment(decision="relevant", confidence=0.94, reason="The paper evaluates decoders for the requested surface code family.")

        async def verify() -> None:
            with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=review)):
                screened = await MinerService._screen_arxiv_documents(
                    [relevant, uncertain, irrelevant], research_question="How do decoders work for rotated surface codes?", diagnostics=trace,
                )
            self.assertEqual(screened, [relevant, uncertain])
            self.assertEqual(trace["candidates"][0]["decision"], "retained_relevant")
            self.assertEqual(trace["candidates"][1]["decision"], "retained_uncertain")
            self.assertEqual(trace["candidates"][2]["decision"], "rejected_irrelevant")
            self.assertEqual(trace["relevance_rejected_count"], 1)
            self.assertEqual(trace["indexed_count"], 2)
            self.assertEqual(uncertain.metadata["relevance_label"], "uncertain")
            self.assertEqual(relevant.metadata["relevance_label"], "relevant")

        asyncio.run(verify())

    def test_arxiv_screening_failure_does_not_index_candidate(self) -> None:
        document = Document(page_content="An extended abstract with enough detail. " * 5, metadata={
            "doc_id": "arxiv:2401.00004v1", "title": "Candidate", "source": "arxiv"})
        trace = {"candidates": [{"arxiv_id": "2401.00004v1"}]}
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=AsyncMock(side_effect=TimeoutError("judge timed out"))), \
             patch("scientific_rag.legacy_pipeline.logger.warning"):
            kept = asyncio.run(MinerService._screen_arxiv_documents([document], research_question="a question", diagnostics=trace))
        self.assertEqual(kept, [])
        self.assertEqual(trace["candidates"][0]["decision"], "screening_failed_not_indexed")
        self.assertEqual(trace["indexed_count"], 0)

    def test_mammography_modality_filter_rejects_substitutes_but_keeps_multimodal(self) -> None:
        question = "How is machine learning used for breast cancer diagnosis in mammography?"
        mismatch = MinerService._mammography_modality_mismatch
        self.assertIn("cytology", mismatch(question, "Wisconsin Diagnostic Breast Cancer Dataset", "Mammography is a screening method."))
        self.assertIn("thermal", mismatch(question, "Thermal imaging for breast cancer detection", "Mammography is also used."))
        self.assertIn("ultrasound", mismatch(question, "Ultrasound-based breast cancer detection", "Mammography is also used."))
        self.assertIsNone(mismatch(question, "Joint classification of mammography and ultrasound", "Both modalities were analyzed."))
        self.assertIsNone(mismatch("How is breast cancer diagnosed?", "Thermal imaging for breast cancer", "A study of thermal images."))

    def test_arxiv_mammography_screen_records_modality_rejection_without_llm_call(self) -> None:
        question = "How is machine learning used for breast cancer diagnosis in mammography?"
        bad = Document(page_content="We train on the Wisconsin Diagnostic Breast Cancer Dataset using FNA cytology data. " * 3,
                       metadata={"doc_id": "arxiv:2501.00001v1", "title": "Wisconsin Diagnostic Breast Cancer Dataset classification", "source": "arxiv"})
        good = Document(page_content="We jointly analyze screening mammograms and ultrasound images for breast lesion classification. " * 3,
                        metadata={"doc_id": "arxiv:2501.00002v1", "title": "Joint neural analysis of mammography and ultrasound", "source": "arxiv"})
        trace = {"candidates": [{"arxiv_id": "2501.00001v1"}, {"arxiv_id": "2501.00002v1"}]}
        review = AsyncMock(return_value=ArxivRelevanceAssessment(decision="relevant", confidence=0.95, reason="Studies mammograms."))
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=review):
            kept = asyncio.run(MinerService._screen_arxiv_documents([bad, good], research_question=question, diagnostics=trace))
        self.assertEqual(kept, [good])
        self.assertEqual(review.await_count, 1)
        self.assertEqual(trace["candidates"][0]["decision"], "rejected_modality_mismatch")
        self.assertEqual(trace["candidates"][1]["decision"], "retained_relevant")

    def test_pubmed_mammography_screen_records_modality_rejection(self) -> None:
        question = "How is machine learning used for breast cancer diagnosis in mammography?"
        bad = Document(page_content="We use thermal imaging to identify breast cancer; mammography is a conventional comparator. " * 2,
                       metadata={"doc_id": "PMID:999001", "title": "Thermal imaging for breast cancer detection", "source": "pubmed", "type": "abstract"})
        trace = {"candidates": [{"pmid": "999001"}]}
        review = AsyncMock()
        with patch("scientific_rag.legacy_pipeline.ainvoke_structured", new=review):
            kept = asyncio.run(MinerService._screen_pubmed_documents([bad], research_question=question, diagnostics=trace))
        self.assertEqual(kept, [])
        review.assert_not_awaited()
        self.assertEqual(trace["candidates"][0]["decision"], "rejected_modality_mismatch")

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
