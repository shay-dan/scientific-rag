from pathlib import Path
import sys
import unittest
import asyncio
from unittest.mock import patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.evaluation import NativeGeminiEmbeddings, diagnose_reference_claims, diagnose_retrieved_contexts, score_reference_context_semantically, summarize_context_precision_views, summarize_scores
from app.models import EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchEvidence
from app.services import _timeline_phase


class EvaluationModelTests(unittest.TestCase):
    def test_timeline_phase_does_not_misclassify_research_as_retrieval(self) -> None:
        self.assertEqual(_timeline_phase("Planning the research approach", "orchestrator"), "planning")
        self.assertEqual(_timeline_phase("Preparing your answer from the evidence", "expert"), "synthesis")
        self.assertEqual(_timeline_phase("Searching the evidence database", "embedding"), "indexing")

    def test_evaluation_tables_and_required_columns_are_declared(self) -> None:
        self.assertIn("evaluation_datasets", EvaluationDataset.metadata.tables)
        self.assertIn("evaluation_cases", EvaluationCase.metadata.tables)
        self.assertIn("evaluation_runs", EvaluationRun.metadata.tables)
        self.assertIn("evaluation_results", EvaluationResult.metadata.tables)
        self.assertIn("research_evidence", ResearchEvidence.metadata.tables)
        self.assertIn("reference_answer", EvaluationCase.__table__.columns)
        self.assertIn("reference_claims", EvaluationCase.__table__.columns)
        self.assertIn("metrics", EvaluationRun.__table__.columns)
        self.assertIn("judge_scores", EvaluationResult.__table__.columns)
        self.assertIn("ordinal", ResearchEvidence.__table__.columns)

    def test_research_evidence_is_ordered_and_linked_to_a_research_run(self) -> None:
        self.assertEqual(next(iter(ResearchEvidence.__table__.c.research_run_id.foreign_keys)).column.table.name, "research_runs")
        self.assertTrue(any("uq_research_evidence_ordinal" == item.name for item in ResearchEvidence.__table__.constraints))

    def test_cases_are_linked_to_datasets_and_runs_are_linked_to_datasets(self) -> None:
        self.assertEqual(next(iter(EvaluationCase.__table__.c.dataset_id.foreign_keys)).column.table.name, "evaluation_datasets")
        self.assertEqual(next(iter(EvaluationRun.__table__.c.dataset_id.foreign_keys)).column.table.name, "evaluation_datasets")

    def test_score_summary_averages_only_judged_results(self) -> None:
        summary = summarize_scores([
            {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6},
            {"faithfulness": 0.7, "answer_relevancy": 0.6, "context_precision": 0.5, "context_recall": 0.4},
        ])
        self.assertEqual(summary["scored_cases"], 2)
        self.assertEqual(summary["faithfulness"], 0.8)
        self.assertEqual(summary["context_recall"], 0.5)

    def test_context_precision_views_keep_ragas_and_claim_aware_scores_separate(self) -> None:
        summary = summarize_context_precision_views([
            {
                "ragas_context_precision": 0.0,
                "retrieval_diagnostics": {
                    "strict_ranked_precision": 1.0,
                    "partial_credit_ranked_precision": 0.8,
                },
            },
            {
                "context_precision": 0.2,
                "retrieval_diagnostics": {
                    "strict_ranked_precision": 0.5,
                    "partial_credit_ranked_precision": 0.6,
                },
            },
        ])
        self.assertEqual(summary["ragas_context_precision"], 0.1)
        self.assertEqual(summary["claim_aware_strict_precision"], 0.75)
        self.assertEqual(summary["claim_aware_partial_precision"], 0.7)
        self.assertEqual(summary["claim_aware_scored_cases"], 2)

    def test_ragas_is_a_declared_runtime_dependency(self) -> None:
        requirements = (project_root / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("ragas>=0.2,<0.3", requirements)
        self.assertIn("rapidfuzz>=3,<4", requirements)

    def test_reference_context_is_used_by_the_ragas_evaluation_path(self) -> None:
        evaluation = (project_root / "app" / "evaluation.py").read_text(encoding="utf-8")
        worker = (project_root / "app" / "worker.py").read_text(encoding="utf-8")
        self.assertIn("score_reference_context_semantically", evaluation)
        self.assertIn("REFERENCE_CONTEXT_PROMPT", evaluation)
        self.assertIn("reference_context=case.reference_context", worker)
        self.assertIn("context_metrics_mode", worker)

    def test_reference_context_semantic_judge_parses_scores(self) -> None:
        class FakeResponse:
            parsed = {"context_precision": 0.8, "context_recall": 0.6, "rationale": "Relevant overlap."}

        class FakeModels:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        with patch("app.evaluation.genai.Client", return_value=FakeClient()):
            scores = asyncio.run(score_reference_context_semantically(
                question="What is a surface code?",
                reference_context="A verified surface-code excerpt.",
                retrieved_contexts=["A retrieved surface-code excerpt."],
                api_key="unused",
                judge_model="gemini-test",
            ))
        self.assertEqual(scores, {"context_precision": 0.8, "context_recall": 0.6})

    def test_claim_diagnostic_returns_auditable_coverage(self) -> None:
        claims = ["Claim one.", "Claim two."]

        class FakeResponse:
            parsed = {"verdicts": [
                {"claim": claims[0], "verdict": "supported", "supporting_contexts": [1], "rationale": "Direct support."},
                {"claim": claims[1], "verdict": "partial", "supporting_contexts": [2], "rationale": "One element is missing."},
            ]}

        class FakeModels:
            def generate_content(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        with patch("app.evaluation.genai.Client", return_value=FakeClient()):
            result = asyncio.run(diagnose_reference_claims(
                question="Question?",
                reference_claims=claims,
                retrieved_contexts=["Evidence one.", "Evidence two."],
                api_key="unused",
                judge_model="gemini-test",
            ))
        self.assertEqual(result["supported_claims"], 1)
        self.assertEqual(result["partial_claims"], 1)
        self.assertEqual(result["strict_claim_recall"], 0.5)
        self.assertEqual(result["partial_credit_claim_recall"], 0.75)
        self.assertEqual(result["verdicts"][0]["supporting_contexts"], [1])

    def test_retrieval_diagnostic_explains_each_rank_and_computes_precision(self) -> None:
        class FakeResponse:
            parsed = {"verdicts": [
                {"rank": 1, "verdict": "relevant", "supported_claims": [1], "rationale": "Direct evidence."},
                {"rank": 2, "verdict": "irrelevant", "supported_claims": [], "rationale": "Different topic."},
                {"rank": 3, "verdict": "partial", "supported_claims": [2], "rationale": "Background only."},
            ]}

        class FakeModels:
            def generate_content(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        with patch("app.evaluation.genai.Client", return_value=FakeClient()):
            result = asyncio.run(diagnose_retrieved_contexts(
                question="Question?",
                reference_answer="Reference.",
                reference_claims=["Claim one.", "Claim two."],
                retrieved_contexts=["Direct.", "Noise.", "Background."],
                api_key="unused",
                judge_model="gemini-test",
            ))
        self.assertEqual(result["relevant_excerpts"], 1)
        self.assertEqual(result["partial_excerpts"], 1)
        self.assertEqual(result["irrelevant_excerpts"], 1)
        self.assertEqual(result["strict_ranked_precision"], 1.0)
        self.assertGreater(result["partial_credit_ranked_precision"], 0.7)
        self.assertEqual(result["verdicts"][0]["supported_claims"], [1])

    def test_evaluation_result_template_exposes_diagnostic_evidence(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "evaluation_result.html").read_text(encoding="utf-8")
        self.assertIn("Generated answer", template)
        self.assertIn("Retrieved excerpts", template)
        self.assertIn("Source records", template)
        self.assertIn("Download JSON", template)
        self.assertIn("case-metrics", template)
        self.assertIn("Claim-level retrieval coverage", template)
        self.assertIn("Per-excerpt retrieval audit", template)
        self.assertIn("ragas_context_precision", template)
        self.assertIn("verified_context_precision", template)
        self.assertIn("strict_claim_recall", template)
        self.assertIn("citation_id", template)
        self.assertIn("Source retrieval diagnostics", template)
        self.assertIn("cache_diagnostics", template)
        self.assertIn("Context precision view", template)
        self.assertIn("Claim-aware (Recommended)", template)
        self.assertIn("Original Ragas", template)
        self.assertIn("Compare both", template)
        self.assertIn('data-precision-mode="claim"', template)
        self.assertIn('data-precision-mode="ragas"', template)
        self.assertIn("useful partial evidence can receive zero", template)

    def test_research_result_template_displays_stable_citation_ids(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "research_result.html").read_text(encoding="utf-8")
        self.assertIn("source.citation_id", template)
        self.assertIn("Source retrieval diagnostics", template)
        self.assertIn("cache_diagnostics", template)
        self.assertIn("LLM provider diagnostics", template)
        self.assertIn("response.llm_diagnostics", template)

    def test_provider_diagnostics_page_exposes_operational_metrics(self) -> None:
        template = (project_root / "app" / "templates" / "diagnostics.html").read_text(encoding="utf-8")
        self.assertIn("Total calls", template)
        self.assertIn("average latency", template)
        self.assertIn("fallbacks", template)

    def test_retrieval_inspector_shows_pubmed_search_candidates_and_decisions(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "ops_retrieval.html").read_text(encoding="utf-8")
        self.assertIn("PubMed search and mining", template)
        self.assertIn("arXiv search and relevance screening", template)
        self.assertIn("arxiv_traces", template)
        self.assertIn("ArxivRelevanceAssessment", (project_root / "src" / "scientific_rag" / "legacy_pipeline.py").read_text(encoding="utf-8"))
        self.assertIn("matched_query_terms", template)
        self.assertIn("abstract_preview", template)
        self.assertIn("decision_reason", template)
        self.assertIn("relevance_confidence", template)
        self.assertIn("relevance_rejected_count", template)
        self.assertIn("screened_count", template)
        self.assertIn("uncertain papers are retained", template)
        self.assertIn("PostgreSQL hybrid retrieval", template)

    def test_synthesis_prompt_requires_evidence_for_every_technical_claim(self) -> None:
        pipeline = (project_root / "src" / "scientific_rag" / "legacy_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("Every factual or technical claim must be directly supported", pipeline)
        self.assertIn("Do not add general background", pipeline)
        self.assertIn("Sources explicitly marked relevance uncertain are secondary evidence", pipeline)
        self.assertIn("Do not combine rates from different denominators", pipeline)
        self.assertIn("may describe a PRIOR CITED STUDY", pipeline)
        self.assertIn("arXiv benchmark as a prospective clinical outcome study", pipeline)
        self.assertIn("call it a historical comparison", pipeline)

    def test_evaluation_page_exposes_recent_run_history(self) -> None:
        template = (project_root / "app" / "templates" / "evaluation.html").read_text(encoding="utf-8")
        self.assertIn("Recent evaluations", template)
        self.assertIn("/ui/evaluations/{{ run.id }}/status", template)
        self.assertIn('name="mode" value="standard"', template)

    def test_evaluation_worker_uses_the_persisted_research_mode(self) -> None:
        worker = (project_root / "app" / "worker.py").read_text(encoding="utf-8")
        self.assertIn('ResearchMode(run.metrics.get("mode", "fast"))', worker)
        self.assertIn("run.metrics[\"mode\"] = mode.value", worker)
        self.assertIn("case.reference_claims", worker)
        self.assertIn('result.judge_scores["claim_diagnostics"]', worker)
        self.assertIn('result.judge_scores["retrieval_diagnostics"]', worker)
        self.assertIn("evaluation_case_timeout_seconds", worker)
        self.assertIn('result.judge_scores["claim_diagnostics_error"]', worker)

    def test_evaluation_progress_template_displays_elapsed_time_and_estimate(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "evaluation_progress.html").read_text(encoding="utf-8")
        self.assertIn("Elapsed", template)
        self.assertIn("Estimated remaining", template)
        self.assertIn("research_completed", template)
        self.assertIn("evaluation_completed", template)

    def test_evaluation_worker_has_a_separate_timeout_and_persists_progress(self) -> None:
        worker = (project_root / "app" / "worker.py").read_text(encoding="utf-8")
        self.assertIn("evaluation_job_timeout_seconds", worker)
        self.assertIn("max_tries=1", worker)
        self.assertIn("_mark_interrupted_evaluation_runs", worker)
        self.assertIn("evaluation_job_timeout_seconds + 300", worker)
        self.assertIn('"research_completed"', worker)
        self.assertIn('"evaluation_completed"', worker)

    def test_native_gemini_embeddings_use_the_current_api_shape(self) -> None:
        class FakeVector:
            values = [0.1, 0.2]
        class FakeResponse:
            embeddings = [FakeVector()]
        class FakeModels:
            def embed_content(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()
        class FakeClient:
            models = FakeModels()
        embeddings = NativeGeminiEmbeddings("unused", "gemini-embedding-2", 768, client=FakeClient())
        self.assertEqual(embeddings.embed_query("test"), [0.1, 0.2])
        self.assertEqual(embeddings.client.models.kwargs["model"], "gemini-embedding-2")
