from pathlib import Path
import sys
import unittest
import asyncio
from unittest.mock import patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.evaluation import NativeGeminiEmbeddings, score_reference_context_semantically, summarize_scores
from app.models import EvaluationCase, EvaluationDataset, EvaluationResult, EvaluationRun, ResearchEvidence


class EvaluationModelTests(unittest.TestCase):
    def test_evaluation_tables_and_required_columns_are_declared(self) -> None:
        self.assertIn("evaluation_datasets", EvaluationDataset.metadata.tables)
        self.assertIn("evaluation_cases", EvaluationCase.metadata.tables)
        self.assertIn("evaluation_runs", EvaluationRun.metadata.tables)
        self.assertIn("evaluation_results", EvaluationResult.metadata.tables)
        self.assertIn("research_evidence", ResearchEvidence.metadata.tables)
        self.assertIn("reference_answer", EvaluationCase.__table__.columns)
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

    def test_evaluation_result_template_exposes_diagnostic_evidence(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "evaluation_result.html").read_text(encoding="utf-8")
        self.assertIn("Generated answer", template)
        self.assertIn("Retrieved excerpts", template)
        self.assertIn("Source records", template)
        self.assertIn("citation_id", template)
        self.assertIn("Source retrieval diagnostics", template)
        self.assertIn("cache_diagnostics", template)

    def test_research_result_template_displays_stable_citation_ids(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "research_result.html").read_text(encoding="utf-8")
        self.assertIn("source.citation_id", template)
        self.assertIn("Source retrieval diagnostics", template)
        self.assertIn("cache_diagnostics", template)

    def test_synthesis_prompt_requires_evidence_for_every_technical_claim(self) -> None:
        pipeline = (project_root / "src" / "scientific_rag" / "legacy_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("Every factual or technical claim must be directly supported", pipeline)
        self.assertIn("Do not add general background", pipeline)

    def test_evaluation_page_exposes_recent_run_history(self) -> None:
        template = (project_root / "app" / "templates" / "evaluation.html").read_text(encoding="utf-8")
        self.assertIn("Recent evaluations", template)
        self.assertIn("/ui/evaluations/{{ run.id }}/status", template)
        self.assertIn('name="mode" value="standard"', template)

    def test_evaluation_worker_uses_the_persisted_research_mode(self) -> None:
        worker = (project_root / "app" / "worker.py").read_text(encoding="utf-8")
        self.assertIn('ResearchMode(run.metrics.get("mode", "fast"))', worker)
        self.assertIn("run.metrics[\"mode\"] = mode.value", worker)

    def test_evaluation_progress_template_displays_elapsed_time_and_estimate(self) -> None:
        template = (project_root / "app" / "templates" / "partials" / "evaluation_progress.html").read_text(encoding="utf-8")
        self.assertIn("Elapsed", template)
        self.assertIn("Estimated remaining", template)

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
