"""R4 reproduction: aggregation, provenance and prompt-integrity defects (pre-fix APIs).

Offline and deterministic; no provider is contacted.
"""
import asyncio
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app import main as api
from app.evaluation import (
    REFERENCE_CLAIM_DIAGNOSTIC_PROMPT,
    REFERENCE_CONTEXT_PROMPT,
    RETRIEVAL_DIAGNOSTIC_PROMPT,
    JUDGE_PROMPT,
    summarize_context_precision_views,
    summarize_scores,
)
from app.schemas import EvaluationDatasetInput

RAGAS = {"engine": "ragas", "faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0}
FALLBACK = {"engine": "gemini_fallback", "faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}


class R4AggregationReproductionTests(unittest.TestCase):
    def test_ragas_and_fallback_scores_are_not_averaged_into_one_value(self) -> None:
        summary = summarize_scores([RAGAS, FALLBACK])
        self.assertIsNone(summary["faithfulness"], "mixed-engine scores were averaged into a single value")

    def test_ragas_precision_view_excludes_fallback_judge_scores(self) -> None:
        views = summarize_context_precision_views([RAGAS, FALLBACK])
        self.assertEqual(views["ragas_context_precision"], 1.0)

    def test_missing_metric_is_counted_not_crashing(self) -> None:
        partial = {"engine": "ragas", "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}
        summary = summarize_scores([dict(RAGAS), partial])
        self.assertEqual(summary["context_recall"], 1.0)

    def test_invalid_metric_values_are_excluded(self) -> None:
        invalid = {**RAGAS, "faithfulness": float("nan")}
        summary = summarize_scores([dict(RAGAS), invalid])
        self.assertEqual(summary["faithfulness"], 1.0)


class R4ProvenanceReproductionTests(unittest.TestCase):
    def test_api_dataset_provenance_reflects_the_declared_input(self) -> None:
        payload = EvaluationDatasetInput.model_validate({"name": "r4-provenance", "cases": [{
            "question": "Does X help?", "reference_answer": "Unclear.", "provenance": "model_generated"}]})
        added = []
        session = MagicMock()
        session.add = lambda item: setattr(item, "id", "dataset-id")
        session.add_all = lambda items: added.extend(items)
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        with patch.object(api, "require_api_key", return_value=SimpleNamespace(role="admin", name="admin")):
            asyncio.run(api.create_evaluation_dataset(SimpleNamespace(), payload, session))
        self.assertEqual([case.provenance for case in added], ["model_generated"])


class R4PromptIntegrityReproductionTests(unittest.TestCase):
    def test_judge_prompts_contain_no_benchmark_topic_wording(self) -> None:
        for prompt in (JUDGE_PROMPT, REFERENCE_CONTEXT_PROMPT, REFERENCE_CLAIM_DIAGNOSTIC_PROMPT, RETRIEVAL_DIAGNOSTIC_PROMPT):
            for topic in ("quantum", "CRISPR", "cancer"):
                with self.subTest(topic=topic, prompt=prompt[:40]):
                    self.assertNotIn(topic, prompt)


if __name__ == "__main__":
    unittest.main()
