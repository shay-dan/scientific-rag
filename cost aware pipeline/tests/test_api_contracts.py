from pathlib import Path
import json
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.main import app
from app.schemas import EvaluationDatasetInput, EvaluationRunInput, ResearchMode, ResearchRequest


class ApiContractTests(unittest.TestCase):
    def test_expected_routes_are_exposed(self) -> None:
        routes = {
            (route.path, tuple(sorted(route.methods or [])))
            for route in app.routes
            if hasattr(route, "methods")
        }
        self.assertIn(("/healthz", ("GET",)), routes)
        self.assertIn(("/", ("GET",)), routes)
        self.assertIn(("/evaluation", ("GET",)), routes)
        self.assertIn(("/ui/evaluations", ("POST",)), routes)
        self.assertIn(("/ui/research", ("POST",)), routes)
        self.assertIn(("/v1/research", ("POST",)), routes)
        self.assertIn(("/v1/research/{run_id}", ("GET",)), routes)
        self.assertIn(("/v1/evaluations/runs/{evaluation_run_id}", ("GET",)), routes)

    def test_research_request_defaults_to_fast_mode(self) -> None:
        payload = ResearchRequest(question="What is the evidence for intervention X?")
        self.assertEqual(payload.mode, ResearchMode.fast)

    def test_api_evaluation_runs_default_to_standard_mode(self) -> None:
        self.assertEqual(EvaluationRunInput().mode, ResearchMode.standard)

    def test_versioned_benchmark_matches_the_evaluation_dataset_contract(self) -> None:
        benchmark_path = project_root / "evaluation" / "benchmark-v1.json"
        benchmark = EvaluationDatasetInput.model_validate(json.loads(benchmark_path.read_text(encoding="utf-8")))
        self.assertEqual(benchmark.name, "scientific-rag-benchmark-v1")
        self.assertEqual(len(benchmark.cases), 10)
