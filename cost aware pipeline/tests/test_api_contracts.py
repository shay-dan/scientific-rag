from pathlib import Path
import json
import sys
import unittest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.main import _aggregate_provider_diagnostics, app
from app.schemas import EvidenceJobRequest, EvaluationDatasetInput, EvaluationRunInput, ResearchCacheMode, ResearchMode, ResearchRequest
from app.services import PIPELINE_VERSION, ResearchService


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
        self.assertIn(("/diagnostics", ("GET",)), routes)
        self.assertIn(("/ui/evaluations", ("POST",)), routes)
        self.assertIn(("/ui/research", ("POST",)), routes)
        self.assertIn(("/ui/research/{run_id}/cancel", ("POST",)), routes)
        self.assertIn(("/v1/research", ("POST",)), routes)
        self.assertIn(("/v1/evidence/jobs", ("POST",)), routes)
        self.assertIn(("/v1/research/{run_id}", ("GET",)), routes)
        self.assertIn(("/v1/evaluations/runs/{evaluation_run_id}", ("GET",)), routes)
        self.assertIn(("/ui/evaluations/{evaluation_run_id}/export", ("GET",)), routes)

    def test_provider_diagnostics_aggregate_persisted_gateway_traces(self) -> None:
        runs = [SimpleNamespace(model_metadata={"llm_gateway": {
            "fallback_count": 1,
            "by_provider": {
                "groq": {"attempts": 2, "successes": 1, "failures": 1, "latency_ms": 300.0},
                "openrouter": {"attempts": 1, "successes": 1, "failures": 0, "latency_ms": 250.0},
            },
            "attempts": [
                {"provider": "groq", "model": "openai/gpt-oss-120b"},
                {"provider": "openrouter", "model": "openrouter/free"},
            ],
        }})]
        result = _aggregate_provider_diagnostics(runs)
        self.assertEqual(result["runs"], 1)
        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["providers"]["groq"]["average_latency_ms"], 150.0)
        self.assertEqual(result["providers"]["openrouter"]["models"], ["openrouter/free"])

    def test_provider_diagnostics_separate_expert_latency_and_throttle_skips(self) -> None:
        runs = [SimpleNamespace(model_metadata={"llm_gateway": {
            "by_provider": {"openrouter": {"attempts": 1, "successes": 1, "failures": 0, "skips": 1, "latency_ms": 1200.0}},
            "attempts": [
                {"provider": "openrouter", "model": "openrouter/free", "role": "expert", "success": False, "skipped": True},
                {"provider": "openrouter", "model": "openrouter/free", "role": "expert", "success": True, "latency_ms": 1200.0},
            ],
        }})]
        result = _aggregate_provider_diagnostics(runs)["providers"]["openrouter"]
        self.assertEqual(result["skips"], 1)
        self.assertEqual(result["expert"]["calls"], 1)
        self.assertEqual(result["expert"]["average_latency_ms"], 1200.0)

    def test_research_request_defaults_to_fast_mode(self) -> None:
        payload = ResearchRequest(question="What is the evidence for intervention X?")
        self.assertEqual(payload.mode, ResearchMode.fast)
        self.assertEqual(payload.cache_mode, ResearchCacheMode.use_cached_answer)
        self.assertFalse(payload.fresh)

    def test_shared_evidence_job_contract_rejects_extra_profile_fields(self) -> None:
        from pydantic import ValidationError
        payload = EvidenceJobRequest(topic_key="a" * 64, question="What is known about recovery after an arm fracture?")
        self.assertEqual(payload.mode, ResearchMode.fast)
        with self.assertRaises(ValidationError):
            EvidenceJobRequest(topic_key="a" * 64, question="short")
        with self.assertRaises(ValidationError):
            EvidenceJobRequest(topic_key="a" * 64, question=payload.question, medical_info="private")

    def test_deep_ui_is_explicit_about_time_cost_and_cancellation(self) -> None:
        index = (project_root / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        progress = (project_root / "app" / "templates" / "partials" / "research_progress.html").read_text(encoding="utf-8")
        self.assertIn("Deep · Experimental", index)
        self.assertIn("6–10 min", index)
        self.assertIn("highest API usage", index)
        self.assertIn("Cancel research", progress)

    def test_cached_research_lookup_requires_current_pipeline_version(self) -> None:
        async def verify() -> None:
            session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)))
            await ResearchService._find_cached(object.__new__(ResearchService), session, "a question", "tester", ResearchMode.fast)
            statement = session.execute.await_args.args[0]
            sql = str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
            self.assertIn(PIPELINE_VERSION, sql)
            self.assertIn("retrieval_version", sql)

        import asyncio
        asyncio.run(verify())

    def test_research_source_response_includes_bounded_passage_for_review(self) -> None:
        async def verify() -> None:
            passage = SimpleNamespace(content="Specific study passage " * 100,
                ordinal=0, metadata_json={"citation_id": "S1", "title": "Study",
                                           "source": "pubmed", "source_url": "https://pubmed.ncbi.nlm.nih.gov/123/"})
            session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: [passage])))
            sources = await ResearchService.sources_for_run(object.__new__(ResearchService), session, uuid4())
            self.assertEqual(sources[0].citation_id, "S1")
            self.assertEqual(len(sources[0].excerpt), 1200)
        import asyncio
        asyncio.run(verify())

    def test_fresh_api_research_skips_cache_and_forces_new_mining(self) -> None:
        async def verify() -> None:
            service = object.__new__(ResearchService)
            service._validate_request = MagicMock()
            service._find_cached = AsyncMock()
            service._metadata = MagicMock(return_value={})
            service._execute = AsyncMock(return_value="new run")
            session = SimpleNamespace(add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock())
            result = await service.run(session, "What is CAR-T evidence?", ResearchMode.fast, False, actor="tester", fresh=True)
            self.assertEqual(result, "new run")
            service._find_cached.assert_not_awaited()
            self.assertEqual(service._execute.await_args.kwargs["cache_mode"], ResearchCacheMode.refresh_sources)

        import asyncio
        asyncio.run(verify())

    def test_api_evaluation_runs_default_to_standard_mode(self) -> None:
        self.assertEqual(EvaluationRunInput().mode, ResearchMode.standard)

    def test_versioned_benchmark_matches_the_evaluation_dataset_contract(self) -> None:
        benchmark_path = project_root / "evaluation" / "benchmark-v1.json"
        benchmark = EvaluationDatasetInput.model_validate(json.loads(benchmark_path.read_text(encoding="utf-8")))
        self.assertEqual(benchmark.name, "scientific-rag-benchmark-v1")
        self.assertEqual(len(benchmark.cases), 10)
        self.assertEqual(len(benchmark.cases[0].reference_claims), 6)

    def test_benchmark_runner_defaults_to_a_single_case(self) -> None:
        runner = (project_root / "scripts" / "run_benchmark.ps1").read_text(encoding="utf-8")
        self.assertIn("[int]$CaseCount = 1", runner)
        self.assertIn("Select-Object -First $CaseCount", runner)
