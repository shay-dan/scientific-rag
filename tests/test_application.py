import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag.application import ResearchApplication
from scientific_rag.settings import Settings


class ApplicationTests(unittest.TestCase):
    def test_translates_legacy_result_to_application_result(self) -> None:
        class FakeConfig:
            def validate(self) -> None:
                return None

        class FakeVectorService:
            pass

        class FakeResearchAgent:
            def __init__(self, vector_service) -> None:
                self.vector_service = vector_service

        class FakeOrchestrator:
            def __init__(self, vector_service, research_agent) -> None:
                self.vector_service = vector_service

            async def execute_full_study(self, **kwargs):
                document = SimpleNamespace(metadata={"title": "Paper", "source": "pubmed", "doc_id": "PMID:1"})
                return {
                    "report": {"answer": "Answer", "raw_docs_used": [document]},
                    "metadata": {"plan_executed": ["sub-question"]},
                    "evaluation_dataset": [{"question": "Q", "answer": "A"}],
                }

        pipeline = SimpleNamespace(
            config=FakeConfig(),
            VectorService=FakeVectorService,
            ResearchAgent=FakeResearchAgent,
            Orchestrator=FakeOrchestrator,
        )
        settings = Settings(google_api_key="test", entrez_email="user@example.com")
        with patch.object(ResearchApplication, "_load_pipeline", return_value=pipeline):
            result = asyncio.run(ResearchApplication(settings).run("Question"))

        self.assertEqual(result.answer, "Answer")
        self.assertEqual(result.plan, ["sub-question"])
        self.assertEqual(result.sources[0]["id"], "PMID:1")

    def test_rejects_blank_question(self) -> None:
        settings = Settings(google_api_key="test", entrez_email="user@example.com")
        with self.assertRaisesRegex(ValueError, "research question"):
            asyncio.run(ResearchApplication(settings).run("  "))

    def test_rejects_database_reset_when_not_enabled(self) -> None:
        settings = Settings(google_api_key="test", entrez_email="user@example.com")
        with self.assertRaisesRegex(PermissionError, "disabled"):
            asyncio.run(ResearchApplication(settings).run("Question", clear_knowledge_base=True))
