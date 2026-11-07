"""Application service boundary around the preserved research pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .settings import Settings


@dataclass(frozen=True)
class ResearchResult:
    answer: str
    sources: list[dict[str, str]]
    plan: list[str]
    evaluation_dataset: list[dict[str, Any]]


class ResearchApplication:
    """Owns one long-lived instance of the full RAG pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._orchestrator: Any | None = None
        self._run_lock = asyncio.Lock()

    @staticmethod
    def _load_pipeline() -> Any:
        from . import legacy_pipeline

        return legacy_pipeline

    def _get_orchestrator(self) -> Any:
        if self._orchestrator is None:
            # The package copy is refactored incrementally while the root file remains a reference.
            pipeline = self._load_pipeline()
            pipeline.config.validate()
            vector_service = pipeline.VectorService()
            research_agent = pipeline.ResearchAgent(vector_service)
            self._orchestrator = pipeline.Orchestrator(vector_service, research_agent)
        return self._orchestrator

    async def run(
        self,
        question: str,
        *,
        clear_knowledge_base: bool = False,
        generate_evaluation_dataset: bool = False,
    ) -> ResearchResult:
        if not question or not question.strip():
            raise ValueError("A research question is required.")
        if clear_knowledge_base and not self.settings.allow_database_reset:
            raise PermissionError("Knowledge-base clearing is disabled by server configuration.")
        async with self._run_lock:
            result = await asyncio.wait_for(
                self._get_orchestrator().execute_full_study(
                    original_query=question.strip(),
                    clear_db=clear_knowledge_base,
                    generate_evaluation_dataset=generate_evaluation_dataset,
                ),
                timeout=self.settings.request_timeout_seconds,
            )
        report = result["report"]
        sources = [
            {
                "title": str(document.metadata.get("title", "Untitled source")),
                "source": str(document.metadata.get("source", "unknown")),
                "id": str(document.metadata.get("doc_id", document.metadata.get("id", ""))),
            }
            for document in report.get("raw_docs_used", [])
        ]
        return ResearchResult(
            answer=str(report.get("answer", "No answer was generated.")),
            sources=sources,
            plan=list(result.get("metadata", {}).get("plan_executed", [])),
            evaluation_dataset=list(result.get("evaluation_dataset", [])),
        )
