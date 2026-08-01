"""API request and response contracts."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class ResearchMode(str, Enum):
    fast = "fast"
    standard = "standard"
    deep = "deep"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=10_000)
    mode: ResearchMode = ResearchMode.fast
    generate_evaluation_dataset: bool = False


class SourceResponse(BaseModel):
    citation_id: str | None = None
    title: str
    source: str
    external_id: str | None = None
    source_url: str | None = None


class ResearchResponse(BaseModel):
    id: UUID
    status: str
    answer: str | None = None
    sources: list[SourceResponse] = []
    plan: list[str] = []
    created_at: datetime
    completed_at: datetime | None = None
    mode: ResearchMode = ResearchMode.fast
    cached: bool = False
    source_diagnostics: dict = Field(default_factory=dict)
    cache_diagnostics: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    database: str


class EvaluationCaseInput(BaseModel):
    question: str = Field(min_length=3)
    reference_answer: str = Field(min_length=3)
    reference_context: str | None = None


class EvaluationDatasetInput(BaseModel):
    name: str = Field(min_length=3, max_length=256)
    cases: list[EvaluationCaseInput] = Field(min_length=1, max_length=100)


class EvaluationRunInput(BaseModel):
    """Controls the research budget used for every case in an evaluation run."""

    mode: ResearchMode = ResearchMode.standard
