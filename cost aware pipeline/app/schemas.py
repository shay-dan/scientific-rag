"""API request and response contracts."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ResearchMode(str, Enum):
    fast = "fast"
    standard = "standard"
    deep = "deep"


class ResearchCacheMode(str, Enum):
    use_cached_answer = "use_cached_answer"
    regenerate_answer = "regenerate_answer"
    refresh_sources = "refresh_sources"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=10_000)
    mode: ResearchMode = ResearchMode.fast
    generate_evaluation_dataset: bool = False
    cache_mode: ResearchCacheMode = ResearchCacheMode.use_cached_answer
    # Deprecated compatibility field. New clients should use cache_mode.
    fresh: bool = False


class EvidenceJobRequest(BaseModel):
    """Only a generic, non-user-specific research question may enter the shared store."""

    model_config = ConfigDict(extra="forbid")
    topic_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    question: str = Field(min_length=10, max_length=600)
    mode: ResearchMode = ResearchMode.fast


class SourceResponse(BaseModel):
    citation_id: str | None = None
    title: str
    source: str
    external_id: str | None = None
    source_url: str | None = None
    study_design: str | None = None
    excerpt: str | None = None


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
    retrieval_version: str | None = None
    source_diagnostics: dict = Field(default_factory=dict)
    cache_diagnostics: dict = Field(default_factory=dict)
    llm_diagnostics: dict = Field(default_factory=dict)
    source_sufficiency: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    database: str


class EvaluationCaseInput(BaseModel):
    question: str = Field(min_length=3)
    reference_answer: str = Field(min_length=3)
    reference_context: str | None = None
    reference_claims: list[str] = Field(default_factory=list, max_length=50)


class EvaluationDatasetInput(BaseModel):
    name: str = Field(min_length=3, max_length=256)
    cases: list[EvaluationCaseInput] = Field(min_length=1, max_length=100)


class EvaluationRunInput(BaseModel):
    """Controls the research budget used for every case in an evaluation run."""

    mode: ResearchMode = ResearchMode.standard
