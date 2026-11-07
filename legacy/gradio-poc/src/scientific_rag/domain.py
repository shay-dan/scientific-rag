"""Validated domain models shared across the research pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    question: str = Field(description="A specific question grounded in the source text.")
    answer: str = Field(description="An answer fully verifiable from the source text.")


class ChunkQAResponse(BaseModel):
    qa_pairs: list[QAPair]


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(description="Search-friendly sub-questions derived from the main query.")


class SubQuery(BaseModel):
    query: str = Field(description="A refined, technical sub-question optimized for search.")
    source_type: Literal["academic", "internal_knowledge"] = Field(
        description="Whether to use external academic APIs or internal model knowledge."
    )
    domains: list[Literal["medical", "physics", "technical", "education", "general"]] = Field(
        description="Scientific or general domains involved."
    )
    external_sources: list[Literal["arxiv", "pubmed"]] = Field(
        default_factory=list,
        description="External APIs to query for academic sub-questions.",
    )
    reasoning: str = Field(description="Why this sub-query is necessary.")


class QueryPlan(BaseModel):
    plan: list[SubQuery]


class GatekeeperReview(BaseModel):
    relevance_score: float = Field(description="Score from 0.0 to 1.0")
    should_skip: bool = Field(description="Whether the query is redundant or irrelevant")
    critique: str = Field(description="Brief rationale for the score")
