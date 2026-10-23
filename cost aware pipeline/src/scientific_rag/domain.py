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


class PubMedRelevanceAssessment(BaseModel):
    decision: Literal["relevant", "irrelevant", "uncertain"] = Field(
        description="Whether this paper's title and abstract materially address the research question. Use uncertain when relevance cannot be established confidently."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the relevance decision, from 0.0 to 1.0.")
    reason: str = Field(description="Short reason grounded only in the supplied title and abstract.")
    intervention_match: Literal["match", "mismatch", "unclear", "not_applicable"] = Field(
        description="Whether the treatment or diagnostic intervention actually studied matches the question."
    )
    population_match: Literal["match", "mismatch", "unclear", "not_applicable"] = Field(
        description="Whether the studied disease or patient population matches the question."
    )
    evidence_role: Literal["direct", "background", "unrelated"] = Field(
        description="Direct evidence studies the requested intervention and population; background is related but cannot establish its outcomes."
    )


class ArxivRelevanceAssessment(BaseModel):
    decision: Literal["relevant", "irrelevant", "uncertain"] = Field(
        description="Whether the arXiv paper materially helps answer the research question. Use uncertain when the connection is plausible but unclear."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the relevance decision, from 0.0 to 1.0.")
    reason: str = Field(description="Short rationale grounded only in the supplied title and abstract.")
