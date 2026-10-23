"""Explicit resource limits for user-facing research requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .schemas import ResearchMode


@dataclass(frozen=True)
class ResearchBudget:
    max_plan_steps: int
    papers_per_source: int
    retrieval_documents: int
    max_context_documents: int
    max_synthesis_subquestions: int
    estimated_max_generation_calls: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


_BUDGETS = {
    ResearchMode.fast: ResearchBudget(1, 2, 4, 5, 1, 5),
    ResearchMode.standard: ResearchBudget(2, 3, 6, 8, 2, 8),
    # Deep deliberately increases breadth without sending an unbounded context
    # to the Expert. Two-source questions may collect up to 12 papers; MMR and
    # parent/child retrieval still select the final 14 evidence documents.
    ResearchMode.deep: ResearchBudget(4, 6, 10, 14, 4, 15),
}


def budget_for(mode: ResearchMode) -> ResearchBudget:
    return _BUDGETS[mode]
