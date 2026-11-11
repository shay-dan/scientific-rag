"""Independent single-call judge for the frozen RAG comparison."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .untrusted_content import UNTRUSTED_DATA_RULE, render_untrusted

# R3: the rubric text is unchanged; retrieved evidence and the candidate answer are
# framed as untrusted data blocks. Recorded with each score so results stay comparable.
PROMPT_SECURITY_VERSION = "r3_untrusted_blocks"


class ReferenceClaimAssessment(BaseModel):
    claim_index: int = Field(ge=1)
    status: Literal["covered", "partially_covered", "missing"]
    supporting_citations: list[str] = Field(default_factory=list)
    explanation: str


class ComparisonJudgeScore(BaseModel):
    answer_correctness: float = Field(ge=0, le=1)
    reference_claim_coverage: float = Field(ge=0, le=1)
    evidence_groundedness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    abstention_quality: float = Field(ge=0, le=1)
    overall_quality: float = Field(ge=0, le=1)
    reference_claim_assessments: list[ReferenceClaimAssessment]
    unsupported_candidate_claims: list[str] = Field(default_factory=list)
    rationale: str


JUDGE_TEMPLATE = """You are an independent scientific RAG evaluator. This is rubric version 2.
Judge only from the verified silver reference and the exact evidence supplied to the candidate.
Do not reward fluent unsupported additions. Do not require the candidate to copy the reference wording.
Treat the silver reference as the target scope, but treat the retrieved evidence as the only factual support
available to the candidate. Do not infer support from the paper title or from your own knowledge.
{untrusted_rule}
Each evidence block's label attribute is its citation label (S1, S2, ...).

QUESTION:
{question}

SILVER REFERENCE ANSWER:
{reference_answer}

ATOMIC REFERENCE CLAIMS:
{reference_claims}

EXACT RETRIEVED EVIDENCE AND CITATION LABELS:
{retrieved_context}

CANDIDATE ANSWER:
{answer}

Score from 0 to 1:
- answer_correctness: correctness of factual claims the candidate actually makes. Unsupported or contradicted
  claims reduce this score. A cautious omission does not make an asserted claim incorrect.
- reference_claim_coverage: fraction and degree of the numbered reference claims actually answered. A cautious
  omission still lowers coverage. Do not give full credit merely because the answer is correct but incomplete.
- evidence_groundedness: degree to which the candidate's material factual claims are directly supported by the
  exact retrieved excerpts. Unsupported additions reduce this even when they are plausible.
- answer_relevancy: how directly and completely the answer addresses the question.
- citation_correctness: whether each cited label actually supports the nearby claim.
- citation_completeness: whether material factual claims carry an appropriate citation.
- abstention_quality: whether uncertainty or missing evidence is stated accurately. Give 1 when no abstention is
  needed and the evidence is sufficient; penalize invented certainty and unnecessary refusal.
- overall_quality: holistic evidence-grounded usefulness. Weight correctness, reference-claim coverage, and
  groundedness more than style. A materially incomplete answer cannot receive full overall quality.

For every numbered reference claim, return exactly one reference_claim_assessment with the same 1-based index:
- covered: the candidate addresses the claim and the retrieved evidence supports it;
- partially_covered: only part of the claim is addressed or support is incomplete;
- missing: the claim is not addressed.
List only citation labels such as S1 or S2 that directly support that assessment. Also list each material candidate
claim that lacks support in unsupported_candidate_claims. Return the structured score and a concise rationale."""


def _exact_context_rows(case: dict, result: dict, *, corpus_contexts: dict[str, dict] | None = None) -> list[tuple[str, str, str]]:
    """(citation label, title, excerpt) for each exact context used during synthesis."""
    by_context = corpus_contexts or {row.get("context_id"): row for row in case.get("reference_context", [])}
    labels: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []
    for source_id, context_id in zip(
        result.get("selected_source_ids", []),
        result.get("selected_context_ids", []),
        strict=True,
    ):
        row = by_context.get(context_id)
        if row is None:
            raise ValueError(f"Result references unknown benchmark context: {context_id}")
        label = labels.setdefault(str(source_id), f"S{len(labels) + 1}")
        rows.append((label, str(row.get("title", "Unknown source")), str(row["excerpt"])))
    if not rows:
        raise ValueError("Result has no exact retrieved contexts")
    return rows


def exact_contexts_for_result(case: dict, result: dict, *, corpus_contexts: dict[str, dict] | None = None) -> list[str]:
    """Reconstruct exact contexts and the citation labels used during synthesis."""
    return [f"[{label}: {title}]\n{excerpt}" for label, title, excerpt in
            _exact_context_rows(case, result, corpus_contexts=corpus_contexts)]


def build_judge_prompt(case: dict, result: dict, *, corpus_contexts: dict[str, dict] | None = None) -> str:
    rows = _exact_context_rows(case, result, corpus_contexts=corpus_contexts)
    return JUDGE_TEMPLATE.format(
        untrusted_rule=UNTRUSTED_DATA_RULE,
        question=case["question"],
        reference_answer=case["reference_answer"],
        reference_claims="\n".join(
            f"{index}. {claim}" for index, claim in enumerate(case.get("reference_claims", []), start=1)
        ),
        retrieved_context="\n\n".join(
            render_untrusted("EVIDENCE", excerpt, attributes={"label": label, "title": title})
            for label, title, excerpt in rows
        ),
        answer=render_untrusted("CANDIDATE_ANSWER", str(result["answer"])),
    )
