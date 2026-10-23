"""Admin-only, persisted RAG evaluation primitives."""
from __future__ import annotations
import asyncio
import math

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    rationale: str


class ReferenceContextScore(BaseModel):
    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    rationale: str


class ReferenceClaimVerdict(BaseModel):
    claim: str
    verdict: str = Field(pattern="^(supported|partial|unsupported)$")
    supporting_contexts: list[int] = Field(default_factory=list)
    rationale: str


class ReferenceClaimDiagnostics(BaseModel):
    verdicts: list[ReferenceClaimVerdict]


class ExcerptRelevanceVerdict(BaseModel):
    rank: int = Field(ge=1)
    verdict: str = Field(pattern="^(relevant|partial|irrelevant)$")
    supported_claims: list[int] = Field(default_factory=list)
    rationale: str


class RetrievalDiagnostics(BaseModel):
    verdicts: list[ExcerptRelevanceVerdict]


JUDGE_PROMPT = """You are a strict RAG evaluator. Score the candidate answer from 0 to 1.
Question: {question}
Reference answer: {reference_answer}
Reference context: {reference_context}
Retrieved context: {retrieved_context}
Candidate answer: {answer}
Faithfulness: claims supported by retrieved context. Answer relevancy: answer addresses question.
Context precision: retrieved context is useful. Context recall: retrieved context covers reference answer.
Return scores and a concise rationale."""

REFERENCE_CONTEXT_PROMPT = """You are a strict RAG retrieval evaluator.
Question: {question}
Verified reference source excerpt: {reference_context}
Retrieved excerpts: {retrieved_context}

Score semantically, not by exact word overlap.
Context precision: how directly useful and on-topic the retrieved excerpts are for answering the question in light of the verified reference excerpt.
Context recall: how well the retrieved excerpts cover the important factual content and approaches in the verified reference excerpt.
Do not reward unrelated quantum-computing material merely because it shares broad terms. Return scores and a concise rationale."""

REFERENCE_CLAIM_DIAGNOSTIC_PROMPT = """You are auditing retrieval coverage for a RAG system.
Question: {question}

Human-reviewed reference claims:
{claims}

Retrieved excerpts (numbered from 1):
{retrieved_context}

For every reference claim, return exactly one verdict in the same order:
- supported: the retrieved excerpts directly establish the complete claim;
- partial: they establish a meaningful part, but at least one material element is missing;
- unsupported: they do not establish the claim.

List only the one-based excerpt numbers that actually support the verdict. Be strict about scope: a general CRISPR or cancer statement does not establish a specific use, limitation, population, or clinical conclusion. Keep the rationale concise."""

RETRIEVAL_DIAGNOSTIC_PROMPT = """You are auditing ranked retrieval results for a RAG system.
Question: {question}
Reviewed reference answer: {reference_answer}

Atomic reference claims (numbered):
{claims}

Retrieved excerpts in rank order:
{retrieved_context}

Return exactly one verdict for every excerpt, in the same rank order:
- relevant: directly useful evidence for answering the question and supports at least one material reference claim;
- partial: on-topic and potentially useful, but supports only part of a claim or provides limited/background evidence;
- irrelevant: does not materially help answer the question.

For each excerpt, list the one-based reference-claim numbers it supports. Use an empty list when none are supported. Judge semantic evidence rather than exact wording and give a concise reason."""


METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class NativeGeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the current Google GenAI embedding API."""

    def __init__(self, api_key: str, model: str, dimensions: int, client=None) -> None:
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=[types.Content(parts=[types.Part.from_text(text=text)]) for text in texts],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimensions),
        )
        vectors = [list(item.values) for item in result.embeddings]
        if len(vectors) != len(texts):
            raise ValueError("Gemini embedding provider returned an unexpected number of vectors")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def summarize_scores(scores: list[dict]) -> dict[str, float | int | None]:
    """Return reproducible, rounded aggregates for one evaluation run."""
    return {
        "scored_cases": len(scores),
        **{
            metric: round(sum(float(score[metric]) for score in scores) / len(scores), 4)
            if scores
            else None
            for metric in METRIC_NAMES
        },
    }


def summarize_context_precision_views(scores: list[dict]) -> dict[str, float | int | None]:
    """Summarize the strict Ragas and claim-aware precision views separately."""
    ragas_values = []
    for score in scores:
        value = score.get("ragas_context_precision")
        if value is None:
            value = score.get("context_precision")
        if value is not None:
            ragas_values.append(float(value))
    diagnostics = [score.get("retrieval_diagnostics") for score in scores if score.get("retrieval_diagnostics")]

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "ragas_context_precision": average(ragas_values),
        "claim_aware_strict_precision": average([
            float(item["strict_ranked_precision"])
            for item in diagnostics
            if item.get("strict_ranked_precision") is not None
        ]),
        "claim_aware_partial_precision": average([
            float(item["partial_credit_ranked_precision"])
            for item in diagnostics
            if item.get("partial_credit_ranked_precision") is not None
        ]),
        "claim_aware_scored_cases": len(diagnostics),
    }


async def score_reference_context_semantically(
    *,
    question: str,
    reference_context: str,
    retrieved_contexts: list[str],
    api_key: str,
    judge_model: str,
) -> dict[str, float]:
    """Use one small judge call to compare evidence semantically to a verified excerpt."""
    client = genai.Client(api_key=api_key)
    prompt = REFERENCE_CONTEXT_PROMPT.format(
        question=question,
        reference_context=reference_context,
        retrieved_context="\n\n".join(retrieved_contexts),
    )
    response = await asyncio.to_thread(
        lambda: client.models.generate_content(
            model=judge_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ReferenceContextScore,
            ),
        )
    )
    score = ReferenceContextScore.model_validate(response.parsed)
    return {
        "context_precision": score.context_precision,
        "context_recall": score.context_recall,
    }


async def diagnose_reference_claims(
    *,
    question: str,
    reference_claims: list[str],
    retrieved_contexts: list[str],
    api_key: str,
    judge_model: str,
) -> dict:
    """Return auditable, claim-level retrieval coverage without re-running research."""
    if not reference_claims:
        raise ValueError("At least one reference claim is required")
    if not retrieved_contexts:
        raise ValueError("At least one retrieved context is required")
    client = genai.Client(api_key=api_key)
    prompt = REFERENCE_CLAIM_DIAGNOSTIC_PROMPT.format(
        question=question,
        claims="\n".join(f"{index}. {claim}" for index, claim in enumerate(reference_claims, start=1)),
        retrieved_context="\n\n".join(
            f"EXCERPT {index}:\n{context}" for index, context in enumerate(retrieved_contexts, start=1)
        ),
    )
    response = await asyncio.to_thread(
        lambda: client.models.generate_content(
            model=judge_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ReferenceClaimDiagnostics,
            ),
        )
    )
    parsed = ReferenceClaimDiagnostics.model_validate(response.parsed)
    if len(parsed.verdicts) != len(reference_claims):
        raise ValueError("Claim diagnostic returned an unexpected number of verdicts")
    verdicts = []
    for expected_claim, verdict in zip(reference_claims, parsed.verdicts, strict=True):
        if verdict.claim.strip() != expected_claim.strip():
            raise ValueError("Claim diagnostic changed or reordered a reference claim")
        if any(index < 1 or index > len(retrieved_contexts) for index in verdict.supporting_contexts):
            raise ValueError("Claim diagnostic referenced an unknown excerpt")
        verdicts.append(verdict.model_dump())
    supported = sum(item["verdict"] == "supported" for item in verdicts)
    partial = sum(item["verdict"] == "partial" for item in verdicts)
    total = len(verdicts)
    return {
        "supported_claims": supported,
        "partial_claims": partial,
        "unsupported_claims": total - supported - partial,
        "total_claims": total,
        "strict_claim_recall": supported / total,
        "partial_credit_claim_recall": (supported + 0.5 * partial) / total,
        "verdicts": verdicts,
    }


async def diagnose_retrieved_contexts(
    *,
    question: str,
    reference_answer: str,
    reference_claims: list[str],
    retrieved_contexts: list[str],
    api_key: str,
    judge_model: str,
) -> dict:
    """Explain each ranked excerpt and compute auditable precision variants."""
    if not retrieved_contexts:
        raise ValueError("At least one retrieved context is required")
    client = genai.Client(api_key=api_key)
    claims = reference_claims or [reference_answer]
    prompt = RETRIEVAL_DIAGNOSTIC_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        claims="\n".join(f"{index}. {claim}" for index, claim in enumerate(claims, start=1)),
        retrieved_context="\n\n".join(
            f"EXCERPT {index}:\n{context}" for index, context in enumerate(retrieved_contexts, start=1)
        ),
    )
    response = await asyncio.to_thread(
        lambda: client.models.generate_content(
            model=judge_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=RetrievalDiagnostics,
            ),
        )
    )
    parsed = RetrievalDiagnostics.model_validate(response.parsed)
    if len(parsed.verdicts) != len(retrieved_contexts):
        raise ValueError("Retrieval diagnostic returned an unexpected number of verdicts")
    verdicts = []
    for expected_rank, verdict in enumerate(parsed.verdicts, start=1):
        if verdict.rank != expected_rank:
            raise ValueError("Retrieval diagnostic changed or reordered excerpt ranks")
        if any(index < 1 or index > len(claims) for index in verdict.supported_claims):
            raise ValueError("Retrieval diagnostic referenced an unknown claim")
        verdicts.append(verdict.model_dump())

    binary = [1.0 if item["verdict"] == "relevant" else 0.0 for item in verdicts]
    weighted = [1.0 if item["verdict"] == "relevant" else 0.5 if item["verdict"] == "partial" else 0.0 for item in verdicts]

    def average_precision(values: list[float]) -> tuple[float, list[float]]:
        denominator = sum(values)
        if denominator == 0:
            return 0.0, [0.0] * len(values)
        cumulative = 0.0
        contributions = []
        for rank, relevance in enumerate(values, start=1):
            cumulative += relevance
            contributions.append(((cumulative / rank) * relevance) / denominator if relevance else 0.0)
        return sum(contributions), contributions

    strict_precision, strict_contributions = average_precision(binary)
    partial_precision, partial_contributions = average_precision(weighted)
    for item, strict_contribution, partial_contribution in zip(
        verdicts, strict_contributions, partial_contributions, strict=True
    ):
        item["strict_rank_contribution"] = strict_contribution
        item["partial_rank_contribution"] = partial_contribution
    return {
        "relevant_excerpts": sum(item["verdict"] == "relevant" for item in verdicts),
        "partial_excerpts": sum(item["verdict"] == "partial" for item in verdicts),
        "irrelevant_excerpts": sum(item["verdict"] == "irrelevant" for item in verdicts),
        "total_excerpts": len(verdicts),
        "strict_ranked_precision": strict_precision,
        "partial_credit_ranked_precision": partial_precision,
        "verdicts": verdicts,
    }


async def score_ragas_context_precision(
    *, question: str, reference_answer: str, retrieved_contexts: list[str], api_key: str, judge_model: str,
) -> float:
    """Run only Ragas reference-based context precision for calibration."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference

    sample = SingleTurnSample(
        user_input=question,
        response=reference_answer,
        retrieved_contexts=retrieved_contexts,
        reference=reference_answer,
    )
    metric = LLMContextPrecisionWithReference(
        llm=LangchainLLMWrapper(ChatGoogleGenerativeAI(model=judge_model, google_api_key=api_key, temperature=0))
    )
    score = float(await metric.single_turn_ascore(sample))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Ragas returned an invalid context-precision score")
    return score


async def score_with_ragas(
    *,
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    reference_answer: str,
    reference_context: str | None,
    api_key: str,
    judge_model: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> dict[str, float]:
    """Score one result with Ragas, without adding its import cost to API startup."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=judge_model, google_api_key=api_key, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(NativeGeminiEmbeddings(api_key, embedding_model, embedding_dimensions))
    sample_args = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference_answer,
    }
    sample = SingleTurnSample(
        **sample_args,
    )
    uses_reference_context = bool(reference_context and reference_context.strip())
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
    }
    scores = {name: float(await metric.single_turn_ascore(sample)) for name, metric in metrics.items()}
    scores["ragas_context_precision"] = scores["context_precision"]
    scores["ragas_context_recall"] = scores["context_recall"]
    if uses_reference_context:
        verified_scores = await score_reference_context_semantically(
            question=question,
            reference_context=reference_context.strip(),
            retrieved_contexts=retrieved_contexts,
            api_key=api_key,
            judge_model=judge_model,
        )
        scores["verified_context_precision"] = verified_scores["context_precision"]
        scores["verified_context_recall"] = verified_scores["context_recall"]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Ragas returned an invalid metric score")
    return scores
