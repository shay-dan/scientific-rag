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
    if uses_reference_context:
        scores.update(await score_reference_context_semantically(
            question=question,
            reference_context=reference_context.strip(),
            retrieved_contexts=retrieved_contexts,
            api_key=api_key,
            judge_model=judge_model,
        ))
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Ragas returned an invalid metric score")
    return scores
