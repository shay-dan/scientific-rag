"""PostgreSQL full-text and pgvector hybrid retrieval adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import uuid

from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings
from .models import DocumentChunk, SourceDocument


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: DocumentChunk
    source: SourceDocument


def reciprocal_rank_fusion(
    vector_results: list[RetrievalCandidate],
    text_results: list[RetrievalCandidate],
    *,
    limit: int,
    rank_constant: int = 60,
) -> list[RetrievalCandidate]:
    """Combine independently ranked vector and full-text results."""
    scores: dict[uuid.UUID, tuple[float, RetrievalCandidate]] = {}
    for results in (vector_results, text_results):
        for rank, candidate in enumerate(results, start=1):
            score, _ = scores.get(candidate.chunk.id, (0.0, candidate))
            scores[candidate.chunk.id] = (score + 1 / (rank_constant + rank), candidate)
    return [candidate for _, candidate in sorted(scores.values(), key=lambda item: item[0], reverse=True)[:limit]]


class PostgresVectorService:
    """Implements the legacy vector-service interface using PostgreSQL/pgvector."""

    def __init__(self, session: AsyncSession, settings: Settings, research_run_id: uuid.UUID) -> None:
        self.session = session
        self.settings = settings
        self.research_run_id = research_run_id
        self._session_lock = asyncio.Lock()
        self.vector_store = None  # Compatibility with the legacy orchestrator's optional reset path.
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1_500, chunk_overlap=200)
        self.client = genai.Client(api_key=settings.google_api_key)

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        def embed() -> list[list[float]]:
            result = self.client.models.embed_content(
                model=self.settings.embedding_model,
                # A raw list of strings can be interpreted by the SDK as parts
                # of one content item.  Use explicit Content objects so the API
                # returns one embedding for every chunk.
                contents=[
                    types.Content(parts=[types.Part.from_text(text=text)])
                    for text in texts
                ],
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.settings.embedding_dimensions,
                ),
            )
            return [list(item.values) for item in result.embeddings]

        vectors = await asyncio.to_thread(embed)
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        if any(len(vector) != self.settings.embedding_dimensions for vector in vectors):
            raise ValueError("Embedding provider returned an unexpected vector dimension")
        return vectors

    async def add_documents(self, documents: list[Document]) -> None:
        for document in documents:
            metadata = dict(document.metadata)
            source = SourceDocument(
                research_run_id=self.research_run_id,
                external_id=str(metadata.get("doc_id") or metadata.get("id") or "") or None,
                title=str(metadata.get("title") or "Untitled source"),
                source=str(metadata.get("source") or "unknown"),
                source_url=str(metadata.get("url") or "") or None,
                metadata_json=metadata,
            )
            self.session.add(source)
            await self.session.flush()
            chunk_texts = self.splitter.split_text(document.page_content)
            vectors = await self._embed(chunk_texts, "RETRIEVAL_DOCUMENT")
            for text, vector in zip(chunk_texts, vectors, strict=True):
                self.session.add(
                    DocumentChunk(
                        source_document_id=source.id,
                        content=text,
                        metadata_json=metadata,
                        search_vector=func.to_tsvector("english", text),
                        embedding=vector,
                    )
                )
        await self.session.flush()

    async def retrieve_context(self, query: str, top_n: int = 10) -> list[Document]:
        query_embedding = (await self._embed([query], "RETRIEVAL_QUERY"))[0]
        # The legacy pipeline may issue several retrievals concurrently. An
        # AsyncSession is intentionally single-operation, so serialize only
        # the database portion while embeddings can still be prepared first.
        async with self._session_lock:
            return await self._retrieve_context_from_database(query, query_embedding, top_n)

    async def _retrieve_context_from_database(self, query: str, query_embedding: list[float], top_n: int) -> list[Document]:
        candidate_limit = max(top_n * 3, 20)
        vector_statement = (
            select(DocumentChunk, SourceDocument)
            .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(candidate_limit)
        )
        tsquery = func.websearch_to_tsquery("english", query)
        text_statement = (
            select(DocumentChunk, SourceDocument)
            .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
            .where(DocumentChunk.search_vector.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(DocumentChunk.search_vector, tsquery).desc())
            .limit(candidate_limit)
        )
        vector_results = [RetrievalCandidate(chunk=row[0], source=row[1]) for row in (await self.session.execute(vector_statement)).all()]
        text_results = [RetrievalCandidate(chunk=row[0], source=row[1]) for row in (await self.session.execute(text_statement)).all()]
        fused_results = reciprocal_rank_fusion(vector_results, text_results, limit=top_n)
        return [
            Document(
                page_content=candidate.chunk.content,
                metadata={
                    **candidate.chunk.metadata_json,
                    "doc_id": candidate.source.external_id,
                    "title": candidate.source.title,
                    "source": candidate.source.source,
                    "url": candidate.source.source_url,
                },
            )
            for candidate in fused_results
        ]
