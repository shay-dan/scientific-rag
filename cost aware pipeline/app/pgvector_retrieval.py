"""PostgreSQL full-text and pgvector hybrid retrieval adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import uuid

from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings
from .models import DocumentChunk, ParentDocument, SourceDocument


PARENT_CHUNK_SIZE = 2_500
PARENT_CHUNK_OVERLAP = 500
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 75


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: DocumentChunk
    source: SourceDocument


def source_fingerprint(document: Document) -> str:
    """Create a stable per-paper identity when a provider repeats a document."""
    metadata = document.metadata
    external_id = str(metadata.get("doc_id") or metadata.get("id") or "").strip()
    if external_id:
        return f"external:{external_id}"
    source = str(metadata.get("source") or "unknown")
    title = str(metadata.get("title") or "")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    return f"content:{source}:{title}:{content_hash}"


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


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def maximal_marginal_relevance(
    candidates: list[RetrievalCandidate],
    query_embedding: list[float],
    *,
    limit: int,
    lambda_mult: float = 0.5,
) -> list[RetrievalCandidate]:
    """Select relevant semantic hits while penalizing near-duplicate chunks."""
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    remaining = [candidate for candidate in candidates if candidate.chunk.embedding]
    selected: list[RetrievalCandidate] = []
    while remaining and len(selected) < limit:
        def score(candidate: RetrievalCandidate) -> tuple[float, float, float]:
            relevance = _cosine_similarity(query_embedding, candidate.chunk.embedding)
            redundancy = max(
                (_cosine_similarity(candidate.chunk.embedding, item.chunk.embedding) for item in selected),
                default=0.0,
            )
            return (lambda_mult * relevance - (1 - lambda_mult) * redundancy, -redundancy, relevance)

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return selected


def parent_context_key(candidate: RetrievalCandidate) -> uuid.UUID:
    """Use one synthesis context per parent while retaining legacy chunk fallback."""
    return candidate.chunk.parent_document_id or candidate.chunk.id


class PostgresVectorService:
    """Implements the legacy vector-service interface using PostgreSQL/pgvector."""

    def __init__(self, session: AsyncSession, settings: Settings, research_run_id: uuid.UUID) -> None:
        self.session = session
        self.settings = settings
        self.research_run_id = research_run_id
        # Global retrieval is used only for the pre-mining cache check.  The
        # orchestrator enables this after indexing so synthesis cannot mix
        # unrelated evidence from earlier research runs.
        self.restrict_to_current_run = False
        self._session_lock = asyncio.Lock()
        self.vector_store = None  # Compatibility with the legacy orchestrator's optional reset path.
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
        self.client = genai.Client(api_key=settings.google_api_key)

    def restrict_final_retrieval_to_current_run(self) -> None:
        self.restrict_to_current_run = True

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
        existing_sources = list(
            (
                await self.session.execute(
                    select(SourceDocument).where(SourceDocument.research_run_id == self.research_run_id)
                )
            ).scalars()
        )
        known_fingerprints = {
            source.metadata_json.get("source_fingerprint")
            for source in existing_sources
            if source.metadata_json.get("source_fingerprint")
        }
        grouped_documents: dict[str, list[Document]] = {}
        for document in documents:
            grouped_documents.setdefault(source_fingerprint(document), []).append(document)

        for fingerprint, source_segments in grouped_documents.items():
            if fingerprint in known_fingerprints:
                continue
            canonical_document = source_segments[0]
            metadata = dict(canonical_document.metadata)
            metadata["source_fingerprint"] = fingerprint
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
            known_fingerprints.add(fingerprint)
            seen_segment_hashes: set[str] = set()
            parent_ordinal = 0
            for segment in source_segments:
                segment_hash = hashlib.sha256(segment.page_content.encode("utf-8")).hexdigest()
                if segment_hash in seen_segment_hashes:
                    continue
                seen_segment_hashes.add(segment_hash)
                segment_metadata = {**metadata, **segment.metadata}
                for parent_text in self.parent_splitter.split_text(segment.page_content):
                    parent = ParentDocument(
                        source_document_id=source.id,
                        ordinal=parent_ordinal,
                        content=parent_text,
                        metadata_json=segment_metadata,
                    )
                    self.session.add(parent)
                    await self.session.flush()
                    child_texts = self.child_splitter.split_text(parent_text)
                    vectors = await self._embed(child_texts, "RETRIEVAL_DOCUMENT")
                    for child_ordinal, (text, vector) in enumerate(zip(child_texts, vectors, strict=True)):
                        child_metadata = {**segment_metadata, "parent_ordinal": parent_ordinal, "child_ordinal": child_ordinal}
                        self.session.add(
                            DocumentChunk(
                                source_document_id=source.id,
                                parent_document_id=parent.id,
                                content=text,
                                metadata_json=child_metadata,
                                search_vector=func.to_tsvector("english", text),
                                embedding=vector,
                            )
                        )
                    parent_ordinal += 1
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
        vector_fetch_limit = max(top_n * 8, 50)
        run_scope = SourceDocument.research_run_id == self.research_run_id if self.restrict_to_current_run else true()
        vector_statement = (
            select(DocumentChunk, SourceDocument)
            .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
            .where(DocumentChunk.embedding.is_not(None), run_scope)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(vector_fetch_limit)
        )
        tsquery = func.websearch_to_tsquery("english", query)
        text_statement = (
            select(DocumentChunk, SourceDocument)
            .join(SourceDocument, DocumentChunk.source_document_id == SourceDocument.id)
            .where(DocumentChunk.search_vector.op("@@")(tsquery), run_scope)
            .order_by(func.ts_rank_cd(DocumentChunk.search_vector, tsquery).desc())
            .limit(candidate_limit)
        )
        vector_candidates = [RetrievalCandidate(chunk=row[0], source=row[1]) for row in (await self.session.execute(vector_statement)).all()]
        vector_results = maximal_marginal_relevance(vector_candidates, query_embedding, limit=candidate_limit)
        text_results = [RetrievalCandidate(chunk=row[0], source=row[1]) for row in (await self.session.execute(text_statement)).all()]
        fused_candidates = reciprocal_rank_fusion(vector_results, text_results, limit=candidate_limit)
        fused_results: list[RetrievalCandidate] = []
        seen_contexts: set[uuid.UUID] = set()
        for candidate in fused_candidates:
            context_key = parent_context_key(candidate)
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)
            fused_results.append(candidate)
            if len(fused_results) == top_n:
                break
        parent_ids = [candidate.chunk.parent_document_id for candidate in fused_results if candidate.chunk.parent_document_id]
        parents_by_id = {
            parent.id: parent
            for parent in (await self.session.execute(select(ParentDocument).where(ParentDocument.id.in_(parent_ids)))).scalars()
        } if parent_ids else {}
        return [
            Document(
                page_content=parents_by_id[candidate.chunk.parent_document_id].content if candidate.chunk.parent_document_id in parents_by_id else candidate.chunk.content,
                metadata={
                    **candidate.chunk.metadata_json,
                    "doc_id": candidate.source.external_id,
                    "title": candidate.source.title,
                    "source": candidate.source.source,
                    "url": candidate.source.source_url,
                    "parent_document_id": str(candidate.chunk.parent_document_id) if candidate.chunk.parent_document_id else None,
                },
            )
            for candidate in fused_results
        ]
