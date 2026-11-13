"""R6C: replay the production hybrid retriever on frozen, label-free benchmark packs.

Opt-in evaluation adapter. It reuses the production retriever unchanged:
``PostgresVectorService.add_documents`` (parent/child splitting, embedding, PostgreSQL
``to_tsvector`` and pgvector storage) and ``retrieve_context(strategy="enhanced")``
(pgvector cosine + MMR, PostgreSQL full-text search, RRF fusion, parent-context
expansion). Depth comes from the production research budget (``standard`` by default),
never from benchmark tuning. The R6A BM25 reference baseline is a separate, unrelated path.

Isolation and identity:
- runs only against a dedicated evaluation database: it must differ from the
  application ``DATABASE_URL``, be migrated to the current Alembic head, and hold no
  research run that this adapter did not create;
- each pack is a frozen corpus of exactly its own text segments, ingested inside its own
  transaction under its own research run, with production run scoping enabled, and
  **rolled back** after retrieval: nothing persists, and no pack can see another;
- every result must come from that pack, otherwise the record is ``invalid``;
- ``corpus_sha256`` binds the corpus (evidence IDs and text hashes) to the embedding model,
  dimensions, chunking and pipeline version.

Exports ranked evidence IDs (and source paper IDs where the benchmark supplies them),
configuration, corpus identity, timings, embedding usage and explicit failures, by opaque
input ID, unscored. Embeddings are provider calls: ``dry_run`` counts the production
chunks and estimates tokens and cost without any call; ``run_retrieval`` refuses to embed
without an explicit authorization (``production_verifier_replay.check_authorization``) and
never starts a pack whose embedding requests or estimated cost would exceed it.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from langchain_core.documents import Document
from sqlalchemy import func, select

from ..config import Settings
from ..cost_policy import ResearchMode, budget_for
from ..evaluation_report import sha256_hex
from ..models import ResearchRun, SourceDocument
from ..pgvector_retrieval import (
    CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PostgresVectorService,
)
from .prediction_export import TEXT_REASON, _record
from .production_verifier_replay import AuthorizationError, check_authorization

ADAPTER = "production_hybrid_retriever"
EVAL_REQUESTER = "r6c-evaluation-adapter"
PRODUCTION_STRATEGY = "enhanced"
SOURCE_LABEL = "benchmark_pack"
CEILING_REASON = "authorized embedding cost ceiling reached before this input"
ISOLATION_VIOLATION = "corpus isolation violated"


class IsolationError(RuntimeError):
    """The evaluation database or a retrieved result is not isolated to the frozen corpus."""


def configuration(settings: Settings, research_mode: str = "standard") -> dict:
    from ..services import PIPELINE_VERSION
    return {"adapter": ADAPTER, "strategy": PRODUCTION_STRATEGY, "research_mode": research_mode,
            "top_n": budget_for(ResearchMode(research_mode)).retrieval_documents,
            "parent_chunk": [PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP], "child_chunk": [CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP],
            "rrf_rank_constant": 60, "mmr_lambda": 0.5, "fts_config": "english",
            "embedding_model": settings.embedding_model, "embedding_dimensions": settings.embedding_dimensions,
            "pipeline_version": PIPELINE_VERSION, "run_scope": "per-pack research run; transaction rolled back"}


def pack_embedding_plan(item: dict, settings: Settings) -> dict:
    """Production chunking of one pack and the embedding estimate production will record."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from scientific_rag.costing import calculate_cost, estimate_text_tokens
    parents = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
    children = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    parent_texts = [parent for s in text_segments(item) for parent in parents.split_text(s["text"])]
    chunks = [child for parent in parent_texts for child in children.split_text(parent)]
    tokens = estimate_text_tokens(chunks) + estimate_text_tokens([item["question"]])
    cost, source = calculate_cost("gemini", settings.embedding_model, tokens)
    # production embeds the children of each parent in one request, and the query in one request
    return {"parents": len(parent_texts), "child_chunks": len(chunks), "requests": len(parent_texts) + 1,
            "tokens": tokens, "cost_usd": cost, "rate_source": source}


def text_segments(item: dict) -> list[dict]:
    return sorted((s for s in item.get("segments", []) if s["availability"] == "text"), key=lambda s: s["evidence_index"])


def corpus_documents(item: dict) -> list[Document]:
    """One production ``Document`` per text segment, carrying the benchmark identities."""
    return [Document(page_content=s["text"], metadata={
        "doc_id": s.get("paper_id") or None, "title": s.get("title") or "Benchmark evidence segment",
        "source": SOURCE_LABEL, "benchmark_input_id": item["input_id"], "evidence_id": s["evidence_id"],
        "evidence_index": s["evidence_index"], "paper_id": s.get("paper_id")}) for s in text_segments(item)]


def corpus_identity(item: dict, config: dict) -> str:
    return sha256_hex({"input_id": item["input_id"], "input_hash": item["input_hash"],
                       "segments": [[s["evidence_id"], sha256_hex(s["text"])] for s in text_segments(item)],
                       "embedding": [config["embedding_model"], config["embedding_dimensions"]],
                       "chunking": [config["parent_chunk"], config["child_chunk"]],
                       "pipeline_version": config["pipeline_version"]})


def retrieval_items(fixed_items: list[dict]) -> list[dict]:
    return sorted((item for item in fixed_items if item["kind"] == "fixed_evidence_question"),
                  key=lambda item: item["input_id"])


def dry_run(items: list[dict], settings: Settings, *, research_mode: str = "standard") -> dict:
    """Count production chunks and estimate embedding tokens and cost. No provider or database call."""
    config = configuration(settings, research_mode)
    status, plans = Counter(), []
    for item in retrieval_items(items):
        if not text_segments(item):
            status["not_executed"] += 1
            continue
        status["planned"] += 1
        plans.append(pack_embedding_plan(item, settings))
    costs = [p["cost_usd"] for p in plans]
    return {"adapter": ADAPTER, "provider_calls_made": 0, "database_calls_made": 0,
            "inputs": sum(status.values()), "status": dict(sorted(status.items())),
            "child_chunks": {"total": sum(p["child_chunks"] for p in plans),
                             "max_per_pack": max((p["child_chunks"] for p in plans), default=0)},
            "embedding_requests": {"documents": sum(p["parents"] for p in plans), "queries": len(plans)},
            "estimated_embedding_tokens": sum(p["tokens"] for p in plans),
            "token_estimate_method": "production estimate_text_tokens over production child chunks and queries",
            "provider": "gemini", "model": settings.embedding_model,
            "estimated_cost_usd": round(sum(costs), 8) if plans and None not in costs else (0.0 if not plans else None),
            "rate_source": plans[0]["rate_source"] if plans else None, "configuration": config}


def _plain_url(url: str) -> tuple:
    parts = urlsplit(url.replace("+asyncpg", ""))
    return (parts.hostname, parts.port or 5432, parts.path.lstrip("/"))


async def verify_isolated_database(session, *, evaluation_url: str, application_url: str | None,
                                   migration_head: str | None = None) -> dict:
    """Refuse any database that is not a dedicated, current, evaluation-only database."""
    if application_url and _plain_url(evaluation_url) == _plain_url(application_url):
        raise IsolationError("the evaluation database must not be the application database")
    if migration_head is None:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        root = Path(__file__).resolve().parents[2]
        migration_head = ScriptDirectory.from_config(Config(str(root / "alembic.ini"))).get_current_head()
    revision = (await session.execute(_alembic_revision())).scalar()
    if revision != migration_head:
        raise IsolationError(f"evaluation database is not at the migration head ({revision} != {migration_head})")
    foreign = (await session.execute(select(func.count()).select_from(ResearchRun).where(
        func.coalesce(ResearchRun.requested_by, "") != EVAL_REQUESTER))).scalar()
    if foreign:
        raise IsolationError(f"evaluation database holds {foreign} research run(s) not created by this adapter")
    return {"migration_head": revision, "foreign_research_runs": 0}


def _alembic_revision():
    from sqlalchemy import text
    return text("SELECT version_num FROM alembic_version")


def _trace_summary(service) -> dict:
    trace = service.retrieval_traces[-1] if service.retrieval_traces else {}
    return {key: trace.get(key) for key in ("vector_count", "mmr_count", "fulltext_count", "selected_count")} | {
        "fts_decision": (trace.get("fts_diagnostics") or {}).get("decision", "applied" if trace.get("fts_query") else None)}


async def retrieve_one(item: dict, *, session, settings: Settings, config: dict,
                       service_factory: Callable = PostgresVectorService) -> dict:
    """Ingest one frozen pack in its own transaction, retrieve, verify isolation, roll back.

    Always returns a record; embedding usage and timings are kept on failures too."""
    documents = corpus_documents(item)
    corpus_hash = corpus_identity(item, config)
    pack_ids = {document.metadata["evidence_id"] for document in documents}
    timings = {"ingest": None, "retrieve": None}
    service, error, results, sources = None, None, [], None
    started = time.perf_counter()
    try:
        run = ResearchRun(question="[R6C evaluation corpus]", requested_by=EVAL_REQUESTER, status="evaluation_corpus",
                          model_metadata={"adapter": ADAPTER, "corpus_sha256": corpus_hash, "input_id": item["input_id"]})
        session.add(run)
        await session.flush()
        service = service_factory(session, settings, run.id)
        service.restrict_final_retrieval_to_current_run()
        await service.add_documents(documents)
        ingested = time.perf_counter()
        timings["ingest"] = round((ingested - started) * 1000, 1)
        sources = (await session.execute(select(func.count()).select_from(SourceDocument).where(
            SourceDocument.research_run_id == run.id))).scalar()
        results = await service.retrieve_context(item["question"], top_n=config["top_n"], strategy=PRODUCTION_STRATEGY)
        timings["retrieve"] = round((time.perf_counter() - ingested) * 1000, 1)
    except Exception as exc:   # embedding, database or retrieval failure: recorded, never retried
        error = exc
    finally:
        await session.rollback()   # the frozen corpus never persists
    embedding = service.embedding_usage if service is not None else {"calls": 0, "input_tokens_estimated": 0,
                                                                      "estimated_cost_usd": 0.0, "model": settings.embedding_model}
    usage = {"cost_status": "estimated" if embedding["estimated_cost_usd"] is not None else "usage_incomplete",
             "provider_calls": embedding["calls"], "input_tokens_estimated": embedding["input_tokens_estimated"],
             "estimated_cost_usd": embedding["estimated_cost_usd"],
             "provider_models": [f"gemini:{embedding['model']}"] if embedding["calls"] else [], "timings_ms": timings}
    if error is not None:
        return _record(item, "fixed_evidence_retrieval", "failed", usage=usage,
                       reason=f"retrieval failure ({type(error).__name__})")
    ranked_evidence, ranked_papers = [], []
    for document in results:
        metadata = document.metadata
        if metadata.get("benchmark_input_id") != item["input_id"] or metadata.get("evidence_id") not in pack_ids:
            return _record(item, "fixed_evidence_retrieval", "invalid", usage=usage,
                           reason=f"{ISOLATION_VIOLATION}: retrieved a passage outside the frozen corpus")
        if metadata["evidence_id"] not in ranked_evidence:
            ranked_evidence.append(metadata["evidence_id"])
        paper = metadata.get("paper_id")
        if paper and paper not in ranked_papers:
            ranked_papers.append(paper)
    partial = len(documents) < len(item.get("segments", []))
    return _record(item, "fixed_evidence_retrieval", "completed", usage=usage, prediction={
        "ranked_evidence_ids": ranked_evidence, "ranked_paper_ids": ranked_papers, "results_returned": len(results),
        "corpus_sha256": corpus_hash, "corpus_segments": len(documents), "corpus_sources": sources,
        "segments_without_text": len(item.get("segments", [])) - len(documents), "trace": _trace_summary(service)},
        reason="some segments were not redistributed and are not in the corpus" if partial else None)


async def run_retrieval(items: list[dict], *, session_factory, settings: Settings, authorization: dict,
                        dataset_hash: str, evaluation_url: str, application_url: str | None,
                        research_mode: str = "standard", service_factory: Callable = PostgresVectorService,
                        migration_head: str | None = None) -> tuple[list[dict], dict]:
    """Run the production retriever on each frozen pack within the authorized embedding ceiling. Unscored."""
    auth = check_authorization(authorization, adapter=ADAPTER, dataset_hash=dataset_hash,
                               route={"provider_order": ["gemini"], "models": {"gemini": settings.embedding_model}})
    config = configuration(settings, research_mode)
    async with session_factory() as session:
        isolation = await verify_isolated_database(session, evaluation_url=evaluation_url,
                                                   application_url=application_url, migration_head=migration_head)
    records, spent, calls, stop = [], 0.0, 0, None
    for item in retrieval_items(items):
        if not text_segments(item):
            records.append(_record(item, "fixed_evidence_retrieval", "not_executed", reason=TEXT_REASON))
            continue
        upcoming = pack_embedding_plan(item, settings)
        if (stop or calls + upcoming["requests"] > auth["max_calls"]
                or spent + (upcoming["cost_usd"] or 0.0) > auth["max_cost_usd"]):
            records.append(_record(item, "fixed_evidence_retrieval", "not_executed", reason=stop or CEILING_REASON))
            continue
        async with session_factory() as session:
            record = await retrieve_one(item, session=session, settings=settings, config=config,
                                        service_factory=service_factory)
        if record["status"] == "invalid" and ISOLATION_VIOLATION in record.get("reason", ""):
            stop = "stopped: corpus isolation was violated"
        usage = record["usage"]
        calls += usage.get("provider_calls", 0)
        if usage.get("estimated_cost_usd") is None and usage.get("provider_calls"):
            stop = stop or "stopped: embedding cost could not be estimated, so the ceiling could not be enforced"
        else:
            spent += usage.get("estimated_cost_usd") or 0.0
        records.append(record)
    records.sort(key=lambda record: record["input_id"])
    return records, {"authorization_id": auth["authorization_id"], "embedding_calls": calls,
                     "estimated_cost_usd": round(spent, 8), "stopped": stop, "isolation": isolation,
                     "configuration": config}


__all__ = ["ADAPTER", "AuthorizationError", "IsolationError", "configuration", "corpus_documents", "corpus_identity",
           "dry_run", "retrieval_items", "retrieve_one", "run_retrieval", "verify_isolated_database"]
