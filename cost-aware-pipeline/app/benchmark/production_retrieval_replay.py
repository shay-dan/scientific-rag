"""R6C: replay the production hybrid retriever on frozen, label-free benchmark packs.

Opt-in evaluation adapter. It reuses the production retriever unchanged:
``PostgresVectorService.add_documents`` (parent/child splitting, embedding, PostgreSQL
``to_tsvector`` and pgvector storage) and ``retrieve_context(strategy="enhanced")``
(pgvector cosine + MMR, PostgreSQL full-text search, RRF fusion, parent-context
expansion). Depth comes from the production research budget (``standard`` by default),
never from benchmark tuning. The R6A BM25 reference baseline is a separate, unrelated path.

Isolation and identity (fail closed):
- runs only against a dedicated evaluation database. The application database is the
  *effective* ``Settings().database_url`` (environment, ``.env`` or default), never only
  the process environment; a missing or unparsable identity refuses the run. The two URLs
  must not name the same server and database under any equivalent spelling (driver,
  ``postgres``/``postgresql``, host case, loopback aliases, socket, default port), and a
  live check must show they are different physical databases (database name, OID and
  server start time); if the application database cannot be reached, isolation is
  unverifiable and the run is refused. The evaluation database must be at the current
  Alembic head and hold no research run that this adapter did not create;
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
without an explicit authorization (``production_verifier_replay.check_authorization``).

Budget (R4 accounting): before a pack starts, its full allowance is reserved: every
production embedding request (one per parent plus the query) and a conservative cost
(one token per 3 characters). A started pack is always charged its whole reservation,
because production counts embedding requests only after they succeed, so failed or
uncertain requests are never treated as zero. Gemini returns no usage, so successful cost
is the production *estimate*; failed packs are ``incomplete``, with the estimate of the
requests that succeeded as a known lower bound.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable
import math
from urllib.parse import parse_qs, urlsplit

from langchain_core.documents import Document
from sqlalchemy import func, select, text

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
RESERVE_CHARS_PER_TOKEN = 3
LOOPBACK_HOSTS = {"", "localhost", "127.0.0.1", "::1", "local"}
IDENTITY_SQL = text("SELECT current_database() AS database, "
                    "(SELECT oid FROM pg_database WHERE datname = current_database()) AS database_oid, "
                    "pg_postmaster_start_time()::text AS server_started")


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
    reserve_tokens = sum(max(1, math.ceil(len(t) / RESERVE_CHARS_PER_TOKEN)) for t in [*chunks, item["question"]])
    reserve, _ = calculate_cost("gemini", settings.embedding_model, reserve_tokens)
    # production embeds the children of each parent in one request, and the query in one request
    return {"parents": len(parent_texts), "child_chunks": len(chunks), "requests": len(parent_texts) + 1,
            "tokens": tokens, "cost_usd": cost, "reserve_tokens": reserve_tokens, "reserve_cost_usd": reserve,
            "rate_source": source}


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
            "reserved_cost_usd": (round(sum(p["reserve_cost_usd"] for p in plans), 8)
                                  if all(p["reserve_cost_usd"] is not None for p in plans) else None),
            "rate_source": plans[0]["rate_source"] if plans else None, "configuration": config}


def effective_application_database_url(settings: Settings | None = None) -> str:
    """The application database the API and worker would actually use (env, .env or default)."""
    return (settings or Settings()).database_url


def database_url_identity(url: str | None) -> tuple:
    """Spelling-independent (host class, port, database) for a PostgreSQL URL."""
    if not isinstance(url, str) or not url.strip():
        raise IsolationError("application database identity is missing; isolation cannot be verified")
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.split("+", 1)[0].lower()
        if scheme not in {"postgresql", "postgres"}:
            raise ValueError(scheme)
        query = parse_qs(parts.query)
        host = (parts.hostname or (query.get("host") or [""])[0] or "").lower().strip("[]")
        port = parts.port or int((query.get("port") or ["5432"])[0])
        database = parts.path.lstrip("/") or (query.get("dbname") or [""])[0] or (parts.username or "")
    except (ValueError, TypeError) as error:
        raise IsolationError(f"database URL is not a verifiable PostgreSQL URL ({type(error).__name__})") from None
    if not database:
        raise IsolationError("database URL names no database; isolation cannot be verified")
    host_class = "local" if host in LOOPBACK_HOSTS or host.startswith("/") else host
    return (host_class, port, database)


def check_database_urls(*, evaluation_url: str, application_url: str | None) -> None:
    """Static, fail-closed comparison before any connection."""
    application = database_url_identity(application_url)
    try:
        evaluation = database_url_identity(evaluation_url)
    except IsolationError as error:
        raise IsolationError(f"evaluation database URL: {error}") from None
    if application == evaluation:
        raise IsolationError("the evaluation database must not be the application database "
                             "(same server and database under an equivalent URL spelling)")


async def database_identity(session_or_connection) -> dict:
    row = (await session_or_connection.execute(IDENTITY_SQL)).mappings().one()
    return {"database": row["database"], "database_oid": row["database_oid"],
            "server_started": None if row["server_started"] is None else str(row["server_started"])}


def async_database_url(url: str) -> str:
    """Use the asyncpg driver for plain PostgreSQL URLs (as the application does)."""
    url = url.strip()
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


async def probe_database_identity(url: str) -> dict:
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(async_database_url(url))
    try:
        async with engine.connect() as connection:
            return await database_identity(connection)
    finally:
        await engine.dispose()


def _complete_identity(identity) -> bool:
    return isinstance(identity, dict) and all(identity.get(key) not in (None, "")
                                              for key in ("database", "database_oid", "server_started"))


async def verify_isolated_database(session, *, evaluation_url: str, application_url: str | None,
                                   migration_head: str | None = None, application_identity=None) -> dict:
    """Refuse any database that is not a verified, dedicated, current, evaluation-only database."""
    check_database_urls(evaluation_url=evaluation_url, application_url=application_url)
    evaluation = await database_identity(session)
    try:
        application = await (application_identity() if application_identity else probe_database_identity(application_url))
    except Exception as error:
        raise IsolationError(f"application database identity could not be verified ({type(error).__name__}); "
                             "refusing to run without verified isolation") from None
    if not _complete_identity(evaluation) or not _complete_identity(application):
        raise IsolationError("database identity could not be verified (missing name, OID or server start time)")
    if all(evaluation[key] == application[key] for key in ("database", "database_oid", "server_started")):
        raise IsolationError("evaluation and application URLs reach the same physical database")
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
    return {"migration_head": revision, "foreign_research_runs": 0, "evaluation_database": evaluation,
            "application_database": application}


def _alembic_revision():
    return text("SELECT version_num FROM alembic_version")


def _trace_summary(service) -> dict:
    trace = service.retrieval_traces[-1] if service.retrieval_traces else {}
    return {key: trace.get(key) for key in ("vector_count", "mmr_count", "fulltext_count", "selected_count")} | {
        "fts_decision": (trace.get("fts_diagnostics") or {}).get("decision", "applied" if trace.get("fts_query") else None)}


async def retrieve_one(item: dict, *, session, settings: Settings, config: dict,
                       service_factory: Callable = PostgresVectorService, reservation: dict | None = None) -> dict:
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
    reservation = reservation or {}
    usage = {"provider_calls_recorded": embedding["calls"], "input_tokens_estimated": embedding["input_tokens_estimated"],
             "provider_models": [f"gemini:{embedding['model']}"], "timings_ms": timings,
             "reserved_requests": reservation.get("requests"), "reserved_cost_usd": reservation.get("reserve_cost_usd")}
    if error is None and embedding["estimated_cost_usd"] is not None:
        # Gemini embeddings return no usage: production's estimate, never a measured cost
        usage.update(cost_status="estimated", estimated_cost_usd=embedding["estimated_cost_usd"])
    else:   # failed or unpriced requests: unknown spend, never zero
        usage.update(cost_status="incomplete", known_cost_lower_bound_usd=embedding["estimated_cost_usd"] or 0.0)
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
                        migration_head: str | None = None, application_identity=None) -> tuple[list[dict], dict]:
    """Run the production retriever on each frozen pack within the authorized embedding ceiling. Unscored."""
    auth = check_authorization(authorization, adapter=ADAPTER, dataset_hash=dataset_hash,
                               route={"provider_order": ["gemini"], "models": {"gemini": settings.embedding_model}})
    config = configuration(settings, research_mode)
    check_database_urls(evaluation_url=evaluation_url, application_url=application_url)   # before any connection
    async with session_factory() as session:
        isolation = await verify_isolated_database(session, evaluation_url=evaluation_url,
                                                   application_url=application_url, migration_head=migration_head,
                                                   application_identity=application_identity)
    budget = {"provider_requests_charged": 0, "cost_complete_usd": None, "cost_estimated_usd": 0.0,
              "known_cost_lower_bound_usd": 0.0, "uncertain_reserved_usd": 0.0, "budget_charged_usd": 0.0}
    records, stop = [], None
    for item in retrieval_items(items):
        if not text_segments(item):
            records.append(_record(item, "fixed_evidence_retrieval", "not_executed", reason=TEXT_REASON))
            continue
        reservation = pack_embedding_plan(item, settings)
        if reservation["reserve_cost_usd"] is None:
            stop = stop or "stopped: the embedding model has no known price, so the budget cannot be enforced"
        if (stop or budget["provider_requests_charged"] + reservation["requests"] > auth["max_calls"]
                or budget["budget_charged_usd"] + reservation["reserve_cost_usd"] > auth["max_cost_usd"]):
            records.append(_record(item, "fixed_evidence_retrieval", "not_executed", reason=stop or CEILING_REASON))
            continue
        async with session_factory() as session:
            record = await retrieve_one(item, session=session, settings=settings, config=config,
                                        service_factory=service_factory, reservation=reservation)
        if record["status"] == "invalid" and ISOLATION_VIOLATION in record.get("reason", ""):
            stop = "stopped: corpus isolation was violated"
        usage = record["usage"]
        # A started pack consumes its whole reservation: failed requests are not counted by production.
        budget["provider_requests_charged"] += max(reservation["requests"], usage["provider_calls_recorded"])
        budget["budget_charged_usd"] += reservation["reserve_cost_usd"]
        if usage["cost_status"] == "estimated":
            budget["cost_estimated_usd"] += usage["estimated_cost_usd"]
        else:
            budget["known_cost_lower_bound_usd"] += usage["known_cost_lower_bound_usd"]
            budget["uncertain_reserved_usd"] += reservation["reserve_cost_usd"]
        records.append(record)
    records.sort(key=lambda record: record["input_id"])
    budget = {key: round(value, 8) if isinstance(value, float) else value for key, value in budget.items()}
    return records, {"authorization_id": auth["authorization_id"], "budget": budget, "stopped": stop,
                     "isolation": isolation, "configuration": config}


__all__ = ["ADAPTER", "AuthorizationError", "IsolationError", "check_database_urls", "configuration",
           "corpus_documents", "corpus_identity", "database_url_identity", "dry_run",
           "effective_application_database_url", "retrieval_items", "retrieve_one", "run_retrieval",
           "verify_isolated_database"]
