"""Build a shared, label-free corpus for the real 12-case comparison."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

from langchain_core.documents import Document
from scientific_rag.costing import calculate_cost, estimate_text_tokens


def frozen_corpus_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten source excerpts without carrying question/answer labels."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for case in payload.get("cases", []):
        for context in case.get("reference_context", []):
            source_id = str(context.get("source_id") or "").strip()
            excerpt = str(context.get("excerpt") or "").strip()
            fingerprint = (source_id, hashlib.sha256(excerpt.encode("utf-8")).hexdigest())
            if not source_id or not excerpt or fingerprint in seen:
                continue
            seen.add(fingerprint)
            url = str(context.get("url") or "")
            source = "pubmed" if source_id.upper().startswith("PMID:") or "pubmed" in url else "arxiv"
            rows.append(
                {
                    "context_id": str(context.get("context_id") or ""),
                    "source_id": source_id,
                    "title": str(context.get("title") or "Untitled source"),
                    "url": url,
                    "source": source,
                    "excerpt": excerpt,
                }
            )
    return sorted(rows, key=lambda row: (row["source_id"], row["context_id"], row["excerpt"]))


def snapshot_hash(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def snapshot_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = frozen_corpus_rows(payload)
    estimated_embedding_tokens = estimate_text_tokens([row["excerpt"] for row in rows])
    estimated_embedding_cost, rate_source = calculate_cost(
        "gemini", "gemini-embedding-2", estimated_embedding_tokens
    )
    grouped: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[row["source"]] += 1
    return {
        "snapshot_version": f"{payload.get('dataset_name', 'silver-dataset')}-shared-corpus",
        "snapshot_hash": snapshot_hash(rows),
        "case_count": len(payload.get("cases", [])),
        "segment_count": len(rows),
        "unique_source_count": len({row["source_id"] for row in rows}),
        "segments_by_source": dict(sorted(grouped.items())),
        "embedding_dimensions": 768,
        "estimated_embedding_tokens": estimated_embedding_tokens,
        "estimated_embedding_cost_usd": estimated_embedding_cost,
        "embedding_rate_source": rate_source,
        "contains_reference_answers": False,
        "contains_reference_claims": False,
        "corpus_source_ids": sorted({row["source_id"] for row in rows}),
    }


def snapshot_documents(payload: dict[str, Any]) -> list[Document]:
    return [
        Document(
            page_content=row["excerpt"],
            metadata={
                "doc_id": row["source_id"],
                "title": row["title"],
                "url": row["url"],
                "source": row["source"],
                "benchmark_context_id": row["context_id"],
                "benchmark_snapshot": True,
            },
        )
        for row in frozen_corpus_rows(payload)
    ]
