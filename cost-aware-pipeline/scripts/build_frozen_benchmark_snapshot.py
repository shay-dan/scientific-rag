"""Create or inspect the shared PostgreSQL/pgvector benchmark snapshot."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.benchmark_dataset import validate_pilot_dataset
from app.config import get_settings
from app.db import session_factory
from app.frozen_benchmark import snapshot_documents, snapshot_manifest
from app.models import ResearchRun, SourceDocument
from app.pgvector_retrieval import PostgresVectorService


async def materialize(payload: dict, manifest: dict) -> dict:
    async with session_factory() as session:
        existing = list(
            (
                await session.execute(
                    select(ResearchRun).where(ResearchRun.requested_by == "benchmark-snapshot")
                )
            ).scalars()
        )
        for run in existing:
            if run.model_metadata.get("benchmark_snapshot_hash") == manifest["snapshot_hash"]:
                source_count = len(
                    list(
                        (
                            await session.execute(
                                select(SourceDocument).where(SourceDocument.research_run_id == run.id)
                            )
                        ).scalars()
                    )
                )
                return {**manifest, "database_status": "reused", "research_run_id": str(run.id), "database_source_count": source_count}

        run = ResearchRun(
            question="Frozen corpus for 12-case RAG architecture comparison",
            requested_by="benchmark-snapshot",
            status="completed",
            answer=None,
            completed_at=datetime.now(timezone.utc),
            model_metadata={
                "benchmark_snapshot_hash": manifest["snapshot_hash"],
                "snapshot_version": manifest["snapshot_version"],
                "immutable_benchmark_corpus": True,
            },
        )
        session.add(run)
        await session.flush()
        service = PostgresVectorService(session, get_settings(), run.id)
        await service.add_documents(snapshot_documents(payload))
        await session.commit()
        return {
            **manifest,
            "database_status": "created",
            "research_run_id": str(run.id),
            "embedding_usage": service.embedding_usage,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation" / "frozen-12-corpus-manifest.json")
    parser.add_argument("--materialize", action="store_true", help="Create embeddings and rows after local tests pass")
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    errors = validate_pilot_dataset(payload)
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, indent=2))
        return 2
    manifest = snapshot_manifest(payload)
    if args.materialize:
        manifest = asyncio.run(materialize(payload, manifest))
    else:
        manifest["database_status"] = "not_materialized_dry_run"
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
