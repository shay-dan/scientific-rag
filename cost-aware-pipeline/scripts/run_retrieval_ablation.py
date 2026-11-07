"""Run a provider-light retrieval-only ablation against the frozen snapshot."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.config import get_settings
from app.db import session_factory
from app.pgvector_retrieval import PostgresVectorService


STRATEGIES = ("naive", "hybrid_basic", "mmr", "hybrid_mmr", "enhanced")


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_recall(returned: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    return round(len(set(returned) & expected_set) / len(expected_set), 4) if expected_set else 0.0


async def run(args: argparse.Namespace) -> int:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = dataset["cases"][: args.case_limit] if args.case_limit else dataset["cases"]
    report = {
        "experiment": "retrieval-only-ablation-12x5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_hash": manifest["snapshot_hash"],
        "snapshot_run_id": manifest["research_run_id"],
        "strategies": list(STRATEGIES),
        "answer_generation_calls": 0,
        "llm_calls": 0,
        "query_embedding_calls_expected": len(cases),
        "results": [],
    }
    settings = get_settings()
    for case in cases:
        print(f"retrieving {case['id']}", flush=True)
        async with session_factory() as session:
            repository = PostgresVectorService(
                session, settings, UUID(manifest["research_run_id"])
            )
            repository.restrict_final_retrieval_to_current_run()
            started = time.perf_counter()
            documents_by_strategy = await repository.retrieve_context_strategies(
                case["question"], STRATEGIES, top_n=args.top_n
            )
            total_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        traces = {trace["strategy"]: trace for trace in repository.retrieval_traces}
        expected = list(dict.fromkeys(row["source_id"] for row in case["reference_context"]))
        for strategy, documents in documents_by_strategy.items():
            source_ids = [str(doc.metadata.get("doc_id") or "") for doc in documents]
            report["results"].append({
                "case_id": case["id"],
                "source_group": case["source_group"],
                "question": case["question"],
                "strategy": strategy,
                "selected_source_ids": source_ids,
                "selected_context_ids": [doc.metadata.get("benchmark_context_id") for doc in documents],
                "source_recall": source_recall(source_ids, expected),
                "returned_context_count": len(documents),
                "fulltext_count": traces[strategy].get("fulltext_count"),
                "fts_query": traces[strategy].get("fts_query"),
                "retrieval_trace": traces[strategy],
                "shared_case_latency_ms": total_latency_ms,
            })
        report.setdefault("embedding_usage", {"calls": 0, "input_tokens_estimated": 0, "estimated_cost_usd": 0.0})
        usage = repository.embedding_usage
        report["embedding_usage"]["calls"] += int(usage.get("calls") or 0)
        report["embedding_usage"]["input_tokens_estimated"] += int(usage.get("input_tokens_estimated") or 0)
        report["embedding_usage"]["estimated_cost_usd"] = round(
            float(report["embedding_usage"]["estimated_cost_usd"]) + float(usage.get("estimated_cost_usd") or 0), 8
        )
        _write(args.output, report)
    summary = {}
    for strategy in STRATEGIES:
        rows = [row for row in report["results"] if row["strategy"] == strategy]
        summary[strategy] = {
            "cases": len(rows),
            "mean_source_recall": round(sum(row["source_recall"] for row in rows) / len(rows), 4),
            "queries_with_fts_hits": sum(int(row["fulltext_count"] or 0) > 0 for row in rows),
            "mean_fts_candidates": round(sum(int(row["fulltext_count"] or 0) for row in rows) / len(rows), 2),
        }
    report["summary"] = summary
    _write(args.output, report)
    print(json.dumps({"summary": summary, "embedding_usage": report["embedding_usage"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation" / "frozen-12-corpus-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "retrieval-ablation-12x5.json")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--top-n", type=int, default=5)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
