"""Replay fixed benchmark contexts through pinned Expert providers/models."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.real_comparison import FrozenResearchAgent
from scientific_rag.costing import calculate_cost, estimate_text_tokens
from scientific_rag.llm_gateway import begin_trace, end_trace, summarize_trace


CANDIDATES = {
    "groq_gpt_oss_120b": ("groq", "openai/gpt-oss-120b"),
    "openrouter_gpt_oss_120b_paid": ("openrouter", "openai/gpt-oss-120b"),
    "openrouter_claude_haiku_5_5": ("openrouter", "anthropic/claude-haiku-5.5"),
    "gemini_3_8_flash": ("gemini", "gemini-3.8-flash"),
}
DEFAULT_CASES = ("pm-01", "pm-03", "ax-01", "ax-03", "mx-03", "mx-04")


class FixedRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    async def retrieve_context(self, query: str, top_n: int = 5) -> list[Document]:
        return list(self.documents[:top_n])


@contextmanager
def pinned_provider(provider: str, model: str):
    keys = ("LLM_PROVIDER_ORDER", "GROQ_MODEL_EXPERT", "OPENROUTER_MODEL_EXPERT", "MODEL_EXPERT",
            "LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS", "LLM_OPENROUTER_TIMEOUT_SECONDS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LLM_PROVIDER_ORDER"] = provider
        os.environ["GROQ_MODEL_EXPERT"] = model
        os.environ["OPENROUTER_MODEL_EXPERT"] = model
        os.environ["MODEL_EXPERT"] = model
        os.environ["LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS"] = "0"
        os.environ["LLM_OPENROUTER_TIMEOUT_SECONDS"] = "60"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(dataset_path: Path, comparison_path: Path, case_ids: list[str]) -> list[dict]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in dataset["cases"]}
    contexts = {context["context_id"]: context for case in dataset["cases"]
                for context in case.get("reference_context", [])}
    prepared = []
    for case_id in case_ids:
        case = cases[case_id]
        baseline = next(row for row in comparison["results"]
                        if row["case_id"] == case_id and row["variant"] == "hybrid_basic")
        documents = []
        for source_id, context_id in zip(
            baseline["selected_source_ids"], baseline["selected_context_ids"], strict=True
        ):
            context = contexts[context_id]
            documents.append(Document(
                page_content=context["excerpt"],
                metadata={
                    "doc_id": source_id,
                    "title": context.get("title", "Unknown source"),
                    "source": context.get("source", case.get("source_group", "benchmark")),
                    "type": "benchmark_excerpt",
                    "benchmark_context_id": context_id,
                },
            ))
        prepared.append({"case": case, "documents": documents})
    return prepared


def preflight(prepared: list[dict], repetitions: int, candidates: list[str]) -> dict:
    # Includes a conservative allowance for the static synthesis instructions.
    per_case_inputs = []
    for item in prepared:
        context = "\n\n".join(document.page_content for document in item["documents"])
        per_case_inputs.append(estimate_text_tokens([item["case"]["question"], context, "x" * 6500]))
    output_tokens_per_call = 1800
    by_candidate = {}
    total: float | None = 0.0
    for candidate in candidates:
        provider, model = CANDIDATES[candidate]
        input_tokens = sum(per_case_inputs) * repetitions
        output_tokens = len(prepared) * repetitions * output_tokens_per_call
        cost, source = calculate_cost(provider, model, input_tokens, output_tokens)
        by_candidate[candidate] = {
            "provider": provider,
            "model": model,
            "calls": len(prepared) * repetitions,
            "estimated_input_tokens": input_tokens,
            "budgeted_output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "rate_source": source,
        }
        if cost is None:
            total = None
        elif total is not None:
            total += cost
    return {
        "cases": len(prepared),
        "repetitions": repetitions,
        "total_calls": len(prepared) * repetitions * len(candidates),
        "by_candidate": by_candidate,
        "estimated_generation_cost_usd": round(total, 8) if total is not None else None,
    }


async def execute(item: dict, candidate: str, repetition: int) -> dict:
    provider, model = CANDIDATES[candidate]
    case = item["case"]
    started = time.perf_counter()
    token = begin_trace()
    trace_closed = False
    try:
        with pinned_provider(provider, model):
            agent = FrozenResearchAgent(
                FixedRetriever(item["documents"]),
                benchmark_subquestions=[case["question"]],
                max_synthesis_subquestions=1,
                retrieval_documents=5,
                max_context_documents=5,
            )
            report = await agent.run_research(case["question"])
        attempts = end_trace(token)
        trace_closed = True
        selected = report.get("raw_docs_used", [])
        return {
            "status": "completed",
            "case_id": case["id"],
            "question": case["question"],
            "variant": candidate,
            "candidate": candidate,
            "provider": provider,
            "model": model,
            "repetition": repetition,
            "answer": report["answer"],
            "selected_source_ids": [str(doc.metadata["doc_id"]) for doc in selected],
            "selected_context_ids": [doc.metadata["benchmark_context_id"] for doc in selected],
            "source_count": len({str(doc.metadata["doc_id"]) for doc in selected}),
            "context_count": len(selected),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm": summarize_trace(attempts),
        }
    except Exception as error:
        attempts = end_trace(token)
        trace_closed = True
        return {
            "status": "failed", "case_id": case["id"], "question": case["question"],
            "variant": candidate, "candidate": candidate, "provider": provider, "model": model,
            "repetition": repetition, "error": f"{type(error).__name__}: {error}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm": summarize_trace(attempts),
        }
    finally:
        if not trace_closed:
            end_trace(token)


async def run(args) -> int:
    case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    unknown = [item for item in candidates if item not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    prepared = prepare(args.dataset, args.comparison, case_ids)
    estimate = preflight(prepared, args.repetitions, candidates)
    print(json.dumps({"preflight": estimate}, indent=2), flush=True)
    if args.estimate_only:
        return 0
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists() else {
        "experiment": "fixed-context-expert-provider-comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": case_ids, "candidates": candidates, "repetitions": args.repetitions,
        "preflight": estimate, "results": [],
    }
    completed = {(row["case_id"], row["candidate"], row["repetition"])
                 for row in report["results"] if row.get("status") == "completed"}
    for item in prepared:
        for candidate in candidates:
            for repetition in range(1, args.repetitions + 1):
                key = (item["case"]["id"], candidate, repetition)
                if key in completed:
                    continue
                print(f"running {key}", flush=True)
                result = await execute(item, candidate, repetition)
                report["results"] = [row for row in report["results"]
                                     if (row["case_id"], row["candidate"], row["repetition"]) != key]
                report["results"].append(result)
                _write(args.output, report)
    report["summary"] = {
        "planned": estimate["total_calls"],
        "completed": sum(row.get("status") == "completed" for row in report["results"]),
        "failed": sum(row.get("status") == "failed" for row in report["results"]),
    }
    _write(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v3.json")
    parser.add_argument("--comparison", type=Path, default=ROOT / "evaluation" / "real-frozen-v3-12x4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "expert-provider-comparison.json")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASES))
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
