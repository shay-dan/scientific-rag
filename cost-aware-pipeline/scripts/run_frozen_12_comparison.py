"""Deprecated comparison runner retained only for auditability.

This runner approximated retrieval with in-memory token overlap and exposed each
case only to its own reference excerpts.  It therefore did not compare the real
application architectures and must not be used for benchmark conclusions.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scientific_rag.llm_gateway import ainvoke, ainvoke_structured, begin_trace, end_trace, summarize_trace  # noqa: E402


class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=2, max_length=4)


VARIANTS = ("naive", "hybrid_basic", "hybrid_enhanced", "agentic_fast")
STOP = {"what", "are", "the", "and", "for", "how", "is", "in", "of", "to", "a", "an", "with", "on", "current", "known"}


def terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in STOP}


def source_key(row: dict) -> str:
    return str(row.get("source_id") or row.get("url") or row.get("title"))


def rank_rows(question: str, rows: list[dict], variant: str) -> list[dict]:
    q = terms(question)
    scored = []
    for row in rows:
        text = f"{row.get('title','')} {row.get('excerpt','')}".lower()
        overlap = len(q & terms(text))
        phrase = sum(1 for token in q if token in text)
        exact = sum(1 for token in ("crispr", "cancer", "surface", "quantum", "microbiome", "transformer") if token in text and token in question.lower())
        score = overlap + 0.2 * phrase + exact
        if variant == "hybrid_basic": score += 0.5 * exact
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [row for _, row in scored]
    if variant == "hybrid_enhanced":
        chosen, seen = [], set()
        for row in ranked:
            key = source_key(row)
            if key not in seen or len(chosen) < 2:
                chosen.append(row); seen.add(key)
            if len(chosen) >= 5: break
        return chosen
    return ranked[:5]


def context(rows: list[dict]) -> str:
    return "\n\n".join(f"--- [S{i+1}: {row.get('title','Unknown')}]\n{row.get('excerpt','')}\nURL: {row.get('url','')}" for i, row in enumerate(rows))


async def answer(question: str, rows: list[dict], variant: str) -> tuple[str, dict]:
    prompt = ChatPromptTemplate.from_template("""Answer the scientific question using only the supplied evidence.
Use cautious wording. Cite every material claim with one or more labels such as [S1].
If evidence is insufficient, say so explicitly.
Question: {question}
Evidence:
{context}""")
    trace_token = begin_trace()
    try:
        result = await ainvoke("expert", 0.0, lambda model: prompt | model | StrOutputParser(), {"question": question, "context": context(rows)})
        return str(result), summarize_trace(end_trace(trace_token))
    except Exception:
        attempts = end_trace(trace_token)
        raise


async def agentic_rows(question: str, rows: list[dict]) -> list[dict]:
    prompt = "Break this question into 2-3 focused retrieval sub-questions; return only JSON. Question: " + question
    token = begin_trace()
    try:
        plan = await ainvoke_structured("planner", 0.0, prompt, Plan)
    finally:
        end_trace(token)
    selected = []
    seen = set()
    for sub in plan.sub_questions:
        for row in rank_rows(sub, rows, "hybrid_enhanced")[:3]:
            if source_key(row) not in seen:
                selected.append(row); seen.add(source_key(row))
    return selected[:6]


def recall(rows: list[dict], expected: list[dict]) -> float:
    expected_ids = {source_key(row) for row in expected}
    found = {source_key(row) for row in rows[:5]}
    return round(len(found & expected_ids) / len(expected_ids), 4) if expected_ids else 0.0


async def run_case(case: dict, variant: str) -> dict:
    rows = case.get("reference_context", [])
    started = time.perf_counter()
    planning_trace = None
    if variant == "agentic_fast":
        rows = await agentic_rows(case["question"], rows)
    else:
        rows = rank_rows(case["question"], rows, variant)
    try:
        generated, trace = await answer(case["question"], rows, variant)
        return {"case_id": case["id"], "variant": variant, "status": "completed", "answer": generated,
                "selected_sources": [source_key(row) for row in rows], "reference_sources": [source_key(row) for row in case.get("reference_context", [])],
                "recall_at_5": recall(rows, case.get("reference_context", [])), "latency_ms": round((time.perf_counter()-started)*1000, 1),
                "llm": trace, "cost_usd": trace.get("estimated_cost_usd")}
    except Exception as error:
        return {"case_id": case["id"], "variant": variant, "status": "failed", "error": f"{type(error).__name__}: {error}",
                "selected_sources": [source_key(row) for row in rows], "latency_ms": round((time.perf_counter()-started)*1000, 1), "cost_usd": None}


async def main_async(dataset: Path, output: Path) -> None:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if len(cases) != 12: raise SystemExit(f"Expected exactly 12 frozen cases, found {len(cases)}")
    results = []
    for case in cases:
        for variant in VARIANTS:
            print(f"{case['id']} {variant}", flush=True)
            results.append(await run_case(case, variant))
    completed = [r for r in results if r["status"] == "completed"]
    known_costs = [r["cost_usd"] for r in completed if r.get("cost_usd") is not None]
    report = {"dataset": str(dataset), "frozen_case_count": len(cases), "variants": VARIANTS,
              "run_count": len(results), "completed_count": len(completed), "failed_count": len(results)-len(completed),
              "known_cost_total_usd": round(sum(known_costs), 8) if len(known_costs) == len(completed) else None,
              "cost_status": "complete" if len(known_costs) == len(completed) else "incomplete_usage_metadata",
              "results": results}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("run_count", "completed_count", "failed_count", "known_cost_total_usd", "cost_status")}, indent=2))


def main() -> None:
    raise SystemExit(
        "This legacy runner is intentionally disabled: it used approximate "
        "in-memory retrieval instead of the application's pgvector/FTS/MMR and "
        "agentic implementations. Use scripts/run_real_frozen_comparison.py "
        "after the real benchmark harness has passed its tests."
    )


if __name__ == "__main__": main()
