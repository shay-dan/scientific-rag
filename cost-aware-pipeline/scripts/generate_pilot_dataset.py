"""Generate and independently validate a 12-case silver evaluation set.

The dataset is produced from live PubMed/arXiv evidence. It is deliberately
not human-reviewed; every case carries provenance and an independent validator
decision so it is clearly distinguishable from the reviewed benchmark bank.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scientific_rag.costing import calculate_cost  # noqa: E402
from scientific_rag.legacy_pipeline import MinerService  # noqa: E402
from scientific_rag.llm_gateway import ainvoke_structured, model_for, primary_route_label  # noqa: E402


class ClaimSupport(BaseModel):
    claim: str
    source_id: str
    excerpt: str = Field(min_length=20)


class GeneratedCase(BaseModel):
    question: str
    reference_answer: str = Field(min_length=80)
    reference_claims: list[str] = Field(min_length=3, max_length=6)
    claim_support: list[ClaimSupport] = Field(min_length=3, max_length=6)


class ValidationResult(BaseModel):
    approved: bool
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)


CaseSource = Literal["pubmed", "arxiv", "mixed"]

CASES: list[tuple[str, CaseSource, str]] = [
    ("pm-01", "pubmed", "What is the current evidence for GLP-1 receptor agonists in obesity treatment?"),
    ("pm-02", "pubmed", "How does gut microbiome dysbiosis contribute to colorectal cancer?"),
    ("pm-03", "pubmed", "What are the clinical benefits and limitations of CAR-T cell therapy?"),
    ("pm-04", "pubmed", "What are the main causes and prevention strategies for antimicrobial resistance in hospitals?"),
    ("ax-01", "arxiv", "What are current approaches to quantum error correction with surface codes?"),
    ("ax-02", "arxiv", "How are diffusion models used for image generation, and what are their limitations?"),
    ("ax-03", "arxiv", "How do linear attention and state-space models improve efficiency on long-context sequence modeling?"),
    ("ax-04", "arxiv", "How is reinforcement learning used for robotic manipulation?"),
    ("mx-01", "mixed", "How is machine learning being used for early cancer detection, and what limits clinical adoption?"),
    ("mx-02", "mixed", "What are the opportunities and risks of AI-assisted drug discovery?"),
    ("mx-03", "mixed", "How can foundation models support scientific discovery while remaining reproducible?"),
    ("mx-04", "mixed", "What are current uses of CRISPR in cancer research and what barriers limit translation?"),
]


def _model_label(role: str) -> str:
    return primary_route_label(role) or model_for("groq", role)


async def _structured_with_order(order: str, role: str, temperature: float, prompt: str, schema: type):
    """Call one role through an explicit provider order for this pilot step."""
    previous = os.environ.get("LLM_PROVIDER_ORDER")
    os.environ["LLM_PROVIDER_ORDER"] = order
    try:
        return await ainvoke_structured(role, temperature, prompt, schema)
    finally:
        if previous is None:
            os.environ.pop("LLM_PROVIDER_ORDER", None)
        else:
            os.environ["LLM_PROVIDER_ORDER"] = previous


def estimate_cost() -> dict:
    """Conservative upper-bound estimate before any network/model call."""
    # UnifiedMiner can screen up to three candidates per source. We budget two
    # source searches plus three 600/180-token relevance calls per source.
    source_calls = sum(1 if source != "mixed" else 2 for _, source, _ in CASES)
    screen_calls = source_calls * 3
    generator_input = 4200
    generator_output = 700
    validator_input = 4200
    validator_output = 450
    gatekeeper_model = model_for("groq", "gatekeeper")
    generator_model = model_for("groq", "qa_generator")
    validator_model = model_for("gemini", "query_repair")
    gate_cost, _ = calculate_cost("groq", gatekeeper_model, screen_calls * 600, screen_calls * 180)
    gen_cost, _ = calculate_cost("groq", generator_model, len(CASES) * generator_input, len(CASES) * generator_output)
    val_cost, _ = calculate_cost("gemini", validator_model, len(CASES) * validator_input, len(CASES) * validator_output)
    total = sum(value or 0 for value in (gate_cost, gen_cost, val_cost))
    return {
        "cases": len(CASES), "source_calls": source_calls, "screen_calls_budgeted": screen_calls,
        "generator": f"groq:{generator_model}", "validator": f"gemini:{validator_model}",
        "estimated_usd_upper_bound": round(total, 4),
        "components_usd": {"source_screening": gate_cost or 0, "generation": gen_cost or 0, "validation": val_cost or 0},
        "daily_budget_usd": float(os.getenv("COST_DAILY_BUDGET_USD", "5")),
        "assumptions": "3 relevance screens per source, 4.2k input tokens per generation/validation; provider retries and embeddings excluded",
    }


def _doc_payload(docs) -> tuple[str, list[dict]]:
    rows = []
    for index, doc in enumerate(docs, 1):
        meta = dict(doc.metadata)
        source_id = str(meta.get("doc_id") or meta.get("arxiv_id") or f"doc-{index}")
        rows.append({"source_id": source_id, "title": meta.get("title", ""), "url": meta.get("url") or meta.get("source_url", ""), "excerpt": doc.page_content[:1800]})
    rows = rows[:8]
    context = "\n\n".join(f"[{row['source_id']}] {row['title']}\n{row['excerpt']}" for row in rows)
    return context, rows


async def generate_one(case_id: str, source: CaseSource, question: str) -> tuple[dict | None, dict]:
    diagnostics: dict = {"case_id": case_id, "source_group": source, "question": question, "mining": {}}
    source_list = [source] if source != "mixed" else ["pubmed", "arxiv"]
    docs = []
    for source_name in source_list:
        found = await MinerService.mine(question, source_name, limit=4, anchor_query=question, diagnostics=diagnostics["mining"].setdefault(source_name, {}))
        docs.extend(found)
    context, rows = _doc_payload(docs)
    diagnostics["retrieved_sources"] = rows
    if len(rows) < 2 or len(context) < 500:
        diagnostics.update({"status": "rejected", "reason": "insufficient_source_evidence"})
        return None, diagnostics

    generation_prompt = f"""You are the benchmark question-answer generator. Use ONLY the source excerpts below.
Create a self-contained scientific question and answer based on the requested topic. Make 3-6 precise claims.
Every claim must be supported by one source excerpt. Do not invent citations, numbers, or consensus.
Topic question: {question}
Source excerpts:
{context}
Return the requested structured object. The source_id values must exactly match the bracketed IDs."""
    # Prefer the cheap/fast Groq route, but permit the configured gateway
    # fallback when Groq is rate-limited. Validation remains explicitly Gemini.
    generated = await _structured_with_order("groq,gemini,openrouter", "qa_generator", 0.1, generation_prompt, GeneratedCase)

    validation_prompt = f"""You are an independent evidence auditor. Validate this generated Q&A against the source excerpts.
Approve only if every material claim is directly supported and the answer does not overstate the evidence.
Give a score from 0 to 1 and list unsupported claims or missing context.
QUESTION: {generated.question}
ANSWER: {generated.reference_answer}
CLAIMS: {json.dumps(generated.reference_claims)}
SOURCE EXCERPTS:
{context}"""
    validated = await _structured_with_order("gemini", "query_repair", 0.0, validation_prompt, ValidationResult)
    diagnostics["validation"] = validated.model_dump()
    if not validated.approved or validated.score < 0.80:
        diagnostics.update({"status": "rejected", "reason": "independent_validation_failed"})
        return None, diagnostics

    by_id = {row["source_id"]: row for row in rows}
    reference_context = []
    persisted_support = []
    for support in generated.claim_support:
        candidates = [row for row in rows if row["source_id"] == support.source_id]
        if candidates:
            # Persist the mined excerpt, never an LLM-rewritten quotation.
            row = max(
                candidates,
                key=lambda item: len(set(support.excerpt.lower().split()) & set(item["excerpt"].lower().split())),
            )
            context_id = f"{case_id}-ctx-{len(reference_context) + 1:02d}"
            reference_context.append({"context_id": context_id, "source_id": support.source_id, "title": row["title"], "url": row["url"], "excerpt": row["excerpt"]})
            persisted_support.append({"claim": support.claim, "context_id": context_id, "source_id": support.source_id, "mapping_method": "generator_explicit_source_mapping"})
    result = {
        "id": case_id, "source_group": source, "question": generated.question,
        "reference_answer": generated.reference_answer, "reference_claims": generated.reference_claims,
        "reference_context": reference_context, "claim_support": persisted_support,
        "claim_support_status": "generator_explicit_mapping", "provenance": "silver_auto_generated",
        "human_review": False, "generator": "groq:" + model_for("groq", "qa_generator"),
        "validator": "gemini:" + model_for("gemini", "query_repair"), "validation_score": validated.score,
    }
    diagnostics["status"] = "approved"
    return result, diagnostics


async def run(output: Path, report: Path, case_id: str | None = None) -> None:
    estimate = estimate_cost()
    print(json.dumps({"preflight": estimate}, indent=2))
    if estimate["estimated_usd_upper_bound"] > estimate["daily_budget_usd"]:
        raise SystemExit("Preflight refused: conservative estimate exceeds COST_DAILY_BUDGET_USD")
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"cases": []}
    results = [item for item in existing.get("cases", []) if case_id is None or item.get("id") != case_id]
    diagnostics = []
    selected_cases = [item for item in CASES if case_id is None or item[0] == case_id]
    for case_id, source, question in selected_cases:
        try:
            item, trace = await generate_one(case_id, source, question)
        except Exception as error:
            item, trace = None, {"case_id": case_id, "status": "error", "error": f"{type(error).__name__}: {error}"}
        diagnostics.append(trace)
        if item:
            results.append(item)
        print(f"{case_id}: {trace.get('status')}", flush=True)
    payload = {"dataset_name": "scientific-rag-pilot-silver-v1", "created_at": datetime.now(timezone.utc).isoformat(), "case_count": len(results), "required_case_count": 12, "human_review": False, "cases": results}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report.write_text(json.dumps({"preflight": estimate, "cases": diagnostics, "approved": len(results)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)}/{len(CASES)} approved cases to {output}")
    if case_id is None and len(results) < 12:
        raise SystemExit("Pilot incomplete: fewer than 12 independently validated cases")


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1-report.json")
    parser.add_argument("--case-id", help="Generate or replace one case in an existing pilot dataset")
    args = parser.parse_args()
    if args.estimate_only:
        print(json.dumps(estimate_cost(), indent=2))
        return
    asyncio.run(run(args.output, args.report, args.case_id))


if __name__ == "__main__":
    main()
