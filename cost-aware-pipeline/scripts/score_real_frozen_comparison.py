"""Score saved comparison answers with one independent Gemini judge call each."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.comparison_judge import PROMPT_SECURITY_VERSION, ComparisonJudgeScore, build_judge_prompt
from app.config import get_settings
from scientific_rag.costing import calculate_cost, estimate_text_tokens


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _usage(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_token_count", None)
    output = getattr(usage, "candidates_token_count", None)
    return (int(prompt), int(output)) if prompt is not None and output is not None else (None, None)


def prepare(
    input_path: Path,
    dataset_path: Path,
    *,
    case_ids: set[str] | None = None,
    variants: set[str] | None = None,
) -> tuple[dict, dict[str, dict], list[tuple[dict, str]]]:
    comparison = json.loads(input_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in dataset["cases"]}
    corpus_contexts = {
        context["context_id"]: context
        for case in dataset["cases"]
        for context in case.get("reference_context", [])
    }
    pending = []
    for result in comparison["results"]:
        if case_ids and result.get("case_id") not in case_ids:
            continue
        if variants and result.get("variant") not in variants:
            continue
        if result.get("status") != "completed" or result.get("judge"):
            continue
        case = cases[result["case_id"]]
        pending.append((result, build_judge_prompt(case, result, corpus_contexts=corpus_contexts)))
    return comparison, cases, pending


def estimate(
    input_path: Path,
    dataset_path: Path,
    model: str,
    *,
    case_ids: set[str] | None = None,
    variants: set[str] | None = None,
) -> dict:
    _, _, pending = prepare(input_path, dataset_path, case_ids=case_ids, variants=variants)
    input_tokens = estimate_text_tokens([prompt for _, prompt in pending])
    # Rubric v2 returns per-reference-claim diagnostics in addition to aggregate scores.
    output_tokens = len(pending) * 700
    cost, rate_source = calculate_cost("gemini", model, input_tokens, output_tokens)
    return {
        "pending_results": len(pending),
        "judge_model": model,
        "estimated_input_tokens": input_tokens,
        "budgeted_output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "rate_source": rate_source,
        "calls_per_result": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "evaluation" / "real-frozen-12x4.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "real-frozen-12x4-scored.json")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-ids", help="Optional comma-separated case filter")
    parser.add_argument("--variants", help="Optional comma-separated variant filter")
    args = parser.parse_args()
    settings = get_settings()
    case_ids = {item.strip() for item in args.case_ids.split(",") if item.strip()} if args.case_ids else None
    variants = {item.strip() for item in args.variants.split(",") if item.strip()} if args.variants else None
    source = args.output if args.resume and args.output.exists() else args.input
    preflight = estimate(
        source, args.dataset, settings.ragas_judge_model,
        case_ids=case_ids, variants=variants,
    )
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.estimate_only:
        return 0
    comparison, cases, pending = prepare(source, args.dataset, case_ids=case_ids, variants=variants)
    client = genai.Client(api_key=settings.google_api_key)
    for result, prompt in pending:
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=settings.ragas_judge_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=ComparisonJudgeScore,
                ),
            )
            score = ComparisonJudgeScore.model_validate(response.parsed).model_dump()
            input_tokens, output_tokens = _usage(response)
            cost, rate_source = calculate_cost("gemini", settings.ragas_judge_model, input_tokens, output_tokens) if input_tokens is not None else (None, None)
            result["judge"] = {
                **score,
                "rubric_version": "comparison_judge_v2",
                "prompt_security": PROMPT_SECURITY_VERSION,
                "model": settings.ragas_judge_model,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": cost,
                "rate_source": rate_source,
            }
            print(f"judged {result['case_id']} {result['variant']}", flush=True)
        except Exception as error:
            result["judge_error"] = f"{type(error).__name__}: {error}"
            print(f"judge failed {result['case_id']} {result['variant']}: {error}", flush=True)
        _write(args.output, comparison)
    selected = [
        row for row in comparison["results"]
        if (not case_ids or row.get("case_id") in case_ids)
        and (not variants or row.get("variant") in variants)
        and row.get("status") == "completed"
    ]
    judged = [row for row in selected if row.get("judge")]
    failed = [row for row in selected if row.get("judge_error") and not row.get("judge")]
    costs = [row["judge"].get("estimated_cost_usd") for row in judged]
    comparison["judge_summary"] = {
        "judged": len(judged),
        "failed": len(failed),
        "estimated_cost_usd": round(sum(costs), 8) if costs and all(value is not None for value in costs) else None,
        "cost_complete": bool(costs) and all(value is not None for value in costs),
        "preflight": preflight,
    }
    _write(args.output, comparison)
    print(json.dumps(comparison["judge_summary"], indent=2))
    return 0 if not failed and len(judged) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
