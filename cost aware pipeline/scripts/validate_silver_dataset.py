"""Verify Silver benchmark sources and semantically audit every atomic claim."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.silver_dataset_validation import (  # noqa: E402
    CaseJudgment,
    SourceDocument,
    build_case_prompt,
    normalize_judgments,
    normalize_scientific_typography,
    rebuild_case,
    source_audit_record,
    validate_rebuilt_cases,
)
from scientific_rag.costing import calculate_cost, estimate_text_tokens  # noqa: E402
from scientific_rag.llm_gateway import (  # noqa: E402
    ainvoke_structured,
    begin_trace,
    end_trace,
    summarize_trace,
)


PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_URL = "https://export.arxiv.org/api/query"
CACHE_PATH = ROOT / "data" / "silver-source-cache.json"
JUDGE_PROVIDER = "gemini"
JUDGE_MODEL = "gemini-2.5-flash"


def _request_xml(url: str, params: dict[str, str], *, attempts: int = 3) -> bytes:
    target = f"{url}?{urlencode(params)}"
    headers = {"User-Agent": "scientific-rag-silver-audit/2.0 (benchmark source verification)"}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(target, headers=headers), timeout=45) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                import time
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def fetch_pubmed(source_ids: list[str]) -> dict[str, SourceDocument]:
    pmids = [source_id.removeprefix("PMID:") for source_id in source_ids]
    if not pmids:
        return {}
    payload = _request_xml(PUBMED_URL, {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    root = ET.fromstring(payload)
    results: dict[str, SourceDocument] = {}
    for item in root.findall(".//PubmedArticle"):
        pmid = _text(item.find(".//MedlineCitation/PMID"))
        article = item.find(".//Article")
        if not pmid or article is None:
            continue
        title = _text(article.find("ArticleTitle"))
        parts = []
        for abstract in article.findall(".//Abstract/AbstractText"):
            label = abstract.attrib.get("Label")
            value = _text(abstract)
            parts.append(f"{label}: {value}" if label and value else value)
        source_id = f"PMID:{pmid}"
        results[source_id] = SourceDocument(
            source_id=source_id,
            title=title,
            abstract=" ".join(part for part in parts if part),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        )
    return results


def fetch_arxiv(source_ids: list[str]) -> dict[str, SourceDocument]:
    raw_ids = [source_id.removeprefix("arxiv:") for source_id in source_ids]
    if not raw_ids:
        return {}
    payload = _request_xml(ARXIV_URL, {"id_list": ",".join(raw_ids), "max_results": str(len(raw_ids))})
    root = ET.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results: dict[str, SourceDocument] = {}
    for entry in root.findall("atom:entry", namespace):
        entry_url = _text(entry.find("atom:id", namespace))
        raw_id = entry_url.rstrip("/").split("/")[-1]
        source_id = f"arxiv:{raw_id}"
        results[source_id] = SourceDocument(
            source_id=source_id,
            title=" ".join(_text(entry.find("atom:title", namespace)).split()),
            abstract=" ".join(_text(entry.find("atom:summary", namespace)).split()),
            url=entry_url,
        )
    return results


def _load_source_cache(path: Path) -> dict[str, SourceDocument]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {source_id: SourceDocument(**item) for source_id, item in raw.items()}


def _save_source_cache(path: Path, documents: dict[str, SourceDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({source_id: asdict(item) for source_id, item in sorted(documents.items())}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def fetch_all_sources(payload: dict[str, Any], cache_path: Path = CACHE_PATH) -> tuple[dict[str, SourceDocument], list[str]]:
    source_ids = sorted({
        context["source_id"]
        for case in payload.get("cases", [])
        for context in case.get("reference_context", [])
    })
    documents = _load_source_cache(cache_path)
    pubmed = [item for item in source_ids if item.startswith("PMID:") and item not in documents]
    arxiv = [item for item in source_ids if item.startswith("arxiv:") and item not in documents]
    try:
        documents.update(fetch_pubmed(pubmed))
        _save_source_cache(cache_path, documents)
        documents.update(fetch_arxiv(arxiv))
    finally:
        _save_source_cache(cache_path, documents)
    documents = {source_id: documents[source_id] for source_id in source_ids if source_id in documents}
    return documents, [item for item in source_ids if item not in documents]


@contextmanager
def _gemini_judge_environment():
    previous_order = os.environ.get("LLM_PROVIDER_ORDER")
    previous_model = os.environ.get("MODEL_QUERY_REPAIR")
    os.environ["LLM_PROVIDER_ORDER"] = JUDGE_PROVIDER
    os.environ["MODEL_QUERY_REPAIR"] = JUDGE_MODEL
    try:
        yield
    finally:
        if previous_order is None:
            os.environ.pop("LLM_PROVIDER_ORDER", None)
        else:
            os.environ["LLM_PROVIDER_ORDER"] = previous_order
        if previous_model is None:
            os.environ.pop("MODEL_QUERY_REPAIR", None)
        else:
            os.environ["MODEL_QUERY_REPAIR"] = previous_model


def estimate_judge_cost(prompts: list[str]) -> dict[str, Any]:
    input_tokens = estimate_text_tokens(prompts)
    # The first live audit averaged roughly 1.55k output tokens per case even
    # with structured output. Keep a conservative margin for future preflights.
    output_tokens = len(prompts) * 1800
    amount, source = calculate_cost(JUDGE_PROVIDER, JUDGE_MODEL, input_tokens, output_tokens)
    return {
        "provider": JUDGE_PROVIDER,
        "model": JUDGE_MODEL,
        "calls": len(prompts),
        "estimated_input_tokens": input_tokens,
        "budgeted_output_tokens": output_tokens,
        "estimated_upper_bound_usd": amount,
        "rate_source": source,
    }


async def audit_claims(cases: list[dict[str, Any]], documents: dict[str, SourceDocument]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [build_case_prompt(case, documents) for case in cases]
    estimate = estimate_judge_cost(prompts)
    print(json.dumps({"preflight_cost_estimate": estimate}, indent=2))
    if estimate["estimated_upper_bound_usd"] is None:
        raise RuntimeError("No trusted price is configured for the selected judge model")
    if estimate["estimated_upper_bound_usd"] > float(os.getenv("SILVER_AUDIT_MAX_USD", "0.25")):
        raise RuntimeError("Silver audit preflight exceeds SILVER_AUDIT_MAX_USD")

    trace_token = begin_trace()
    audited: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        with _gemini_judge_environment():
            for position, (case, prompt) in enumerate(zip(cases, prompts), 1):
                print(f"Auditing case {position}/{len(cases)}: {case['id']}")
                try:
                    result = await ainvoke_structured("query_repair", 0.0, prompt, CaseJudgment)
                    judgments = normalize_judgments(case, result)
                except Exception as error:
                    errors.append({"case_id": case["id"], "error": f"{type(error).__name__}: {error}"})
                    judgments = normalize_judgments(case, CaseJudgment(judgments=[]))
                audited.append(rebuild_case(case, judgments, documents))
    finally:
        trace = end_trace(trace_token)
    return audited, {"preflight": estimate, "actual": summarize_trace(trace), "errors": errors}


def build_source_audit(payload: dict[str, Any], documents: dict[str, SourceDocument], missing: list[str]) -> dict[str, Any]:
    contexts_by_source: dict[str, list[dict[str, Any]]] = {}
    for case in payload.get("cases", []):
        for context in case.get("reference_context", []):
            contexts_by_source.setdefault(context["source_id"], []).append(context)
    records = [source_audit_record(document, contexts_by_source.get(source_id, [])) for source_id, document in sorted(documents.items())]
    checks = [check for record in records for check in record["original_excerpt_checks"]]
    return {
        "sources_expected": len(contexts_by_source),
        "sources_fetched": len(documents),
        "sources_missing": missing,
        "original_excerpts_checked": len(checks),
        "verbatim_verified": sum(check["status"] == "verbatim_verified" for check in checks),
        "near_verbatim": sum(check["status"] == "near_verbatim" for check in checks),
        "not_found": sum(check["status"] == "not_found" for check in checks),
        "records": records,
    }


def write_report(path: Path, dataset: dict[str, Any], audit: dict[str, Any]) -> None:
    cases = dataset["cases"]
    judgments = [item for case in cases for item in case.get("claim_audit", [])]
    source = audit["source_audit"]
    model_audits = [*audit.get("model_audit_history", []), audit["model_audit"]]
    repair_generation = audit.get("repair_generation") or {}
    total_preflight = sum(float(item.get("preflight", {}).get("estimated_upper_bound_usd") or 0) for item in model_audits)
    total_actual = sum(float(item.get("actual", {}).get("estimated_cost_usd") or 0) for item in model_audits)
    total_calls = sum(int(item.get("preflight", {}).get("calls") or 0) for item in model_audits)
    total_preflight += float(repair_generation.get("preflight", {}).get("estimated_usd") or 0)
    total_actual += float(repair_generation.get("actual", {}).get("estimated_cost_usd") or 0)
    total_calls += int(repair_generation.get("preflight", {}).get("calls") or 0)
    estimate_delta = total_actual - total_preflight
    lines = [
        f"# {dataset.get('dataset_name', 'Silver benchmark')} — independent source and claim audit",
        "",
        "This dataset remains **Silver**: it was automatically generated and independently audited, but not human-reviewed.",
        "",
        "## Outcome",
        "",
        f"- Cases audited: **{len(cases)}**",
        f"- Eligible repaired cases: **{sum(case['eligible_for_benchmark'] for case in cases)}**",
        f"- Rejected cases: **{sum(not case['eligible_for_benchmark'] for case in cases)}**",
        f"- Atomic claims audited: **{len(judgments)}**",
        f"- Supported: **{sum(item['decision'] == 'supported' for item in judgments)}**",
        f"- Contradicted: **{sum(item['decision'] == 'contradicted' for item in judgments)}**",
        f"- Insufficient: **{sum(item['decision'] == 'insufficient' for item in judgments)}**",
        "",
        "## Source verification",
        "",
        f"- Original sources fetched: **{source['sources_fetched']} / {source['sources_expected']}**",
        f"- Original excerpts checked: **{source['original_excerpts_checked']}**",
        f"- Verbatim matches: **{source['verbatim_verified']}**",
        f"- Near-verbatim matches: **{source['near_verbatim']}**",
        f"- Not found verbatim: **{source['not_found']}**",
        *( 
            [
                "",
                "### Targeted repair source verification",
                "",
                f"- Repair excerpts checked: **{audit['latest_source_audit']['original_excerpts_checked']}**",
                f"- Repair excerpts verified verbatim: **{audit['latest_source_audit']['verbatim_verified']}**",
                f"- Repair excerpts not found: **{audit['latest_source_audit']['not_found']}**",
            ]
            if audit.get("latest_source_audit") else []
        ),
        "",
        "## Cost and method",
        "",
        f"- Independent judge: `{audit['model_audit']['preflight']['provider']}:{audit['model_audit']['preflight']['model']}`",
        f"- Model calls across initial audit and targeted retry: **{total_calls}**",
        f"- Combined preflight estimates: **${total_preflight:.4f}**",
        f"- Recorded cumulative model cost: **${total_actual:.4f}**",
        f"- Historical estimate error: **${estimate_delta:+.4f}**; future runs reserve 1,800 output tokens per case.",
        "- Each atomic claim received a separate structured decision, although claims from one case were batched into one call to reduce cost.",
        "- Repaired answers are deterministic joins of supported claims; the audit model did not write replacement scientific claims.",
        "",
        "## Case status",
        "",
        "| Case | Status | Supported | Contradicted | Insufficient |",
        "|---|---|---:|---:|---:|",
    ]
    for case in cases:
        items = case["claim_audit"]
        lines.append(
            f"| {case['id']} | {case['benchmark_status']} | "
            f"{sum(item['decision'] == 'supported' for item in items)} | "
            f"{sum(item['decision'] == 'contradicted' for item in items)} | "
            f"{sum(item['decision'] == 'insufficient' for item in items)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    report_path: Path,
    *,
    estimate_only: bool,
    rebuild_only: bool,
    finalize_only: bool,
    case_ids: list[str],
    seed_dataset_path: Path | None,
    seed_audit_path: Path | None,
    dataset_name: str,
    dataset_version: int,
    repair_audit_path: Path | None,
) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if finalize_only:
        if not output_path.exists() or not audit_path.exists():
            raise FileNotFoundError("Finalize-only requires existing dataset and audit files")
        dataset = json.loads(output_path.read_text(encoding="utf-8"))
        for case in dataset.get("cases", []):
            case["question"] = normalize_scientific_typography(str(case.get("question", "")))
            case["reference_claims"] = [normalize_scientific_typography(item) for item in case.get("reference_claims", [])]
            case["reference_answer"] = " ".join(case["reference_claims"])
            for support in case.get("claim_support", []):
                if support.get("claim"):
                    support["claim"] = normalize_scientific_typography(support["claim"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if repair_audit_path:
            audit["repair_generation"] = json.loads(repair_audit_path.read_text(encoding="utf-8"))
        dataset["generated_at"] = datetime.now(timezone.utc).isoformat()
        audit["generated_at"] = dataset["generated_at"]
        output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_report(report_path, dataset, audit)
        print(json.dumps({"finalize_only": True, "cases": len(dataset.get("cases", []))}, indent=2))
        return
    selected_cases = [case for case in payload.get("cases", []) if not case_ids or case["id"] in set(case_ids)]
    missing_case_ids = sorted(set(case_ids) - {case["id"] for case in selected_cases})
    if missing_case_ids:
        raise ValueError(f"Unknown case ids: {', '.join(missing_case_ids)}")
    fetch_payload = payload if rebuild_only or not case_ids else {"cases": selected_cases}
    documents, missing = await asyncio.to_thread(fetch_all_sources, fetch_payload)
    source_audit = build_source_audit(fetch_payload, documents, missing)
    prompts = [build_case_prompt(case, documents) for case in selected_cases]
    estimate = estimate_judge_cost(prompts)
    print(json.dumps({"source_audit_summary": {key: value for key, value in source_audit.items() if key != "records"}, "preflight_cost_estimate": estimate}, indent=2))
    if estimate_only:
        return

    if rebuild_only:
        if not output_path.exists() or not audit_path.exists():
            raise FileNotFoundError("Rebuild-only requires existing v2 dataset and audit files")
        existing_dataset = json.loads(output_path.read_text(encoding="utf-8"))
        existing_by_id = {case["id"]: case for case in existing_dataset.get("cases", [])}
        cases = []
        for base_case in payload.get("cases", []):
            existing = existing_by_id.get(base_case["id"])
            if existing is None or not existing.get("claim_audit"):
                continue
            cases.append(rebuild_case(base_case, existing["claim_audit"], documents))
        validation_errors = validate_rebuilt_cases(cases, documents)
        if validation_errors:
            raise RuntimeError("Rebuilt dataset validation failed: " + "; ".join(validation_errors))
        dataset = dict(existing_dataset)
        dataset["cases"] = cases
        dataset["generated_at"] = datetime.now(timezone.utc).isoformat()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["source_audit"] = source_audit
        audit["generated_at"] = dataset["generated_at"]
        output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_report(report_path, dataset, audit)
        print(json.dumps({"rebuild_only": True, "cases": len(cases), "validation_errors": 0}, indent=2))
        return

    audited_cases, model_audit = await audit_claims(selected_cases, documents)
    audited_by_id = {case["id"]: case for case in audited_cases}
    existing_path = output_path if output_path.exists() else seed_dataset_path
    if case_ids and existing_path and existing_path.exists():
        existing_dataset = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_by_id = {case["id"]: case for case in existing_dataset.get("cases", [])}
        existing_by_id.update(audited_by_id)
        cases = []
        for base_case in payload.get("cases", []):
            existing = existing_by_id.get(base_case["id"])
            if existing is None:
                continue
            cases.append(existing)
    else:
        cases = audited_cases
    generated_at = datetime.now(timezone.utc).isoformat()
    dataset = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "generated_at": generated_at,
        "provenance": "silver_auto_generated_independently_audited",
        "human_review": False,
        "source_dataset": input_path.name,
        "cases": cases,
    }
    validation_errors = validate_rebuilt_cases(audited_cases, documents)
    if validation_errors:
        raise RuntimeError("Audited dataset validation failed: " + "; ".join(validation_errors))
    previous_model_audits = []
    previous_source_audit = None
    existing_audit_path = audit_path if audit_path.exists() else seed_audit_path
    if case_ids and existing_audit_path and existing_audit_path.exists():
        previous = json.loads(existing_audit_path.read_text(encoding="utf-8"))
        previous_model_audits = list(previous.get("model_audit_history", []))
        previous_source_audit = previous.get("source_audit")
        if previous.get("model_audit"):
            previous_model_audits.append(previous["model_audit"])
    audit = {
        "generated_at": generated_at,
        "source_audit": previous_source_audit or source_audit,
        "latest_source_audit": source_audit if previous_source_audit else None,
        "model_audit": model_audit,
        "model_audit_history": previous_model_audits,
    }
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(report_path, dataset, audit)
    print(json.dumps({
        "output": str(output_path),
        "eligible_cases": sum(case["eligible_for_benchmark"] for case in cases),
        "rejected_cases": sum(not case["eligible_for_benchmark"] for case in cases),
        "model_cost_usd": model_audit["actual"].get("estimated_cost_usd"),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "pilot-silver-v2.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "evaluation" / "pilot-silver-v2-audit.json")
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation" / "pilot-silver-v2-report.md")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--seed-dataset", type=Path)
    parser.add_argument("--seed-audit", type=Path)
    parser.add_argument("--dataset-name", default="pilot-silver-v2")
    parser.add_argument("--dataset-version", type=int, default=2)
    parser.add_argument("--repair-audit", type=Path)
    args = parser.parse_args()
    asyncio.run(run(
        args.input,
        args.output,
        args.audit,
        args.report,
        estimate_only=args.estimate_only,
        rebuild_only=args.rebuild_only,
        finalize_only=args.finalize_only,
        case_ids=args.case_id,
        seed_dataset_path=args.seed_dataset,
        seed_audit_path=args.seed_audit,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        repair_audit_path=args.repair_audit,
    ))


if __name__ == "__main__":
    main()
