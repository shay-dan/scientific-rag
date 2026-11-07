"""Run a reviewed dataset through the live API without evaluating it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case.get("eligible_for_benchmark", True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "standard", "deep"), default="fast")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs; empty runs every eligible case")
    parser.add_argument("--base-url", default=os.getenv("COMPARISON_BASE_URL", "http://api:8000"))
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    selected_ids = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    if selected_ids:
        cases = [case for case in cases if case["id"] in selected_ids]
        missing = selected_ids - {case["id"] for case in cases}
        if missing:
            parser.error(f"Unknown or ineligible case IDs: {', '.join(sorted(missing))}")
    research_reservation = {"fast": 0.05, "standard": 0.20, "deep": 0.50}[args.mode]
    preflight = {
        "case_count": len(cases),
        "mode": args.mode,
        "research_reservation_usd": round(len(cases) * research_reservation, 2),
        "later_judge_reservation_usd": round(len(cases) * 0.10, 2),
        "combined_conservative_reservation_usd": round(len(cases) * (research_reservation + 0.10), 2),
    }
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.estimate_only:
        return

    configured = os.environ["APP_API_KEYS"].split(",")[0]
    api_key = configured.split(":", 1)[1]
    rows: list[dict] = []
    payload = {
        "kind": "live-dataset-research",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(args.dataset),
        "preflight": preflight,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        body = json.dumps({
            "question": case["question"],
            "mode": args.mode,
            "cache_mode": "refresh_sources",
            "generate_evaluation_dataset": False,
        }).encode()
        request = Request(
            f"{args.base_url.rstrip('/')}/v1/research",
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": api_key, "Host": "localhost"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=650) as response:
                status = response.status
                result = json.load(response)
        except HTTPError as error:
            status = error.code
            raw = error.read().decode("utf-8", errors="replace")
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"detail": raw or str(error)}
        except (TimeoutError, URLError) as error:
            status = 0
            result = {"detail": f"{type(error).__name__}: {error}"}

        row = {
            "case_id": case["id"],
            "source_group": case.get("source_group"),
            "question": case["question"],
            "http_status": status,
            "duration_seconds": round(time.perf_counter() - started, 1),
            "run_id": result.get("id"),
            "status": result.get("status"),
            "error": result.get("detail"),
            "source_count": len(result.get("sources", [])),
            "fallback_count": (result.get("llm_diagnostics") or {}).get("fallback_count"),
        }
        rows.append(row)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
