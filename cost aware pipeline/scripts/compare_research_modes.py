"""Run a small, fixed Fast/Standard reliability comparison; no Ragas calls."""

from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CASES = (
    ("pubmed", "What are the clinical outcomes and toxicities of CAR-T therapy in B-cell lymphoma?"),
    ("arxiv", "What are current approaches to quantum error correction using surface codes?"),
    ("mixed", "How is machine learning used for breast cancer diagnosis in mammography, and what are the limitations?"),
)


def main() -> None:
    configured = os.environ["APP_API_KEYS"].split(",")[0]
    api_key = configured.split(":", 1)[1]
    base_url = os.getenv("COMPARISON_BASE_URL", "http://localhost:8000").rstrip("/")
    for corpus, question in CASES:
        for mode in ("fast", "standard"):
            body = json.dumps({"question": question, "mode": mode, "fresh": True, "generate_evaluation_dataset": False}).encode()
            request = Request(
                f"{base_url}/v1/research",
                data=body,
                headers={"Content-Type": "application/json", "X-API-Key": api_key},
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urlopen(request, timeout=650) as response:
                    status = response.status
                    result = json.load(response)
            except HTTPError as error:
                status = error.code
                result = json.load(error)
            except (TimeoutError, URLError) as error:
                status = 0
                result = {"detail": f"Request failed: {type(error).__name__}"}
            gateway = result.get("llm_diagnostics") or {}
            row = {
                "corpus": corpus,
                "mode": mode,
                "question": question,
                "http_status": status,
                "duration_seconds": round(time.perf_counter() - started, 1),
                "run_id": result.get("id"),
                "error": result.get("detail"),
                "sources": [
                    {"title": source.get("title"), "source": source.get("source"),
                     "external_id": source.get("external_id"), "url": source.get("source_url")}
                    for source in result.get("sources", [])
                ],
                "source_diagnostics": result.get("source_diagnostics"),
                "fallback_count": gateway.get("fallback_count"),
                "provider_attempts": gateway.get("by_provider"),
                "answer": result.get("answer"),
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
