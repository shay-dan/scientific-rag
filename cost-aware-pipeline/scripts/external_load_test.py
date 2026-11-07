"""Host-side HTTP load probe; runs outside the application containers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=25)
    args = parser.parse_args()

    def request_once(_: int) -> tuple[bool, float, str | None]:
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(f"{args.base_url}/healthz", timeout=10) as response:
                if response.status != 200:
                    return False, (time.perf_counter() - started) * 1000, f"HTTP {response.status}"
            return True, (time.perf_counter() - started) * 1000, None
        except Exception as error:
            return False, (time.perf_counter() - started) * 1000, type(error).__name__

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = [future.result() for future in as_completed([pool.submit(request_once, item) for item in range(args.requests)])]
    latencies = [row[1] for row in rows]
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successes": sum(row[0] for row in rows),
        "failures": sum(not row[0] for row in rows),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "mean_ms": round(sum(latencies) / len(latencies), 2),
        "errors": sorted({row[2] for row in rows if row[2]}),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
