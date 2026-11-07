"""Restore explicit claim-to-context links omitted by the original generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmark_dataset import add_explicit_claim_support, validate_pilot_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "pilot-silver-v1.json")
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    repaired = add_explicit_claim_support(payload)
    errors = validate_pilot_dataset(repaired)
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 2
    args.dataset.write_text(json.dumps(repaired, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": len(repaired["cases"]), "dataset": str(args.dataset)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
