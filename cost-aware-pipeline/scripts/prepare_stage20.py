"""Offline manifest and exact budget fingerprint; no live execution command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.benchmark.e2e_budget import CASES, protocol


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def load_manifest(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "cases"} \
            or value["schema"] != "scientific-rag-e2e-development/v1":
        raise ValueError("invalid development manifest")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("expected five cases")
    questions = []
    for case, case_id in zip(cases, CASES, strict=True):
        if not isinstance(case, dict) or set(case) != {"case_id", "question"} or case["case_id"] != case_id:
            raise ValueError("invalid case order or fields")
        question = case["question"]
        if not isinstance(question, str) or not 20 <= len(question) <= 1200 or question != question.strip():
            raise ValueError("invalid question")
        questions.append(" ".join(question.casefold().split()))
    if len(set(questions)) != len(questions):
        raise ValueError("duplicate questions")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    config = protocol()
    print(json.dumps({
        "manifest_sha256": canonical_sha256(manifest), "protocol_sha256": canonical_sha256(config),
        "protocol": config, "live_authorized": False, "provider_calls": 0,
        "live_readiness": "requires_verified_pinned_sdk_and_operator_authorization",
    }, indent=2))


if __name__ == "__main__":
    main()
