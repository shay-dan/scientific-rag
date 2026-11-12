"""R6A: offline bridges from a benchmark split to synthesis replay and R5 verification.

- ``replay_evidence(benchmark, pack, question_id)`` gives the fixed evidence for a
  question in the shape the structured-synthesis pipeline consumes (one source ID
  per paper, S1..Sn, with title, excerpt, section and type). It reads only the frozen pack,
  so a replay never downloads literature. Running synthesis on it is a separate,
  explicitly authorized step (it calls a model).
- ``to_verifier_dataset(benchmark, split)`` converts claim records to R5's
  ``scientific-rag-verifier-dataset/v1``; the R5 harness then computes every verifier
  metric. Label provenance and reviewer status pass through unchanged, and the
  benchmark's label mappings are cited, never re-applied.
"""
from __future__ import annotations

from ..evaluation_report import sha256_hex
from ..verifier_evaluation import DATASET_SCHEMA as VERIFIER_DATASET_SCHEMA
from .contracts import TARGET_LABEL_SCHEME, Benchmark, fold

REPLAY_SCHEMA = "scientific-rag-synthesis-replay/v1"


def replay_evidence(benchmark: Benchmark, pack: dict, question_id: str) -> dict:
    """Fixed evidence for one question: its segments, ordered by evidence_id, with one
    source label (S1..Sn) per paper in order of first appearance."""
    papers = {str(p["paper_id"]): p for p in pack["papers"]}
    linked = {str(r["evidence_id"]) for r in benchmark.of("relevance")
              if str(r.get("question_id")) == question_id and r.get("evidence_id") and int(r.get("grade", 0)) >= 1}
    segments = [e for e in pack["evidence"]
                if question_id in {str(q) for q in e.get("question_ids") or []} or str(e["evidence_id"]) in linked]
    segments.sort(key=lambda e: str(e["evidence_id"]))
    blocks = []
    labels: dict[str, str] = {}
    for segment in segments:
        paper = papers.get(str(segment["paper_id"]), {})
        # One source label per paper, as production build_evidence_blocks assigns by document identity.
        label = labels.setdefault(str(segment["paper_id"]), f"S{len(labels) + 1}")
        blocks.append({
            "source_id": label, "evidence_id": segment["evidence_id"], "paper_id": segment["paper_id"],
            "citation_id": segment["citation_id"], "title": paper.get("title", ""),
            "excerpt": segment["text"], "section": segment.get("section"), "page": segment.get("page"),
            "evidence_type": segment["evidence_type"], "provider": paper.get("provider"),
            "source_verification": segment.get("source_verification"),
        })
    return {"schema": REPLAY_SCHEMA, "question_id": question_id, "pack_id": pack["pack_id"],
            "pack_hash": pack["content_hash"], "network_access": False, "blocks": blocks}


def to_verifier_dataset(benchmark: Benchmark, *, split: str) -> dict:
    """R5 verifier dataset for one split. One item per claim; answer = question.

    A claim citing several evidence segments becomes one item whose evidence text is
    the segments joined in order (``evidence_id`` lists them with '+'). Targets are
    ``claims.N`` in claim_id order within each question.
    """
    manifest = benchmark.manifest
    questions = {str(q["question_id"]) for q in benchmark.of("question") if q.get("split") == split}
    evidence = {str(e["evidence_id"]): e for e in benchmark.of("evidence")}
    mappings = {str(m["mapping_id"]): m for m in benchmark.of("label_mapping")}
    claims = sorted((c for c in benchmark.of("claim") if str(c.get("question_id")) in questions),
                    key=lambda c: (str(c["question_id"]), str(c["claim_id"])))
    counters: dict[str, int] = {}
    used_mappings: set[str] = set()
    items = []
    for claim in claims:
        qid = str(claim["question_id"])
        counters[qid] = counters.get(qid, 0) + 1
        evidence_ids = [str(eid) for eid in claim["evidence_ids"]]
        segments = [evidence[eid] for eid in evidence_ids]
        if claim.get("label_mapping_id"):
            used_mappings.add(str(claim["label_mapping_id"]))
        quote = claim.get("quote")
        if quote:
            # Cite the segment that actually contains the quote as the source.
            source = next((s for s in segments if fold(quote) in fold(s["text"])), segments[0])
        else:
            source = segments[0]
        items.append({
            "item_id": str(claim["claim_id"]), "answer_id": qid, "target": f"claims.{counters[qid]}",
            "claim_text": claim["claim_text"], "source_id": str(source["citation_id"]),
            "evidence_id": "+".join(evidence_ids), "evidence_text": "\n\n".join(s["text"] for s in segments),
            "evidence_quote": quote, "expected_verdict": claim["label"],
            "label_provenance": claim["label_provenance"], "reviewer_status": claim["reviewer_status"],
            "evidence_type": segments[0]["evidence_type"] if len({s["evidence_type"] for s in segments}) == 1 else "mixed",
            "error_category": claim.get("error_category") or "none",
            "critical": bool(claim.get("critical", False)), "essential": bool(claim.get("essential", False)),
            "rationale": claim.get("rationale"), "source_label": claim.get("original_label"),
        })
    dataset = {
        "schema": VERIFIER_DATASET_SCHEMA,
        "dataset_id": f"{manifest.get('benchmark_id')}:{split}",
        "version": str(manifest.get("version")),
        "label_scheme": TARGET_LABEL_SCHEME,
        "items": items,
    }
    if used_mappings:
        selected = [mappings[mid] for mid in sorted(used_mappings) if mid in mappings]
        dataset["source_label_scheme"] = "benchmark-mappings:" + ",".join(sorted({m["source_scheme"] for m in selected}))
        dataset["label_mapping"] = {
            "mapping_version": sha256_hex([{k: m.get(k) for k in ("mapping_id", "version", "rules", "method")}
                                           for m in selected])[:16],
            "reviewed_by": ", ".join(sorted({str(m.get("reviewed_by")) for m in selected})),
            "rules": "; ".join(f"{m['mapping_id']}@{m['version']} ({m['method']})" for m in selected),
        }
    return dataset
