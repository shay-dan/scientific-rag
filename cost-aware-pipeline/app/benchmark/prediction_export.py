"""R6B: offline baselines on label-free inputs, and the owner-side prediction export.

Every prediction here is **unscored**: the implementation workspace has no reference
labels, so no accuracy, F1, Recall@K, nDCG or acceptance rate is computed. Scoring is
done by the benchmark owner outside the repository (``owner_scoring``).

Baselines (deterministic; no model, network or database):
- open discovery: not executable offline (it needs live PubMed/arXiv search), so every
  question is recorded ``not_executed``;
- within-pack evidence retrieval: R6A's BM25 reference ranks the text segments of the
  question's own pack. This is a reference baseline, **not** the production retriever
  (PostgreSQL FTS + pgvector with RRF, MMR and parent/child retrieval);
- claim verification: R5's ``lexical_numeric`` rules applied to the claim and its
  evidence text (the handoff supplies no citation quote, so the whole evidence text
  stands in for it);
- answer tasks (yes/no/maybe decisions, trial-result inference, synthesis) need a
  model and are recorded ``not_executed``.

Export contract ``scientific-rag-benchmark-predictions/v1`` (one JSON document):
    run         run_id, code {sha, source}, dataset {version, dataset_hash,
                manifest_sha256, inputs_hash per file}, configurations {name: sha256},
                protocol {discovery_frozen_hash, fixed_evidence_loaded_after_discovery}
    predictor   name, kind, version, components {task: method}
    records     sorted by (task, input_id); each {input_id, task, stage, status,
                reason?, prediction?, usage, input_hash, record_hash}
    export_hash SHA-256 over the canonical records
Status is one of completed | not_executed | failed | invalid | missing.
"""
from __future__ import annotations

from ..evaluation_report import canonical_json, sha256_hex
from ..verifier_evaluation import lexical_numeric_baseline
from .protocol import EvaluationSession
from .retrieval import lexical_baseline_run

EXPORT_SCHEMA = "scientific-rag-benchmark-predictions/v1"
TASKS = {
    "open_discovery": "discovery",
    "fixed_evidence_retrieval": "fixed_evidence",
    "claim_verification": "fixed_evidence",
    "fixed_evidence_answer": "fixed_evidence",
}
STATUSES = ("completed", "not_executed", "failed", "invalid", "missing")
NO_USAGE = {"cost_status": "no_calls", "provider_calls": 0}
LIVE_SEARCH_REASON = "open discovery requires live PubMed/arXiv search, which is not authorized"
MODEL_REASON = "requires a language model, which is not authorized"
TEXT_REASON = "evidence text is not redistributed; fetching it is an external download, which is not authorized"
ANSWER_ITEM_TYPES = {"abstract_question_answering_yes_no_maybe", "trial_result_inference_intervention_comparator_outcome"}


def _record(input_item: dict, task: str, status: str, *, prediction=None, reason=None) -> dict:
    if task not in TASKS or status not in STATUSES:
        raise ValueError(f"invalid task/status {task!r}/{status!r}")
    record = {"input_id": input_item["input_id"], "task": task, "stage": TASKS[task], "status": status,
              "input_hash": input_item["input_hash"], "usage": dict(NO_USAGE)}
    if prediction is not None:
        record["prediction"] = prediction
    if reason is not None:
        record["reason"] = reason
    record["record_hash"] = sha256_hex(record)
    return record


def within_pack_retrieval(item: dict, *, top_k: int = 10) -> dict:
    """Rank the text segments of one question's own pack with R6A's BM25 reference."""
    texts = [segment for segment in item.get("segments", []) if segment["availability"] == "text"]
    if not texts:
        return _record(item, "fixed_evidence_retrieval", "not_executed", reason=TEXT_REASON)
    pack = {"pack_id": f"inline:{item['input_id']}", "content_hash": item["input_hash"],
            "questions": [{"question_id": item["input_id"], "text": item.get("question") or item.get("claim", "")}],
            "evidence": [{"evidence_id": s["evidence_id"], "paper_id": s.get("paper_id") or f"{item['input_id']}#paper",
                          "text": s["text"]} for s in texts], "papers": []}
    ranked = lexical_baseline_run(pack, top_k=top_k)["results"][0]["ranked"]
    order = [entry["evidence_id"] for entry in ranked]
    order += [s["evidence_id"] for s in sorted(texts, key=lambda s: s["evidence_index"]) if s["evidence_id"] not in order]
    partial = len(texts) < len(item.get("segments", []))
    return _record(item, "fixed_evidence_retrieval", "completed",
                   prediction={"ranked_evidence_ids": order[:top_k], "segments_ranked": len(texts),
                               "segments_without_text": len(item.get("segments", [])) - len(texts)},
                   reason="some segments were not redistributed and could not be ranked" if partial else None)


def claim_verification(items: list[dict]) -> list[dict]:
    """R5 lexical_numeric rules on (claim, evidence text). Only input fields are read."""
    usable, records = [], []
    for item in items:
        claim = item.get("claim") or item.get("question")
        texts = [s["text"] for s in item.get("segments", []) if s["availability"] == "text"]
        if not texts:
            records.append(_record(item, "claim_verification", "not_executed", reason=TEXT_REASON))
            continue
        evidence = "\n\n".join(texts)
        usable.append((item, {"item_id": item["input_id"], "claim_text": claim, "evidence_text": evidence,
                              "evidence_quote": evidence}))
    if usable:
        output = lexical_numeric_baseline({"items": [fields for _, fields in usable]})["predictions"]
        verdicts = {row["item_id"]: row for row in output}
        for item, _ in usable:
            row = verdicts[item["input_id"]]
            records.append(_record(item, "claim_verification", "completed",
                                   prediction={"verdict": row["verdict"], "rationale": row["rationale"]}))
    return records


def offline_baseline_records(session: EvaluationSession) -> tuple[list[dict], dict]:
    """Run the enforced protocol with offline baselines; return records and protocol facts."""
    discovery_hash = session.skip_discovery(LIVE_SEARCH_REASON)
    fixed = session.load_fixed_evidence()
    discovery_records, verified_hash = session.frozen_discovery()
    by_id = {item.input_id: item for item in session.discovery_inputs}
    records = []
    for frozen in discovery_records:
        stub = {"input_id": frozen["input_id"], "input_hash": sha256_hex(
            {"input_id": frozen["input_id"], "question": by_id[frozen["input_id"]].question})}
        records.append(_record(stub, "open_discovery", frozen["status"], reason=frozen.get("reason"),
                               prediction=frozen.get("prediction")))
    claims = []
    for item in fixed:
        if item["kind"] == "claim_verification_input" or item.get("item_type") == "claim_verification":
            claims.append(item)
        if item["kind"] == "fixed_evidence_question":
            records.append(within_pack_retrieval(item))
            if item.get("item_type") != "claim_verification":
                records.append(_record(item, "fixed_evidence_answer", "not_executed", reason=MODEL_REASON))
    records += claim_verification(claims)
    session.mark_exported()
    records.sort(key=lambda record: (record["task"], record["input_id"]))
    return records, {"discovery_frozen_hash": verified_hash, "discovery_hash_at_freeze": discovery_hash,
                     "fixed_evidence_loaded_after_discovery": True}


def build_export(*, records: list[dict], protocol: dict, import_report: dict, run_id: str, git: dict,
                 predictor: dict) -> dict:
    handoff = import_report["handoff"]
    export = {
        "schema": EXPORT_SCHEMA,
        "state": "unscored",
        "run": {
            "run_id": run_id,
            "code": {"sha": git.get("sha"), "source": git.get("source")},
            "dataset": {"dataset_version": handoff["dataset_version"], "dataset_hash": handoff["dataset_hash"],
                        "manifest_sha256": handoff["manifest_sha256"],
                        "inputs_hash": {path: f["inputs_hash"] for path, f in import_report["files"].items()}},
            "configurations": {meta["name"]: meta["sha256"] for meta in handoff["configs"].values()},
            "protocol": protocol,
        },
        "predictor": predictor,
        "counts": _counts(records),
        "records": records,
        "statement": "Unscored predictions. The implementation workspace has no reference labels; scores are "
                     "computed by the benchmark owner, per scoring population, outside the repository.",
    }
    export["export_hash"] = export_hash(export)
    return export


def export_hash(export: dict) -> str:
    return sha256_hex(sorted(canonical_json(record) for record in export["records"]))


def verify_export(export: dict) -> None:
    """Contract and integrity check used before an export is shared or scored."""
    if export.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"export schema must be {EXPORT_SCHEMA}")
    keys = set()
    for record in export.get("records", []):
        if record.get("task") not in TASKS or record.get("status") not in STATUSES:
            raise ValueError(f"invalid task/status in record {record.get('input_id')!r}")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        if sha256_hex(body) != record.get("record_hash"):
            raise ValueError(f"record hash mismatch for {record.get('input_id')!r}")
        key = (record["task"], record["input_id"])
        if key in keys:
            raise ValueError(f"duplicate record {key}")
        keys.add(key)
    if export_hash(export) != export.get("export_hash"):
        raise ValueError("export hash mismatch")


def _counts(records: list[dict]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        task = counts.setdefault(record["task"], {status: 0 for status in STATUSES})
        task[record["status"]] += 1
    return dict(sorted(counts.items()))
