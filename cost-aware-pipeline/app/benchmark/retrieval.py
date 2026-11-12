"""R6A: frozen evidence packs and retrieval-only evaluation.

An evidence pack is a fixed corpus (papers + evidence segments + question
associations) built from a validated benchmark split. Retrieval is evaluated
against it from a ranked run (``scientific-rag-retrieval-run/v1``), produced either
by the built-in deterministic lexical baseline or imported from another system.

Retrieval metrics measure whether relevant papers/passages were ranked; they do
NOT show that any passage supports a claim (that is claim verification, R5) or that
the evidence suffices to answer the question.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..evaluation_report import bootstrap_mean_interval, canonical_json, sha256_hex, wilson_interval
from .contracts import Benchmark, records_hash

PACK_SCHEMA = "scientific-rag-evidence-pack/v1"
RUN_SCHEMA = "scientific-rag-retrieval-run/v1"
REPORT_SCHEMA = "scientific-rag-retrieval-report/v1"
DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_SEED = 20261112
_TOKEN = re.compile(r"[a-z0-9]+")


# --- evidence packs -----------------------------------------------------------------------------
def build_evidence_pack(benchmark: Benchmark, *, split: str, pack_id: str | None = None) -> dict:
    """Freeze one split's retrieval corpus into a portable pack.

    The pack holds every evidence segment except those tied to another split's
    questions (by association, claim or relevance label): unassociated segments stay
    in as distractors, so retrieval is not artificially easy. Papers are those of the
    included segments plus papers labeled for this split's questions.
    """
    questions_in = {str(q["question_id"]) for q in benchmark.of("question") if q.get("split") == split}
    other = {str(q["question_id"]) for q in benchmark.of("question") if q.get("split") != split}
    tied_elsewhere: set[str] = set()
    for record in benchmark.of("evidence"):
        if other & {str(q) for q in record.get("question_ids") or []}:
            tied_elsewhere.add(str(record["evidence_id"]))
    for claim in benchmark.of("claim"):
        if str(claim.get("question_id")) in other:
            tied_elsewhere |= {str(eid) for eid in claim.get("evidence_ids") or []}
    for relevance in benchmark.of("relevance"):
        if str(relevance.get("question_id")) in other and relevance.get("evidence_id"):
            tied_elsewhere.add(str(relevance["evidence_id"]))
    evidence = sorted((dict(e) for e in benchmark.of("evidence") if str(e["evidence_id"]) not in tied_elsewhere),
                      key=lambda e: str(e["evidence_id"]))
    paper_ids = {str(e["paper_id"]) for e in evidence} | {
        str(r["paper_id"]) for r in benchmark.of("relevance") if str(r.get("question_id")) in questions_in}
    papers = sorted((dict(p) for p in benchmark.of("paper") if str(p["paper_id"]) in paper_ids),
                    key=lambda p: str(p["paper_id"]))
    questions = sorted(({"question_id": str(q["question_id"]), "text": q["text"]}
                        for q in benchmark.of("question") if str(q["question_id"]) in questions_in),
                       key=lambda q: q["question_id"])
    manifest = benchmark.manifest
    pack = {
        "schema": PACK_SCHEMA,
        "pack_id": pack_id or f"{manifest.get('benchmark_id')}:{manifest.get('version')}:{split}",
        "benchmark_id": manifest.get("benchmark_id"), "benchmark_version": manifest.get("version"),
        "split": split, "papers": papers, "evidence": evidence, "questions": questions,
    }
    pack["content_hash"] = sha256_hex({"papers": records_hash(papers), "evidence": records_hash(evidence),
                                       "questions": records_hash(questions)})
    return pack


def pack_paper_of(pack: dict) -> dict[str, str]:
    return {str(e["evidence_id"]): str(e["paper_id"]) for e in pack["evidence"]}


# --- baseline retriever -------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text or "").casefold())


def lexical_baseline_run(pack: dict, *, top_k: int = 10, k1: float = 1.2, b: float = 0.75) -> dict:
    """Deterministic BM25 over the pack's evidence segments (ties broken by evidence_id).

    A controlled reference for the harness, not the production retriever.
    """
    docs = {str(e["evidence_id"]): _tokens(e["text"]) for e in pack["evidence"]}
    lengths = {eid: len(tokens) for eid, tokens in docs.items()}
    average = sum(lengths.values()) / len(lengths) if lengths else 0.0
    frequency = Counter(term for tokens in docs.values() for term in set(tokens))
    total = len(docs)
    results = []
    for question in pack["questions"]:
        terms = _tokens(question["text"])
        scored = []
        for eid, tokens in docs.items():
            counts = Counter(tokens)
            score = 0.0
            for term in set(terms):
                if term not in counts:
                    continue
                idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
                tf = counts[term]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[eid] / (average or 1)))
            if score > 0:
                scored.append((-round(score, 10), eid))
        ranked = [{"evidence_id": eid, "score": -negative} for negative, eid in sorted(scored)[:top_k]]
        results.append({"question_id": question["question_id"], "status": "ok", "ranked": ranked})
    return {"schema": RUN_SCHEMA, "run_id": f"lexical-bm25:{pack['pack_id']}",
            "system": {"name": "lexical_bm25", "version": "v1", "kind": "baseline",
                       "config": {"k1": k1, "b": b, "top_k": top_k}},
            "evidence_pack": {"pack_id": pack["pack_id"], "content_hash": pack["content_hash"]},
            "results": results}


# --- evaluation ---------------------------------------------------------------------------------
def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _dcg(gains: list[int]) -> float:
    return sum((2 ** gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))


def _question_metrics(paper_ranking: list[str], passage_ranking: list[str], paper_grades: dict[str, int],
                      passage_relevant: set[str], ks: tuple[int, ...]) -> dict:
    relevant_papers = {pid for pid, grade in paper_grades.items() if grade >= 1}
    ideal = sorted((grade for grade in paper_grades.values() if grade >= 1), reverse=True)
    metrics = {}
    for k in ks:
        top_papers = paper_ranking[:k]
        hits = sum(pid in relevant_papers for pid in top_papers)
        ideal_dcg = _dcg(ideal[:k])
        metrics[k] = {
            "paper_recall": hits / len(relevant_papers) if relevant_papers else None,
            "paper_precision": hits / k,
            "paper_success": hits > 0 if relevant_papers else None,
            "paper_ndcg": (_dcg([paper_grades.get(pid, 0) for pid in top_papers]) / ideal_dcg) if ideal_dcg else None,
        }
        if passage_relevant:
            passage_hits = sum(eid in passage_relevant for eid in passage_ranking[:k])
            metrics[k]["passage_recall"] = passage_hits / len(passage_relevant)
            metrics[k]["passage_precision"] = passage_hits / k
        else:
            metrics[k]["passage_recall"] = metrics[k]["passage_precision"] = None
    return metrics


def evaluate_retrieval(benchmark: Benchmark, pack: dict, run: dict, *, ks: tuple[int, ...] = DEFAULT_KS,
                       seed: int = DEFAULT_SEED) -> dict:
    """Retrieval report for the questions of the pack's split.

    Definitions (per question, then macro-averaged over *evaluable* questions —
    those with at least one relevance label of grade >= 1):
      paper ranking         = distinct papers in order of first appearance among
                              the ranked evidence segments;
      Recall@K (paper)      = relevant papers (grade >= 1) in the top K / relevant papers;
      Precision@K (paper)   = relevant papers in the top K / K (fewer than K retrieved
                              still divides by K);
      Success@K             = at least one relevant paper in the top K (a rate, Wilson CI);
      nDCG@K                = DCG of graded gains (2^grade - 1, log2 discount) over the
                              top K papers / ideal DCG from the labeled grades;
      passage Recall/Precision@K use evidence-level relevance labels (grade >= 1) and
                              are averaged over questions that have passage labels.
    Results that are missing, failed or duplicated are listed. Two variants:
      ``completed``: only questions with a valid ``ok`` result;
      ``all_questions``: missing/failed/duplicate results count as empty rankings.
    Empty rankings score 0. Questions without relevant labels are reported, not averaged.
    """
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"retrieval run schema must be {RUN_SCHEMA}")
    split = pack["split"]
    paper_of = pack_paper_of(pack)
    questions = sorted(str(q["question_id"]) for q in pack["questions"])
    grades: dict[str, dict[str, int]] = {qid: {} for qid in questions}
    passages: dict[str, set[str]] = {qid: set() for qid in questions}
    for relevance in benchmark.of("relevance"):
        qid = str(relevance.get("question_id"))
        if qid not in grades:
            continue
        if relevance.get("evidence_id"):
            if int(relevance["grade"]) >= 1:
                passages[qid].add(str(relevance["evidence_id"]))
        else:
            grades[qid][str(relevance["paper_id"])] = int(relevance["grade"])

    by_question: dict[str, list[dict]] = {}
    unknown_questions = 0
    for result in run.get("results") or []:
        qid = str((result or {}).get("question_id"))
        if qid in grades:
            by_question.setdefault(qid, []).append(result)
        else:
            unknown_questions += 1

    status = {}
    rankings: dict[str, tuple[list[str], list[str]]] = {}
    unknown_evidence = 0
    for qid in questions:
        results = by_question.get(qid, [])
        if not results:
            status[qid] = "missing"
        elif len(results) > 1:
            status[qid] = "duplicate"
        elif results[0].get("status", "ok") != "ok":
            status[qid] = "failed"
        else:
            status[qid] = "ok"
        ranked_ids = []
        if status[qid] == "ok":
            for entry in results[0].get("ranked") or []:
                eid = str(entry.get("evidence_id") if isinstance(entry, dict) else entry)
                if eid not in paper_of:
                    unknown_evidence += 1
                    continue
                if eid not in ranked_ids:
                    ranked_ids.append(eid)
        papers = []
        for eid in ranked_ids:
            if paper_of[eid] not in papers:
                papers.append(paper_of[eid])
        rankings[qid] = (papers, ranked_ids)

    evaluable = [qid for qid in questions if any(grade >= 1 for grade in grades[qid].values())]
    per_question = {qid: _question_metrics(*rankings[qid], grades[qid], passages[qid], ks) for qid in questions}
    variants = {"completed": [qid for qid in evaluable if status[qid] == "ok"], "all_questions": evaluable}
    aggregates = {}
    for variant, members in variants.items():
        aggregates[variant] = {"questions": len(members), "by_k": {}}
        for k in ks:
            row = {}
            for metric in ("paper_recall", "paper_precision", "paper_ndcg", "passage_recall", "passage_precision"):
                values = [per_question[qid][k][metric] for qid in members if per_question[qid][k][metric] is not None]
                interval = bootstrap_mean_interval(values, seed=int(sha256_hex([seed, variant, k, metric])[:12], 16))
                row[metric] = {"n": len(values), "mean": _round(sum(values) / len(values)) if values else None,
                               "ci95_bootstrap": list(interval) if interval else None}
            successes = sum(bool(per_question[qid][k]["paper_success"]) for qid in members)
            interval = wilson_interval(successes, len(members))
            row["paper_success"] = {"numerator": successes, "denominator": len(members),
                                    "rate": _round(successes / len(members)) if members else None,
                                    "ci95_wilson": list(interval) if interval else None}
            aggregates[variant]["by_k"][str(k)] = row

    status_counts = Counter(status.values())
    warnings = ["Retrieval metrics show whether labeled papers/passages were ranked; they do not show that "
                "evidence supports any claim or suffices to answer the question."]
    if len(evaluable) < 30:
        warnings.append("Fewer than 30 evaluable questions: intervals are wide and differences are inconclusive.")
    if run.get("evidence_pack", {}).get("content_hash") not in (None, pack["content_hash"]):
        warnings.append("The run was produced against a different evidence pack (content hash mismatch).")
    if unknown_evidence or unknown_questions:
        warnings.append("Some ranked evidence IDs or result question IDs are not in the pack; they were ignored.")
    report = {
        "schema": REPORT_SCHEMA,
        "split": split,
        "provenance": {"benchmark_id": benchmark.manifest.get("benchmark_id"),
                       "benchmark_version": benchmark.manifest.get("version"),
                       "pack_id": pack["pack_id"], "pack_hash": pack["content_hash"],
                       "run_id": run.get("run_id"), "system": run.get("system"),
                       "run_hash": sha256_hex(sorted(canonical_json(r) for r in run.get("results") or [])),
                       "seed": seed, "ks": list(ks)},
        "population": {
            "questions": len(questions), "evaluable": len(evaluable),
            "without_relevant_labels": sorted(set(questions) - set(evaluable)),
            "result_status": {key: status_counts.get(key, 0) for key in ("ok", "missing", "failed", "duplicate")},
            "unknown_result_questions": unknown_questions, "unknown_ranked_evidence": unknown_evidence,
        },
        "aggregates": aggregates,
        "per_question": {qid: {"status": status[qid], "evaluable": qid in evaluable,
                               "ranked_papers": rankings[qid][0][:max(ks)],
                               "by_k": {str(k): {name: _round(value) if isinstance(value, float) else value
                                                 for name, value in per_question[qid][k].items()} for k in ks}}
                         for qid in questions},
        "warnings": warnings,
    }
    return report


def render_retrieval_markdown(report: dict) -> str:
    provenance, population = report["provenance"], report["population"]
    lines = ["# Retrieval evaluation (frozen evidence pack)", "",
             f"- Split `{report['split']}`; pack `{provenance['pack_id']}` (hash `{provenance['pack_hash'][:16]}…`)",
             f"- Run `{provenance['run_id']}`; system `{(provenance['system'] or {}).get('name')}`",
             f"- Questions {population['questions']}, evaluable {population['evaluable']}; "
             f"results {population['result_status']}", ""]
    for variant, aggregate in report["aggregates"].items():
        lines += [f"## Variant `{variant}` ({aggregate['questions']} questions)", "",
                  "| K | Paper recall | Paper precision | nDCG | Success (n/N) | Passage recall | Passage precision |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for k, row in aggregate["by_k"].items():
            def show(metric):
                value = row[metric]["mean"]
                return "—" if value is None else f"{value:.4f} (n={row[metric]['n']})"
            success = row["paper_success"]
            lines.append(f"| {k} | {show('paper_recall')} | {show('paper_precision')} | {show('paper_ndcg')} | "
                         f"{success['numerator']}/{success['denominator']} | {show('passage_recall')} | "
                         f"{show('passage_precision')} |")
        lines.append("")
    lines += ["## Warnings", ""] + [f"- {warning}" for warning in report["warnings"]] + [""]
    return "\n".join(lines)
