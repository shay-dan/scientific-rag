"""R6A: discovery-evaluation interface (contract + evaluator; no R9-D changes).

``scientific-rag-discovery-run/v1`` records, per question, five distinct stages:

    discovered          candidate papers found by search (provider, query)
    ranked              candidate order used to choose what to screen
    screened            relevance-screening decisions: accepted | rejected | failed
    accepted            final accepted sources
    retrieved_evidence  evidence segments retrieved downstream from accepted sources

The evaluator reports each stage separately against graded paper relevance labels
(grade >= 1 is relevant) and counts where relevant papers were lost. Discovery
quality is not retrieval quality, evidence sufficiency or claim support.

``discovery_result_from_research_metadata`` adapts the traces the current pipeline
already records (PubMed/arXiv inspectors). Stages a trace does not record are marked
``not_observed`` rather than guessed.
"""
from __future__ import annotations

from ..evaluation_report import wilson_interval
from .contracts import Benchmark

RUN_SCHEMA = "scientific-rag-discovery-run/v1"
REPORT_SCHEMA = "scientific-rag-discovery-report/v1"
STAGES = ("discovered", "ranked", "screened", "accepted", "retrieved_evidence")
SCREEN_DECISIONS = ("accepted", "rejected", "failed")


def _rate(numerator: int, denominator: int, definition: str) -> dict:
    interval = wilson_interval(numerator, denominator)
    return {"numerator": numerator, "denominator": denominator,
            "rate": round(numerator / denominator, 4) if denominator else None,
            "ci95_wilson": list(interval) if interval else None, "definition": definition}


def discovery_result_from_research_metadata(question_id: str, model_metadata: dict) -> dict:
    """Map stored PubMed/arXiv inspector traces to one discovery-run result."""
    discovered, screened = [], []
    for source, key, prefix in (("pubmed", "pubmed_inspector", "pmid:"), ("arxiv", "arxiv_inspector", "arxiv:")):
        for trace in (model_metadata or {}).get(key) or []:
            for candidate in trace.get("candidates") or []:
                identifier = candidate.get("pmid") if source == "pubmed" else candidate.get("arxiv_id")
                if not identifier:
                    continue
                paper_id = f"{prefix}{identifier}"
                discovered.append({"paper_id": paper_id, "provider": source, "rank": candidate.get("rank"),
                                   "query": trace.get("query")})
                decision = str(candidate.get("decision") or "")
                if decision.startswith("retained_"):
                    screened.append({"paper_id": paper_id, "decision": "accepted", "reason": decision})
                elif decision.startswith("rejected"):
                    screened.append({"paper_id": paper_id, "decision": "rejected", "reason": decision})
                elif decision.startswith("screening_failed"):
                    screened.append({"paper_id": paper_id, "decision": "failed", "reason": decision})
    ranked = [item["paper_id"] for item in sorted(
        discovered, key=lambda item: (item["rank"] is None, item["rank"] if item["rank"] is not None else 0))]
    return {"question_id": question_id, "status": "ok", "stages": {
        "discovered": discovered, "ranked": list(dict.fromkeys(ranked)), "screened": screened,
        "accepted": [item["paper_id"] for item in screened if item["decision"] == "accepted"],
        "retrieved_evidence": "not_observed",
    }}


def evaluate_discovery(benchmark: Benchmark, run: dict, *, split: str, rank_cutoff: int = 20) -> dict:
    """Stage-by-stage discovery report for one split (micro-averaged over relevant papers).

    Denominators are relevant (question, paper) pairs from evaluable questions, except
    screening precision (accepted papers) and evidence recall (relevant passages).
    A missing or failed result counts as nothing discovered.
    """
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"discovery run schema must be {RUN_SCHEMA}")
    questions = sorted(str(q["question_id"]) for q in benchmark.of("question") if q.get("split") == split)
    relevant = {qid: set() for qid in questions}
    labeled = {qid: set() for qid in questions}
    passages = {qid: set() for qid in questions}
    for record in benchmark.of("relevance"):
        qid = str(record.get("question_id"))
        if qid not in relevant:
            continue
        if record.get("evidence_id"):
            if int(record["grade"]) >= 1:
                passages[qid].add(str(record["evidence_id"]))
            continue
        labeled[qid].add(str(record["paper_id"]))
        if int(record["grade"]) >= 1:
            relevant[qid].add(str(record["paper_id"]))
    results: dict[str, list[dict]] = {}
    for result in run.get("results") or []:
        results.setdefault(str((result or {}).get("question_id")), []).append(result)

    totals = {key: 0 for key in ("relevant", "discovered", "ranked_in_cutoff", "screened", "accepted",
                                 "lost_not_discovered", "lost_below_cutoff_or_unscreened", "lost_rejected",
                                 "lost_screening_failed", "accepted_papers", "accepted_relevant",
                                 "accepted_unlabeled", "passages_relevant", "passages_retrieved")}
    status_counts = {"ok": 0, "missing": 0, "failed": 0, "duplicate": 0}
    evidence_observed = 0
    for qid in questions:
        entries = results.get(qid, [])
        status = "missing" if not entries else "duplicate" if len(entries) > 1 else \
            ("ok" if entries[0].get("status", "ok") == "ok" else "failed")
        status_counts[status] += 1
        stages = entries[0].get("stages", {}) if status == "ok" else {}
        discovered = {str(item.get("paper_id") if isinstance(item, dict) else item) for item in stages.get("discovered") or []}
        ranked = [str(pid) for pid in stages.get("ranked") or []][:rank_cutoff]
        decisions = {str(item.get("paper_id")): item.get("decision") for item in stages.get("screened") or []
                     if isinstance(item, dict)}
        accepted = {str(pid) for pid in stages.get("accepted") or []}
        evidence = stages.get("retrieved_evidence")
        for pid in relevant[qid]:
            totals["relevant"] += 1
            if pid not in discovered:
                totals["lost_not_discovered"] += 1
                continue
            totals["discovered"] += 1
            totals["ranked_in_cutoff"] += pid in ranked
            if pid in decisions:
                totals["screened"] += 1
            if pid in accepted:
                totals["accepted"] += 1
            elif decisions.get(pid) == "rejected":
                totals["lost_rejected"] += 1
            elif decisions.get(pid) == "failed":
                totals["lost_screening_failed"] += 1
            else:
                totals["lost_below_cutoff_or_unscreened"] += 1
        totals["accepted_papers"] += len(accepted)
        totals["accepted_relevant"] += len(accepted & relevant[qid])
        totals["accepted_unlabeled"] += len(accepted - labeled[qid])
        if isinstance(evidence, list):
            evidence_observed += 1
            totals["passages_relevant"] += len(passages[qid])
            totals["passages_retrieved"] += len(passages[qid] & {str(eid) for eid in evidence})
    return {
        "schema": REPORT_SCHEMA, "split": split, "rank_cutoff": rank_cutoff,
        "provenance": {"benchmark_id": benchmark.manifest.get("benchmark_id"),
                       "benchmark_version": benchmark.manifest.get("version"),
                       "run_id": run.get("run_id"), "system": run.get("system")},
        "population": {"questions": len(questions), "result_status": status_counts,
                       "evaluable_questions": sum(bool(relevant[q]) for q in questions)},
        "stages": {
            "discovery_recall": _rate(totals["discovered"], totals["relevant"],
                                      "relevant papers discovered / relevant papers"),
            "ranking_recall_at_cutoff": _rate(totals["ranked_in_cutoff"], totals["relevant"],
                                              f"relevant papers in the top {rank_cutoff} ranked / relevant papers"),
            "screening_recall": _rate(totals["accepted"], totals["screened"],
                                      "relevant screened papers accepted / relevant screened papers"),
            "screening_precision": _rate(totals["accepted_relevant"], totals["accepted_papers"],
                                         "accepted papers labeled relevant / accepted papers"),
            "final_source_recall": _rate(totals["accepted"], totals["relevant"],
                                         "relevant papers accepted / relevant papers"),
            "evidence_recall": _rate(totals["passages_retrieved"], totals["passages_relevant"],
                                     "relevant passages retrieved / relevant passages (questions with observed evidence)"),
        },
        "losses": {key: totals[key] for key in ("lost_not_discovered", "lost_below_cutoff_or_unscreened",
                                                 "lost_rejected", "lost_screening_failed")},
        "accepted_unlabeled_papers": totals["accepted_unlabeled"],
        "questions_with_observed_evidence": evidence_observed,
        "warnings": [warning for warning in (
            "Accepted papers without a relevance label are not counted as irrelevant; label them before "
            "drawing conclusions about screening precision." if totals["accepted_unlabeled"] else None,
            "Discovery stages are reported separately; they say nothing about evidence sufficiency or claim support.",
        ) if warning],
    }
