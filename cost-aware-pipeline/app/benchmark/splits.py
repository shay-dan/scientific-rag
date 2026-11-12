"""R6A: split integrity and leakage checks.

Checks, each reported as an ``Issue``:
- every question is in exactly one known split;
- question-level leakage: the same question ID, or near-duplicate question text,
  appears in more than one split;
- paper-level leakage: a paper (or a paper family) is labeled or cited for
  questions in more than one split;
- near-duplicate claim leakage across splits;
- a split's records (claims, relevance, evidence associations) only reference
  questions of that split.

Near-duplicates use word-shingle Jaccard similarity on case- and
punctuation-folded text (default threshold 0.8, 3-word shingles; texts shorter
than the shingle size use word sets). Exact normalized matches always count.
"""
from __future__ import annotations

import re
from itertools import combinations

from .contracts import SPLITS, Benchmark, Issue

NEAR_DUPLICATE_THRESHOLD = 0.8
SHINGLE = 3
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(str(text or "").casefold()))


def shingles(text: str, size: int = SHINGLE) -> frozenset:
    words = normalize(text).split()
    if len(words) < size:
        return frozenset(words)
    return frozenset(tuple(words[index:index + size]) for index in range(len(words) - size + 1))


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _near_duplicates(items: list[tuple[str, str, str]], threshold: float) -> list[tuple[str, str, float]]:
    """items: (id, split, text). Returns cross-split pairs at or above the threshold."""
    prepared = [(item_id, split, normalize(text), shingles(text)) for item_id, split, text in items]
    pairs = []
    for (left_id, left_split, left_norm, left_sh), (right_id, right_split, right_norm, right_sh) in combinations(prepared, 2):
        if left_split == right_split:
            continue
        score = 1.0 if left_norm == right_norm else jaccard(left_sh, right_sh)
        if score >= threshold:
            pairs.append((min(left_id, right_id), max(left_id, right_id), round(score, 4)))
    return sorted(pairs)


def check_splits(benchmark: Benchmark, *, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> list[Issue]:
    issues: list[Issue] = []
    questions = [q for q in benchmark.of("question") if isinstance(q, dict) and q.get("question_id")]
    split_of: dict[str, set[str]] = {}
    for question in questions:
        split_of.setdefault(str(question["question_id"]), set()).add(str(question.get("split")))
    for qid, splits in sorted(split_of.items()):
        if len(splits) > 1:
            issues.append(Issue("error", "split_leakage_question", "question", qid,
                                f"question appears in splits {sorted(splits)}"))
    single = {qid: next(iter(splits)) for qid, splits in split_of.items() if len(splits) == 1}

    for left, right, score in _near_duplicates(
            [(str(q["question_id"]), str(q.get("split")), str(q.get("text") or "")) for q in questions], threshold):
        issues.append(Issue("error", "near_duplicate_leakage", "question", f"{left}~{right}",
                            f"near-duplicate questions across splits (similarity {score})"))

    papers = {str(p["paper_id"]): p for p in benchmark.of("paper") if isinstance(p, dict) and p.get("paper_id")}
    family = {pid: str(paper.get("family_id") or pid) for pid, paper in papers.items()}
    uses: dict[str, set[str]] = {}

    def use(paper_id, question_id) -> None:
        split = single.get(str(question_id))
        if split:
            uses.setdefault(family.get(str(paper_id), str(paper_id)), set()).add(split)

    evidence = {str(e["evidence_id"]): e for e in benchmark.of("evidence") if isinstance(e, dict) and e.get("evidence_id")}
    for relevance in benchmark.of("relevance"):
        if isinstance(relevance, dict):
            use(relevance.get("paper_id"), relevance.get("question_id"))
    for item in evidence.values():
        for qid in item.get("question_ids") or []:
            use(item.get("paper_id"), qid)
    for claim in benchmark.of("claim"):
        if not isinstance(claim, dict):
            continue
        for eid in claim.get("evidence_ids") or []:
            if str(eid) in evidence:
                use(evidence[str(eid)].get("paper_id"), claim.get("question_id"))
    for paper_family, splits in sorted(uses.items()):
        if len(splits) > 1:
            issues.append(Issue("error", "split_leakage_paper", "paper", paper_family,
                                f"paper family used by questions in splits {sorted(splits)}"))

    claims = [c for c in benchmark.of("claim") if isinstance(c, dict) and c.get("claim_id")]
    for left, right, score in _near_duplicates(
            [(str(c["claim_id"]), single.get(str(c.get("question_id")), "?"), str(c.get("claim_text") or ""))
             for c in claims if single.get(str(c.get("question_id")))], threshold):
        issues.append(Issue("error", "near_duplicate_leakage", "claim", f"{left}~{right}",
                            f"near-duplicate claims across splits (similarity {score})"))
    return sorted(set(issues))


def split_summary(benchmark: Benchmark) -> dict:
    counts = {split: 0 for split in SPLITS}
    for question in benchmark.of("question"):
        if isinstance(question, dict) and question.get("split") in counts:
            counts[question["split"]] += 1
    return {"questions_by_split": counts}


def filter_split(benchmark: Benchmark, split: str) -> Benchmark:
    """Records belonging to one split; papers/evidence kept only if referenced by it."""
    qids = {str(q["question_id"]) for q in benchmark.of("question") if isinstance(q, dict) and q.get("split") == split}
    claims = [c for c in benchmark.of("claim") if isinstance(c, dict) and str(c.get("question_id")) in qids]
    relevance = [r for r in benchmark.of("relevance") if isinstance(r, dict) and str(r.get("question_id")) in qids]
    evidence_ids = {str(eid) for c in claims for eid in c.get("evidence_ids") or []}
    evidence_ids |= {str(r["evidence_id"]) for r in relevance if r.get("evidence_id")}
    evidence = [e for e in benchmark.of("evidence") if isinstance(e, dict) and (
        str(e.get("evidence_id")) in evidence_ids or qids & {str(q) for q in e.get("question_ids") or []})]
    paper_ids = {str(e.get("paper_id")) for e in evidence} | {str(r.get("paper_id")) for r in relevance}
    return Benchmark(manifest=benchmark.manifest, records={
        "question": [q for q in benchmark.of("question") if isinstance(q, dict) and str(q.get("question_id")) in qids],
        "paper": [p for p in benchmark.of("paper") if isinstance(p, dict) and str(p.get("paper_id")) in paper_ids],
        "evidence": evidence, "claim": claims, "relevance": relevance,
        "label_mapping": list(benchmark.of("label_mapping")),
    })
