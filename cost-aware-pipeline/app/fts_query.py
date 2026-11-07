"""Deterministic PostgreSQL FTS query construction for scientific questions."""

from __future__ import annotations

import re

from scientific_rag.query_quality import aliases_for, extract_anchors


_GENERIC_FTS_TERMS = {
    "approach", "approaches", "associated", "benefits", "causes", "clinical",
    "contribute", "current", "does", "enhanced", "ensuring", "evidence",
    "explored", "limitations", "main", "models", "opportunities", "prevention",
    "research", "risks", "strategies", "strategy", "support", "traditional",
    "used", "using", "utilized", "what", "while", "with", "reduced", "being",
    "their", "their", "into", "from", "that", "this", "these", "those", "how",
}


def _websearch_term(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().lower()).replace('"', "")
    if not cleaned:
        return ""
    # websearch_to_tsquery treats a bare hyphen as NOT. Quoting both phrases
    # and hyphenated entities preserves them as positive lexical concepts.
    return f'"{cleaned}"' if " " in cleaned or "-" in cleaned else cleaned


def build_postgres_fts_query(question: str) -> tuple[str, dict]:
    """Build a broad, rankable web-search query instead of ANDing a full question.

    Lexical retrieval is a recall channel that RRF will combine with dense
    retrieval. OR is intentional: requiring every natural-language term made
    all 12 pilot FTS searches return zero rows.
    """
    anchors = extract_anchors(question)
    retained_anchors = [
        concept for concept in anchors.required
        if concept.lower() not in _GENERIC_FTS_TERMS
    ]
    raw_terms: list[str] = []
    for concept in retained_anchors:
        raw_terms.extend(aliases_for(concept))
    # If deterministic concept extraction found no useful anchor, fall back to
    # distinctive words from the original question rather than planner-style
    # boilerplate. RRF can reward documents matching several OR terms.
    if not raw_terms:
        raw_terms.extend(
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", question)
            if len(token) >= 4 and token.lower() not in _GENERIC_FTS_TERMS
        )
    rendered: list[str] = []
    seen: set[str] = set()
    for value in raw_terms:
        term = _websearch_term(value)
        if term and term not in seen:
            seen.add(term)
            rendered.append(term)
    if not rendered:
        raise ValueError("Could not derive PostgreSQL FTS terms from the query")
    return " OR ".join(rendered), {
        "required_anchors": list(anchors.required),
        "retained_anchors": retained_anchors,
        "optional_terms": list(anchors.optional),
        "operator": "OR",
        "builder": "deterministic_anchor_websearch_v1",
    }
