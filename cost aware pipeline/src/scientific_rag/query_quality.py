"""Deterministic scientific query planning with a bounded LLM repair escape hatch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "be", "been", "being", "by", "can",
    "current", "do", "does", "evidence", "for", "from", "how", "in", "into", "is",
    "known", "latest", "main", "of", "on", "recent", "research", "studies", "the",
    "their", "to", "use", "used", "uses", "using", "what", "which", "with",
}

# Canonical concepts and useful spelling variants. Long/specific phrases are
# intentionally listed before generic token extraction.
_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("retrieval augmented generation", ("retrieval augmented generation", "retrieval-augmented generation", "rag")),
    ("quantum error correction", ("quantum error correction", "qec")),
    ("fourier neural operator", ("fourier neural operator", "fourier neural operators", "fno")),
    ("partial differential equation", ("partial differential equation", "partial differential equations", "pde", "pdes")),
    ("large language model", ("large language model", "large language models", "llm", "llms")),
    ("machine learning", ("machine learning", "deep learning", "artificial intelligence", "ai")),
    ("surface code", ("surface code", "surface codes", "rotated surface code", "xzzx")),
    ("breast cancer", ("breast cancer", "mammary cancer")),
    ("lung cancer", ("lung cancer", "pulmonary cancer")),
    ("colorectal cancer", ("colorectal cancer", "colon cancer", "rectal cancer")),
    ("drug discovery", ("drug discovery", "drug design")),
    ("cancer", ("cancer", "tumor", "tumour", "oncology", "neoplasm")),
    ("crispr", ("crispr", "crispr-cas9", "crispr cas9", "cas9")),
    ("car-t", ("car-t", "car t", "car t-cell", "chimeric antigen receptor t")),
    ("semaglutide", ("semaglutide", "glp-1")),
    ("alphafold", ("alphafold", "protein structure prediction")),
    ("mammography", ("mammography", "mammogram", "tomosynthesis")),
    ("quantum machine learning", ("quantum machine learning", "qml")),
    ("gut microbiome", ("gut microbiome", "intestinal microbiome", "gut microbiota")),
)


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", value.lower()).strip()


@dataclass(frozen=True)
class QueryAnchors:
    required: tuple[str, ...]
    optional: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    method: str = "deterministic"

    def as_dict(self) -> dict:
        return asdict(self)


def aliases_for(concept: str) -> tuple[str, ...]:
    for canonical, aliases in _CONCEPTS:
        if canonical == concept:
            return aliases
    return (concept,)


def extract_anchors(question: str, planner_query: str = "") -> QueryAnchors:
    """Extract stable topic concepts without spending an LLM call."""
    normalized = _normalized_text(question)
    required: list[str] = []
    consumed: set[str] = set()
    for canonical, aliases in _CONCEPTS:
        if any(re.search(rf"(?<![a-z0-9]){re.escape(_normalized_text(alias))}(?![a-z0-9])", normalized) for alias in aliases):
            # Avoid retaining generic cancer when a more specific cancer concept exists.
            if canonical == "cancer" and any(item.endswith(" cancer") for item in required):
                continue
            required.append(canonical)
            consumed.update(token for alias in aliases for token in alias.split())

    # Preserve distinctive acronyms/hyphenated entities that are not in the lexicon.
    for raw in re.findall(r"\b(?:[A-Z][A-Z0-9]{2,}|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b", question):
        term = raw.lower()
        term_parts = set(re.findall(r"[a-z0-9]+", term))
        # Do not duplicate a recognized phrase solely because the user wrote it
        # with a hyphen ("lung-cancer" vs "lung cancer").
        if term not in consumed and not term_parts.intersection(consumed) and term not in required:
            required.append(term)

    fallback = [
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", question)
        if len(token) >= 4 and token.lower() not in _STOP_WORDS and token.lower() not in consumed
    ]
    if not required:
        required.extend(dict.fromkeys(fallback[:3]))

    planner_terms = [
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", planner_query)
        if len(token) >= 4 and token.lower() not in _STOP_WORDS
    ]
    optional = [item for item in dict.fromkeys([*fallback, *planner_terms]) if item not in required][:5]
    return QueryAnchors(tuple(required[:4]), tuple(optional))


def _pubmed_clause(concept: str) -> str:
    aliases = aliases_for(concept)
    terms = [f'"{alias}"[tiab]' for alias in aliases]
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def build_pubmed_query(anchors: QueryAnchors, *, broad: bool = False) -> str:
    required = [_pubmed_clause(item) for item in anchors.required]
    if broad or not anchors.optional:
        return " AND ".join(required)
    optional = " OR ".join(_pubmed_clause(item) for item in anchors.optional[:3])
    return " AND ".join([*required, f"({optional})"])


def _arxiv_clause(concept: str) -> str:
    aliases = aliases_for(concept)
    return "(" + " OR ".join(f'all:"{alias}"' for alias in aliases) + ")"


def build_arxiv_query(anchors: QueryAnchors, *, broad: bool = False) -> str:
    required = [_arxiv_clause(item) for item in anchors.required]
    if broad or not anchors.optional:
        return " AND ".join(required)
    optional = " OR ".join(_arxiv_clause(item) for item in anchors.optional[:2])
    return " AND ".join([*required, f"({optional})"])


def concepts_covered(text: str, concepts: tuple[str, ...]) -> list[str]:
    normalized = _normalized_text(text)
    return [
        concept for concept in concepts
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(_normalized_text(alias))}(?![a-z0-9])", normalized)
            for alias in aliases_for(concept)
        )
    ]


def query_is_valid(query: str, anchors: QueryAnchors) -> bool:
    return len(concepts_covered(query, anchors.required)) == len(anchors.required)
