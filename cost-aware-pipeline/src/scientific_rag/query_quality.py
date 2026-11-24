"""Deterministic scientific query planning with a bounded LLM repair escape hatch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "be", "been", "being", "by", "can",
    "causes", "challenges", "current", "do", "does", "evidence", "for", "from", "how",
    "in", "into", "is", "known", "latest", "limitations", "main", "of", "on",
    "between", "relationship", "relationships", "trade", "offs", "tradeoffs",
    "opportunities", "prevention", "recent", "research", "risks", "roles", "studies",
    "strategies", "the", "their", "to", "use", "used", "uses", "using", "what",
    "which", "with",
}

# Canonical concepts and useful spelling variants. Long/specific phrases are
# intentionally listed before generic token extraction.
_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("robotic manipulation", ("robotic manipulation", "robot manipulation", "manipulation robot", "manipulation robots")),
    ("reinforcement learning", ("reinforcement learning", "offline reinforcement learning", "online reinforcement learning", "offline rl", "online rl")),
    ("retrieval augmented generation", ("retrieval augmented generation", "retrieval-augmented generation", "rag")),
    ("quantum error correction", ("quantum error correction", "qec")),
    ("fourier neural operator", ("fourier neural operator", "fourier neural operators", "fno")),
    ("partial differential equation", ("partial differential equation", "partial differential equations", "pde", "pdes")),
    ("large language model", ("large language model", "large language models", "llm", "llms")),
    ("sleep duration", ("sleep duration", "sleep durations", "sleep time", "sleep length")),
    ("dementia", ("dementia", "dementias")),
    ("quantization", ("quantization", "quantisation", "quantized", "quantised", "quantize", "quantise")),
    ("time restricted eating", ("time restricted eating", "time-restricted eating", "time restricted feeding")),
    ("older adults", ("older adults", "elderly", "older people")),
    ("adults", ("adults", "adult")),
    ("children", ("children", "pediatric", "paediatric")),
    ("natural remedies", ("natural remedies", "natural remedy", "herbal", "dietary", "supplements", "complementary therapies")),
    ("diabetic retinopathy", ("diabetic retinopathy",)),
    ("machine learning", ("machine learning", "deep learning", "artificial intelligence", "ai")),
    ("surface code", ("surface code", "surface codes", "rotated surface code", "xzzx")),
    ("breast cancer", ("breast cancer", "mammary cancer")),
    ("lung cancer", ("lung cancer", "pulmonary cancer")),
    ("colorectal cancer", ("colorectal cancer", "colon cancer", "rectal cancer")),
    ("drug discovery", ("drug discovery", "drug design")),
    ("cancer", ("cancer", "tumor", "tumour", "oncology", "neoplasm")),
    ("crispr", ("crispr", "crispr-cas9", "crispr cas9", "cas9")),
    ("car-t", ("car-t", "car t", "car t-cell", "chimeric antigen receptor t")),
    ("semaglutide", ("semaglutide",)),
    ("alphafold", ("alphafold", "protein structure prediction")),
    ("mammography", ("mammography", "mammogram", "tomosynthesis")),
    ("quantum machine learning", ("quantum machine learning", "qml")),
    ("gut microbiome", ("gut microbiome", "intestinal microbiome", "gut microbiota")),
    ("antimicrobial resistance", ("antimicrobial resistance", "antibiotic resistance", "drug-resistant infection", "drug resistant infection", "amr")),
    ("healthcare setting", ("healthcare setting", "healthcare settings", "health care setting", "health care settings", "hospital", "hospitals", "clinical setting", "clinical settings")),
)


# Descriptive comparisons and qualifiers are useful search details, but should
# never become mandatory identity anchors merely because they are hyphenated.
_NON_ENTITY_HYPHENS = frozenset({"trade-off", "trade-offs", "post-training"})


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", value.lower()).strip()


@dataclass(frozen=True)
class QueryAnchors:
    required: tuple[str, ...]
    optional: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    method: str = "deterministic"
    dimensions: tuple[str, ...] = ()
    study_design: tuple[str, ...] = ()
    intent: tuple[str, ...] = ()
    comparison: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


def aliases_for(concept: str) -> tuple[str, ...]:
    for canonical, aliases in _CONCEPTS:
        if canonical == concept:
            return tuple(dict.fromkeys((*aliases, *(a + "s" for a in aliases if not a.endswith("s") and len(a) >= 4))))
    # Conservative regular plural alternatives; never strip disease endings
    # such as psoriasis, or equate an individual drug with its whole class.
    if concept.endswith("s") or len(concept) < 4:
        return (concept,)
    return (concept, concept + "s")


def extract_anchors(question: str, planner_query: str = "") -> QueryAnchors:
    """Extract stable topic concepts without spending an LLM call."""
    from .evidence_adequacy import requested_dimensions, DIMENSION_ALIASES
    normalized = _normalized_text(question)
    dimensions = requested_dimensions(question)
    design_words = ("randomized", "randomised", "observational", "cohort", "trial", "trials", "review", "meta-analysis")
    design = tuple(word for word in design_words if re.search(rf"\b{re.escape(word)}\b", question.lower()))
    dimension_words = {word for dim in dimensions for alias in DIMENSION_ALIASES[dim] for word in _normalized_text(alias).split()}
    generic = _STOP_WORDS | dimension_words | set(design_words) | {
        "help", "helps", "benefit", "benefits", "reduce", "reduces", "affect", "affects", "effect", "effects",
        "compare", "compared", "comparison", "versus", "whether", "improve", "improves", "improved",
        "applications", "application", "approaches", "approach", "utilized", "enhanced", "tasks", "offline", "online", "strengths", "style",
        "extent", "confounding", "reverse", "causation", "explain", "observed", "associations", "incident",
        "terms", "consumption", "inference", "post", "training", "natural", "remedies", "clinical",
        "randomised", "systematic", "controlled", "results", "patients", "adults", "children", "efficacy",
        "not", "any", "may", "has", "was", "its", "but", "get", "yet", "also",
    }
    required: list[str] = []
    consumed: set[str] = set()
    for canonical, aliases in _CONCEPTS:
        if any(re.search(rf"(?<![a-z0-9]){re.escape(_normalized_text(alias))}(?![a-z0-9])", normalized) for alias in aliases):
            # Avoid retaining generic cancer when a more specific cancer concept exists.
            if canonical == "adults" and "older adults" in required:
                continue
            if canonical == "cancer" and any(item.endswith(" cancer") for item in required):
                continue
            required.append(canonical)
            consumed.update(token for alias in aliases for token in _normalized_text(alias).split())

    # Preserve distinctive acronyms/hyphenated entities that are not in the lexicon.
    for raw in re.findall(r"\b(?:[A-Z][A-Z0-9]{2,}|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b", question):
        term = raw.lower()
        if term in _NON_ENTITY_HYPHENS or term in design_words:
            continue
        term_parts = set(re.findall(r"[a-z0-9]+", term))
        # Do not duplicate a recognized phrase solely because the user wrote it
        # with a hyphen ("lung-cancer" vs "lung cancer").
        if term not in consumed and not term_parts.intersection(consumed) and term not in required:
            required.append(term)

    fallback = [
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", question)
        if len(token) >= 3 and token.lower() not in generic and token.lower() not in consumed
    ]
    # Unknown topics remain constraints even alongside a recognized concept.
    # Dropping them because the lexicon recognized another word loses diseases
    # and interventions. No silent four-anchor truncation.
    required.extend(item for item in dict.fromkeys(fallback) if item not in required)

    planner_terms = [
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", planner_query)
        if len(token) >= 4 and token.lower() not in _STOP_WORDS
    ]
    optional = [item for item in dict.fromkeys([*fallback, *planner_terms]) if item not in required][:5]
    return QueryAnchors(
        tuple(required), tuple(optional), dimensions=dimensions, study_design=design,
        intent=tuple(word for word in ("reduce", "help", "affect", "compare", "explain") if re.search(rf"\b{word}\b", normalized)),
        comparison=tuple(word for word in ("versus", "between", "compared") if re.search(rf"\b{word}\b", normalized)),
    )


def _pubmed_clause(concept: str) -> str:
    aliases = aliases_for(concept)
    terms = [f'"{alias}"[tiab]' for alias in aliases]
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def build_pubmed_query(anchors: QueryAnchors, *, broad: bool = False) -> str:
    required = [_pubmed_clause(item) for item in sorted(anchors.required)]
    if broad or not anchors.optional:
        return " AND ".join(required)
    optional = " OR ".join(_pubmed_clause(item) for item in anchors.optional[:3])
    return " AND ".join([*required, f"({optional})"])


def _arxiv_clause(concept: str) -> str:
    aliases = aliases_for(concept)
    return "(" + " OR ".join(f'all:"{alias}"' for alias in aliases) + ")"


def build_arxiv_query(anchors: QueryAnchors, *, broad: bool = False) -> str:
    required = [_arxiv_clause(item) for item in sorted(anchors.required)]
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
