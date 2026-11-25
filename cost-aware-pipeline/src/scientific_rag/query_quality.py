"""Two-tier scientific query constraints; descriptors never become mandatory entities.

Grounded entities from the existing planner supplement a small terminology lexicon.
The deterministic fallback uses noun positions, not all non-stopwords. Unresolved
subjects remain explicit and cannot authorize cached-source admission.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import re

_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("long covid", ("long covid", "post covid 19 condition", "post covid condition", "post covid 19 syndrome")),
    ("chronic kidney disease", ("chronic kidney disease", "ckd")),
    ("heart failure", ("heart failure",)),
    ("hypertension", ("hypertension", "hypertensive")),
    ("emergency department", ("emergency department", "emergency departments")),
    ("sodium reduction", ("sodium reduction", "salt reduction", "reduced salt intake", "reduced sodium intake")),
    ("aspirin", ("aspirin",)),
    ("sepsis", ("sepsis",)),
    ("tomosynthesis", ("tomosynthesis",)),
    ("graph neural network", ("graph neural network", "graph neural networks", "gnn")),
    ("ceramic coating", ("ceramic coating", "ceramic coatings")),
    ("robotic manipulation", ("robotic manipulation", "robot manipulation", "manipulation robot", "manipulation robots")),
    ("offline reinforcement learning", ("offline reinforcement learning", "offline rl")),
    ("online reinforcement learning", ("online reinforcement learning", "online rl")),
    ("reinforcement learning", ("reinforcement learning",)),
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

    ("diabetic retinopathy", ("diabetic retinopathy",)),
    ("deep learning", ("deep learning",)),
    ("machine learning", ("machine learning",)),
    ("artificial intelligence", ("artificial intelligence", "ai")),
    ("surface code", ("surface code", "surface codes", "rotated surface code", "xzzx")),
    ("breast cancer", ("breast cancer", "mammary cancer")),
    ("lung cancer", ("lung cancer", "pulmonary cancer")),
    ("colorectal cancer", ("colorectal cancer", "colon cancer", "rectal cancer")),
    ("drug discovery", ("drug discovery", "drug design")),
    ("cancer", ("cancer", "tumor", "tumour", "oncology", "neoplasm")),
    ("crispr", ("crispr", "crispr-cas9", "crispr cas9", "cas9")),
    ("car-t", ("car-t", "car t", "car t-cell", "chimeric antigen receptor t")),
    ("semaglutide", ("semaglutide",)),
    ("alphafold", ("alphafold",)),
    ("mammography", ("mammography", "mammogram")),
    ("quantum machine learning", ("quantum machine learning", "qml")),
    ("gut microbiome", ("gut microbiome", "intestinal microbiome", "gut microbiota")),
    ("antimicrobial resistance", ("antimicrobial resistance", "antibiotic resistance", "drug-resistant infection", "drug resistant infection", "amr")),
    ("healthcare setting", ("healthcare setting", "healthcare settings", "health care setting", "health care settings", "hospital", "hospitals", "clinical setting", "clinical settings")),
)



_INTERVENTIONS = {"aspirin", "crispr", "car-t", "semaglutide", "mammography", "tomosynthesis", "alphafold", "sodium reduction", "offline reinforcement learning", "online reinforcement learning"}
_POPULATIONS = {"adults", "children", "older adults", "healthcare setting", "emergency department"}
_DISEASES = {"cancer", "breast cancer", "lung cancer", "colorectal cancer", "dementia", "diabetic retinopathy", "hypertension", "sepsis", "heart failure", "chronic kidney disease", "long covid"}
# Ranking descriptors, never an entity ontology. Unknown words are not promoted
# to constraints by absence from this set.
_DISCOURSE = set("a an the which what how why does do can could may would should is are was were be been of for in on with without from after before at to and or than more less between versus compare compared comparison effect effects associated persistent recent current evidence research study studies trial trials randomized observational review clinical reduce reduces reduction lower higher improve improves improved help helps ease approaches approach natural remedies complementary herbal dietary supplements generic unknown rare newly described applications application toxicity adverse events performance predictive models model type terms consumption inference fatigue flares symptoms limitations property prediction under cyclic loading".split())


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", value.lower()).strip()


def _singular(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "is", "us", "ics")):
        return word[:-1]
    return word


def _morph_text(value):
    return " ".join(_singular(w) for w in re.sub(r"\boffline (?:and|versus) online reinforcement learning\b", "offline reinforcement learning and online reinforcement learning", _normalized_text(value)).split())


@dataclass(frozen=True)
class QueryAnchors:
    required: tuple[str, ...]
    optional: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    method: str = "entities-and-descriptors-v2"
    dimensions: tuple[str, ...] = ()
    study_design: tuple[str, ...] = ()
    intent: tuple[str, ...] = ()
    comparison: tuple[str, ...] = ()
    entity_roles: tuple[tuple[str, str], ...] = ()
    unresolved_subject: bool = False

    def as_dict(self):
        return asdict(self)


def aliases_for(concept: str) -> tuple[str, ...]:
    for canonical, aliases in _CONCEPTS:
        if _morph_text(canonical) == _morph_text(concept):
            return aliases
    return (concept,)


def concepts_covered(text: str, concepts: tuple[str, ...]) -> list[str]:
    normalized = _morph_text(text)
    return [concept for concept in concepts if any(
        re.search(r"(?<![a-z0-9])" + re.escape(_morph_text(alias)) + r"(?![a-z0-9])", normalized)
        for alias in aliases_for(concept)
    )]


def extract_anchors(question: str, planner_query: str = "", *, entity_hints=()) -> QueryAnchors:
    from .evidence_adequacy import requested_dimensions, dimension_aliases
    normalized = _normalized_text(question)
    dimensions = requested_dimensions(question)
    dimension_words = {w for d in dimensions for a in dimension_aliases(question)[d] for w in _normalized_text(a).split()}
    required = []
    roles = {}
    consumed = set()
    for canonical, aliases in _CONCEPTS:
        matching = [alias for alias in aliases if concepts_covered(question, (alias,))]
        if not matching:
            continue
        if canonical == "reinforcement learning" and any(x.endswith(" reinforcement learning") for x in required):
            continue
        if canonical == "adults" and "older adults" in required:
            continue
        if canonical == "cancer" and any(x.endswith(" cancer") for x in required):
            continue
        required.append(canonical)
        roles[canonical] = "intervention" if canonical in _INTERVENTIONS else "population" if canonical in _POPULATIONS else "disease" if canonical in _DISEASES else "subject"
        consumed.update(w for a in matching for w in _normalized_text(a).split())

    if re.search(r"offline\s+(?:and|versus|vs)\s+online reinforcement learning", question, re.I):
        for arm in ("offline reinforcement learning", "online reinforcement learning"):
            if arm not in required:
                required.append(arm)
                roles[arm] = "intervention"
    def add_grounded(term, role):
        raw_term = term.lower()
        term = _normalized_text(term)
        words = [w for w in term.split() if w not in _DISCOURSE and w not in consumed and w not in dimension_words]
        if not words:
            return
        term = raw_term if re.fullmatch(r"[a-z]+-(?:ion|oxide|alloy)", raw_term) else " ".join(words)
        if term not in required and concepts_covered(question, (term,)):
            required.append(term)
            roles[term] = role
            consumed.update(words)

    # Existing planner output, no new model invocation. Require verbatim/morph-
    # grounded text and an explicit entity role, never invented synonyms.
    for hint in entity_hints:
        term = getattr(hint, "text", "")
        role = getattr(hint, "role", "subject")
        if term and concepts_covered(question, (term,)):
            add_grounded(term, role)
    # Unlisted drug names/classes and scientific acronyms have identifiable
    # positions or morphology; ordinary hyphenated descriptors do not.
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", question):
        if token.lower().endswith(("mab", "nib", "stat", "purinol", "cycline", "cillin")) or (token.isupper() and len(token) >= 3):
            add_grounded(token, "intervention" if not token.isupper() else "subject")
    for token in re.findall(r"\b\d+-bit\b", question):
        required.append(token)
        roles[token] = "intervention"
    for term in re.findall(r"\b[a-z]+-(?:ion|oxide|alloy)\b", question, re.I):
        add_grounded(term, "exposure")
    for term in re.findall(r"\b(?:batteries|battery|coatings|detectors)\b", question, re.I):
        add_grounded(term, "subject")
    # Verb objects / explicit population relations are conservative bootstrap
    # noun spans, useful for unlisted diseases and exposures (not disease names).
    patterns = (
        (r"\b(?:help|helps|ease|treat|treats|prevent|prevents|affect|affects)\s+(.+?)(?=\b(?:without|with|and|or|versus|in|on)\b|[?.!]|$)", "disease"),
        (r"\b(?:with|for|in|on)\s+(.+?)(?=\b(?:without|and|or|versus|in|on|under)\b|[?.!]|$)", "subject"),
        (r"^(?:does|do|can|could|may)\s+(.+?)(?=\s+(?:help|reduce|affect|improve|prevent|cause|lower))", "intervention"),
    )
    dimensions = requested_dimensions(question)
    dimension_words = {w for d in dimensions for a in dimension_aliases(question)[d] for w in _normalized_text(a).split()}
    for pattern, role in patterns:
        for match in re.finditer(pattern, question, re.I):
            phrase = match.group(1).strip()
            words = [w for w in _normalized_text(phrase).split() if w not in dimension_words]
            if len(words) <= 4:
                add_grounded(" ".join(words), role)
    # Medical nominal endings can identify an unlisted condition in a compact
    # topic list. Unknown descriptors never become required by elimination.
    for token in normalized.split():
        if token not in _DISCOURSE and token.endswith(("itis", "osis", "iasis", "emia")):
            add_grounded(token, "disease")
    terms = [w for w in _normalized_text(question + " " + planner_query).split() if w not in _DISCOURSE and w not in consumed and w not in dimension_words]
    rankable = {"performance", "prediction", "persistent", "cardiovascular", "efficient", "fatigue", "flares", "safety", "accuracy", "latency"}
    terms.extend(w for w in normalized.split() if w in rankable and w not in consumed)
    optional = tuple(dict.fromkeys(terms))[:8]
    return QueryAnchors(tuple(dict.fromkeys(required)), optional, dimensions=dimensions,
        study_design=tuple(w for w in ("randomized", "observational", "cohort", "review") if w in normalized.split()),
        entity_roles=tuple(roles.items()), unresolved_subject=not bool(required))


def _variants(concept):
    aliases = aliases_for(concept)
    # API phrase searches need plural/singular alternatives too. Only transform
    # explicit plural nouns, never invent oncologys/dietarys.
    return tuple(dict.fromkeys((*aliases, *(_morph_text(a) for a in aliases))))


def _clause(concept, source, relaxed=False):
    if source == "pubmed" and relaxed:
        # Unquoted, untagged biomedical phrases permit PubMed Automatic Term
        # Mapping (MeSH and morphology); essential concepts remain conjuncts.
        return "(" + " OR ".join(_variants(concept)) + ")"
    if source == "arxiv" and relaxed:
        return "(" + " OR ".join("(" + " AND ".join(f"all:{word}" for word in _morph_text(alias).split()) + ")" for alias in _variants(concept)) + ")"
    field = "[tiab]" if source == "pubmed" else ""
    prefix = "" if source == "pubmed" else "all:"
    return "(" + " OR ".join(f'{prefix}"{alias}"{field}' for alias in _variants(concept)) + ")"


def _build(anchors, source, broad=False, relaxed=False):
    required = [_clause(t, source, relaxed) for t in sorted(anchors.required)]
    if not required:
        return " OR ".join(_clause(t, source, relaxed) for t in anchors.optional[:4])
    if broad or not anchors.optional:
        return " AND ".join(required)
    return " AND ".join([*required, "(" + " OR ".join(_clause(t, source) for t in anchors.optional[:3]) + ")"])


def build_pubmed_query(anchors, *, broad=False, relaxed=False):
    return _build(anchors, "pubmed", broad, relaxed)


def build_arxiv_query(anchors, *, broad=False, relaxed=False):
    return _build(anchors, "arxiv", broad, relaxed)


def query_is_valid(query, anchors):
    content = re.sub(r"\b(?:AND|OR)\b|\[tiab\]|all:", " ", query)
    return bool(query) and len(concepts_covered(content, anchors.required)) == len(anchors.required)


def topical_documents(documents, anchors):
    roles = dict(anchors.entity_roles)
    arms = tuple(t for t in anchors.required if roles.get(t) == "intervention")
    context = tuple(t for t in anchors.required if t not in arms)
    eligible = []
    for doc in documents:
        text = str(doc.metadata.get("title", "")) + " " + doc.page_content
        if not anchors.required:
            if not doc.metadata.get("reused_from_postgres") and doc.metadata.get("relevance_label") in {"relevant", "uncertain"}:
                eligible.append(doc)
        elif len(concepts_covered(text, context)) == len(context) and (not arms or concepts_covered(text, arms)):
            eligible.append(doc)
    return eligible
