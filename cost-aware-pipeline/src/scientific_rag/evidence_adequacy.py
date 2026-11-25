"""Versioned passage-level retrieval adequacy, never a semantic entailment judge.

Direct candidates require a result-bearing passage; they remain candidates until
claim verification. Unknown dimensions fail open to qualified synthesis, never to
an assertion of full sufficiency. No model calls or private-question logging.
"""
from __future__ import annotations
import hashlib
import re
from app.untrusted_content import find_instruction_like_spans

DIMENSION_ALIASES = {
    'cardiovascular mortality': ('cardiovascular mortality', 'cardiovascular death', 'cardiovascular deaths'),
    'mortality': ('mortality', 'death', 'deaths', 'survival'),
    'hospitalization': ('hospitalization', 'hospitalisation', 'hospitalizations', 'hospitalisations', 'hospital admission'),
    'bleeding': ('bleeding', 'haemorrhage', 'hemorrhage'),
    'accuracy': ('accuracy', 'predictive performance', 'discrimination', 'auroc'),
    'memory': ('memory', 'storage', 'model size'),
    'latency': ('latency', 'inference time', 'throughput'),
    'external validation': ('external validation', 'externally validated', 'external generalization', 'external generalisation', 'generalizability', 'generalisability'),
    'energy efficiency': ('energy efficiency', 'energy consumption', 'power consumption'),
    'capacity': ('capacity', 'charge capacity'),
    'durability': ('durability', 'cycle life', 'fatigue life'),
    'cost': ('cost', 'costs', 'price'),
    'pain': ('pain', 'pain severity'),
    'blood pressure': ('blood pressure', 'systolic pressure', 'diastolic pressure'),
    'flares': ('flares', 'flare', 'flare rates'),
    'safety': ('safety', 'adverse events', 'adverse effects', 'side effects'),
    'confounding': ('confounding', 'confounders', 'reverse causation'),
}
_RESULT = re.compile(r'\b(result\w*|found|observed|associated|achieved|measured|reduced|lower|higher|increased|decreased|hazard ratio|risk ratio|no difference)\b', re.I)
_MISSING_EVIDENCE = re.compile(r'no (?:eligible )?(?:studies|trials|evidence)|(?:search|review) (?:did not|failed to) (?:find|identify)', re.I)
_CONTEXT = re.compile(r'\b(background|introduction|future|should|hypothes\w*|aims?|objective|protocol)\b', re.I)
_COMPARATIVE = re.compile(r'\b(versus|vs\.?|compar\w*|relative to|than|between|head.to.head)\b', re.I)
_INDIRECT = re.compile(r'\b(indirect|network meta.analysis|separate (?:studies|trials))\b', re.I)


def mentions(text: str, aliases) -> bool:
    from .query_quality import concepts_covered
    return any(concepts_covered(text, (alias,)) for alias in aliases)


def dimension_aliases(question):
    # Ambiguous terms have domain-specific meanings. Storage is not cognitive
    # memory; model capacity is not battery charge capacity; care prices are not
    # computational complexity. An alias never imports an unrequested domain.
    tech = bool(re.search(r"quantiz|language model|neural|comput|inference|algorithm|machine learning", question, re.I))
    materials = bool(re.search(r"batter|electrod|ceramic|coating|material|fatigue life|cycle life", question, re.I))
    aliases = dict(DIMENSION_ALIASES)
    aliases["memory"] = DIMENSION_ALIASES["memory"] if tech else ("memory", "memory loss", "cognitive function")
    if not tech:
        aliases.pop("latency")
    if not materials:
        aliases["capacity"] = ("capacity",)
    aliases["cost"] = ("cost", "costs")  # price alone does not request costs
    if re.search(r"\b(?:care|healthcare|medical|treatment|financial|economic)\b.*\bcosts?\b|\bcosts?\b.*\b(?:care|healthcare|medical|treatment)\b", question, re.I):
        aliases["cost"] = ("care costs", "cost of care", "healthcare costs", "medical costs", "treatment costs", "financial costs", "healthcare expenditure")
    measurement = tech or bool(re.search(r"predict|diagnos|model|classifier|imaging|detection", question, re.I))
    if not measurement:
        aliases["accuracy"] = ("accuracy",)
    return aliases


def requested_dimensions(question: str) -> tuple[str, ...]:
    dims = tuple(key for key, aliases in dimension_aliases(question).items() if mentions(question, aliases))
    if "cardiovascular mortality" in dims and not re.search(r"all.cause|overall mortality", question, re.I):
        dims = tuple(d for d in dims if d != "mortality")
    # Compound names (memory care, memory T cells) are not memory outcomes.
    if re.search(r"memory (?:care|clinic|t cells)", question, re.I) and not re.search(r"memory (?:loss|function|performance)", question, re.I):
        dims = tuple(d for d in dims if d != "memory")
    return dims


def passage_spans(text):
    """Raw offsets; decimal points do not end a passage."""
    for match in re.finditer(r".+?(?:[.!?](?=\s+[A-Z]|\s*$)|\n|$)", text, re.S):
        raw = match.group()
        passage = raw.strip()
        if passage:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield start, start+len(passage), passage


def assess_adequacy(question, documents, *, external_unavailable=False, anchors=None):
    from .query_quality import extract_anchors, concepts_covered, topical_documents
    docs = list(documents)
    anchors = anchors or extract_anchors(question)
    dimensions = requested_dimensions(question)
    aliases = dimension_aliases(question)
    topical_ids = {id(doc) for doc in topical_documents(docs, anchors)}
    roles = dict(anchors.entity_roles)
    interventions = tuple(t for t in anchors.required if roles.get(t) in {"intervention", "exposure"})
    def covered(text, terms):
        return concepts_covered(text, terms, entity_aliases=anchors.entity_aliases)
    topical_docs = [doc for doc in docs if id(doc) in topical_ids]
    corpus = " ".join(doc.page_content for doc in topical_docs)
    missing_entities = [t for t in anchors.required if t not in covered(corpus, anchors.required)]
    comparison_arms = anchors.comparison
    covered_arms = covered(corpus, comparison_arms)
    direct_comparison = False
    findings = []
    for dimension in dimensions:
        passages = []
        for doc in docs:
            metadata = doc.metadata or {}
            for start,end,passage in passage_spans(doc.page_content):
                if not mentions(passage, aliases[dimension]) or find_instruction_like_spans(passage):
                    continue
                topical = id(doc) in topical_ids
                # A whole abstract mentioning the requested drug cannot attribute
                # results of another intervention or a cited prior study to it.
                local_arm = not interventions or bool(covered(passage, interventions))
                # One-arm results remain usable context. A direct comparative
                # candidate needs every arm and a comparison relation in the
                # SAME result passage, never a corpus-wide union.
                comparative = not comparison_arms or (
                    len(covered(passage, comparison_arms)) == len(comparison_arms)
                    and bool(_COMPARATIVE.search(passage)) and not _INDIRECT.search(passage)
                )
                prior = re.search(r"\b(previous|prior|earlier|another) (?:study|trial|studies)|et al\.", passage, re.I)
                background = metadata.get("evidence_role") == "background" or str(metadata.get("section", "")).lower() in {"introduction","background"}
                result_candidate = topical and local_arm and _RESULT.search(passage) and not (prior or _CONTEXT.search(passage) or _MISSING_EVIDENCE.search(passage) or background or anchors.unresolved_subject)
                status = "direct_candidate" if result_candidate and comparative else "contextual"
                direct_comparison |= bool(comparison_arms and status == "direct_candidate")
                passages.append({
                    "source_id":str(metadata.get("citation_id") or metadata.get("doc_id") or metadata.get("url") or hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]),
                    "document_digest":hashlib.sha256(doc.page_content.encode()).hexdigest(),
                    "passage_id":hashlib.sha256(f"{start}:{end}:{passage}".encode()).hexdigest()[:16],
                    "start":start,"end":end,"study_design":metadata.get("study_design") or "unknown","status":status,
                    "covered_arms":covered(passage, comparison_arms),
                })
        status = "direct_candidate" if any(p["status"] == "direct_candidate" for p in passages) else "contextual" if passages else "missing"
        findings.append({"dimension":dimension,"status":status,"passages":passages})
    unanswered = [d["dimension"] for d in findings if d["status"] != "direct_candidate"]
    count = sum(d["status"] == "direct_candidate" for d in findings)
    status = ("external_retrieval_failure" if external_unavailable else "missing") if not docs else "candidate_coverage_complete" if dimensions and not unanswered else "partial" if count else "contextual_only" if any(d["passages"] for d in findings) else "uncertain"
    comparison = {"requested_arms":list(comparison_arms), "covered_arms":covered_arms,
        "missing_arms":[arm for arm in comparison_arms if arm not in covered_arms],
        "status":"not_requested" if not comparison_arms else "direct_candidate" if direct_comparison else "indirect_candidates" if len(covered_arms) == len(comparison_arms) else "partial"}
    if missing_entities or anchors.unresolved_subject:
        status = "partial" if count else "uncertain" if docs else status
    return {"version":"passage-adequacy-v3","status":status,"scope":"retrieval_candidates_only",
        "full_sufficiency_established":False,"dimensions":findings,"unanswered_dimensions":unanswered,
        "comparison":comparison,"missing_entities":missing_entities,"entity_resolution_uncertain":anchors.unresolved_subject,
        "unrecognized_dimensions_possible":True,"external_retrieval_failure":bool(external_unavailable),
        "reason":"Candidate coverage is neither semantic support nor proof of search completeness. Missing candidate matches remain uncertain.","additional_model_calls":0}
