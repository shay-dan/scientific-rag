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
    'safety': ('safety', 'adverse events', 'adverse effects', 'side effects'),
    'confounding': ('confounding', 'confounders', 'reverse causation'),
}
_RESULT = re.compile(r'\b(result\w*|found|observed|associated|achieved|measured|reduced|lower|higher|increased|decreased|hazard ratio|risk ratio|no difference)\b', re.I)
_MISSING_EVIDENCE = re.compile(r'no (?:eligible )?(?:studies|trials|evidence)|(?:search|review) (?:did not|failed to) (?:find|identify)', re.I)
_CONTEXT = re.compile(r'\b(background|introduction|future|should|hypothes\w*|aims?|objective|protocol)\b', re.I)


def mentions(text: str, aliases) -> bool:
    normalized = re.sub(r'[^a-z0-9]+', ' ', text.lower())
    return any(re.search(r'(?<![a-z0-9])' + re.escape(re.sub(r'[^a-z0-9]+', ' ', a.lower())) + r'(?![a-z0-9])', normalized) for a in aliases)


def requested_dimensions(question: str) -> tuple[str, ...]:
    dims = tuple(key for key, aliases in DIMENSION_ALIASES.items() if mentions(question, aliases))
    # Do not invent a second all-cause endpoint from cardiovascular mortality.
    if 'cardiovascular mortality' in dims and not re.search(r'all.cause|overall mortality', question, re.I):
        dims = tuple(d for d in dims if d != 'mortality')
    return dims


def assess_adequacy(question: str, documents, *, external_unavailable=False) -> dict:
    from .query_quality import extract_anchors, concepts_covered
    docs = list(documents)
    anchors = extract_anchors(question)
    dimensions = requested_dimensions(question)
    findings = []
    for dimension in dimensions:
        passages = []
        for doc in docs:
            metadata = doc.metadata or {}
            for match in re.finditer(r'[^.!?\n]+(?:[.!?]|$)', doc.page_content):
                passage = match.group().strip()
                if not mentions(passage, DIMENSION_ALIASES[dimension]):
                    continue
                # Embedded instructions must never count as scientific results.
                if find_instruction_like_spans(passage):
                    continue
                topical = len(concepts_covered(doc.page_content, anchors.required)) == len(anchors.required)
                background = metadata.get('evidence_role') == 'background' or str(metadata.get('section', '')).lower() in {'introduction','background'}
                status = 'direct_candidate' if topical and _RESULT.search(passage) and not (_CONTEXT.search(passage) or _MISSING_EVIDENCE.search(passage) or background) else 'contextual'
                passages.append({
                    'source_id': str(metadata.get('citation_id') or metadata.get('doc_id') or metadata.get('url') or hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]),
                    'passage_id': hashlib.sha256(doc.page_content.encode()).hexdigest()[:16],
                    'start': match.start(), 'end': match.end(),
                    'study_design': metadata.get('study_design') or 'unknown',
                    'status': status,
                })
        status = 'direct_candidate' if any(p['status'] == 'direct_candidate' for p in passages) else 'contextual' if passages else 'missing'
        findings.append({'dimension':dimension, 'status':status, 'passages':passages})
    unanswered = [d['dimension'] for d in findings if d['status'] != 'direct_candidate']
    candidate_count = sum(d['status'] == 'direct_candidate' for d in findings)
    if not docs:
        status = 'external_retrieval_failure' if external_unavailable else 'missing'
    elif candidate_count and unanswered:
        status = 'partial'
    else:
        status = 'uncertain'
    return {
        'version':'passage-adequacy-v1', 'status':status,
        'scope':'retrieval_candidates_only', 'full_sufficiency_established':False,
        'dimensions':findings, 'unanswered_dimensions':unanswered,
        'unrecognized_dimensions_possible':True,
        'external_retrieval_failure':bool(external_unavailable),
        'reason':'Search coverage is not semantic support or evidence of absence. Direct candidates require claim verification; unrecognized question parts remain uncertain.',
        'additional_model_calls':0,
    }
