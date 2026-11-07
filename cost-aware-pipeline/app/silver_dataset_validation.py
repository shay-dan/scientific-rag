"""Source and claim validation for versioned Silver benchmark datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import html
import re
from typing import Any, Literal

from pydantic import BaseModel, Field


_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def normalize_scientific_typography(value: str) -> str:
    """Repair common model formatting losses without changing claim meaning."""
    value = value.replace("\u2011", "-").replace("\u00a0", " ")
    replacements = (
        (r"\bCRISPRCas9\b", "CRISPR-Cas9"),
        (r"\bCART\b", "CAR-T"),
        (r"\bTcells\b", "T cells"),
        (r"\bTcell\b", "T cell"),
        (r"\bgenomewide\b", "genome-wide"),
        (r"\bsolidtumor\b", "solid-tumor"),
        (r"\bchromatinremodeling\b", "chromatin-remodeling"),
        (r"\btumorkilling\b", "tumor killing"),
        (r"\bexhaustionrelated\b", "exhaustion-related"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    title: str
    abstract: str
    url: str

    @property
    def sha256(self) -> str:
        payload = f"{self.title}\n{self.abstract}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ClaimJudgment(BaseModel):
    claim_index: int = Field(ge=1)
    decision: Literal["supported", "contradicted", "insufficient"]
    confidence: float = Field(ge=0, le=1)
    source_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=5, max_length=1000)


class CaseJudgment(BaseModel):
    judgments: list[ClaimJudgment]


def normalize_text(value: str) -> str:
    """Normalize publisher formatting without changing scientific content."""
    value = html.unescape(value or "")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def excerpt_verification(excerpt: str, document: SourceDocument) -> dict[str, Any]:
    """Verify whether an excerpt is present in the fetched source abstract."""
    expected = normalize_text(excerpt)
    actual = normalize_text(document.abstract)
    if not expected or not actual:
        return {"status": "not_found", "similarity": 0.0}
    if expected in actual:
        return {"status": "verbatim_verified", "similarity": 1.0}

    sentences = [normalize_text(item) for item in _SENTENCE_RE.split(document.abstract) if item.strip()]
    similarity = max((SequenceMatcher(None, expected, sentence).ratio() for sentence in sentences), default=0.0)
    status = "near_verbatim" if similarity >= 0.90 else "not_found"
    return {"status": status, "similarity": round(similarity, 4)}


def best_evidence_excerpt(claim: str, document: SourceDocument, *, max_chars: int = 1400) -> str:
    """Return exact source sentences with the strongest lexical claim overlap."""
    claim_terms = set(_WORD_RE.findall(claim.lower()))
    sentences = [item.strip() for item in _SENTENCE_RE.split(document.abstract) if item.strip()]
    if not sentences:
        return document.abstract[:max_chars].strip()

    ranked: list[tuple[float, int, int, str]] = []
    for start in range(len(sentences)):
        for size in (1, 2):
            if start + size > len(sentences):
                continue
            excerpt = " ".join(sentences[start:start + size])
            excerpt_terms = set(_WORD_RE.findall(excerpt.lower()))
            overlap = len(claim_terms & excerpt_terms) / max(1, len(claim_terms))
            ranked.append((overlap, -size, start, excerpt))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    excerpt = ranked[0][3][:max_chars].strip()
    if normalize_text(excerpt) not in normalize_text(document.abstract):
        raise ValueError(f"Selected evidence excerpt is not contiguous in {document.source_id}")
    return excerpt


def validate_rebuilt_cases(cases: list[dict[str, Any]], documents: dict[str, SourceDocument]) -> list[str]:
    """Return structural/source errors in an independently audited dataset."""
    errors: list[str] = []
    for case in cases:
        case_id = str(case.get("id", "unknown"))
        claims = case.get("reference_claims", [])
        contexts = case.get("reference_context", [])
        support = case.get("claim_support", [])
        if case.get("eligible_for_benchmark") and len(claims) < 3:
            errors.append(f"{case_id}: eligible case has fewer than three supported claims")
        if len(claims) != len(contexts) or len(claims) != len(support):
            errors.append(f"{case_id}: claim/context/support counts differ")
        for context in contexts:
            source_id = context.get("source_id", "")
            document = documents.get(source_id)
            if document is None:
                errors.append(f"{case_id}: missing original source {source_id}")
                continue
            if normalize_text(context.get("excerpt", "")) not in normalize_text(document.abstract):
                errors.append(f"{case_id}: excerpt is not contiguous in {source_id}")
            if context.get("source_sha256") != document.sha256:
                errors.append(f"{case_id}: source hash mismatch for {source_id}")
    return errors


def build_case_prompt(case: dict[str, Any], documents: dict[str, SourceDocument]) -> str:
    """Build one bounded prompt whose output still judges every claim separately."""
    claims = "\n".join(f"{index}. {claim}" for index, claim in enumerate(case["reference_claims"], 1))
    source_ids = {item["source_id"] for item in case.get("reference_context", [])}
    sources = []
    for source_id in sorted(source_ids):
        document = documents.get(source_id)
        if not document:
            continue
        sources.append(
            f"SOURCE {source_id}\nTITLE: {document.title}\nABSTRACT:\n{document.abstract[:6500]}"
        )
    return f"""You are an independent scientific evidence auditor. Judge EACH atomic claim separately.
Use only the supplied original-source titles and abstracts. Do not use outside knowledge.

Decision rules:
- supported: the source directly entails the material claim, including its population, intervention, comparison, numbers, and level of certainty.
- contradicted: the source directly conflicts with the claim.
- insufficient: the source is related but does not establish the full claim, or the required source is missing.
- Do not treat a paper's background statement as that paper's experimental result.
- For every judgment, return only source_id values from the supplied sources that support the decision.
- Copy each source_id exactly, including its `PMID:` or `arxiv:` prefix.
- Return exactly one judgment for every numbered claim, in the same order.
- Keep each rationale to one concise sentence under 300 characters.
- Confidence is confidence in the decision, not confidence that the claim is true in the world.

QUESTION:
{case['question']}

CLAIMS:
{claims}

ORIGINAL SOURCES:
{chr(10).join(sources)}
"""


def normalize_judgments(case: dict[str, Any], judgment: CaseJudgment) -> list[dict[str, Any]]:
    """Guarantee exactly one decision for each expected claim index."""
    expected = len(case.get("reference_claims", []))
    by_index = {item.claim_index: item for item in judgment.judgments if 1 <= item.claim_index <= expected}
    normalized = []
    for index in range(1, expected + 1):
        item = by_index.get(index)
        if item is None:
            normalized.append({
                "claim_index": index,
                "decision": "insufficient",
                "confidence": 1.0,
                "source_ids": [],
                "rationale": "The independent judge did not return a decision for this claim.",
                "judge_returned": False,
            })
        else:
            normalized.append({**item.model_dump(), "judge_returned": True})
    return normalized


def resolve_source_id(source_id: str, documents: dict[str, SourceDocument]) -> str | None:
    """Resolve harmless provider formatting changes without guessing identities."""
    candidate = source_id.strip()
    if candidate in documents:
        return candidate
    lowered = candidate.lower()
    for known in documents:
        if lowered == known.lower():
            return known
        if known.startswith("PMID:") and lowered == known.removeprefix("PMID:").lower():
            return known
        if known.startswith("arxiv:") and lowered == known.removeprefix("arxiv:").lower():
            return known
    return None


def rebuild_case(
    case: dict[str, Any],
    judgments: list[dict[str, Any]],
    documents: dict[str, SourceDocument],
    *,
    min_confidence: float = 0.75,
    min_supported_claims: int = 3,
) -> dict[str, Any]:
    """Create a repaired case using only independently supported atomic claims."""
    original_claims = list(case.get("reference_claims", []))
    supported = [
        item for item in judgments
        if item["decision"] == "supported" and float(item["confidence"]) >= min_confidence
    ]
    new_claims: list[str] = []
    new_contexts: list[dict[str, Any]] = []
    new_support: list[dict[str, Any]] = []

    for output_index, item in enumerate(supported, 1):
        claim_index = int(item["claim_index"])
        claim = normalize_scientific_typography(original_claims[claim_index - 1])
        available_ids = [
            resolved for source_id in item.get("source_ids", [])
            if (resolved := resolve_source_id(source_id, documents)) is not None
        ]
        if not available_ids:
            continue
        source_id = available_ids[0]
        document = documents[source_id]
        context_id = f"{case['id']}-v2-ctx-{output_index:02d}"
        new_claims.append(claim)
        new_contexts.append({
            "context_id": context_id,
            "source_id": source_id,
            "title": document.title,
            "url": document.url,
            "excerpt": best_evidence_excerpt(claim, document),
            "excerpt_status": "verbatim_from_original_source",
            "source_sha256": document.sha256,
        })
        new_support.append({
            "claim_index": output_index,
            "original_claim_index": claim_index,
            "claim": claim,
            "context_id": context_id,
            "source_id": source_id,
            "decision": item["decision"],
            "confidence": item["confidence"],
            "rationale": item["rationale"],
            "mapping_method": "independent_semantic_source_audit",
        })

    repaired = dict(case)
    repaired["question"] = normalize_scientific_typography(str(case.get("question", "")))
    repaired["reference_answer_original"] = case.get("reference_answer", "")
    repaired["reference_claims_original"] = original_claims
    repaired["reference_claims"] = new_claims
    repaired["reference_answer"] = " ".join(new_claims)
    repaired["reference_context"] = new_contexts
    repaired["claim_support"] = new_support
    repaired["claim_support_status"] = "independently_semantically_validated"
    repaired["claim_audit"] = judgments
    repaired["benchmark_status"] = "accepted_repaired" if len(new_claims) >= min_supported_claims else "rejected_insufficient_supported_claims"
    repaired["eligible_for_benchmark"] = len(new_claims) >= min_supported_claims
    repaired["provenance"] = "silver_auto_generated_independently_audited"
    repaired["human_review"] = False
    return repaired


def source_audit_record(document: SourceDocument, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [excerpt_verification(item.get("excerpt", ""), document) for item in contexts]
    return {
        **asdict(document),
        "abstract": None,
        "sha256": document.sha256,
        "abstract_chars": len(document.abstract),
        "original_excerpt_checks": checks,
    }
