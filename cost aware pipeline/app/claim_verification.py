"""Deterministic contracts and evidence mapping for semantic claim verification.

This module does not call an LLM and is not wired into production synthesis. It
prepares structurally validated claim/evidence pairs and validates a future
semantic verifier response without defaulting missing judgments to supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .weak_model_synthesis import (
    EvidenceBlock,
    StructuredEvidenceAnswerV2,
    _normalise,
    validate_answer_v2,
)


TARGET_PATTERN = r"^(claims|limitations)\.[1-9]\d*$"
MAX_PASSAGE_CONTEXT_CHARS = 2400
_TARGET_RE = re.compile(TARGET_PATTERN)


class ClaimVerificationError(ValueError):
    """Raised when evidence mapping or assessment validation is incomplete."""


class ClaimVerdict(str, Enum):
    supported = "supported"
    contradicted = "contradicted"
    insufficient_evidence = "insufficient_evidence"


class ClaimAssessment(BaseModel):
    """One semantic verdict for one stable answer-field target."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(pattern=TARGET_PATTERN)
    verdict: ClaimVerdict
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def reason_must_contain_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must contain non-whitespace text")
        return stripped


class ClaimVerificationBatch(BaseModel):
    """Structured response contract for one bounded verification batch."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[ClaimAssessment] = Field(default_factory=list, max_length=20)


@dataclass(frozen=True, slots=True)
class ClaimVerificationInput:
    """A claim paired with one exact passage from its cited source."""

    target: str
    claim_text: str
    source_id: str
    evidence_quote: str
    matched_source_title: str
    matched_passage_context: str


def _normalise_with_positions(text: str) -> tuple[str, tuple[int, ...]]:
    """Mirror the existing whitespace/case normalization and retain raw offsets."""
    characters: list[str] = []
    positions: list[int] = []
    whitespace_pending = False
    whitespace_position = 0

    for raw_index, character in enumerate(text):
        if character.isspace():
            if characters and not whitespace_pending:
                whitespace_pending = True
                whitespace_position = raw_index
            continue

        if whitespace_pending:
            characters.append(" ")
            positions.append(whitespace_position)
            whitespace_pending = False

        folded = character.casefold()
        characters.extend(folded)
        positions.extend([raw_index] * len(folded))

    return "".join(characters), tuple(positions)


def _quote_span(excerpt: str, quote: str) -> tuple[int, int] | None:
    normalized_excerpt, positions = _normalise_with_positions(excerpt)
    normalized_quote = _normalise(quote)
    if not normalized_quote:
        return None
    normalized_start = normalized_excerpt.find(normalized_quote)
    if normalized_start < 0:
        return None
    normalized_end = normalized_start + len(normalized_quote) - 1
    return positions[normalized_start], positions[normalized_end] + 1


def _bounded_context(excerpt: str, quote_start: int, quote_end: int) -> str:
    if len(excerpt) <= MAX_PASSAGE_CONTEXT_CHARS:
        return excerpt

    quote_length = quote_end - quote_start
    if quote_length >= MAX_PASSAGE_CONTEXT_CHARS:
        return excerpt[quote_start:quote_end]

    available = MAX_PASSAGE_CONTEXT_CHARS - quote_length
    left = min(quote_start, available // 2)
    right = min(len(excerpt) - quote_end, available - left)
    left = min(quote_start, available - right)
    context_start = quote_start - left
    context_end = quote_end + right
    return excerpt[context_start:context_end]


def _format_structural_errors(errors: dict[str, tuple[str, ...]]) -> str:
    return "; ".join(
        f"{target}: {', '.join(target_errors)}"
        for target, target_errors in sorted(errors.items(), key=lambda item: _target_sort_key(item[0]))
    )


def _target_sort_key(target: str) -> tuple[int, int]:
    match = _TARGET_RE.fullmatch(target)
    if match is None:
        raise ClaimVerificationError(f"invalid target {target!r}")
    section, raw_index = target.split(".", 1)
    return (0 if section == "claims" else 1, int(raw_index))


def build_verification_inputs(
    answer: StructuredEvidenceAnswerV2,
    blocks: Iterable[EvidenceBlock],
) -> tuple[ClaimVerificationInput, ...]:
    """Map each structurally valid answer item to one quote-containing passage."""
    block_list = list(blocks)
    structural_errors = validate_answer_v2(answer, block_list)
    if structural_errors:
        raise ClaimVerificationError(
            "cannot build semantic verification inputs from a structurally invalid answer: "
            + _format_structural_errors(structural_errors)
        )

    blocks_by_source: dict[str, list[EvidenceBlock]] = {}
    for block in block_list:
        blocks_by_source.setdefault(block.source_id, []).append(block)

    mapped: list[ClaimVerificationInput] = []
    for section, claims in (("claims", answer.claims), ("limitations", answer.limitations)):
        for index, claim in enumerate(claims, start=1):
            target = f"{section}.{index}"
            candidates = blocks_by_source.get(claim.source_id, [])
            matched_block: EvidenceBlock | None = None
            matched_span: tuple[int, int] | None = None
            for block in candidates:
                span = _quote_span(block.excerpt, claim.evidence_quote)
                if span is not None:
                    matched_block = block
                    matched_span = span
                    break
            if matched_block is None or matched_span is None:
                # validate_answer_v2 should already have caught this. Keep a
                # fail-closed guard so future contract changes cannot mis-map it.
                raise ClaimVerificationError(
                    f"{target}: no passage in {claim.source_id} contains the evidence quote"
                )
            mapped.append(ClaimVerificationInput(
                target=target,
                claim_text=claim.text,
                source_id=claim.source_id,
                evidence_quote=claim.evidence_quote,
                matched_source_title=matched_block.title,
                matched_passage_context=_bounded_context(
                    matched_block.excerpt, matched_span[0], matched_span[1]
                ),
            ))

    return tuple(mapped)


def validate_assessment_batch(
    batch: ClaimVerificationBatch | dict,
    expected_targets: Iterable[str],
) -> tuple[ClaimAssessment, ...]:
    """Require exactly one valid assessment for every expected target.

    An empty expected-target collection is valid only with an empty response and
    returns an empty tuple. Any malformed, duplicate, missing, or unknown target
    raises ClaimVerificationError.
    """
    expected = tuple(expected_targets)
    invalid_expected = [target for target in expected if _TARGET_RE.fullmatch(target) is None]
    if invalid_expected:
        raise ClaimVerificationError(
            f"invalid expected target(s): {', '.join(repr(target) for target in invalid_expected)}"
        )
    if len(expected) != len(set(expected)):
        raise ClaimVerificationError("expected targets contain duplicates")

    try:
        parsed = batch if isinstance(batch, ClaimVerificationBatch) else ClaimVerificationBatch.model_validate(batch)
    except ValidationError as error:
        raise ClaimVerificationError(f"invalid assessment batch: {error}") from error

    submitted = [assessment.target for assessment in parsed.assessments]
    duplicates = sorted({target for target in submitted if submitted.count(target) > 1}, key=_target_sort_key)
    if duplicates:
        raise ClaimVerificationError(f"duplicate targets in assessment batch: {', '.join(duplicates)}")

    expected_set = set(expected)
    submitted_set = set(submitted)
    problems: list[str] = []
    missing = sorted(expected_set - submitted_set, key=_target_sort_key)
    unknown = sorted(submitted_set - expected_set, key=_target_sort_key)
    if missing:
        problems.append(f"missing assessments: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown assessments: {', '.join(unknown)}")
    if problems:
        raise ClaimVerificationError("; ".join(problems))

    by_target = {assessment.target: assessment for assessment in parsed.assessments}
    return tuple(by_target[target] for target in sorted(expected_set, key=_target_sort_key))
