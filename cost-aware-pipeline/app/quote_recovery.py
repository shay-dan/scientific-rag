"""Source-anchored quotation repair. No inference, content rewriting or validator changes."""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from typing import Iterable, Literal

from .weak_model_synthesis import (
    EvidenceBlock, EvidenceClaim, StructuredEvidenceAnswerV2, V2RepairBatch,
    validate_answer_v2,
)

QuoteRecoveryMode = Literal["off", "unicode", "unicode_ellipsis"]
MODES = frozenset(("off", "unicode", "unicode_ellipsis"))
HYPHENS = frozenset(("-", "\u2011"))
ELLIPSIS = re.compile(r"\u2026|(?<!\.)\.{3}(?!\.)")
MIN_ANCHOR = 12
MAX_GAP = 160
MAX_EXPANSION = 600


def validate_quote_recovery_mode(mode: str, semantic_mode: str) -> None:
    if mode not in MODES:
        raise ValueError("Unsupported quotation recovery mode")
    if mode != "off" and semantic_mode != "enforce":
        raise ValueError("Quotation recovery requires semantic verification in enforce mode")


def _pattern(text: str, *, hyphens: bool = False) -> str:
    """Literal, case-sensitive anchors; tolerate existing whitespace equivalence.

    In word compounds and number-unit compounds only, permit the two tested
    hyphen code points. Do not equate mathematical minus, a leading sign,
    letter-digit exponent/identifier separators, or numeric range separators.
    """
    pieces = []
    for index, char in enumerate(text):
        if char.isspace():
            if index == 0 or not text[index - 1].isspace():
                pieces.append(r"\s+")
        elif (hyphens and char in HYPHENS and 0 < index < len(text) - 1
              and text[index - 1].isalnum() and text[index + 1].isalpha()):
            pieces.append(r"[-\u2011]")
        else:
            pieces.append(re.escape(char))
    return "".join(pieces)


def _positions(pattern: str, excerpt: str):
    # Lookahead includes overlapping literal anchors. Anchors are stripped;
    # internal whitespace runs terminate at their next literal character.
    for match in re.finditer("(?=(" + pattern + "))", excerpt):
        yield match.start(1), match.end(1)


def _unique_span(quote: str, excerpts: list[str], method: str) -> tuple[str | None, int]:
    """Return a unique raw source span, with match count capped at two.

    Two denotes ambiguity (two or more), including repeated identical strings
    in different positions/passage blocks. No cross-block concatenation.
    """
    quote = quote.strip()
    spans = []
    if method == "unicode":
        if not any(c in HYPHENS for c in quote):
            return None, 0
        pattern = _pattern(quote, hyphens=True)
        for excerpt in excerpts:
            for start, end in _positions(pattern, excerpt):
                spans.append(excerpt[start:end])
                if len(spans) > 1:
                    return None, 2
        if len(spans) != 1:
            return None, len(spans)
        candidate = spans[0]
        # Require a genuine allowed-hyphen substitution, not whitespace-only
        # matching or broad punctuation normalization.
        original = re.sub(r"\s+", " ", quote)
        located = re.sub(r"\s+", " ", candidate)
        if (original == located or len(original) != len(located)
                or any(a != b and {a, b} != HYPHENS
                       for a, b in zip(original, located))):
            return None, 1
        return (candidate if 8 <= len(candidate) <= 1600 else None), 1

    if method != "ellipsis":
        raise ValueError("Unsupported quotation recovery method")
    anchors = ELLIPSIS.split(quote)
    if len(anchors) != 2:
        return None, 0
    left, right = (anchor.strip() for anchor in anchors)
    if min(len(left), len(right)) < MIN_ANCHOR:
        return None, 0
    for excerpt in excerpts:
        right_positions = list(_positions(_pattern(right), excerpt))
        starts = [position[0] for position in right_positions]
        for start, left_end in _positions(_pattern(left), excerpt):
            # Explicitly enumerate every eligible endpoint, not the first regex match.
            # Bound the search to the allowed gap, even for highly repetitive text.
            lo = bisect_left(starts, left_end + 1)
            hi = bisect_right(starts, left_end + MAX_GAP)
            for index in range(lo, hi):
                _, end = right_positions[index]
                spans.append(excerpt[start:end])
                if len(spans) > 1:
                    return None, 2
    if len(spans) != 1:
        return None, len(spans)
    candidate = spans[0]
    if candidate == quote or not 8 <= len(candidate) <= MAX_EXPANSION:
        return None, 1
    return candidate, 1


def _eligible(errors: Iterable[str]) -> bool:
    values = tuple(errors)
    return bool(values) and all(
        error.startswith("evidence_quote is not a verbatim substring of ")
        for error in values
    )


def _recover_field(
    claim: EvidenceClaim, blocks: list[EvidenceBlock], mode: QuoteRecoveryMode,
    target: str, audit: list[dict],
) -> EvidenceClaim:
    excerpts = [block.excerpt for block in blocks if block.source_id == claim.source_id]
    methods = ("unicode", "ellipsis") if mode == "unicode_ellipsis" else ("unicode",)
    for method in methods:
        candidate, count = _unique_span(claim.evidence_quote, excerpts, method)
        event = {"target": target, "source_id": claim.source_id, "method": method,
                 "match_count_capped_at_two": count, "status": "ambiguous" if count > 1 else "not_recovered"}
        if candidate is not None:
            replacement = EvidenceClaim.model_validate({
                **claim.model_dump(), "evidence_quote": candidate,
            })
            # Do not bypass the existing instruction-like quotation guard.
            if (any(candidate in excerpt for excerpt in excerpts)
                    and not validate_answer_v2(
                        StructuredEvidenceAnswerV2(claims=[replacement]), blocks
                    )):
                event["status"] = "recovered"
                event["exact_source_match"] = True
                audit.append(event)
                return replacement
        audit.append(event)
        # A detected ambiguity or an unsafe unique candidate may not be retried
        # under a different equivalence rule.
        if count > 1 or candidate is not None:
            return claim
    return claim


def recover_answer_quotes(
    answer: StructuredEvidenceAnswerV2, blocks: Iterable[EvidenceBlock],
    mode: QuoteRecoveryMode,
) -> tuple[StructuredEvidenceAnswerV2, list[dict], dict[str, tuple[str, ...]]]:
    """Repair only fields with quotation-substring errors; full revalidation."""
    if mode not in MODES:
        raise ValueError("Unsupported quotation recovery mode")
    blocks = list(blocks)
    errors = validate_answer_v2(answer, blocks)
    if mode == "off" or not errors:
        return answer, [], errors
    data = answer.model_dump()
    audit: list[dict] = []
    for target, field_errors in errors.items():
        if not _eligible(field_errors):
            continue
        section, index = target.split(".")
        idx = int(index) - 1
        claim = getattr(answer, section)[idx]
        updated = _recover_field(claim, blocks, mode, target, audit)
        data[section][idx] = updated.model_dump()
    result = StructuredEvidenceAnswerV2.model_validate(data)
    remaining = validate_answer_v2(result, blocks)
    if any(target not in errors or any(e not in errors[target] for e in values)
           for target, values in remaining.items()):
        raise RuntimeError("Quotation recovery introduced a structural validation error")
    return result, audit, remaining


def recover_repair_quotes(
    batch: V2RepairBatch, blocks: Iterable[EvidenceBlock],
    mode: QuoteRecoveryMode, expected_targets: Iterable[str],
) -> tuple[V2RepairBatch, list[dict]]:
    """Correct eligible replacements only; never change repair target coverage.

    The caller must merge and revalidate the complete answer using the existing
    strict or degrade policy. Do not repair unrelated or duplicate batch items.
    """
    if mode not in MODES:
        raise ValueError("Unsupported quotation recovery mode")
    if mode == "off":
        return batch, []
    blocks = list(blocks)
    expected = set(expected_targets)
    seen: set[str] = set()
    data = batch.model_dump()
    audit: list[dict] = []
    for index, repair in enumerate(batch.repairs):
        if repair.target in seen or repair.target not in expected:
            continue
        seen.add(repair.target)
        if repair.action != "replace" or repair.replacement is None:
            continue
        errors = validate_answer_v2(
            StructuredEvidenceAnswerV2(claims=[repair.replacement]), blocks
        ).get("claims.1", ())
        if _eligible(errors):
            replacement = _recover_field(
                repair.replacement, blocks, mode, repair.target, audit
            )
            data["repairs"][index]["replacement"] = replacement.model_dump()
    return V2RepairBatch.model_validate(data), audit
