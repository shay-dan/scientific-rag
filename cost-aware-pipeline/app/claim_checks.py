"""Pure, audit-only consistency checks for structurally validated claims.

The checks compare numbers in a claim with numbers in its already validated
verbatim quotation. They do not infer entailment, scientific validity, or
evidence strength, and they never modify an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, ROUND_HALF_UP
import re
from typing import Iterable, Literal

from .weak_model_synthesis import (
    EvidenceBlock,
    EvidenceClaim,
    StructuredEvidenceAnswerV2,
    validate_answer_v2,
)


AuditStatus = Literal["consistent", "inconsistent", "unverifiable", "not_applicable"]
SignKind = Literal["none", "positive", "negative", "ambiguous"]
RoundingResult = Literal[
    "equivalent", "different", "ambiguous_tie",
    "claim_precision_exceeds_evidence", "unsupported_precision",
]
_UNSIGNED_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER = rf"[-+−–]?{_UNSIGNED_NUMBER}"
_VALID_UNSIGNED_NUMBER_RE = re.compile(rf"{_UNSIGNED_NUMBER}\Z")
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+−–]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
# Deliberately small direct-unit vocabulary. Unknown alphabetic words are not
# silently promoted to units; expanding this list requires focused examples.
_UNIT = (
    r"(?:%|percent(?:age)?|"
    r"(?:kg|mg|mcg|ug|[µμ]g|ng|g)\s*/\s*(?:day|week|month|year)s?|"
    r"mg\s*/\s*dl|mmol\s*/\s*l|"
    r"(?:mcg|ug|[µμ]g|ng)\s*/\s*ml|"
    r"kg|mg|mcg|ug|[µμ]g|ng|g|ml|dl|l|mmol|mol|mmhg|bpm|hz|"
    r"years?|months?|weeks?|days?|hours?|minutes?|seconds?|"
    r"patients?|participants?|adults?|subjects?)"
    r"(?![A-Za-zµμ/])"
)
_INTERVAL_RE = re.compile(
    rf"(?<![\w.])(?P<left>{_NUMBER})\s*(?P<separator>to|[-–—])\s*"
    rf"(?P<right>{_NUMBER})(?:\s*(?P<unit>{_UNIT}))?",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(
    rf"(?<![\w.])(?P<left>{_NUMBER})\s*(?P<separator>[:/])\s*"
    rf"(?P<right>{_NUMBER})(?:\s*(?P<unit>{_UNIT}))?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMBER})\s*(?P<unit>%|percent(?:age)?)",
    re.IGNORECASE,
)
_SCALAR_RE = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMBER})(?:\s*(?P<unit>{_UNIT}))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _Quantity:
    form: Literal["scalar", "percentage", "interval", "fraction", "ratio"]
    values: tuple[Decimal, ...]
    precision: tuple[int, ...]
    unit: str | None
    signs: tuple[SignKind, ...]
    raw_signature: str

    def diagnostic(self) -> dict:
        return {
            "form": self.form,
            "normalized": _normalized_values(self.form, self.values),
            "unit": self.unit,
            "precision": list(self.precision),
            "sign": self.signs[0] if len(self.signs) == 1 else list(self.signs),
        }


def _decimal(raw: str) -> Decimal:
    normalized = raw.replace("−", "-")
    if normalized.startswith("–"):
        normalized = normalized[1:]
    try:
        return Decimal(normalized.replace(",", ""))
    except InvalidOperation as error:  # pragma: no cover - guarded by regex
        raise ValueError(f"Invalid numeric token: {raw!r}") from error


def _precision(raw: str) -> int:
    normalized = raw.lstrip("-+−–").replace(",", "")
    return len(normalized.rsplit(".", 1)[1]) if "." in normalized else 0


def _sign(raw: str) -> SignKind:
    if raw.startswith("+"):
        return "positive"
    if raw.startswith(("-", "−")):
        return "negative"
    if raw.startswith("–"):
        return "ambiguous"
    return "none"


def _normalize_unit(raw: str | None, *, percentage: bool = False) -> str | None:
    if percentage:
        return "%"
    if raw is None:
        return None
    unit = raw.casefold().replace("µ", "u").replace("μ", "u")
    unit = re.sub(r"\s*/\s*", "/", unit)
    if unit in {"%", "percent", "percentage"}:
        return "%"
    singular = {
        "years": "year", "months": "month", "weeks": "week", "days": "day",
        "hours": "hour", "minutes": "minute", "seconds": "second",
        "patients": "patient", "participants": "participant", "adults": "adult",
        "subjects": "subject",
    }
    if "/" in unit:
        numerator, denominator = unit.split("/", 1)
        return f"{numerator}/{singular.get(denominator, denominator)}"
    return singular.get(unit, unit)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0", ""} else text


def _normalized_values(form: str, values: tuple[Decimal, ...]) -> str:
    rendered = [_decimal_text(value) for value in values]
    if form == "interval":
        return "..".join(rendered)
    if form == "ratio":
        return ":".join(rendered)
    if form == "fraction":
        return "/".join(rendered)
    return rendered[0]


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _has_unsupported_numeric_format(text: str) -> bool:
    for match in _NUMERIC_TOKEN_RE.finditer(text):
        raw = match.group(0)
        unsigned = raw.lstrip("-+−–")
        if "e" in unsigned.casefold() or _VALID_UNSIGNED_NUMBER_RE.fullmatch(unsigned) is None:
            return True
    return False


def _is_hyphenated_identifier(text: str, span: tuple[int, int]) -> bool:
    start, _ = span
    return (
        start >= 2
        and text[start - 1] in "-–—"
        and (text[start - 2].isalnum() or text[start - 2] == "_")
    )


def _extract_quantities(text: str) -> tuple[tuple[_Quantity, ...], bool]:
    if _has_unsupported_numeric_format(text):
        return (), True
    found: list[tuple[int, _Quantity]] = []
    occupied: list[tuple[int, int]] = []

    for pattern, form in ((_INTERVAL_RE, "interval"), (_RATIO_RE, "ratio")):
        for match in pattern.finditer(text):
            if _overlaps(match.span(), occupied) or _is_hyphenated_identifier(text, match.span()):
                continue
            left = match.group("left")
            right = match.group("right")
            separator = match.group("separator")
            actual_form = "fraction" if form == "ratio" and separator == "/" else form
            unit = _normalize_unit(match.group("unit"))
            found.append((match.start(), _Quantity(
                actual_form,
                (_decimal(left), _decimal(right)),
                (_precision(left), _precision(right)),
                unit,
                (_sign(left), _sign(right)),
                re.sub(r"\s+", "", match.group(0)).casefold(),
            )))
            occupied.append(match.span())

    for match in _PERCENT_RE.finditer(text):
        if _overlaps(match.span(), occupied) or _is_hyphenated_identifier(text, match.span()):
            continue
        raw = match.group("value")
        found.append((match.start(), _Quantity(
            "percentage", (_decimal(raw),), (_precision(raw),), "%", (_sign(raw),),
            re.sub(r"\s+", "", match.group(0)).casefold(),
        )))
        occupied.append(match.span())

    for match in _SCALAR_RE.finditer(text):
        if _overlaps(match.span(), occupied) or _is_hyphenated_identifier(text, match.span()):
            continue
        raw = match.group("value")
        found.append((match.start(), _Quantity(
            "scalar", (_decimal(raw),), (_precision(raw),),
            _normalize_unit(match.group("unit")),
            (_sign(raw),),
            re.sub(r"\s+", "", match.group(0)).casefold(),
        )))
        occupied.append(match.span())

    return tuple(item for _, item in sorted(found, key=lambda item: item[0])), False


def _rounded_equivalent(left: _Quantity, right: _Quantity) -> RoundingResult:
    try:
        for left_value, right_value, left_precision, right_precision in zip(
            left.values, right.values, left.precision, right.precision, strict=True
        ):
            if left_precision > right_precision:
                return "claim_precision_exceeds_evidence"
            if left_precision == right_precision:
                if left_value != right_value:
                    return "different"
                continue
            quantum = Decimal(1).scaleb(-left_precision)
            half_up = right_value.quantize(quantum, rounding=ROUND_HALF_UP)
            half_even = right_value.quantize(quantum, rounding=ROUND_HALF_EVEN)
            if half_up != half_even:
                return "ambiguous_tie"
            if half_up != left_value:
                return "different"
    except InvalidOperation:
        # Optional diagnostics must not fail research when a numeric token is
        # beyond the active Decimal context's safely supported precision.
        return "unsupported_precision"
    return "equivalent"


def _finding(
    target: str,
    claim: EvidenceClaim,
    status: AuditStatus,
    reason_code: str,
    claim_values: tuple[_Quantity, ...],
    evidence_values: tuple[_Quantity, ...],
) -> dict:
    return {
        "target": target,
        "source_id": claim.source_id,
        "category": "quotation_numeric_consistency",
        "status": status,
        "reason_code": reason_code,
        "claim_values": [item.diagnostic() for item in claim_values],
        "evidence_values": [item.diagnostic() for item in evidence_values],
    }


def _audit_field(target: str, claim: EvidenceClaim) -> dict:
    claim_values, claim_format_error = _extract_quantities(claim.text)
    evidence_values, evidence_format_error = _extract_quantities(claim.evidence_quote)
    if claim_format_error or evidence_format_error:
        return _finding(
            target, claim, "unverifiable", "ambiguous_numeric_format",
            claim_values, evidence_values,
        )
    if not claim_values:
        if evidence_values:
            return _finding(
                target, claim, "unverifiable", "missing_numeric_claim",
                claim_values, evidence_values,
            )
        return _finding(
            target, claim, "not_applicable", "no_numeric_claim", claim_values, evidence_values
        )
    if not evidence_values:
        return _finding(
            target, claim, "unverifiable", "missing_numeric_evidence", claim_values, evidence_values
        )

    claim_forms = tuple(item.form for item in claim_values)
    evidence_forms = tuple(item.form for item in evidence_values)
    if len(claim_values) != len(evidence_values):
        reason = (
            "ambiguous_numeric_alignment"
            if set(claim_forms) == set(evidence_forms)
            else "incomparable_numeric_forms"
        )
        return _finding(
            target, claim, "unverifiable", reason,
            claim_values, evidence_values,
        )
    if claim_forms != evidence_forms:
        return _finding(
            target, claim, "unverifiable", "incomparable_numeric_forms",
            claim_values, evidence_values,
        )
    if len(claim_values) > 1:
        return _finding(
            target, claim, "unverifiable", "ambiguous_numeric_alignment",
            claim_values, evidence_values,
        )
    if any(left.unit != right.unit for left, right in zip(claim_values, evidence_values, strict=True)):
        reason = "missing_unit_context" if any(
            (left.unit is None) != (right.unit is None)
            for left, right in zip(claim_values, evidence_values, strict=True)
        ) else "incompatible_units"
        return _finding(target, claim, "unverifiable", reason, claim_values, evidence_values)

    if any(
        item.form in {"scalar", "interval"} and item.unit is None
        for item in (*claim_values, *evidence_values)
    ):
        return _finding(
            target, claim, "unverifiable", "missing_unit_context",
            claim_values, evidence_values,
        )

    pairs = tuple(zip(claim_values, evidence_values, strict=True))
    if any(sign == "ambiguous" for pair in pairs for item in pair for sign in item.signs):
        return _finding(
            target, claim, "unverifiable", "ambiguous_sign", claim_values, evidence_values
        )
    if any(
        (left_sign == "none") != (right_sign == "none")
        for left, right in pairs
        for left_sign, right_sign in zip(left.signs, right.signs, strict=True)
    ):
        return _finding(
            target, claim, "unverifiable", "implicit_sign_direction",
            claim_values, evidence_values,
        )
    if any(
        left_value.copy_abs() == right_value.copy_abs() and left_value != right_value
        for left, right in pairs
        for left_value, right_value in zip(left.values, right.values, strict=True)
    ):
        return _finding(
            target, claim, "inconsistent", "sign_mismatch", claim_values, evidence_values
        )
    if all(left.values == right.values for left, right in pairs):
        reason = (
            "exact_numeric_match"
            if all(left.raw_signature == right.raw_signature for left, right in pairs)
            else "reformatted_numeric_match"
        )
        return _finding(target, claim, "consistent", reason, claim_values, evidence_values)
    rounded_results = tuple(_rounded_equivalent(left, right) for left, right in pairs)
    if "unsupported_precision" in rounded_results:
        return _finding(
            target, claim, "unverifiable", "numeric_precision_unsupported",
            claim_values, evidence_values,
        )
    if "claim_precision_exceeds_evidence" in rounded_results:
        return _finding(
            target, claim, "unverifiable", "claim_precision_exceeds_evidence",
            claim_values, evidence_values,
        )
    if "ambiguous_tie" in rounded_results:
        return _finding(
            target, claim, "unverifiable", "ambiguous_rounding_tie",
            claim_values, evidence_values,
        )
    if all(result == "equivalent" for result in rounded_results):
        return _finding(
            target, claim, "consistent", "rounded_numeric_match", claim_values, evidence_values
        )
    return _finding(
        target, claim, "inconsistent", "numeric_mismatch", claim_values, evidence_values
    )


def audit_answer_claims(
    answer: StructuredEvidenceAnswerV2,
    evidence_blocks: Iterable[EvidenceBlock],
) -> dict:
    """Return deterministic diagnostics without modifying ``answer``.

    Structural validation is repeated as a fail-closed precondition. This also
    preserves the existing R3 rejection of quotations that overlap embedded
    instruction-like text.
    """
    blocks = tuple(evidence_blocks)
    structural_errors = validate_answer_v2(answer, blocks)
    if structural_errors:
        raise ValueError("Claim audit requires a structurally validated answer")

    findings = [
        _audit_field(f"{section}.{index}", claim)
        for section in ("claims", "limitations")
        for index, claim in enumerate(getattr(answer, section), start=1)
    ]
    summary = {
        status: sum(item["status"] == status for item in findings)
        for status in ("consistent", "inconsistent", "unverifiable", "not_applicable")
    }
    return {
        "mode": "audit",
        "status": "completed",
        "taxonomy_version": "numeric-quote-v1",
        "target_scope": "final_selected_answer",
        "findings": findings,
        "summary": summary,
    }
