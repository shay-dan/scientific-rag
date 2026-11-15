# R7B deterministic claim audit

R7B adds an opt-in, deterministic consistency audit for structurally validated
scientific claims. It is diagnostic only: findings never change answer text, remove
claims, alter semantic-verifier verdicts, or trigger provider calls.

## Configuration and placement

Set `CLAIM_AUDIT_MODE=off|audit`. The production default is `off`. The audit runs after
the existing structural validation, optional R7A repair, and any semantic filtering,
and remains separate from semantic-verification metadata. Strict and degrade recovery
modes and semantic `off|shadow|enforce` behavior are unchanged.

For a fresh audited run, internal metadata records `claim_audit_mode: audit` and the
claim-validation record includes `deterministic_audit`. The audit report has this stable
shape:

```json
{
  "mode": "audit",
  "status": "completed",
  "taxonomy_version": "numeric-quote-v1",
  "target_scope": "final_selected_answer",
  "findings": [
    {
      "target": "claims.1",
      "source_id": "S1",
      "category": "quotation_numeric_consistency",
      "status": "inconsistent",
      "reason_code": "numeric_mismatch",
      "claim_values": [{"form": "percentage", "normalized": "31", "unit": "%", "precision": [0]}],
      "evidence_values": [{"form": "percentage", "normalized": "13", "unit": "%", "precision": [0]}]
    }
  ],
  "summary": {"consistent": 0, "inconsistent": 1, "unverifiable": 0, "not_applicable": 0}
}
```

Targets use the one-based `claims.N` and `limitations.N` positions in the final selected
answer stored alongside `claim_validation.claims` and `claim_validation.limitations`.
`target_scope: final_selected_answer` makes this contract explicit. In semantic enforce
mode, removed claims receive no deterministic finding and surviving claims are reindexed
to match those final lists. Reports contain normalized numeric diagnostics, source IDs,
and stable reason codes, not full source passages or quotations.

## Status and reason taxonomy

Statuses deliberately avoid the semantic verdict `supported`:

- `consistent`: the claim and its validated quotation have directly comparable values.
- `inconsistent`: directly comparable values differ, including a lost or reversed sign.
- `unverifiable`: a safe deterministic comparison cannot be made.
- `not_applicable`: the claim contains no recognized numeric quantity.

Stable reason codes in `numeric-quote-v1` are:

- Consistent: `exact_numeric_match`, `reformatted_numeric_match`,
  `rounded_numeric_match`.
- Inconsistent: `sign_mismatch`, `numeric_mismatch`.
- Unverifiable: `incomparable_numeric_forms`, `ambiguous_numeric_alignment`,
  `incompatible_units`, `missing_unit_context`, `missing_numeric_evidence`,
  `numeric_precision_unsupported`.
- Not applicable: `no_numeric_claim`.

These categories describe quotation/numeric consistency only. They do not establish
scientific validity, evidence quality, attribution, or semantic entailment.

## Conservative comparison rules

The auditor compares quantities in each claim only with quantities in that claim's
already validated verbatim evidence quotation. It does not search adjacent passage text
or infer that an unrelated number supports a measurement.

It recognizes signed integers and decimals, percentages, intervals, ratios/fractions,
and a deliberately small direct-unit vocabulary (common mass, volume, concentration,
amount, pressure, rate, and frequency tokens). Ordinary words such as `participants`
and `patients` are not treated as units. Exact values and benign reformatting such as
`5.0` versus `5.00` are consistent. Same-form rounding is consistent only when rounding
the more precise value with round-half-up to the lower stated decimal precision yields
the less precise value. Values beyond the active Decimal context's safe precision are
unverifiable rather than errors. Interval endpoints remain ordered and are compared
independently.

The auditor abstains when forms or alignment are unsafe. Decimal-to-percentage and
ratio/fraction-to-percentage comparisons are unverifiable. It never derives a percentage
from separately stated numerator and denominator values, converts units, infers a
measurement's meaning, or positionally aligns multiple separately stated quantities.
Those multi-quantity sentences are unverifiable even when their numeric sequence is
identical, because numeric equality is not semantic agreement. Missing unit
context and explicitly incompatible unit strings are separate unverifiable outcomes,
not contradictions.

Structural validation is repeated as a fail-closed precondition. This preserves the R3
source-ID, verbatim-quotation, and instruction-like-content protections; source text is
data and is never interpreted as instructions.

## Cache and queue behavior

Audit mode intentionally does not change the answer cache identity. An otherwise valid
historical answer may therefore satisfy a request made while audit mode is enabled. Such
a cached result keeps its original metadata and is never claimed to have been audited.
Use `regenerate_answer` or `refresh_sources` when a newly audited result is required.
The application does not retrofit diagnostics onto old cached answers because the
structured intermediate claims and exact runtime context are not guaranteed to be
available under the same contract.

New queued runs persist `claim_audit_mode: audit`; the persisted value wins over a
worker's current environment. A historical job with no persisted mode resolves to
`off`. Default off/strict jobs retain the historical queue argument shape. Audited jobs
require an R7B-capable worker.

## Limitations, cost, and rollback

The parser is intentionally narrow. Complex scientific notation, confidence-interval
grammar beyond simple numeric ranges, compound/dimensional units, locale-specific number
formats, and unit conversions can be unverifiable or unrecognized. Matching values can
still refer to different measurements; a consistent finding is not semantic support.
Conversely, safe transformations outside the documented rules can abstain. These
false-positive controls favor `unverifiable` over speculative equivalence or conflict.

The audit is local and linear in the claim/quotation text examined. It adds no model,
network, or provider calls and has no provider cost. To roll back behavior, set
`CLAIM_AUDIT_MODE=off`. Before deploying older workers, drain or cancel queued audited
jobs because their extended argument list is not understood by pre-R7B workers. No
database rollback is needed.
