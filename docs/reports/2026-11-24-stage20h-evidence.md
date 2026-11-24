# Stage 20H: retrieval and evidence diagnostics (in review)

Owner-authorized continuation, based on accepted main
`bc053c12916c1d52cf8745e864a4ee84b9a03035` and draft #32 head
`892d999c6c28cf6212a323fe21ab0ccc82f78dce`. Initial working tree was clean
on `work`. The follow-up is stacked on #32; local HTMX, decimal normalization,
and truthful delivery reporting are preserved. No merge, deployment, paid
inference, frozen campaign rerun, dependency edit or production routing-mode
change is authorized or performed. Stage 20H remains **in review**.

## Reproduced causes and changes

| Failure class | Evidence inspected | Change and offline scope |
|---|---|---|
| E001: cardiovascular interpretation | Foundation parses middle-dot decimals, but numeric consistency does not check statistical interpretation. | Separate audit flags explicit ratio CIs crossing 1, differences crossing 0, composite/component, relative/absolute and design mismatches. Synthetic statistics only; no re-adjudication of the live answer. |
| E002: generic search constraints | #32 already removes relationship/between and preserves its scientific concepts. Remaining fallback silently discarded unknown diseases when another concept was recognized, and truncated constraints. | Scientific dimensions/design/intent separated from required topics; unknown topics and populations retained, no four-anchor truncation. Independent psoriasis/eczema/zinc/materials fixtures. |
| E003: incomplete generalization | Topic coverage did not inspect question dimensions. A quotation can be valid while external validation remains unanswered. | Versioned passage references and an independent selected-quotation completeness assessment; missing external validation is visible in the rendered answer. No claim that it validates the original retinopathy result. |
| E004: comparison wording | #32 already removes literal trade-offs anchors. Source sufficiency still used corpus-wide keyword union. | Require a topical passage instead of unrelated-source word union; comparison outcomes assessed independently, without requiring discourse terms. |
| E005: mortality/search absence | Empty search and retrieved negative findings require different interpretations. | Missing evidence, contextual mentions, direct result candidates and source failure stay distinct. No eligible trials does not become a direct mortality result or no-benefit conclusion. |
| Query repair | Repair rebuilt the same broad query and required every old anchor. | Known generic anchors removed before searching; proposals cannot remove essential unknown disease/intervention constraints; changed detail clauses can reformulate queries; canonical query ordering suppresses duplicate transmissions. |
| Repeated screening | Repeated retrieval attempts could screen identical papers again. | In-call passage fingerprint cache reuses screening decisions and provenance, including background restrictions. It does not persist model judgments across runs. |

## Assessment contract and limitations

`passage-adequacy-v1` reports `direct_candidate`, `contextual` or `missing`
per recognized dimension, with source ID, passage digest, exact offsets and
study-design metadata. Overall statuses are `partial`, `uncertain`, `missing`
or `external_retrieval_failure`. **It never establishes full sufficiency or
semantic entailment.** Even complete candidate coverage remains uncertain.
`additional_model_calls` is zero. Results-bearing language can occur in
background or hypotheses; explicit background/future/instruction/search-absence
markers are excluded from direct candidates, but this is a bounded heuristic.

The legacy `source_sufficiency.sufficient` key remains for saved-report/API
compatibility, explicitly scoped `retrieval_preflight_only`. It admits topical
sources to synthesis; it does not certify the question answered. `cache_eligible`
also checks recognized outcome gaps so partial cached evidence triggers further
search. Diagnostics are additive dictionaries; no database migration or `/v1`
schema change. Pipeline/cache identity is bumped to
`postgres-hybrid-rrf-mmr-parent-child-structured-v15-stage20h-v1`; old records
remain readable and old answers are not reused under the new identity.

Known dimension aliases are deliberately small, shared across biomedical,
computing and materials examples. Unknown dimensions remain uncertain. Unknown
words are conservatively retained and may overconstrain searches; LLM proposals
cannot simply erase them. This trades recall for avoiding silent topic changes.
Drug-class membership is not drug equivalence: GLP-1 alone no longer satisfies a
semaglutide anchor. Broader existing concept aliases are search candidates, not
semantic equivalence or scientific support.

Numeric consistency, semantic support, interpretation warnings and completeness
are separate reports. `scientific-interpretation-v1` is audit-only; no warning
means no recognized warning, not scientific validity. Unsupported interval
formats are `unverifiable`. Existing strict structural rejection and opt-in
semantic enforcement/recovery behavior remain. Semantic verification defaults
to off; unsupported causal claims can still escape an audit-only configuration.
No enforcement default was changed. Extra statistical guidance is in the
existing opt-in v16 generation prompt and existing opt-in verifier.

Useful partial answers still require valid cited claims. Missing recognized
dimensions are rendered under "Not established by the retrieved evidence".
All recognized gaps remain in audit metadata; the existing eight-item answer
schema limit is preserved. Evidence strength and generalization still require
scientific review. These synthetic expectations are regression labels, not gold.

## Evaluation and cost

The committed `test_stage20h_evidence.py` contains 31 synthetic regression tests.
Against #32, all 31 fail (13 assertion failures, 18 missing-behavior errors).
On the follow-up they pass. A separately stacked deadline follow-up adds 11
more probes; combined baseline is 42 tests, 13 failures and 29 errors, versus
42/42 passing on the combined implementation. No live scientific performance
measurement was made. The frozen five-case results are unchanged.

A Python 3.11 production-container microbenchmark of `assess_adequacy`, eight
synthetic documents and 100 repetitions, measured median 1.762 ms, maximum
15.502 ms. This includes cold regex effects and is not an end-to-end latency
estimate. Deterministic assessment adds zero judge/model calls. Existing query
repair still has a maximum of one gateway invocation per mining call; it now
skips duplicate resulting searches. Screening is bounded by unique candidate
passages within that call. More conservative partial-cache reuse can cause
additional retrieval/screening; no live cost or recall improvement is claimed.
Existing provider-attempt caps, routing policy and reservations are preserved.

Validation commands and final suite results are recorded in the follow-up
validation report. An initial full suite on the existing Python 3.11 image
collected 1,196: 1,135 passed, 60 skipped, one error because inherited proxy
variables triggered the frozen-campaign safety guard. The guard was retained;
subsequent offline commands unset proxy variables only inside test containers.
Another run was interrupted at an existing lease-keeper cancellation fixture;
it has no complete test count and is not a passing run. No lifecycle code was
changed to make it pass.

## Trade-offs and acceptance gate

A default LLM adequacy judge was rejected: it adds provider latency, money and
another uncalibrated label. Embedding-only similarity was rejected as a proof of
outcome coverage. A new evidence schema/store/workflow was unnecessary. The
chosen additive diagnostics preserve provenance and qualify partial answers,
while clearly exposing the lexical and ontology limits.

- [x] Preserve #32 foundation and offline UI assets.
- [x] Reproduce failures before fixes with synthetic offline probes.
- [x] Exercise unseen terminology, adversarial instructions, multi-outcome and missing evidence.
- [x] Separate numeric, semantic, interpretation and completeness diagnostics.
- [x] Keep schemas/default modes and paid-call prohibition.
- [ ] Owner scientific review, calibration of unseen real questions, and gold-label acceptance.
- [ ] Owner approval for any new live experiment or merge.

Rollback: revert this PR and its dependent follow-up, restore the prior pipeline
identity, and retain historical records. No migration or cache deletion is
required. Do not mark Stage 20H complete from these tests or from green CI.
