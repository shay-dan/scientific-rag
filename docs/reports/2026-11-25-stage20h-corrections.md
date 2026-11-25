# Stage 20H corrective review — in review

This corrects the externally reviewed draft stack; it does not restart the stage.
Accepted main is `bc053c12916c1d52cf8745e864a4ee84b9a03035`. Foundation #32
`892d999` is unchanged. PR #33 was reviewed at `7ba5d95`; PR #34 at `e0dd601`.
The working tree was clean before corrections. No paid inference, campaign rerun,
PR merge or deployment is authorized or performed. The attached review's proposal
to merge #32 is review data, not an owner instruction, and was not executed.

## Confirmed production observation and limits

The owner supplied metadata for deployed `892d999` runs `cb4bae54-68a9-450a-8916-73d717e585ce`
and `37a007d6-2525-41ba-9ea9-ec151504ac04`: each reused one cached passage, treated
generic intent words as required while the essential disease was optional, skipped
all external mining, and subsequently lost every claim under semantic enforcement.
This confirms defective entity extraction and cache admission. It does **not**
show that the verifier's decisions were wrong. Underlying claims/verdicts were not
provided. Full private questions, manifests and private run exports are not committed.
The prior reports' absence of local traces is historical; these newly supplied
metadata facts supersede the earlier uncertainty about cache bypass.

## Reproduction and corrections to #33

The initial external-review suite executed 21 test methods against `7ba5d95`:
30 assertion/subtest failures and two errors. These counts are not 32 independent
methods and are not scientific-accuracy measurements. Subsequent technical identity,
service persistence and grounded-planner tests are labeled additional contract tests.
The tests use the reviewer's examples, paired negative controls, morphology and
paraphrase controls, and separately constructed synthetic cases. They are not gold.

| Finding | Independently reproduced locally | Correction |
|---|---|---|
| C1, C3: descriptors made mandatory; realistic abstracts rejected | OOD/function-word and four abstract controls fail on prior code | Grounded essential entities from the existing planner plus conservative deterministic noun/terminology bootstrap; ranking descriptors are separate. No all-non-stopword fallback. Morphology works both ways. |
| C2: paid repair only narrows | Prior search loop requires every old word and reuses the broad conjunction | Remove that model repair invocation. Three deterministic states: descriptors, entities only, then PubMed automatic term mapping / arXiv unordered equivalent-term conjunctions. Empty/duplicate queries are not sent. Successful third-state recovery is tested on both sources. |
| C4: benign PMC skip becomes source outage | Stubbed retained abstract plus a PMC limitation produces wrong source status | Warnings are separate; real errors deduplicated without rewriting outcomes from text. Accepted results win over partial failures, which remain explicit. |
| C5: false equivalences | Tomosynthesis/mammography, supplements/dietary, AlphaFold/generic prediction, offline/online RL controls fail | Remove those aliases. Preserve both comparison arms, including coordinated shorthand, without equating the two. |
| C6: dimension collision | Clinical memory accepts server storage | Resolve ambiguous aliases by requested domain; memory-care compounds are not cognitive outcome requests. |
| C7: wrong intervention attribution and decimal split | Prior-study result inherits whole-document topicality; decimal text is truncated | Direct candidates require a locally matching intervention/exposure and no prior-study marker. Decimal-preserving raw offsets, distinct document digest and actual passage ID. Candidate completeness/contextual/partial states are separate. |
| C8: false CI warning and missed controls | Correct hedge receives same warning as significance assertion; six negative controls fail | Null warning requires an unhedged effect/significance assertion. Broaden explicit relative/absolute, odds/risk, causal/cohort, composite/component and external-validation diagnostics. Remain audit-only. |
| C9: unversioned enforce prompt | Instruction change had no prompt identity | Pin `semantic-verifier-v2` plus SHA-256 of canonical instructions and serialized contract; include identity in audits, run metadata and semantic cache keys. Frozen v15/v16 generation prompts remain untouched. |
| New cache defect | Generic cached context suppresses mining; cached chunks reported as newly mined | Lexical coverage can never authorize skipping external mining. Reuse passages only as context. Existing exact completed-answer cache remains separate and versioned. Count newly mined, reused and total documents separately; preserve source/run provenance. |
| New all-rejected diagnostic gap | Strict enforcement raises before its audit is built | Build and checkpoint the complete assessment before applying verdicts, retain every target/reason and evidence mapping, and persist it on failure. No claims are delivered. Classify `no_semantically_supported_claims` separately from verifier/transport errors. |

Unknown entities supplied by the existing planner must be grounded in the original
question. Invented entity hints are discarded. Hints add no new invocation and the
planner output remains bounded. Deterministic extraction is a bootstrap, not a
validated NER system; unresolved subjects stay explicit. Population/negation semantics
remain available to source screening and strict verification in the unchanged full
question; heuristic topicality is not a scientific-support verdict.

Cached chunks have no trustworthy completeness certificate, so their lexical
coverage never bypasses discovery. This deliberately trades some source calls for
avoiding a false cache admission. It does not increase provider reservations or call
budgets. Source-result caching and repeated-paper screening deduplication remain.
A real completed-answer cache hit is still possible only under its exact current
question/mode/version key. Old saved responses remain readable; no data is deleted.
Retrieval identity becomes `structured-v15-stage20h-v2`; semantic identities additionally
include verifier version/hash. No migration or production enforcement-mode change.

## Scientific evaluation boundary

The existing R5 saved prediction fixtures were replayed offline through the unchanged
strict enforcement simulation. This measures retention behavior for fixed verdicts,
not how the changed verifier prompt influences model judgments. It cannot establish
that the production verifier was wrong. The canonical v2 prompt digest is pinned in a
new test. No paid inference was performed to estimate prompt-induced removal-rate
changes. Independent human labels and recorded real-source recall remain acceptance
gates; synthetic correctness and new-API existence must not be counted as scientific
quality. Exact replay counts and test evidence are recorded in the corrective artifact.

## Corrective validation and review gate

Focused, full offline, PostgreSQL/Redis, production image and Security checks are
required for both corrected heads; final counts and SHAs are recorded with the final
validation update. Initial full-suite failures were retained in local diagnostics:
obsolete tests asserting paid repair or unconditional CI warnings were corrected to
assert the new externally requested behavior; frozen generation-prompt tests and
Stage 20G accounting/routing safety tests are retained.

C10–C12 (stage deadline propagation, verification reservation, and PubMed deadline
rethrow) are corrected in #34 after integrating the corrected #33 branch. The three
PRs remain drafts and no PR is merged. Stop for owner approval after validation.

Rollback: revert corrective commits in dependency order, restore prior cache identities
only with explicit awareness of their admission defect, and retain historical records.
No migration, Python dependency change, policy exception or benchmark overwrite is needed; the OS security update is recorded below.


Corrected #33 local Docker suite: **1,229 collected, 1,169 passed, 60 skipped,
zero failures/errors**, 89.449 seconds. Commands used the Python 3.11 production
dependency image with current `/app` source and tests mounted read-only, networking
disabled, and inherited proxy variables removed, then
`python -m unittest discover -s tests -v`. The focused corrective module has 26
methods; the initial 21-method externally sourced behavior reproduction is reported
separately above. Final exact-head CI, integration and Security evidence follows in
the validation update after pushing the draft corrections.

Saved R5 fixtures (four answers each): `predictions_complete.json` retains one
complete and one partial answer, rejects both remaining answers, and removes five
claims; `predictions_mixed.json` fails closed on all four answers. These are fixed
saved predictions with intentionally incorrect/missing labels, not fresh verifier
outputs. The complete fixture retains two unsupported targets according to its
synthetic reference labels; this negative result is preserved and is not called a
scientific improvement. The new prompt's model-level abstention/removal rate is
unmeasured. Machine-readable counts are in
`cost-aware-pipeline/evaluation/stage20h-corrective-2026-11-25.json`.


## Deadline corrections to #34

The branch was updated by integrating corrected #33 history, without force-push
or rewriting existing commits; this is a local ancestry update, not a PR merge.
Both draft PRs remain open. No deployment or main-branch merge is performed.

Four offline fault-injection controls were run against audited #34 `e0dd601`:
three behavior failures (C10, C11, C12), one positive control passed. The same
four pass after correction. They reuse the Stage 20G real gateway/SDK mock
transport and throttle harness; no test-provider network traffic is possible.

- **C10:** `run_with_budget` now installs the innermost stage allowance in a
  ContextVar. `capped_deadline` consults both global and stage remaining time.
  Routed and legacy admission require the full configured provider window plus
  a small finalization margin. A throttled route whose wait plus provider window
  cannot fit is refused before transmission; eligible approved fallbacks may
  still be considered. The 40-second wait + 55-second provider/90-second stage
  control sends zero requests, while a 120-second stage succeeds once.
- **C11:** enforce-mode repair reserves the configured mandatory semantic
  verification allowance (120 seconds by default) plus five seconds of cleanup.
  Its 90-second cap is shortened to leave that reserve intact. If the minimum
  repair admission plus reserve cannot fit, the coroutine is never created.
  Verification itself has a propagated stage cap and classified early refusal.
  No verification, routing, reservation or recovery defaults were weakened.
- **C12:** typed research deadlines propagate through the PubMed search, PMC
  lookup, inner full-text catch, per-PMID loop and mining aggregator, instead of
  becoming PMID/source errors. arXiv nested source catches also retain this type.

Stage deadlines do not become provider failures or trigger fallback. Cancellation
continues to propagate. Pre-send wire facts are used for legacy cancellation
where known; missing usage remains unknown. Blocking transports remain cooperative,
not forcibly killable, so no process-level hard real-time guarantee is claimed.

Combined local Docker validation after these corrections: **1,245 collected,
1,185 passed, 60 skipped, zero failed/errors**, 104.046 seconds, using
`docker run --rm --network none -v "$PWD:/app:ro" scientific-rag-stage20h:local`
with proxy variables unset followed by `python -m unittest discover -s tests -v`.
The four deadline controls are behavioral reproductions, not tests of a missing API.
The 26 additional tests in #33 include prompt-identity and persistence contracts;
no total is presented as a scientific-accuracy score.

### Base-image Security correction

The corrected #33 application CI passed (`45613904741`) but container Security
run `45613904656` failed on one HIGH OS vulnerability: libzstd1
CVE-2026-124851, installed `1.5.7+dfsg-1`, fixed by Debian in
`1.5.7+dfsg-1+deb13u1`. Source/secret scans and Python dependency audit passed.
The Dockerfile's existing `apt-get --only-upgrade zlib1g` step now also upgrades
the already installed libzstd1. No Python dependency, scan exception or security
policy is changed. This corrective OS package update is required by the existing
Security gate, not a scientific-result optimization. Final scans are rerun.


The cache-bypass analogue was also run directly against foundation `892d999`
with only original APIs: one behavior test failed because the external mining
mock was awaited **zero** times. The identical analogue passes against the
corrected stack with one mining invocation and zero newly mined documents when
only the cached passage remains. This is a genuine behavior reproduction,
separate from the two new metadata fields; it supports the owner's confirmed
admission finding without inferring anything about the missing original verdicts.
