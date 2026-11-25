# Stage 20H final P1 corrections — in review

Starting checkout: `codex/stage20h-final-p1`, clean at
`387177965ecbe25016ec334ef9eccc611499baeb` (PR #33). Remote `main` is
`bc053c12916c1d52cf8745e864a4ee84b9a03035`; foundation #32 remains
`892d999c6c28cf6212a323fe21ab0ccc82f78dce`; #34 starts at
`5662f7742bdb7d1c3dcb925d0cbe0cd9e5e412cd`. These refs, draft states and
dependency bases were verified remotely. Both starting follow-up heads had
successful CI and Security workflows. The task branch preserves PR #33 ancestry.

## Reproduction before implementation

Commit `fd34263` fixes the independently specified regression expectations before
source changes. Its 16 methods produced **26 assertion/subtest failures, zero
errors** against the original #33 Python 3.11 production dependency image.
These are synthetic behavioral controls, not independent scientific gold labels
or a measured real-source recall benchmark. Additional answer/admission contracts
are identified separately in the test module and are not called precommitted holdout.

| Issue | Minimal observed failure | Correction |
|---|---|---|
| P1-A | Lacosamide in Dravet syndrome/adolescents loses lacosamide; aspirin in pregnant women with migraine loses the population and disease; brivaracetam versus perampanel loses both arms | Use all grounded planner hints, explicit planner instructions for roles/all arms, syntactic subject/comparison spans and every nested population/disease relation. Keep unresolved essential subjects explicit. No added model calls or drug-suffix expansion. |
| P1-B | Supplemental vitamin D / recurrent C. difficile abstracts fail their questions' wording; a compound population hint becomes an exact prose constraint | Preserve entity spans, separate population from disease relations, use bounded biomedical/morphological aliases and question-grounded parenthetical abbreviations. Drug swaps and wrong-disease/population controls still fail. Material/artifact anchors remain separate. |
| P1-C | Separate rimegepant and ubrogepant studies are counted as complete pain coverage; even a one-arm result can mark the dimension complete | Require every requested arm plus an explicit comparison relation in the same result passage for a direct comparative candidate. Expose partial/indirect/direct statuses and arm coverage. Retain indirect/one-arm context for qualified synthesis, report missing entities/comparisons in selected-answer diagnostics, and never let lexical context authorize a cache bypass. |
| P1-D | A relevance-screening exception on either source sets the ladder stop label to external source unavailable | Distinguish the stop reason and repair-exhaustion state. Actual transport outages, empty results, scientific rejection, deadlines and accepted partial results remain distinct. |

The P1-D aggregate final outcome was already correctly derived from candidate
decisions on this head; the reproduced defect was the ladder stop label and its
internal failure flag. Screening-provider/schema failure stops the ladder without
another paid screening attempt: a new source query cannot repair that failure.
Ordinary rejection/empty results still use the existing bounded relaxation ladder
and screening deduplication. Existing source/provider/deadline budgets are unchanged.

## Implementation scope and safeguards

- `query_quality.py`: grounded entities, bounded fallback, roles, aliases and explicit
  resolution/comparison metadata. Search aliases are not semantic support verdicts.
- `evidence_adequacy.py`: passage/arm coverage and incomplete entity diagnostics,
  retaining raw offsets, source IDs and document/passage digests.
- `legacy_pipeline.py`: planner instructions, partial-answer coverage disclosures,
  qualified source admission and truthful screening stop classification.
- `app/services.py`: retrieval/cache identity becomes `structured-v15-stage20h-v3`.
  Existing mode-specific verifier hash and quote/recovery identities remain intact.
- `test_stage20h_final_p1.py`: independent medical/nonmedical controls, review
  regressions, negative/positive paraphrases, fabricated/missing hints, unresolved
  extraction, two/three-arm and alternatives coverage, fixed-verdict enforcement
  and source-screen failure injection. The old explicit cache-version assertion
  in `test_verifier_evaluation_equivalence.py` advances with the version.
- Roadmap and engineering handover link this new record; historical reports and
  frozen benchmark inputs/results/manifests remain intact.

Semantic-verifier instructions/hash, strict enforcement, citation checking, cached
provenance, frozen generation prompts, Stage 20G routing/accounting, provider
priorities/reservations and security rules are unchanged. No new dependency or
schema migration. No inference, live discovery/campaign, provider credentials,
production database access, merge or deployment.

## Validation

Validation uses Python **3.11.18** and the production Docker requirements constrained
by the unchanged `constraints.txt` (all nine exact pins checked). This cloud session
requires its CA bundle for dependency downloads: a temporary Dockerfile mounts it
as a BuildKit secret with TLS verification enabled. No repository Dockerfile change
or embedded session CA is needed.

Offline commands run with networking disabled, source/tests mounted read-only and
temporary writable `/tmp`. For the full suite, inherited proxy variables are removed
inside that network-disabled container so Stage 20G's existing campaign guard can be
tested. The first full diagnostic run had the obsolete version assertion, the
subsequently corrected materials regression, and this proxy-guard setup error;
none was suppressed or reclassified as a passing test.

Commands (inside the production image):

```sh
python -B -m unittest discover -s tests -p 'test_stage20h*.py' -v
env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy python -B -m unittest discover -s tests -v
python -m compileall -q app src tests model_policy.py run_api.py
alembic upgrade head
alembic current --check-heads
python -B -m unittest discover -s tests/integration -t . -v
```

Initial #33 source-mounted offline run at `961b1c8`: **1,252 collected, 1,192 passed, 60 skipped,
zero failures/errors** (104.614 seconds). The focused Stage 20H modules pass
**80/80**, including 20 final-P1 methods. The skipped tests are existing opt-in
integration checks and an existing local artifact-dependent case; the dedicated
PostgreSQL/Redis run is reported separately. Python 3.11 compilation passes.

Final review additionally reproduced the legacy `sufficient` flag returning true
for two separate one-arm papers despite the new indirect-comparison diagnostic.
The strengthened existing comparison test failed once before tightening this flag:
at least one eligible passage must retain all requested comparison entities.
Separate studies now keep `sufficient=false` and `qualified_synthesis_eligible=true`;
they remain available as explicitly qualified indirect evidence. Same-passage
entity coverage alone still does not establish direct comparative support; the
result/relation checks and strict semantic verifier remain separate. Final repeated
counts and workflow conclusions are recorded on the final PR heads.

Integration runs opt in with `RUN_INTEGRATION_TESTS=1` against dedicated ephemeral
PostgreSQL 16/pgvector and Redis 7 containers on an internal Docker network. There
are no production volumes or user data. Production image build/provenance and
`/healthz` smoke are checked separately. Final counts and remote workflow conclusions
are recorded in the PR descriptions; [#33 checks](https://github.com/shay-dan/scientific-rag/pull/33/checks)
and [#34 checks](https://github.com/shay-dan/scientific-rag/pull/34/checks) follow the
reviewed heads. A passing synthetic suite is not scientific acceptance.

## Limits, integration and rollback

Entity extraction and coverage remain bounded heuristics. Unusual noun structures,
unlisted abbreviations/synonyms, negated populations and comparisons whose arms and
results occur in different sentences can remain unresolved/contextual. A comparative
candidate is not confirmed head-to-head evidence or a scientifically supported claim.
Selected quotations can expose missing arms even when every retained individual
claim is semantically supported. There is no new model judge or inference-based
recall, abstention, accuracy, latency or cost measurement.

P2 planner/repair admission refinements and audit-only statistical regex extensions
are deferred; existing #34 deadline/reservation safeguards are preserved and tested.
PubMed automatic-term-mapping translation remains unverified against live eSearch;
no external scientific request was made. Such a request needs separate authorization.

### Additional #34 typed-screening deadline correction

After the P1 integration, one targeted deadline injection exposed an additional
boundary: both relevance screeners swallowed `ResearchDeadlineExceeded` under
their generic screening-error handler. One test method failed both PubMed/arXiv
subtests before the correction. Both handlers now rethrow that exact typed exception,
preserving its stage, identity and `stage_admission_refused` classification through
the existing mining/service deadline handlers. Ordinary screening/provider/schema
errors still use P1-D's screening-failure path. This adds no retry, call, budget,
provider-admission change or new gateway behavior. The original #34 stage admission,
mandatory-verification reservation and uncertain-transmission controls remain.

Initial combined #34 source-mounted offline run at `b0472ed`: **1,271 collected, 1,211 passed,
60 skipped, zero failures/errors** (143.390 seconds). Focused Stage 20H modules:
**99/99**. The integration merge retains both original heads; only the two typed
screening rethrows are additional reliability implementation changes.

### Final comparison-guard validation

After the final legacy comparison-admission guard, source-mounted production
dependency-image results are:

| Source commit | Collected | Passed | Skipped | Failures/errors | Focused |
|---|---:|---:|---:|---:|---:|
| #33 `ce0a4ac` | 1,252 | 1,192 | 60 | 0/0 | 80/80 |
| #34 `344118a` | 1,271 | 1,211 | 60 | 0/0 | 99/99 |

Durations: 106.347 seconds (#33 fresh retry) and 120.813 seconds (#34). The
dedicated #34 final-source PostgreSQL/Redis run passes 59/59 in 16.789 seconds;
Python 3.11 compilation passes. One #33 rerun stalled in the unchanged
lease-heartbeat cancellation test and was explicitly stopped (exit 137), not
counted as a pass. Its log was preserved locally; the fresh full retry above
passes without modifying the lease code, test, cancellation or timing assertions.
Final exact-head production build/provenance, integration, health and Security
conclusions are recorded in the PR descriptions and linked workflows.

Publish #33 by a normal fast-forward update, then merge its corrected ancestry into
#34 without rebase/force-push, preserving `main → #32 → #33 → #34`. Keep all drafts
unmerged and await owner review. Rollback is a new revert in reverse dependency
order; retain records/data and do not silently restore the defective cache identity.
