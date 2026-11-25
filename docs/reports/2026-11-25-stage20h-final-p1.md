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

Final #33 source-mounted offline run: **1,252 collected, 1,192 passed, 60 skipped,
zero failures/errors** (104.614 seconds). The focused Stage 20H modules pass
**80/80**, including 20 final-P1 methods. The skipped tests are existing opt-in
integration checks and an existing local artifact-dependent case; the dedicated
PostgreSQL/Redis run is reported separately. Python 3.11 compilation passes.

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

Publish #33 by a normal fast-forward update, then merge its corrected ancestry into
#34 without rebase/force-push, preserving `main → #32 → #33 → #34`. Keep all drafts
unmerged and await owner review. Rollback is a new revert in reverse dependency
order; retain records/data and do not silently restore the defective cache identity.
