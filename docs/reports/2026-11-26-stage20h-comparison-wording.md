# Stage 20H comparison-wording blocker — in review

Continuation from clean #33 `ce0a4aca68933b813b7f406f3af082851e6f2eb2`
and #34 `bcce835eef02787d9b5244fea0300bb0e022b755`. Both remote draft
heads and their dependency bases were checked. Foundation #32 remains unchanged.

## Reproduction and minimal correction

The regression module was committed before source changes as `416be21`.
Six methods produced **12 assertion/subtest failures, zero errors** before the fix.
Both "compared with" and "compared to" retained the grounded intervention names
but lost comparison arms/intent, allowed separate one-arm sources to set the
legacy sufficiency flag, and failed selected-answer comparison disclosure.

Both detection expressions now use `compar\w*`, consistent with the existing
passage-level comparative expression. Versus, better-than, compare and comparison
controls retain comparison intent. "A or B" remains alternatives; a single
intervention remains noncomparative. A result passage containing both arms and an
explicit comparison remains eligible. No entity-extraction redesign.

The mandatory real-pipeline selected-answer test exposed one additional critical
data-flow gap: the regex-only correction passed the five extraction/admission
methods but still failed both selected-answer subtests. Completeness re-extracted
the question without the grounded planner entities, losing zolpidem's intervention
role. The orchestrator now passes its already-grounded anchors through synthesis
into retrieved/selected-quotation adequacy. The optional argument defaults to the
previous standalone extraction behavior. This is a narrow pass-through, not P2
expansion; it adds no model call and changes neither synthesis/verifier prompts nor
semantic decisions, quotation checks, source provenance or provider budgets.

The selected-answer regression exercises the real orchestrator/research agent with
two separate synthetic one-arm studies and fixed supported verdicts for individual
claims. It requires `sufficient=false`, `indirect_candidates`, and an explicit
statement that direct comparison was not established. All providers are mocked;
networking is disabled for offline unit runs. These are engineering controls,
not evidence of scientific accuracy.

## Validation and publication

Python 3.11 production dependency images use the unchanged requirements/constraints.
Source/tests are mounted read-only. Full offline runs remove inherited proxy
variables inside a network-disabled container so the unchanged Stage 20G guard
can execute. Integration uses isolated PostgreSQL 16/pgvector and Redis 7 test
containers, with no production data or host service exposure.

Commands inside the production dependency images:

```sh
python -B -m unittest discover -s tests -p test_stage20h_comparison_wording.py -v
python -B -m unittest discover -s tests -p 'test_stage20h*.py' -v
env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy python -B -m unittest discover -s tests -v
alembic upgrade head
alembic current --check-heads
python -B -m unittest discover -s tests/integration -t . -v
```

Final counts, exact heads and CI/Security links are recorded in the draft PR
descriptions. Preserve `main → #32 → #33 → #34` by normal fast-forward branch
updates and ancestry integration, without rewriting published commits or merging
PRs. The final GitHub production workflows also verify image build/provenance,
compilation, migrations and health. No new security exceptions.

Local #33 validation: **86/86 focused**, **1,258 collected / 1,198 passed /
60 skipped / zero failures/errors** in the full offline suite (78.629 seconds),
and **59/59 PostgreSQL/Redis integration tests** (9.583 seconds). The 60 full-suite
skips are the existing opt-in integration tests (run separately) and the workflow
file check unavailable inside the application-only image. Exact-head CI results
for both final branches are linked in their PR descriptions.

Rollback is a new revert of these correction commits in dependency order. This
amends the same unmerged Stage 20H release and retains its existing v3 cache
identity; no schema, cache format, data deletion or migration is introduced.
Historical reports, frozen benchmarks and generation prompts remain intact.

No provider calls, paid inference, provider credentials, deployment, PR merge or
deferred P2 implementation. **Stop for owner review.**
