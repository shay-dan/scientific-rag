# R7C versioned scientific generation prompt

R7C restores numeric and attribution guidance only in an opt-in structured-synthesis
prompt version. Set `GENERATION_PROMPT_VERSION=v15|v16`; the production default is
`v15`. No provider, model, temperature, route, budget, output schema, repair prompt,
semantic verifier, deterministic claim audit, or retrieval behavior changes.

## Prompt contract

`generation_prompt_v2(question, context, *, prompt_version="v15")` supports two
versions:

- `v15` is byte-for-byte identical to the pre-R7C production prompt. Its SHA-256 for
  the pinned test question and context is
  `889c949b74ccf4cd1381650154eb7b8adbc116fe947d73f5dc9a7f1704d32bb2`.
- `v16` inserts exactly the block below after the existing generation instructions and
  before `QUESTION:`. The block is application-authored and remains outside the
  untrusted evidence region.

```text
Numeric and attribution rules:
- Copy each number exactly as the quoted passage writes it, with its unit, and with its
  confidence interval, p-value or denominator when the passage states them. Do not
  calculate, convert, round or combine numbers.
- The evidence_quote must contain every number stated in the claim.
- Say which group, outcome, comparator and time point a number refers to, in the
  passage's own terms. Never combine numbers from different populations, subgroups or
  denominators in one claim.
- Report a result as this source's own finding only when the passage presents it that
  way. If the passage attributes it to another study, say so or leave it out.
- Keep the passage's design and certainty. Never describe a retrospective, single-arm,
  preprint, historical or indirect comparison as randomized, head-to-head or
  established, and do not turn an association into a cause.
- If the passage does not state a detail, leave it out; do not infer it.
```

For the same pinned fixture, the v16 prompt SHA-256 is
`fc6b232e28829baa5c87f044def973a61c8097570e2ef9d37dcddd9f4f094415`.
The v16 delta is 976 characters, approximately 244 tokens using the conservative
four-characters-per-token planning estimate. The fixed test fixture caps the complete
v16 prompt at 2,400 characters. Actual request size still depends on the question and
evidence context.

The canonical `StructuredEvidenceAnswerV2` JSON schema is unchanged. Its deterministic
sorted compact-JSON SHA-256 is
`194e586d95e1f6a0f07310ef27446ed5449370334d57769c7b42cb85c993f471`.

## Persistence, queues, and caches

Only v16 runs store `generation_prompt_version: v16` in internal run metadata. Absence
means v15, preserving historical records and default response metadata. A queued run's
persisted version is authoritative across retries, restarts, requeues, and worker
environment changes; a historical queued run with no field always resolves to v15.

V15 keeps the existing v15 pipeline/cache identity exactly. V16 appends
`-genprompt-v16` after any semantic-verification and recovery suffixes. Claim audit mode
remains outside the cache identity because it does not change the answer. Consequently,
v15 and v16 answers cannot satisfy each other's cache lookups, while all historical v15
lookups remain compatible. Opt-in v16 queue arguments are appended after the existing
recovery and audit arguments; historical/default queue shapes are unchanged.

## Scope and limitations

The instructions constrain model behavior but do not prove numeric correctness,
attribution, study quality, or semantic support. Structural source-ID and verbatim-quote
validation remains fail-closed. R7A recovery, R7B audit findings, and semantic
verification retain their existing behavior. The initial generation still makes at most
one call and the existing bounded repair still makes at most one call; R7C adds no call,
retry, network request, or provider cost by itself. The longer prompt can add roughly
244 input tokens when v16 is selected, so a later evaluation must use a v16-appropriate
input-token ceiling without changing the output-token ceiling.

Rollback is configuration-only: set `GENERATION_PROMPT_VERSION=v15`. Existing v16
answers remain visible in history but are excluded from v15 cache hits. Before deploying
workers older than R7C, drain or cancel queued v16 jobs because their appended argument
is not understood by older workers. No database rollback is required.

## R7D evaluation requirements (not implemented or executed)

A separately authorized R7D comparison must:

- freeze the questions, evidence passages, application-authored notes, and their hashes;
- use the same provider, model, temperature, output-token cap, provider budgets, and
  fallback behavior for both arms, with only prompt version differing;
- account for v16's longer input prompt in the request budget and record prompt/template
  hashes without recording private questions or evidence;
- score numeric fidelity and source attribution separately;
- report first-pass structural validity, repair rate, claim coverage, abstention, and
  pre-repair versus post-repair outcomes;
- keep R7B deterministic-audit outcomes separate from evaluator judgments and never
  attribute an observed improvement to R7B, which is audit-only.

R7C performs no R7D instrumentation, replay, provider request, or paid evaluation.

## Verification and decision record

The tests pin both prompt hashes and the schema hash, prove the single-variable v16
delta and placement, exercise the production generation/repair path without providers,
and cover configuration, identity composition, cache lookup, queue shapes, persisted
precedence, and worker forwarding. Local Docker evidence on the R7C branch: 17/17 focused
R7C tests; 845 deterministic tests executed (787 passed, 58 skipped); 57/57 live
PostgreSQL/pgvector and Redis integration tests; compilation passed. Final CI and
Security evidence is recorded in the R7C pull request.

Decision: preserve v15 exactly and keep it as the shipped default; make v16 explicit,
persisted, cache-isolated, and reversible. This provides a controlled comparison for
future R7D without claiming that the new instructions improve scientific accuracy.
