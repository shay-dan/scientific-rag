# Controlled evaluation of the production system (R6C)

R6C adds two independent, opt-in adapters that run **existing production components**
on the label-free development handoff (v0.3.1) and export unscored predictions for the
benchmark owner. Nothing here scores, tunes or changes production behavior.

| | Semantic-verifier replay | Hybrid-retrieval replay |
|---|---|---|
| Module | `app/benchmark/production_verifier_replay.py` | `app/benchmark/production_retrieval_replay.py` |
| Production code reused | `ClaimVerificationInput`, `plan_verification_batches`, `verify_claims` (gateway call, response validation, R3 isolation and instruction-like-support guard) | `PostgresVectorService.add_documents` and `retrieve_context(strategy="enhanced")`: pgvector cosine + MMR, PostgreSQL FTS, RRF, parent/child expansion |
| Export task | `claim_verification` only | `fixed_evidence_retrieval` only |
| Provider calls | gateway role `gatekeeper` | Gemini embeddings (`EMBEDDING_MODEL`) |
| Default | dry-run only | dry-run only |

Both adapters use the R6B stage protocol: open discovery is frozen as `not_executed`
before fixed evidence is loaded. They emit the R6B export contract
(`scientific-rag-benchmark-predictions/v1`), and each export holds one task only.
Verifier and retrieval results are never combined. The R6A BM25 reference baseline
stays a separate path, and no production module imports either adapter.

CLI: `scripts/benchmark_r6c.py` (`verifier-dry-run`, `verifier-run`, `retrieval-dry-run`,
`retrieval-run`). The handoff and all outputs stay outside the (public) repository.

## Semantic-verifier replay

**Representation**, documented and not tuned:
- Each benchmark claim is one target (`claims.1`) in its own gateway call, so no claim
  sees another claim.
- The benchmark gives evidence segments but no citation quote. The claim's text
  segments, in evidence-index order and joined by blank lines, form the matched passage
  and stand in for the quote. This is the R5/R6B convention.
- The question slot carries a fixed neutral sentence, *"Does the supplied evidence
  support the claim?"*. The production prompt treats that slot as untrusted context,
  never as evidence.
- Evidence that does not fit the production prompt bound (24,000 characters) is
  `not_executed` and is never truncated. None of the v0.3.1 claims exceeds it.
- Segments that are not redistributed are never fetched.

**Modes.** The adapter returns verdicts only. It never calls `apply_semantic_verdicts`
and never reads or changes `SEMANTIC_CLAIM_VERIFICATION_MODE`. Production off/shadow/
enforce behavior is untouched, and the default stays `off`.

**Provenance.** `prompt_contract_sha256` fingerprints the production prompt (it changes
if the prompt changes). Each record carries its verdict and the model's reason, whether
the R3 guard applied, the measured provider/model, tokens, estimated cost and latency.

**Failure accounting.**
- `failed`: gateway failure or invalid structured response, with usage kept.
- `invalid`: answered by a provider or model outside the authorization; execution then
  stops.
- `not_executed`: no text, prompt bound, cost or call ceiling, or a stop after
  unmeasurable usage.
- Failed calls are never retried by the adapter.

## Hybrid-retrieval replay

**Corpus.** Each fixed-evidence pack is a frozen corpus of exactly its own text
segments. One production `Document` per segment carries `evidence_id`,
`benchmark_input_id` and `paper_id`, with `doc_id` set to the benchmark paper ID when
supplied. `corpus_sha256` binds the evidence IDs and text hashes to the embedding model,
dimensions, chunking and pipeline version.

**Isolation (fail closed).**
1. The database must be a verified, dedicated evaluation database:
   - **Application identity.** It comes from the effective application configuration,
     `Settings().database_url` (environment, `.env` or default), never from the process
     environment alone. A missing or unparsable identity refuses the run.
   - **URL comparison, before any connection.** The two URLs are normalized first:
     driver suffix, `postgres`/`postgresql`, host case, loopback aliases (`localhost`,
     `127.0.0.1`, `::1`), Unix sockets, default port. They must not name the same server
     and database.
   - **Physical check.** Both databases are queried for name, OID and server start time,
     and these must show different physical databases. If the application database
     cannot be reached, isolation is unverifiable and the run is refused.
   - **Migration head.** The database must be at the current Alembic head, using the
     existing migrations.
   - **Research runs.** It must hold no research run that the adapter did not create.
2. Each pack is ingested in its own transaction under its own research run, with
   production run scoping enabled, and is **rolled back** after retrieval. Nothing
   persists, and no pack can see another.
3. Every returned passage must belong to that pack; otherwise the record is `invalid`
   and execution stops.

**Configuration.** The production `enhanced` strategy with top-N from the production
research budget: `standard` gives 6; `fast` and `deep` are selectable and recorded.
Chunking is parent 2,500/500 and child 500/75; RRF k = 60; MMR λ = 0.5; FTS config
`english`. No parameter is tuned.

**Output.** Each record carries:
- ranked evidence IDs, and ranked paper IDs where supplied;
- result count, corpus segments and sources, and corpus identity;
- a numeric retrieval trace (vector, MMR, full-text and selected counts, plus the FTS
  decision), with no query text;
- ingest and retrieval timings;
- the production embedding-usage estimate.

## Dry runs (no provider, network or database call)

- `verifier-dry-run` plans every call with the production planner. It reports prompt
  sizes and a token estimate: the production 4-characters-per-token rule plus a
  200-token output estimate. It also reports a **planning cost bound** (3 characters per
  token plus the output limit, 2,000 tokens by default, per call) for each route. Finally
  it reports whether the gateway can enforce the limit on that route and whether a live
  run would be eligible after authorization.
- `retrieval-dry-run` splits each pack with the production splitters. It reports chunk
  and embedding-request counts, the production embedding-token estimate and cost, and the
  reserved cost (3 characters per token).

## Paid execution needs an explicit authorization

`verifier-run` and `retrieval-run` refuse to run without `--authorization FILE`, parsed as
**strict JSON** (`NaN`, `Infinity` and `-Infinity` are refused):

```json
{"schema": "scientific-rag-paid-evaluation-authorization/v1", "authorization_id": "…",
 "approved_by": "…", "adapter": "production_semantic_verifier | production_hybrid_retriever",
 "handoff_dataset_hash": "<handoff dataset hash>", "provider": "groq", "model": "openai/gpt-oss-120b",
 "max_calls": 50, "max_cost_usd": 0.10}
```

The run is refused before any call when:
- the schema, adapter or handoff hash does not match;
- `max_calls` is not a positive integer (booleans, fractions, NaN and Infinity are
  refused);
- `max_cost_usd` is not a finite positive number;
- for the verifier, `max_output_tokens` is not a positive integer, or the gateway cannot
  enforce it for the authorized provider (R6D);
- the model has no known price, so the budget could not be enforced;
- for the verifier, the gateway route is not pinned to the single authorized provider and
  model (`LLM_PROVIDER_ORDER_GATEKEEPER=<provider>`);
- the planned calls exceed `max_calls` (unless `allow_partial` is true).

**Verifier output limit (R6D).** The verifier authorization must also carry
`max_output_tokens`. The adapter passes it through the production verifier to the
gateway, which sets it on the provider request and refuses any request without it. Groq
and OpenRouter are supported; Gemini is refused. The limit is a planning bound on output,
not an absolute billing guarantee. See
[output-token-limits-r6d.md](output-token-limits-r6d.md).

**Budget accounting (R4).** Every record states its cost status:

| Status | Meaning | Charged against the budget |
|---|---|---|
| `complete` | measured provider usage (verifier) | measured cost |
| `estimated` | production estimate; Gemini embeddings return no usage | the full reservation |
| `incomplete` | failed or unmeasured requests; any known lower bound is recorded | the full reservation |

- **Before each call or pack:** its full allowance is reserved. For the verifier that is
  one gateway attempt and the planning bound. For retrieval it is every production
  embedding request (one per parent plus the query) and the conservative cost. It starts
  only if the reservation fits in what remains.
- **Calls:** gateway retries and fallbacks count against `max_calls`. Retrieval charges
  each started pack its reserved requests even when production recorded fewer, because
  production counts embedding requests only after they succeed.
- **Stops:** the verifier stops after unmeasured usage, an unauthorized model, an
  unenforced output limit, or reported output above the limit.
- **Run summary:** reports complete, estimated, lower-bound, uncertain-reserved and
  charged amounts separately.

Pilot selection is label-blind: `--limit N` takes the first N inputs by opaque ID, or
`--input-ids FILE` takes a listed subset.

## Tests

- **`tests/test_r6c_production_adapters.py`** (deterministic). These tests run the real
  production verifier with a fake gateway that records usage in the real gateway trace.
  They cover:
  - mapping and the prompt bound;
  - prompt identity with production;
  - a dry run that makes no call;
  - authorization refusals before any call;
  - the cost, call and usage stops;
  - unauthorized models, gateway and response failures;
  - the R3 guard, and that enforcement is never applied.

  For retrieval they cover configuration, corpus identity, rollback, foreign results,
  failure accounting, the database guards and production chunk counting. They also
  cover CLI refusals and that production code does not import the adapters.
- **`tests/test_r6c_review_reproduction.py`** (review of `d97957f`). Each test
  reproduced a defect before the fix:
  - non-finite and non-integer budgets, and NaN in authorization JSON;
  - failed embedding requests escaping the call budget, and failed spend counted as zero;
  - verifier execution without a provider-enforced output limit;
  - an application database from `.env`, equivalent URL spellings, aliases, missing and
    unverifiable identities.
- **`tests/integration/test_r6c_hybrid_retrieval_integration.py`** (PostgreSQL/pgvector).
  It creates and migrates a dedicated database, then runs the unchanged production
  retriever. A deterministic local embedder stands in for the provider only. It covers:
  - ranking within each pack, with pgvector and FTS both executed;
  - parent expansion;
  - nothing persisting after a run;
  - reproducible rankings and corpus identity;
  - refusal of foreign runs and of the application database, including equivalent
    spellings and an unreachable or missing application identity;
  - failure accounting;
  - the cost ceiling.

## Not in R6C

- No paid call has been made.
- No live PubMed/arXiv discovery.
- No migration, production algorithm change or tuning.
- No reference labels or sealed holdout.
- No scoring: the owner scores exports outside the repository with the R6B v2 contract.
