# Optional provider-enforced output-token limits (R6D)

R6D adds an optional, per-request `max_output_tokens` to the LLM gateway. When it is
omitted, every request is built and sent exactly as before. No production caller sets it.
Its first user is the R6C semantic-verifier replay adapter, which needs a bounded output
before live evaluation can be authorized.

## Gateway interface (`src/scientific_rag/llm_gateway.py`)

| Function | Change |
|---|---|
| `create_chat_model(provider, role, temperature, max_output_tokens=None)` | optional limit |
| `ainvoke(role, temperature, chain_builder, payload, *, max_output_tokens=None)` | optional limit |
| `ainvoke_structured(role, temperature, prompt, response_schema, *, max_output_tokens=None)` | optional limit |
| `validate_max_output_tokens`, `output_token_limit_parameters`, `output_token_limit_support` | new helpers |
| `app.claim_verification.verify_claims(..., max_output_tokens=None)` | optional pass-through to the gateway |

Typed refusals, all subclasses of `OutputTokenLimitError`:

- `InvalidOutputTokenLimit`: the limit is not a strictly positive `int`. Booleans,
  fractions, zero, negatives, NaN, Infinity and strings are refused.
- `UnsupportedOutputTokenLimit`: a configured provider cannot enforce a verified limit.
- `OutputTokenLimitNotEnforced`: an outbound request did not carry the limit, or a
  response did not come from a verified bounded request.

## Provider support

| Provider | Bounded requests | Outbound parameter | How it is set |
|---|---|---|---|
| Groq | supported | `max_completion_tokens` | `ChatOpenAI(max_tokens=N)`; the pinned `langchain-openai` 1.7.2 sends it as `max_completion_tokens` |
| OpenRouter | supported | `max_completion_tokens` | same |
| Gemini | **not supported; refused** | none | its outbound request is not verified by the gateway |

The parameter names above were read from the outgoing request body in tests (mocked
transport), not from model attributes. Both providers document `max_completion_tokens`
as the completion limit and `max_tokens` as its deprecated form; the guard accepts either
name, with exactly the requested value.

## Enforcement at the request boundary

A bounded model gets its own HTTP clients with a guard that runs before each request is
sent. The guard refuses the request, and nothing is sent, when:

- the body carries no limit, or a limit that is not exactly the requested integer;
- the body asks for more than one completion (`n > 1`).

Other controls for bounded requests:

- SDK retries stay off (`max_retries=0`, as in production), and redirects are not
  followed, so one gateway attempt is one provider request.
- A response that did not pass through the guard is refused.
- A bounded call is refused before any request if any configured provider for the role
  is unsupported. Fallback between supported providers stays bounded with the same limit,
  so a bounded call can never become an unbounded one.
- A limit refusal is never retried or routed to another provider.

## R6C semantic-verifier adapter

- The authorization file must also carry `max_output_tokens` (positive integer).
- Gates before any provider call, in order: valid authorization and budgets; handoff
  identity; pinned provider and model with a known price; a limit the gateway can enforce
  for that provider; a call reservation; a cost reservation.
- The limit is passed to `verify_claims`, then to the gateway. It also sizes each call's
  cost reservation (input at one token per 3 prompt characters, output at the limit).
- `BudgetLedger` reserves one call and its cost before each call, atomically, and settles
  afterwards:

| Outcome | Call charged | Cost charged | Status |
|---|---|---|---|
| Success with complete usage | yes | measured cost | `complete` |
| Failure, or partial or missing usage | yes | full reservation; run stops | `incomplete`, with known lower bound |
| Cancelled mid-call | yes | full reservation (unsettled) | not recorded; ledger keeps it |
| Would exceed the remaining calls or cost | no call made | nothing | `not_executed` |

- Gateway retries count as extra calls. Runs sharing one ledger cannot overspend.
- A provider that reports more output tokens than the limit stops the run.
- Verdict records, the export contract and owner-side scoring are unchanged. Each
  record's `usage` also states `max_output_tokens`.
- `verifier-dry-run` reports `live_execution.eligible_after_authorization` with any
  blockers. It makes no provider call and needs no API key.

Authorization example:

```json
{"schema": "scientific-rag-paid-evaluation-authorization/v1", "authorization_id": "…",
 "approved_by": "…", "adapter": "production_semantic_verifier",
 "handoff_dataset_hash": "<handoff dataset hash>", "provider": "groq", "model": "openai/gpt-oss-120b",
 "max_calls": 50, "max_cost_usd": 0.10, "max_output_tokens": 2000}
```

## What the limit does not guarantee

The limit bounds the output tokens requested from the provider. It is not an absolute
billing guarantee:

- **Provider behavior is not verified offline.** Whether the provider honors the limit,
  and how it counts reasoning tokens against it, can only be observed in a live call.
- **Input cost is estimated** from prompt characters before the call.
- **Prices** come from the repository's dated rate table, not from the provider at call
  time.
- **Unreported usage** is charged at the full reservation, which is an assumption, not a
  measurement.
- **A cancelled or timed-out request** may still be billed by the provider.
- **Truncation.** A small limit can cut a structured answer short; that call is recorded
  as `failed`, and its cost is still charged.
- **Gemini** stays unsupported until its outbound request can be verified.

## Tests

`tests/test_r6d_output_token_limits.py` (no live request; provider HTTP is mocked):

- limit validation and the support table;
- the limit in the outbound request for Groq and OpenRouter, including structured calls;
- no limit and the original client construction when none is requested;
- unsupported providers, bounded fallback, a single request on failure, dropped and
  unverified limits, multiple completions, redirects, and no key or prompt in errors or logs;
- the verifier pass-through, with the default call unchanged;
- adapter budgets: success, failure before usage, partial usage, repeated failures,
  cancellation, concurrency, reservations that would exceed the budget, and refusals
  before any call;
- the unchanged export contract, and the offline dry run.
