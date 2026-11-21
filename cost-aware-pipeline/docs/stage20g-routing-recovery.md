# Stage 20G — consolidated routing and recovery, and the v2 production E2E protocol

**Status:** in review. No live inference has been run or is authorized by this stage.

## 1. Diagnosis of the Stage 20F v1 stop

The v1 campaign (`422aabe9…308e`) is terminal: it is never resumed, rerun or deleted, and
its budget is never transferred.

On E001, the seventh Groq request (role `expert`) was refused **locally**: the Redis
admission controller logged `reason=tokens retry_after_ms=31804`. Nothing was transmitted.
Three mechanisms combined to cause this:

| Mechanism | Effect |
|---|---|
| The estimate reserved 3,000 output tokens for `expert`, although every Stage 20 request was capped at 2,000 | The request was over-reserved by 1,000 tokens against a 6,000-TPM local window |
| The legacy gateway's "recovery phase" re-checks admission immediately | With Groq as the only route, the call was refused twice in the same minute and then failed, without waiting 31.8 s for the reset |
| The v1 adapter stopped the whole campaign on any gateway exception | One local admission refusal ended all five questions |

This was an admission-policy failure. It says nothing about scientific answer quality or
provider health.

The existing fallback was also unsuitable for this experiment, for four reasons:
- it is driven by `is_retryable_error`, which also treats invalid model output as an outage
  and falls back;
- it falls back after timeouts whose transmission outcome is unknown;
- it does not check compatibility (output limit, reasoning effort, structured output);
- it has no request budget.

## 2. The policy: WAIT / FALLBACK / SEND / STOP

`src/scientific_rag/routing_policy.py` is pure: no I/O, clock or network. The gateway applies
it only when `LLM_ROUTING_POLICY=consolidated`. The default (`legacy`) is unchanged.

**Before each transmission** (`decide_admission`), it works through the approved routes in
order:
1. **Filter routes.** It drops routes that are:
   - incompatible: output cap unsupported or too large, reasoning effort unsupported,
     structured output unsupported, context too long, or unpriced;
   - already transmitted to in this call;
   - in a quota group exhausted by a provider rate limit;
   - refused by the accounting hook.
2. **Preferred route.** It probes admission on the preferred route and sends if admitted.
3. **Short wait.** If the preferred route is refused only by a window (tokens, requests,
   cooldown or concurrency), it waits when all of these hold:
   - the request fits a fresh window;
   - the reset is within 5 s, or there is no other route;
   - the cumulative wait stays within 65 s;
   - at least 10 s of the call deadline remains afterwards.
4. **Fallback.** Otherwise it probes the fallbacks in order and sends on the first one
   admitted.
5. **Longer wait.** Otherwise it waits for the soonest admissible window.
6. **Stop.** Otherwise it stops with a classified reason, for example:
   - `exceeds_provider_window`: the request can never fit the whole TPM window, so there is
     no wait;
   - `capacity_wait_exceeds_budget`, `budget_exhausted`, `no_fallback_route`,
     `no_compatible_route`, `admission_unavailable`, `deadline_exhausted`, `attempt_limit`.

Admission is taken lazily, so a fallback slot is reserved only when it is actually used.

**After a failed transmission** (`failure_disposition`), the next step depends on how far the
request got. The route that failed is never re-sent.

| Transmission state | Example | Next step |
|---|---|---|
| Not transmitted | A guard or budget refused it before sending | Stop (configuration or budget) |
| Definitive response | HTTP 429 | Fall back; the quota group is marked exhausted |
| Definitive response | HTTP 5xx or 408 | Fall back |
| Definitive response | HTTP 413, or `context_length_exceeded` | Fall back (the route is too small for the request) |
| Definitive response | 400 `json_validate_failed` or completion-budget exhaustion | Stop (invalid output); opt-in fallback only |
| Definitive response | 401, 403 or another configuration error | Stop |
| Uncertain | Timeout or connection reset after sending | Stop; never replayed. Opt-in fallback only (`LLM_ROUTING_FALLBACK_AFTER_UNCERTAIN`) |
| Completed but invalid | HTTP 200 whose JSON fails the schema, or local answer validation fails | Stop; opt-in fallback only (`LLM_ROUTING_FALLBACK_ON_INVALID_OUTPUT`) |

The gateway determines the transmission state from wire facts. A request hook runs
immediately before sending and a response hook records the HTTP status. Without those, it
falls back to the error chain. A `TransmissionRefused` raised by an accounting transport is
always final.

**Safety properties:**
- **No hidden retries:** SDK retries are off (`max_retries=0`).
- **OpenRouter request rules:** routed requests carry
  `provider: {require_parameters: true, allow_fallbacks: false, max_price: …}`, and the
  outbound guard refuses any request that does not carry exactly these values.
- **Allowlist:** only `LLM_ROUTING_ROUTES` / `LLM_ROUTING_ROUTES_<ROLE>` entries
  (`provider:model`) with declared capabilities (`ROUTE_CAPABILITIES`) are ever candidates.
- **Attempt limit:** at most 2 physical attempts per logical call (configurable).

## 3. Engineering decisions

| Decision | Why | Pros / cons | Simpler alternative considered | Where | Cost / operations |
|---|---|---|---|---|---|
| One pure decision module, used by one new gateway path behind a setting | Replaces scattered retry/skip logic with one testable policy | + deterministic, fully unit-tested; − a second code path coexists with `legacy` until it is promoted | Patch the legacy loop with a 65 s wait (the earlier v2 patch) | Production code, opt-in | None until enabled |
| Wait only if the request fits a fresh window; stop immediately if it can never fit | Waiting cannot help an oversize request | + bounded latency, honest reason; − estimates are heuristic (about four characters per token) | Always wait 60 s | Production | Latency only |
| Reserve a bounded request's real output cap | It is enforced on the wire; 3,000 over-reserved by 1,000 | + fewer false refusals; − none found | Lower the global expert reserve | Production (bounded calls only) | None |
| Fall back only on definitive provider rejections | Avoids double billing and duplicated effects | + no unsafe replay; − an unknown outcome costs the call | Fall back on any retryable error (legacy) | Production | Avoids duplicate charges |
| No fallback on invalid output by default | A billed, invalid answer is an outcome; re-sampling elsewhere hides model quality and costs more | + honest evaluation; − slightly lower completion | Legacy fallback on `InvalidStructuredResponse` | Production default for the new path | Saves calls |
| Same-model fallback (OpenRouter `openai/gpt-oss-120b`), not another model | Scientific capability must match, not just availability | + comparable answers; − depends on OpenRouter credits and upstream support for every parameter | Gemini chat fallback, refused: the gateway cannot verify a bounded output there | Route table + protocol | Capped at Groq's rates |
| A price ceiling via `max_price` equal to Groq's rates | Turns a modeled reservation into a real per-token upper bound | + worst case unchanged; − may exclude pricier upstreams (a definitive no-endpoint error) | Trust the list price | Evaluation protocol (operators may override) | No budget increase |
| Accounting at the transport, budget hook in the policy | Every physical request is reserved before its bytes are sent; the policy avoids starting attempts the budget would refuse | + no uncounted transmission; − two touch points | Count in the gateway only | Evaluation | — |
| Case-scoped versus campaign-scoped stops | v1 lost four questions to one refusal; the experiment must measure completion and non-delivery | + reliability is measurable; − more questions can run under partial failure (still within the ceilings) | Stop everything (v1) | Evaluation | Bounded by the ceilings |
| No embedding fallback | Another embedding route changes the vector space | + retrieval integrity; − embedding outages fail the case | — | Evaluation | — |
| Refuse to run with proxy variables set | An httpx proxy mount could bypass the guarded transport | + no uncounted path; − an operator behind a proxy must unset it | — | Evaluation runner | — |

## 4. v2 production E2E protocol (proposed; requires owner authorization)

- **Campaign:** `stage20-e2e-v2`, a new protocol with schema `scientific-rag-e2e-protocol/v2`.
- **Manifest:** the same private five-question manifest (`c0a96c97…394a`), located by hash.
- **Scientific configuration (unchanged):**
  - fast mode, `refresh_sources`;
  - prompt v16, Unicode quotation recovery, claim recovery `degrade`;
  - semantic verification `enforce`, claim audit on;
  - Gemini `gemini-embedding-2` at 768 dimensions;
  - no synthetic evaluation calls.
- **Chat routes:**
  - primary: Groq `openai/gpt-oss-120b`;
  - fallback: OpenRouter `openai/gpt-oss-120b`, with low reasoning, temperature 0, at most
    2,000 output tokens and the pinned provider preferences.
  - A `-NoFallback` Groq-only variant exists, with its own fingerprint.
- **Routing values:** waits ≤ 65 s, primary preferred within 5 s, 2 attempts per call,
  150 s per call, provider timeouts of 30 s (Groq) and 45 s (OpenRouter), Groq local budget
  6,000 TPM / 20 RPM.
- **Request ceilings:**
  - chat: 20 per question and 100 per campaign, shared by Groq and OpenRouter;
  - embedding: 25 per question and 125 per campaign;
  - 600 s per question, 3,000 s per campaign.

**Exact worst-case modeled reservation** (fixed per request, never refunded):

| Item | Reservation |
|---|---:|
| Chat request (Groq or capped OpenRouter): 32,768 × $0.15/M + 2,000 × $0.60/M | $0.0061152 |
| Embedding request: 8,192 × $0.20/M | $0.0016384 |
| 100 chat + 125 embedding requests | **$0.81632** |

The campaign limit equals that maximum. The fallback does not increase the budget, because
fallback requests draw on the same chat pool at the same capped rate. These are modeled
reservations, not provider charges.

## 5. Operator workflow (one command per action)

From the existing checkout, once this branch or its merge is checked out:

```powershell
cd C:\src\scientific-rag-ff8954e
git fetch origin
git checkout --detach origin/agent/stage20g-routing-recovery   # or origin/main after merge
.\cost-aware-pipeline\scripts\stage20.ps1 -Mode Test
```

**`-Mode Test`** makes no inference request. It:
- verifies a clean commit;
- builds the image once per commit and checks its provenance;
- checks the credentials file for the required names only (`$HOME\stage20f-provider.env`);
- finds the manifest in `Desktop\stage20-operator`;
- runs the offline suites in the image without network;
- runs the non-inference preflight:
  - Groq and Gemini model metadata;
  - OpenRouter key validity, plus the public list of endpoints that support every parameter
    within the price ceiling;
- writes `stage20g-test-<sha7>.json` and prints the authorization fingerprint.

**`-Mode Run -Authorization <fingerprint>`** runs only after explicit owner approval:
1. It refuses unless the recomputed fingerprint matches and the campaign's results volume is
   new.
2. It creates an isolated network, PostgreSQL, Redis and results volume.
3. It migrates the database, runs the five questions and exports the results.

**`-Mode Report`** exports read-only:
- `…-summary.json`: sanitized;
- `…-review-private.json`: questions, answers, sources, candidates and routing events, for
  your review.

## 6. Tests

All tests are offline.

| Test file | Coverage |
|---|---|
| `tests/test_routing_policy.py` (pure) | every WAIT / FALLBACK / SEND / STOP branch, compatibility, allowlist, configuration |
| `tests/test_stage20g_routing.py` (real gateway, OpenAI SDK serialization, throttle, an in-memory Redis twin of the admission script, fake clock) | see below |
| `tests/test_stage20g_e2e.py` | campaign arithmetic and fingerprints; contract checks; case- and campaign-scoped stops; concurrency; the E001 regression through guarded transports; 429 fallback with both requests reserved; exhausted fallback budget; uncertain sends charged and never replayed; journal failure preventing transmission; real chat SDKs and the Gemini embedding SDK |
| `tests/test_stage20g_runner.py` | manifest location; fingerprint binding; run refusals before any side effect; no overwriting on export |
| `tests/integration/test_provider_throttle.py` (real Redis) | `exceeds_window`; bounded estimate |

`test_stage20g_routing.py` covers:
- the E001 regression: a 31.8 s wait, then exactly one send;
- the legacy failure mode, kept for comparison;
- RPM exhaustion, and a request larger than the whole window;
- a wait the deadline cannot afford;
- an oversize request falling back, with the exact OpenRouter body;
- an incompatible fallback that is never built;
- an exhausted fallback budget;
- the allowlist;
- HTTP 429 and 5xx, timeouts with an unknown outcome, and `json_validate_failed`;
- a valid JSON response that fails the schema, and a failed local output validation;
- a transport refusal;
- concurrent admissions.

## 7. Limitations

- Provider compatibility, pricing, quotas and scientific accuracy are not established by
  fake HTTP responses. Some facts can only be confirmed by a live request:
  - whether OpenRouter accepts the `max_price` number format;
  - whether an upstream endpoint honours `json_schema` together with `reasoning_effort` at
    the ceiling price.

  The preflight checks only the public endpoint metadata.
- Token estimates are character-based heuristics. The local 6,000-TPM budget is kept below
  Groq's published limit as headroom.
- Declared route context and output limits are conservative declarations, not guarantees.
- The PowerShell script was written for Windows PowerShell 5.1 and PowerShell 7, but could
  not be executed in this environment.
- The consolidated policy is opt-in. Promoting it to the production default is a separate
  decision, after the E2E results are reviewed.
