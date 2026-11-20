# Stage 20D — bounded five-question end-to-end preparation

This PR is offline preparation, not permission to execute a live campaign.
The production pipeline and defaults are unchanged. The private five-question
development manifest is owner-held, not committed. Verify novelty against prior
private manifests (public checked-in question datasets inspected; no obvious
topic overlap). This is not a gold or holdout evaluation.

## Exact ceilings

- Five cases, sequential, fast research, fresh source discovery, v16, Unicode
  quotation recovery, degrade, semantic enforce, audit, no synthetic QA/judge.
- Groq GPT-OSS 120B only, temperature zero, low reasoning, 2,000 completion tokens.
- Gemini embedding-2 only, 768 dimensions, text only.
- 20 chat / 25 embedding requests per question, 100 / 125 over the campaign.
- Wire-body byte length plus 512 framing allowance as conservative admission
  rule: 32,768 chat input, 8,192 embedding aggregate input. This is a modeled
  allowance, not verified provider token billing; refuse oversize wire requests.
- Fixed full reservation per admitted request: $0.0061152 chat, $0.0016384
  embedding (dated repo rates $0.15/$0.60 and $0.20 per million tokens).
  Maximum modeled reservation $0.81632; separate cap $0.85. Never refund, even
  on apparent success or uncertain send. Provider receipts and estimated usage
  can be reported separately. No transfer from closed R7D program.
- 600s per question, 3,000s campaign admission deadline; in-flight cancellation
  cannot undo an already admitted transmission. Stop on HTTP error, uncertain
  transmission, gateway failure, limit, deadline or persistence failure.

The adapter injects the existing gateway's bounded low-reasoning path and
requires its original physical-request guard. The separate embedding SDK must
use the guarded HTTP transport. The mocked pinned-SDK test verifies both chat
call forms and actual Gemini serialization and counts. There is no implicit
network transport and no live CLI in this PR. Use a dedicated process: patches
are not safe for a shared API server. The final local operator invocation must
bind exact merge SHA, manifest and protocol SHA, provider prices/account tier,
private exclusive journal path and explicit owner authorization; it must use
the actual production Orchestrator, PostgresVectorService and Redis throttle,
and stop before each new case on any sticky stop. A journal cannot be reused or
resumed; do not copy it or change paths to bypass the bound. Final authorization
and private operator wiring remain gates after CI, not claims of live readiness.

Offline within pinned image (\`--network none\`):
\`\`\`
python -m unittest discover -s tests -p test_stage20_e2e_budget.py -v
python scripts/prepare_stage20.py --manifest /private/stage20-questions.json
\`\`\`

Review exact request shapes and any rejected bounds before live preparation.
No provider compatibility, account pricing/quotas, source recall or scientific
accuracy is established by fake HTTP responses. The private review must include
all attempted cases and non-delivery categories, retrieved and selected evidence,
initial/repair/pre-semantic/final candidates, claim and limitation dispositions,
source inspection, latency, request reservations and known/uncertain usage.

## Next evaluation, not another architecture change

After the five-question production E2E pilot (separate authorization), prioritize
R9 controlled comparisons: retrieval-only ablation with shared embeddings on
reviewed frozen corpora; then five questions x four arms (naive dense, basic
hybrid, enhanced hybrid, agentic planning + enhanced hybrid). Existing
app/real_comparison.py and frozen-corpus runner are reusable but must explicitly
share the same downstream model/prompt/recovery/verifier and context budget;
current defaults and top-k differ. Frozen-corpus agentic planning is NOT live
autonomous discovery; report that separately in a later full comparison.
Expand only if the smoke comparison is sound. Blind owner review, no live judge
calls by default. Do not treat the illustrative expected architecture ranking
as a measured result. R8 verifier calibration/release decision remains a distinct
gate; never change production defaults on the five-question pilot alone.
