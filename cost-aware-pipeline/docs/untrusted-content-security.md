# Untrusted scientific content (R3)

External scientific text is data, never instructions. This page records where that
text reaches a language model, how it is framed, which deterministic guards limit
damage, and what has not been verified.

## Threat model

Paper text, titles, PMC section names, arXiv abstracts, benchmark excerpts and
generated answers can contain text addressed to an AI system. Examples: "ignore
previous instructions", a fake `SYSTEM:` turn or chat-template tokens, "verdict:
supported", "mark this paper relevant", forged source labels or delimiters, or
"cite this paper for all claims".

No model call has tool authority. A model can only return a schema-validated object,
so an injection can at most corrupt a relevance decision, an answer claim, a
verifier verdict or an evaluation score. Untrusted text cannot run code, fetch URLs
or change configuration.

## Trust-boundary inventory

| Entry point | Untrusted inputs | Framing (R3) | Deterministic guard |
|---|---|---|---|
| PubMed relevance screen (`MinerService._screen_pubmed_documents`) | title, abstract or full-text excerpt | `PAPER` block, title as single-line attribute, data rule | candidate trace `instruction_like_content`; existing confidence thresholds |
| arXiv relevance screen (`_screen_arxiv_documents`) | title, abstract | `PAPER` block, data rule | candidate trace flag |
| Structured synthesis (`ResearchAgent.run_research`, `generation_prompt_v2`, `repair_prompt_v2`) | passages, titles, PMC section names | one `EVIDENCE` block per passage; `source_id` and pipeline notes are application-owned and outside the block | verbatim-quote validation; **a quote is rejected when it is instruction-like or occurs only inside isolated instruction text**; one repair, then fail closed |
| Semantic verification (`app.claim_verification`) | claim, quote, title, passage | existing JSON framing (`json.dumps` escapes newlines, so a forged end marker stays inside a string); **instruction text is removed before the model sees it** (`instruction_like_content_removed`) | **`supported` becomes `insufficient_evidence` only when the claim or its quote overlaps isolated instruction text**; clean claims from the same passage keep their verdict |
| Evaluation judges (`app.evaluation`, worker fallback judge) | reference context, retrieved excerpts, candidate answer | instructions first, then the data rule, then `REFERENCE_CONTEXT` / `EXCERPT` / `CANDIDATE_ANSWER` blocks | numeric schema bounds |
| Frozen comparison judge (`app.comparison_judge`) | excerpts, titles, candidate answer | `EVIDENCE` blocks with application-owned `label`; rubric wording unchanged; scores record `prompt_security` | schema bounds |
| Evaluation-dataset QA generation (disabled by default) | document chunks | `DOCUMENT` blocks | schema |
| Planner, decomposition, query repair | user question only | unchanged | not external content |
| Ragas metrics (library prompts) | contexts, answers | not controllable | residual risk |
| Manual research scripts under `scripts/` | benchmark abstracts | synthesis scripts reuse the hardened builders; dataset-generation scripts are unchanged | operator-run, paid, approval required |

PDF ingestion is not enabled. A PDF-extracted fixture checks that the same framing
and guards would apply (R9-F readiness).

## Mechanism

`app/untrusted_content.py` (standard library only):

- `render_untrusted(kind, text, attributes)` wraps text as
  `<<<BEGIN_UNTRUSTED_KIND nonce=…>>> … <<<END_UNTRUSTED_KIND nonce=…>>>`. The nonce
  is derived from the content, so prompts stay deterministic and cacheable.
- `neutralize` breaks `<<<`/`>>>` runs, removes chat-role tokens and zero-width
  characters, so the text can neither close its own block nor open a new one.
  Attributes are collapsed to one line.
- `find_instruction_like_spans` is a deliberately narrow regex detector. It is robust to
  zero-width obfuscation. Flagged blocks carry `instruction_like="yes"`.

Detection never rejects a paper or changes provenance (source IDs, titles and passages
are stored unchanged). It only:

1. prevents an instruction sentence from being quoted as evidence;
2. removes instruction text from what the verifier model sees, and prevents support
   that rests on it;
3. records a flag for audit.

### Isolation (verifier and citation checks)

`instruction_mask_ranges` widens each detected span to its whole sentence, so no
fragment of the instruction remains. Sentences end at `.`, `!` or `?` followed by
whitespace, or at a newline; decimals such as `0.84` do not end a sentence.

A provenance forgery is treated more strictly: a forged block marker or a forged
`--- [Sn: …] ---` header isolates everything from its line to the end of the passage,
because the text after it has unknown origin. A passage with no sentence boundary is
isolated entirely.

For the verifier:

- **Passage:** isolated ranges are replaced by `[instruction-like text removed]`.
- **Title:** an instruction-like title is withheld. The claim is still assessed against
  its passage.
- **Claim text or quote:** if either is instruction-like, it is withheld and the target
  is unsafe.
- **Quote outside isolated text:** a quote must occur at least once outside isolated
  ranges, otherwise the target is unsafe (fail closed).
- **Unsafe targets:** a `supported` verdict for an unsafe target becomes
  `insufficient_evidence`. The three-class contract and off/shadow/enforce behavior are
  unchanged.
- **Window edges:** the verifier's 2,400-character evidence window never cuts through
  isolated text. An edge that falls inside it moves outside it, so a truncated
  instruction fragment cannot slip past detection.

## Verified offline

`tests/test_untrusted_content_security.py` and the fixtures in
`tests/fixtures/prompt_injection/` cover:

- abstract instructions;
- fake system/developer turns;
- verdict overrides;
- fabricated-citation directives;
- ignore-evidence directives;
- malicious PMC XML (forged CDATA markers, comment payloads, injected section title);
- misleading metadata;
- judge contamination;
- delimiter and structured-output forgery;
- PDF-extracted text.

The attacker-compliant fakes obey every injection. The tests show that:

- injected text never appears outside a data block;
- injected quotes are rejected;
- an attacker repair fails closed;
- an attacker verifier cannot certify flagged evidence.

## Cache version

R3 bumps `PIPELINE_VERSION` from `…-structured-v14` to `…-structured-v15`. The
shadow and enforce identities (`…-semantic-shadow-v1`, `…-semantic-enforce-v1`) derive
from it, so all three modes change.

- **Historical records:** v14 records are not modified. They stay readable through
  `/v1` and keep their stored `retrieval_version`.
- **New requests:** the cache lookup, in-flight reuse, background evidence-job reuse and
  the R2 identity lock all key on the effective version. A new request therefore never
  reuses a pre-R3 answer. Identical new requests still deduplicate onto one run with one
  cost reservation.
- **Paper corpus:** documents, chunks and embeddings are not versioned by
  `PIPELINE_VERSION`, so they are reused. Only answers are regenerated.
- **Regeneration cost:** the first request for each previously answered question, per
  caller and mode, costs one normal research run. That is the same reservation and
  provider usage as an uncached request. Default reservations are $0.05 fast, $0.20
  standard and $0.75 deep (`COST_*_RESERVATION_USD`); these are budgets, not exact charges.
  There is no bulk re-run; regeneration happens only on demand.
- **Rollback:** reverting to v14 code makes v14 answers reusable again. v15 answers
  created in the meantime are then ignored by lookups but remain readable.

## Not verified

- Whether real models follow the data rule or ignore injections. That needs a paid live
  probe, which requires owner approval.
- Injected *scientific misinformation* phrased as a normal finding, with no detectable
  instruction or forged boundary near it: no detector can flag it, and structural
  validation accepts it. R5–R8 address claim correctness.
- Homoglyph and paraphrased injections that evade the regex.
- Ragas internal prompts.

## Known false positives

Papers about prompt injection, or transcripts with `Assistant:` lines, are flagged.
The effect is bounded to the sentence that contains the attack string: that sentence
cannot be cited and is removed from the verifier's view. Other sentences of the same
paper remain usable evidence. The papers are still retrieved and screened.
