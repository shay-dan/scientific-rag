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
| Structured synthesis (`ResearchAgent.run_research`, `generation_prompt_v2`, `repair_prompt_v2`) | passages, titles, PMC section names | one `EVIDENCE` block per passage; `source_id` and pipeline notes are application-owned and outside the block | verbatim-quote validation; **quotes that reproduce instruction-like text are rejected**; one repair, then fail closed |
| Semantic verification (`app.claim_verification`) | claim, quote, title, passage | existing JSON framing (`json.dumps` escapes newlines, so a forged end marker stays inside a string); new `instruction_like_content` field | **`supported` is downgraded to `insufficient_evidence`** when the evidence is flagged |
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

Detection never removes evidence and never rejects a paper. It only:

1. prevents an instruction sentence from being quoted as evidence;
2. prevents a verifier from certifying support from flagged evidence;
3. records a flag for audit.

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

## Not verified

- Whether real models follow the data rule or ignore injections. That needs a paid live
  probe, which requires owner approval.
- Injected *scientific misinformation* phrased as a normal finding: no detector can
  flag it, and structural validation accepts it. Only the verifier guard (when the same
  passage also contains flagged text) and future R5–R8 work address it.
- Homoglyph and paraphrased injections that evade the regex.
- Ragas internal prompts.

## Known false positives

Papers about prompt injection, or transcripts with `Assistant:` lines, are flagged.
The effect is bounded: such quotes cannot be cited, and support from them is
downgraded. The papers are still retrieved and screened.
