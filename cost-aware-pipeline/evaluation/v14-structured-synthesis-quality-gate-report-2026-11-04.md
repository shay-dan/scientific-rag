# v14 structured-synthesis quality gate

Date: 2026-11-04

## Scope

- Production synthesis changed from free-form Markdown plus line-level review to a structured evidence contract.
- Every factual claim and factual limitation contains a source ID and a verbatim quote.
- Application code verifies the source and quote before rendering user-facing Markdown.
- At most one targeted repair is allowed; an invalid final contract fails safely.
- Ragas judge usage is now recorded from provider metadata. Embedding token use is separately marked as estimated.
- The live gate used three Fast cases only: one PubMed case (`pm-02`) and two arXiv cases (`ax-02`, `ax-04`).

## Saved-evidence style comparison

No retrieval or mining was repeated. The comparison used the exact evidence saved for earlier ax-02 and ax-04 runs.

- The earlier free-form ax-02 answer scored faithfulness 1.00, answer relevancy 0.92, context precision 1.00, and context recall 0.60 in this calibration run.
- OpenRouter Qwen was not reliable enough as a single provider for the nested structured contract. It produced serialized object text in one run, invalid structured output in another, and timed out on ax-04.
- Invalid coverage additions were correctly detected because their quoted text was not a verbatim substring of the claimed source.
- Decision: keep the structured contract, but do not pin production to Qwen alone. Use the resilient provider chain and reject invalid contracts.

## Live v14 results

| Case | Corpus | Completed | Latency | Fallbacks | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pm-02 | PubMed | yes | 98.4 s | 1 | 1.000 | 0.949 | 1.000 | 1.000 |
| ax-02 | arXiv | yes | 37.0 s | 0 | 1.000 | 0.916 | 0.500 | 0.600 |
| ax-04 | arXiv | yes | 60.8 s | 1 | 1.000 | 0.784 | 1.000 | 0.500 |
| **Mean** | mixed | **100%** | **65.4 s** | — | **1.000** | **0.883** | **0.833** | **0.700** |

Configured gates were faithfulness >= 0.85, answer relevancy >= 0.90, context precision >= 0.65, context recall >= 0.65, and 100% completion.

- Passed: completion, faithfulness, context precision, context recall.
- Missed: mean answer relevancy (0.883), driven by ax-04 (0.784).
- The structured contract eliminated unsupported answer claims in this gate, but it does not solve missing evidence coverage. ax-04 still needs better retrieval coverage or a more direct answer formulation.

## Usage and cost

Ragas usage was collected from actual response metadata across all metric and diagnostic calls.

| Case | Judge calls | Judge input tokens | Judge output tokens | Judge + embedding cost |
|---|---:|---:|---:|---:|
| pm-02 | 12 | 18,317 | 6,352 | $0.02139 |
| ax-02 | 10 | 11,372 | 3,188 | $0.01139 |
| ax-04 | 10 | 11,705 | 5,201 | $0.01653 |
| **Total** | **32** | **41,394** | **14,741** | **$0.04931** |

Research traces account for at least $0.0275 across the three runs. The exact research total is incomplete because two failed OpenRouter attempts returned no token usage metadata; those runs are correctly marked as incomplete rather than presented as exact costs. The measured combined lower bound is therefore approximately $0.0768, well below the conservative $0.45 reservation.

## Conclusion

v14 validates the main safety objective: all three live answers completed with deterministic source/quote validation and faithfulness 1.00. It is not yet a full quality pass because ax-04 lowered answer relevancy and reference-claim coverage. The next focused work should inspect ax-04 evidence selection and answer coverage; it should not broaden the benchmark until that case is understood.
