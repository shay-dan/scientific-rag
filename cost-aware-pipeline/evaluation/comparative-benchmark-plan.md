# Comparative RAG benchmark

This experiment measures whether each layer of retrieval and orchestration is
worth its extra latency and cost. It does not include a cross-encoder and it
does not claim human review unless the dataset explicitly records verified
references.

## Configurations

| Configuration | Retrieval | Orchestration |
| --- | --- | --- |
| Naive RAG | pgvector dense top-k chunks | one answer-generation call |
| Hybrid Basic | pgvector + PostgreSQL full-text search + RRF | one answer-generation call |
| Hybrid Enhanced | Hybrid Basic + MMR + parent/child context expansion | one answer-generation call |
| Agentic Fast | production planner, source routing, gatekeeping, retrieval and synthesis | dynamic Fast budget |

The current production retrieval remains Hybrid Enhanced. The comparison
exposes the three retrieval strategies through
`PostgresVectorService.retrieve_context(..., strategy=...)`; the default is
still `enhanced`, so existing research behavior is unchanged.

## Dataset policy

The requested experiment needs 100 distinct questions, frozen answers, atomic
claims, and independently verified reference excerpts or source identifiers.
The repository currently has 30 reviewed cases (10 PubMed, 10 arXiv, 10
mixed). The runner refuses to treat those 30 as 100 and refuses to fabricate
the missing ground truth. Add the remaining cases before paid execution.

Human review is intentionally excluded from the runtime benchmark. Ground
truth preparation can be manual, but the evaluation run itself is automated.

## Metrics

- Recall@5 and Recall@10 against verified source identifiers.
- Citation/source support from the stored evidence and automated judge.
- Answer quality using the existing Ragas/LLM evaluation path.
- Latency, success rate, timeout rate, fallback rate and estimated/actual cost.
- Mean, median and population standard deviation across three repetitions.

Run the provider-free estimate first:

```powershell
python scripts/compare_rag_variants.py --estimate-only --case-count 100 --repetitions 3
```

Validate a completed frozen dataset without paid work:

```powershell
python scripts/compare_rag_variants.py --dataset evaluation/comparative-rag-v1.json --validate --case-count 100
```

The cost estimate is a conservative upper bound, not an invoice. It assumes
up to five generation calls for Agentic Fast, one generation call for each
non-agentic variant, shared embeddings, and one automated judge call per case
and repetition. External mining is excluded from the frozen-corpus estimate.

## Execution order

1. Complete and independently verify the 100-case frozen dataset.
2. Run the provider-free validation and cost preflight.
3. Run one pilot per variant on 5 cases.
4. Run the full 100-case matrix three times only after reviewing the pilot and
   confirming the cost ceiling.
5. Run a separate open-mining Agentic Fast experiment; do not mix it with the
   frozen-corpus comparison.
6. Publish the report only with the dataset version, model versions, corpus
   snapshot, configuration, and cost assumptions.

