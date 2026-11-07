# Real implementation map — 12-case pilot

This document is the implementation contract for **תכנית 10 השלבים**. It
prevents the benchmark from silently substituting toy retrieval logic for the
application code.

## Shared controls

- All variants query one shared, immutable PostgreSQL snapshot identified by a
  benchmark `research_run_id` and content manifest hash.
- All variants use the configured 768-dimensional Gemini embedding model and
  the same child/parent chunks.
- Reference answers, reference claims, expected source IDs, and validation
  labels are scoring data only. They are never placed in a retrieval or answer
  generation prompt.
- The answer model route and grounded-answer instructions are shared by the
  three non-agentic variants. Agentic Fast uses the production synthesis prompt
  because that prompt is part of the production agent implementation.
- External PubMed/arXiv mining and global evidence reuse are forbidden during
  the frozen comparison.

## Variant mapping

### Naive

- Real entry point: `PostgresVectorService.retrieve_context(..., strategy="naive")`.
- Executes the real Gemini query embedding and pgvector cosine-distance query.
- Returns top-k child chunks.
- Does not execute PostgreSQL full-text search, RRF, MMR, parent expansion,
  planning, or external mining.
- One grounded answer-generation call.

### Hybrid Basic

- Real entry point: `PostgresVectorService.retrieve_context(..., strategy="hybrid_basic")`.
- Executes pgvector dense retrieval and PostgreSQL `websearch_to_tsquery` full-
  text retrieval.
- Combines the two ranked lists with the production Reciprocal Rank Fusion
  implementation (`rank_constant=60`).
- Returns top-k child chunks.
- Does not execute MMR, parent expansion, planning, or external mining.
- One grounded answer-generation call.

### Hybrid Enhanced

- Real entry point: `PostgresVectorService.retrieve_context(..., strategy="enhanced")`.
- Executes dense retrieval, PostgreSQL full-text retrieval, production MMR,
  production RRF, deduplication by parent, and parent-context expansion.
- Does not execute query decomposition or external mining.
- One grounded answer-generation call.
- Cross-Encoder reranking is explicitly excluded.

### Agentic Fast

- Real entry point: `scientific_rag.legacy_pipeline.ResearchAgent.run_research`.
- Uses a strategy-bound adapter that pins every retrieval to the real
  `enhanced` path above.
- Executes the production Orchestrator planner and gatekeeper, then passes the
  accepted benchmark-controlled query to the production ResearchAgent evidence
  deduplication and Expert synthesis path.
- Uses the production Fast budget from `app.cost_policy`:
  `max_synthesis_subquestions=1`, `retrieval_documents=4`, and
  `max_context_documents=5`.
- The production `Orchestrator.execute_full_study` is deliberately not called:
  it performs routing, cache reuse, and external mining, which would mutate the
  corpus and invalidate the frozen-corpus experiment.
- `FrozenResearchAgent.decompose_query_async` returns the already accepted
  query instead of performing another model-driven decomposition. Therefore
  this arm tests real production components, but it is not the complete live
  Fast workflow.
- Agentic Fast returns at most four retrieval contexts under the current Fast
  budget; the non-agentic arms return five. Source recall is therefore useful
  descriptively but is not a perfectly equal-context-budget comparison.

## Measurements

The harness must wrap the complete variant execution in one LLM gateway trace.
For Agentic Fast this includes both Planner and Expert calls. It records real
provider/model, attempts, fallbacks, input/output tokens when returned, known
cost (or `null` when pricing/usage is incomplete), embedding usage, wall-clock
latency, selected source IDs, selected chunk IDs, and retrieval diagnostics.

Recall@K is computed only after execution by comparing retrieved source IDs to
the hidden verified source IDs. A retrieved source is not considered correct
merely because it came from the same question's source group.
