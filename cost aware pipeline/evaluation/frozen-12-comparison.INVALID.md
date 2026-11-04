# Invalid architecture comparison

`frozen-12-comparison.json` is retained only as an audit artifact. Its results
must not be used to compare Naive, Hybrid, Hybrid Enhanced, or Agentic RAG.

The experiment is invalid because it:

- approximated retrieval with in-memory token overlap instead of the real
  pgvector and PostgreSQL full-text paths;
- did not run the production MMR and parent-child implementation;
- used a simplified planner loop instead of the production `ResearchAgent`;
- searched each question only against its own reference excerpts, leaking the
  expected evidence and making Recall@K misleading.

The legacy runner is intentionally disabled. The replacement experiment must
use one shared frozen corpus, the real four implementations, complete LLM usage
tracing, and tests that prove reference answers are not exposed to retrieval or
generation.
