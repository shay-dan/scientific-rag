# Historical implementations

The maintained Scientific RAG application lives in
[`cost-aware-pipeline/`](../cost-aware-pipeline/).

`gradio-poc/` preserves the earlier notebook-to-application migration: a Gradio UI,
Chroma-based retrieval experiments, the original multi-agent pipeline, a small
packaged wrapper, Docker files, and their unit tests. It is retained for provenance,
comparison, and access to the project's design history.

This implementation is not the supported production path. Its dependencies and
commands are preserved together and CI still runs its offline unit tests, but it is
not expected to receive new features or operational hardening. Treat its source and
configuration as historical context; use the root and maintained-application
documentation for current setup.
