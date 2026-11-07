# Why Hybrid Enhanced lost evidence

This is a provider-free post-hoc analysis of the saved 12-case pilot. It did not run retrieval, research, embeddings, or LLM calls.

## Aggregate diagnosis

- Cases with lower source recall: **5 / 12**
- Cases with higher source recall: **0 / 12**
- Expected sources present in Basic but lost by Enhanced: **6**
- Expected sources newly recovered by Enhanced: **1**
- Unexpected sources added by Enhanced: **25**
- Queries with no PostgreSQL FTS hit: Basic **12**, Enhanced **12**
- Candidates rejected as duplicate parents: **0**
- Naive and Hybrid Basic with identical ordered contexts: **12 / 12**

## Per-case evidence changes

| Case | Group | Basic recall | Enhanced recall | Delta | Lost expected sources | Added unexpected sources |
|---|---|---:|---:|---:|---:|---:|
| pm-01 | pubmed | 1.0000 | 0.6667 | -0.3333 | 1 | 3 |
| pm-02 | pubmed | 1.0000 | 1.0000 | +0.0000 | 0 | 3 |
| pm-03 | pubmed | 1.0000 | 1.0000 | +0.0000 | 0 | 2 |
| pm-04 | pubmed | 1.0000 | 0.5000 | -0.5000 | 1 | 3 |
| ax-01 | arxiv | 0.7500 | 0.7500 | +0.0000 | 1 | 1 |
| ax-02 | arxiv | 1.0000 | 0.6667 | -0.3333 | 1 | 3 |
| ax-04 | arxiv | 1.0000 | 1.0000 | +0.0000 | 0 | 2 |
| mx-01 | mixed | 1.0000 | 1.0000 | +0.0000 | 0 | 3 |
| mx-02 | mixed | 0.6667 | 0.6667 | +0.0000 | 0 | 2 |
| mx-03 | mixed | 1.0000 | 1.0000 | +0.0000 | 0 | 0 |
| mx-04 | mixed | 1.0000 | 0.5000 | -0.5000 | 1 | 1 |
| ax-03 | arxiv | 1.0000 | 0.6667 | -0.3333 | 1 | 2 |

## Lost-source trace details

- **pm-01 — Obesity and GLP-1.**: vector rank `5`, MMR rank `7`, relevance `0.770492`, redundancy `0.822349`, MMR score `-0.025928`, FTS rank `None`, reasons `outside_final_top_n_after_rrf`.
- **pm-04 — The threat of antimicrobial resistance in developing countries: causes and control strategies.**: vector rank `5`, MMR rank `9`, relevance `0.654364`, redundancy `0.764453`, MMR score `-0.055044`, FTS rank `None`, reasons `outside_final_top_n_after_rrf`.
- **ax-01 — Latency-Constrained Hardware-Aware Quantum Error Correction Co-Design with Adaptive Confidence-Gated Neural Decoding for the Rotated Surface Code**: vector rank `3`, MMR rank `6`, relevance `0.698681`, redundancy `0.776646`, MMR score `-0.038983`, FTS rank `None`, reasons `outside_final_top_n_after_rrf`.
- **ax-02 — Text-to-image Diffusion Models in Generative AI: A Survey**: vector rank `2`, MMR rank `6`, relevance `0.791515`, redundancy `0.854079`, MMR score `-0.031282`, FTS rank `None`, reasons `outside_final_top_n_after_rrf`.
- **mx-04 — Genome-wide CRISPR screens of T cell exhaustion identify chromatin remodeling factors that limit T cell persistence.**: vector rank `2`, MMR rank `7`, relevance `0.782348`, redundancy `0.851989`, MMR score `-0.03482`, FTS rank `None`, reasons `outside_final_top_n_after_rrf`.
- **ax-03 — Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention**: vector rank `3`, MMR rank `12`, relevance `0.737502`, redundancy `0.8179`, MMR score `-0.040199`, FTS rank `None`, reasons `outside_final_top_n_after_rrf`.

## Interpretation

The saved run did **not** exercise a working lexical+dense fusion: PostgreSQL FTS returned zero candidates for all 12 natural-language queries. Consequently, Hybrid Basic reduced to dense retrieval and returned the same ordered contexts as Naive in every case. Their different answer scores are generation/provider variance, not a retrieval improvement.

Hybrid Basic and Enhanced start from the same dense candidates. With no FTS list, RRF cannot combine independent signals; Enhanced differs primarily because it runs MMR before RRF, followed by parent-level deduplication.

In this snapshot, parent expansion is structurally unable to help: each parent is only one short benchmark excerpt and has one child. If duplicate-parent rejections are zero, parent deduplication also did not cause the measured loss. The remaining differentiator is MMR reordering before RRF.

With `lambda=0.5`, MMR gives relevance and non-redundancy equal weight. The six expected sources that disappeared were originally at dense ranks 2–5 and were moved to MMR ranks 6–12. On this tiny corpus, the diversity penalty therefore replaced answer-bearing sources with diverse but mostly non-reference sources. RRF could not correct that movement because it received no FTS-ranked list.

## Next controlled check

First repair the FTS query builder and add a regression test proving non-zero lexical hits for representative PubMed, arXiv, and mixed questions. Then run a retrieval-only ablation on a realistic full-text corpus: Dense, Dense+FTS+RRF, Dense+MMR, and Dense+FTS+RRF+parent handling. It requires no answer generation and isolates each component.
