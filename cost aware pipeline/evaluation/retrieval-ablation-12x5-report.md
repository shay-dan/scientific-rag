# Refined retrieval-only ablation

This experiment used the same frozen PostgreSQL/pgvector snapshot for all strategies. It made no answer-generation, Planner, Expert, Gatekeeper, or Judge calls. One query embedding was shared by all five strategies for each question.

## Aggregate results

| Strategy | Mean source recall | FTS coverage | Mean FTS candidates | Mean unique sources |
|---|---:|---:|---:|---:|
| Dense | 0.9514 | 0/12 | 0.00 | 3.42 |
| Dense + FTS + RRF | 0.9514 | 12/12 | 5.25 | 3.67 |
| Dense + MMR | 0.7847 | 0/12 | 0.00 | 4.83 |
| Dense + MMR + FTS + RRF | 0.9097 | 12/12 | 5.25 | 4.25 |
| Dense + MMR + FTS + RRF + Parent | 0.9097 | 12/12 | 5.25 | 4.25 |

## Paired effects

- Adding FTS+RRF to Dense: 0 improved, 0 degraded, 12 unchanged.
- Adding FTS+RRF to MMR: 4 improved, 0 degraded, 8 unchanged.
- Adding parent handling after Hybrid+MMR returned identical ordered contexts in 12/12 cases.

## Per-case source recall

| Case | Group | Dense | Hybrid Basic | MMR | Hybrid MMR | Enhanced | FTS candidates |
|---|---|---:|---:|---:|---:|---:|---:|
| ax-01 | arxiv | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 2 |
| ax-02 | arxiv | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 4 |
| ax-03 | arxiv | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 3 |
| ax-04 | arxiv | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3 |
| mx-01 | mixed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14 |
| mx-02 | mixed | 0.6667 | 0.6667 | 0.6667 | 1.0000 | 1.0000 | 8 |
| mx-03 | mixed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2 |
| mx-04 | mixed | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 10 |
| pm-01 | pubmed | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 4 |
| pm-02 | pubmed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 |
| pm-03 | pubmed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4 |
| pm-04 | pubmed | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 8 |

## Interpretation

The refined FTS query builder fixed the zero-hit failure: every question produced lexical candidates. Dense and Hybrid Basic now have equal expected-source recall (0.9514), so this small answer-derived corpus does not yet demonstrate an FTS quality gain, but it does show that lexical retrieval is functioning without harming recall.

MMR alone reduced recall from 0.9514 to 0.7847. FTS+RRF recovered it to 0.9097, showing that an independent lexical signal can repair some diversity-induced losses. It still remained below the Dense baseline, so MMR should not be enabled unconditionally from this evidence.

Parent handling changed nothing in all cases. That is expected: this benchmark stores one short excerpt per parent and one child per parent. Parent-child retrieval must be tested on full-length documents with multiple children before it can be judged.

## Cost and decision

- Query embedding calls: 12.
- Estimated embedding cost: $0.00006560.
- Answer-generation and judge calls: 0.
- Production default: unchanged. The next valid experiment requires a realistic full-text corpus, not more tuning against these answer-selected excerpts.
