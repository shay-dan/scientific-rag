# RAG comparison — current conclusions

This document summarizes the experiments without committing the large raw JSON
outputs. The experiments are exploratory Silver evaluations, not a final
production baseline.

## 1. Retrieval-only ablation

The 12-question frozen PostgreSQL/pgvector retrieval ablation used the same
query embedding for every strategy and made no generation or judge calls.

| Strategy | Mean source recall | FTS coverage | Mean unique sources |
|---|---:|---:|---:|
| Dense pgvector | 0.9514 | 0/12 | 3.42 |
| Dense + PostgreSQL FTS + RRF | 0.9514 | 12/12 | 3.67 |
| Dense + MMR | 0.7847 | 0/12 | 4.83 |
| Dense + MMR + FTS + RRF | 0.9097 | 12/12 | 4.25 |
| Dense + MMR + FTS + RRF + parent context | 0.9097 | 12/12 | 4.25 |

Interpretation:

- PostgreSQL FTS and RRF were functioning on every case, but this small
  answer-derived corpus did not improve recall over dense retrieval.
- MMR alone reduced recall from 0.9514 to 0.7847.
- Adding FTS/RRF recovered part of that loss, to 0.9097, but not all of it.
- Parent expansion made no difference because this snapshot contained almost
  one short passage per parent. It needs a realistic multi-child corpus before
  its value can be measured.

Decision: keep the hybrid components available, but do not claim that MMR or
parent/child expansion improved quality from this experiment alone.

## 2. Four-way RAG comparison

The exploratory 12-question frozen-corpus comparison used four configurations:

- **Naive RAG:** dense pgvector search and one answer call.
- **Hybrid Basic:** pgvector plus PostgreSQL full-text search and RRF.
- **Hybrid Enhanced:** Hybrid Basic plus MMR and parent/child expansion.
- **Agentic Fast:** planning, source routing, gatekeeping, retrieval and
  synthesis under the Fast budget.

| Configuration | Success | Recall@5 | Correctness | Citation correctness | Citation completeness | Overall quality | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive | 100% | 1.0000 | 0.9250 | 0.9333 | 0.9833 | 0.9042 | 11.60s |
| Hybrid Basic | 100% | 0.9792 | 0.9667 | 1.0000 | 1.0000 | **0.9250** | 12.86s |
| Hybrid Enhanced | 100% | 0.9792 | 0.9208 | 0.9708 | 0.9667 | 0.8958 | 14.61s |
| Agentic Fast | 100% | 0.8611 | 0.9125 | 1.0000 | 0.9417 | 0.8625 | 12.75s |

Interpretation:

- Hybrid Basic was the descriptive leader in this frozen pilot.
- Hybrid Enhanced did not beat Hybrid Basic; its additional retrieval logic
  added latency without improving this corpus-level score.
- Agentic Fast did not justify its extra orchestration complexity on this
  frozen corpus.
- This does **not** prove Agentic RAG is unnecessary: external mining was
  disabled, so the experiment did not measure its main advantage—finding new
  evidence when the database is insufficient.

Important confounders: one run per question/configuration, automated Silver
ground truth, provider fallback, and a smaller context budget for Agentic Fast.

## 3. Provider comparison

On the same saved evidence contexts, two repetitions across six cases showed:

| Expert model/provider | Successful generations | Overall score | Mean latency | Known generation cost |
|---|---:|---:|---:|---:|
| Groq GPT-OSS 120B | 6/12 | 0.800 | 2.42s | $0.0036 |
| OpenRouter GPT-OSS 120B | 5/12 | 0.760 | 6.00s | $0.0007 |
| Gemini 3.8 Flash | 12/12 | 0.917 | 13.18s | $0.0726 |
| OpenRouter Claude Haiku 5.5 | 12/12 | **0.925** | **6.85s** | **$0.0104** |

The result suggests that model behavior mattered more than simply changing
providers. Paying for the same GPT-OSS model did not fix citation compliance.
Haiku and Gemini were more reliable in this limited experiment.

The later v14 structured-contract test also showed that Qwen should not be
treated as a guaranteed single provider for nested evidence output. The
provider-agnostic fallback chain is therefore important.

## 4. What can and cannot be claimed

Supported claims:

- The project implements and tests Naive, Hybrid Basic, Hybrid Enhanced and
  Agentic Fast comparison paths.
- Hybrid Basic was the strongest descriptive configuration in the current
  12-question frozen pilot.
- MMR and parent/child expansion were not demonstrated to improve recall on
  the current snapshot.
- The gateway provides fallback and cost/latency observability.

Not yet supported:

- A statistically stable winner across domains.
- A production baseline.
- Proof that Agentic RAG is worse on live open-mining research.
- A 100-question result.
- Exact total cost when providers omit token usage for failed attempts.

## Next experiment

Pin one Expert model, equalize every configuration to the same context budget,
run the same frozen cases at least three times, and report mean, median,
standard deviation, timeout rate, fallback rate, latency and cost. Only after
that controlled pilot should the benchmark expand toward 100 questions.

Source reports:

- `evaluation/retrieval-ablation-12x5-report.md`
- `evaluation/real-frozen-v3-12x4-report.md`
- `evaluation/provider-model-comparison-report.md`
- `evaluation/comparative-benchmark-plan.md`
