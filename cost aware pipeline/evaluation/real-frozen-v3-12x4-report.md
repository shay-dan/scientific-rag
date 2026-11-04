# Repaired Silver v3: real frozen RAG comparison

This is an **exploratory 12-question Silver pilot**, not a production baseline.

## Results

| Variant | Success | Recall@5 | Correctness | Relevancy | Citation correctness | Citation completeness | Overall quality | Mean latency | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 100% | 1.0000 ± 0.0000 | 0.9250 | 0.9792 | 0.9333 | 0.9833 | 0.9042 ± 0.0967 | 11.60s | 5 |
| hybrid_basic | 100% | 0.9792 ± 0.0691 | 0.9667 | 0.9417 | 1.0000 | 1.0000 | 0.9250 ± 0.1010 | 12.86s | 8 |
| hybrid_enhanced | 100% | 0.9792 ± 0.0691 | 0.9208 | 0.9083 | 0.9708 | 0.9667 | 0.8958 ± 0.1282 | 14.61s | 7 |
| agentic_fast | 100% | 0.8611 ± 0.2214 | 0.9125 | 0.9000 | 1.0000 | 0.9417 | 0.8625 ± 0.1721 | 12.75s | 7 |

## Interpretation

- Descriptive leader: **hybrid_basic**.
- Do not change the production default from this pilot alone.
- There is one run per question/configuration, no human review, and provider fallback changed the answer model across runs. Agentic Fast also had a smaller retrieval budget than the other arms.

The comparison suggests that the fixed Hybrid Basic pipeline is the strongest candidate for simple questions in this small frozen-corpus test. Agentic Fast did not earn back its additional orchestration complexity here, but this benchmark deliberately disabled live mining and therefore does not test its main advantage: finding missing evidence dynamically.

## Cohort breakdown

| Cohort | Variant | Recall@5 | Overall quality | Mean latency |
|---|---|---:|---:|---:|
| arxiv | naive | 1.0000 | 0.8125 | 2.14s |
| arxiv | hybrid_basic | 1.0000 | 0.9750 | 13.89s |
| arxiv | hybrid_enhanced | 0.9375 | 0.8125 | 13.11s |
| arxiv | agentic_fast | 0.7708 | 0.8000 | 9.11s |
| mixed | naive | 1.0000 | 0.9250 | 12.53s |
| mixed | hybrid_basic | 0.9375 | 0.8500 | 19.86s |
| mixed | hybrid_enhanced | 1.0000 | 0.9125 | 14.52s |
| mixed | agentic_fast | 0.8125 | 0.7875 | 14.85s |
| pubmed | naive | 1.0000 | 0.9750 | 20.12s |
| pubmed | hybrid_basic | 1.0000 | 0.9500 | 4.82s |
| pubmed | hybrid_enhanced | 1.0000 | 0.9625 | 16.21s |
| pubmed | agentic_fast | 1.0000 | 1.0000 | 14.29s |

## Provider behavior

| Variant | Groq attempts/successes | OpenRouter attempts/successes | Gemini attempts/successes | Runs with fallback |
|---|---:|---:|---:|---:|
| naive | 12/7 | 5/0 | 5/5 | 5/12 |
| hybrid_basic | 12/4 | 8/2 | 6/6 | 8/12 |
| hybrid_enhanced | 12/5 | 7/2 | 5/5 | 7/12 |
| agentic_fast | 36/29 | 7/4 | 3/3 | 7/12 |

Agentic Fast includes Planner and Gatekeeper calls in addition to Expert synthesis, so its provider-attempt count is intentionally not directly comparable with the three single-stage pipelines.

## Cost accounting

- Snapshot embeddings: $0.000917
- Automated judge: $0.037485
- Known research lower bound: $0.157485
- Research cost complete: **False**
- Missing research usage is concentrated in rejected/rate-limited Groq attempts and the free OpenRouter route; the reported research amount remains a conservative measured lower bound.

## Winner counts

- naive: 5 wins including ties; 0 unique wins.
- hybrid_basic: 8 wins including ties; 1 unique wins.
- hybrid_enhanced: 7 wins including ties; 2 unique wins.
- agentic_fast: 7 wins including ties; 0 unique wins.

## Limitations

- The 12 references are automatically repaired and independently model-audited Silver data, not human-reviewed Gold data.
- There is one execution per case and variant, so repeated-run variance is not measured.
- Provider fallback changed the answer-generating model and confounds architecture with model availability.
- Agentic Fast returned at most four contexts while the other variants returned five.
- External mining was disabled to keep one frozen corpus; this does not test the full live Agentic workflow.
- The questions were generated from the frozen evidence, which favors direct retrieval.
- The judge is an automated Gemini judge and not a human accuracy assessment.
- Research cost is a measured lower bound where providers omitted usage metadata.

## Recommended next experiment

Pin one answer model, equalize every arm to five contexts, and repeat the same 12 questions three times. Then analyze simple versus multi-hop questions separately. Only after those controls pass should the benchmark expand toward 100 questions.
