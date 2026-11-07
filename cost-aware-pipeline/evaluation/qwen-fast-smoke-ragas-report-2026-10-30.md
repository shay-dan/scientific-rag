# Qwen Fast smoke — saved-run Ragas validation

Date: 2026-10-30

Three previously completed Fast research runs were scored without repeating research, mining, or retrieval. The cases cover one PubMed-led question, one arXiv-led question, and one mixed PubMed/arXiv question. This is a three-case smoke validation, not a broad benchmark.

| Domain | Faithfulness | Answer relevancy | Ragas context precision | Context recall | Strict claim recall | Partial-credit claim recall |
|---|---:|---:|---:|---:|---:|---:|
| PubMed / CAR-T | 0.929 | 0.885 | 1.000 | 0.500 | 0.000 | 0.300 |
| arXiv / surface codes | 1.000 | 0.996 | 1.000 | 0.250 | 0.400 | 0.700 |
| Mixed / mammography ML | 0.900 | 0.941 | 0.000 | 1.000 | 0.600 | 0.700 |
| **Mean** | **0.943** | **0.941** | **0.333** | **0.583** | **0.333** | **0.567** |

## Interpretation

- The answers are strongly grounded in the evidence they actually use: mean faithfulness is 0.943.
- They remain responsive to the questions: mean answer relevancy is 0.941.
- Evidence coverage is the main weakness. The PubMed run retained one paper focused on non-relapse mortality and toxicities, so it could not cover efficacy and remission claims. The arXiv run covered code variants and concatenation but only partially covered stabilizer measurement and implementation constraints.
- The mixed case exposes a calibration disagreement. Strict Ragas context precision is 0.0, while the separate claim diagnostic marks three of five reference claims supported and one partially supported. The raw Ragas result is preserved; it is not replaced with the claim-aware score.
- With only three cases, these means are diagnostic and must not be presented as a production benchmark or confidence interval.

## Evaluation compatibility fixes

- Ragas 0.4.3 now uses Gemini's OpenAI-compatible async endpoint instead of the incompatible native Google Instructor path.
- Gemini 2.5 Flash reasoning is disabled for the judge (`reasoning_effort=none`) so internal thinking does not consume the structured-output budget.
- The Ragas embedding path now uses a modern `BaseRagasEmbedding` adapter around the current Google GenAI `EmbedContentConfig` API.
- Research answers and evidence were read from PostgreSQL; no research API calls were repeated.

Raw outputs:

- [`results/qwen-smoke-pubmed-ragas.json`](results/qwen-smoke-pubmed-ragas.json)
- [`results/qwen-smoke-arxiv-ragas.json`](results/qwen-smoke-arxiv-ragas.json)
- [`results/qwen-smoke-mixed-ragas.json`](results/qwen-smoke-mixed-ragas.json)
