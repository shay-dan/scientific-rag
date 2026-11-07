# Expert model and provider comparison

Date: 2026-10-28

## Purpose

This experiment asks two separate questions:

1. Does paying for the same GPT-OSS 120B model through OpenRouter improve reliability over Groq?
2. Does a newer paid OpenRouter model or Gemini model improve the Expert synthesis step without changing retrieval evidence?

The experiment is intentionally small. It uses six representative cases and two repetitions per candidate. Every candidate receives the exact same saved Hybrid Basic context. It does not mine new papers, change retrieval, or expand to the planned 100-question benchmark.

## Cost approval before execution

The initial 36-call generation preflight estimated a maximum of **$0.1212**:

| Candidate | Maximum estimated generation cost |
| --- | ---: |
| Groq GPT-OSS 120B | $0.0168 |
| OpenRouter GPT-OSS 120B | $0.0040 |
| Gemini 3.8 Flash | $0.1004 |

The separate OpenRouter Claude Haiku 5.5 experiment estimated a maximum generation cost of **$0.0134**. Automated judging was estimated before each scoring run as well.

## Results

| Expert candidate | Successful generations | Correctness | Reference coverage | Groundedness | Overall | Reliability-adjusted overall | Mean successful latency | Known generation cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Groq · GPT-OSS 120B | 6/12 | 0.875 | 0.925 | 0.792 | 0.800 | 0.400 | 2.42 s | $0.0036 |
| OpenRouter · GPT-OSS 120B | 5/12 | 0.800 | 0.896 | 0.740 | 0.760 | 0.317 | 6.00 s | $0.0007 |
| Gemini · Gemini 3.8 Flash | 12/12 | 0.983 | 0.896 | 0.983 | 0.917 | 0.917 | 13.18 s | $0.0726 |
| OpenRouter · Claude Haiku 5.5 | 12/12 | 0.983 | 0.908 | 0.983 | 0.925 | 0.925 | 6.85 s | $0.0104 |

All quality scores are produced by the same automated Gemini 2.5 Flash comparison judge. They are Silver comparative evidence, not human-verified medical accuracy.

## Interpretation

- Paying for the same GPT-OSS 120B model did **not** fix citation-format compliance. The model frequently produced a plausible answer without the required source identifiers, regardless of whether it ran through Groq or OpenRouter.
- Claude Haiku 5.5 and Gemini 3.8 Flash both completed all 12 fixed-context generations.
- In this small sample, Haiku was slightly stronger on the automated overall score, roughly twice as fast as Gemini 3.8, and materially cheaper.
- Provider reliability and model behavior are different variables. Switching providers does not necessarily fix a model-level instruction-following failure.

## Judge correction

The comparison judge was upgraded before this experiment. Version 2 separates:

- factual correctness;
- coverage of reference claims;
- evidence groundedness;
- abstention quality;
- unsupported candidate claims.

Re-scoring the earlier 48 saved outputs showed that Agentic Fast was generally correct and grounded but omitted more reference claims than the old judge revealed. Its v2 aggregate correctness was 0.96 and groundedness 0.95, while reference-claim coverage was 0.66.

## Real Fast validation

One real Fast research run was executed after adding role-specific provider order:

- Question: AI in lung-cancer imaging and limits to safe clinical adoption.
- Orchestrator: Groq GPT-OSS 120B, 2.16 s.
- Gatekeeper: Groq GPT-OSS 120B, 0.97 s.
- Planner: Groq GPT-OSS 120B, 0.74 s.
- Expert: OpenRouter Claude Haiku 5.5, 8.03 s.
- Fallbacks: 0.
- Sources: 3.
- Evidence sufficiency: passed.
- Estimated total LLM cost: **$0.001886**.

This validates the routing path end to end once. It does not establish reliability under repetition or load.

## Recommendation

Keep the global Groq-first policy for short planner, gatekeeper, and orchestrator calls. Use the role-specific Expert order:

1. OpenRouter Claude Haiku 5.5;
2. Gemini 3.8 Flash;
3. Groq GPT-OSS 120B.

Do not silently treat these values as a production baseline. Repeat the controlled 12-case experiment at least three times before expanding to 100 questions or changing published quality claims.

## Reproducibility artifacts

- `expert-provider-comparison.json`
- `expert-provider-comparison-scored.json`
- `expert-provider-haiku.json`
- `expert-provider-haiku-scored.json`
- `real-frozen-v3-12x4-scored-v2.json`
- `scripts/run_expert_provider_comparison.py`
- `scripts/score_real_frozen_comparison.py`

## Remaining limitations

- Six cases are not representative of the full domain distribution.
- Two repetitions are insufficient for stable variance estimates.
- Context was frozen from Hybrid Basic, so the experiment isolates generation rather than the complete Agentic pipeline.
- Failed-attempt cost can be a lower bound when a provider does not return usage metadata.
- The automated judge can be strict or inconsistent and is not a substitute for expert human review.
