# Model-routing plan status

Date: 2026-10-28

## Completed

1. Upgraded the comparison judge to separate correctness, reference coverage, groundedness, abstention quality, and unsupported claims.
2. Re-scored the existing 48 saved outputs without paying for new research.
3. Added current cost rates and preflight estimates for the compared paid models.
4. Tested the same GPT-OSS 120B model through Groq and paid OpenRouter.
5. Tested Gemini 3.8 Flash on identical frozen evidence.
6. Tested paid OpenRouter Claude Haiku 5.5 on identical frozen evidence.
7. Added role-specific provider order to the LLM gateway.
8. Validated one real Fast run with Groq for orchestration and Haiku for Expert synthesis.

## Decision for the next controlled phase

- Keep Groq first for short orchestration roles.
- Use OpenRouter Claude Haiku 5.5 first for the Expert role.
- Use Gemini 3.8 Flash as the Expert fallback.
- Keep GPT-OSS 120B as the final Expert fallback, not the preferred citation-heavy generator.
- Do not expand to 100 questions yet.

## Saved improvement backlog

1. Diagnose why the Agentic and Enhanced variants lost evidence on specific mixed and arXiv cases.
2. Equalize the number of contexts and token budgets before attributing quality differences to orchestration.
3. Repeat the controlled 12-case comparison three times with one pinned generation model.
4. Keep frozen-corpus and live-mining benchmarks separate.
5. Surface claim-level support and reference coverage in the evaluation dashboard.
6. Improve failed-attempt and reasoning-token cost accounting.
7. Record the upstream model/provider selected behind OpenRouter routing, when available.
8. Tune MMR for coverage before diversity on multi-part questions.
9. Expand to 100 questions only after the 12-case repeated baseline is stable.
