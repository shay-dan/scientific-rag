# Paid Qwen Fast smoke test — 2026-10-29

Three fresh end-to-end Fast research requests were executed after replacing the OpenRouter free router with `qwen/qwen3.6-35b-a3b`, raising the OpenRouter deadline from 12 to 35 seconds, and allowing evidence-heavy Expert contexts up to 12,000 characters. This is an operational smoke test, not a statistically meaningful quality benchmark.

| Cohort | Status | Time | Previous time | Displayed sources | Routing result |
|---|---:|---:|---:|---:|---|
| PubMed / CAR-T | 200 | 43.6 s | 69.6 s | 1 | PubMed only |
| arXiv / surface codes | 200 | 43.1 s | 48.5 s | 2 | arXiv only |
| Mixed / mammography ML | 200 | 86.3 s | 65.3 s | 3 unique | PubMed + arXiv |

Run identifiers:

- PubMed: `540409cf-8bc7-4b24-a3cc-61843a8dedcd`
- arXiv: `0ae24700-6e77-4f45-aa1a-a1c7df06ea01`
- Mixed: `76401538-ed33-483b-b431-4fde8cc9f30f`

## Passed

- All three requests completed with HTTP 200.
- Source routing matched the intended corpus in all three cases.
- OpenRouter Qwen completed **14/14** calls successfully.
- Gemini was not needed for any of these calls.
- No provider failure triggered a failover.
- API source presentation now deduplicates repeated parent/child passages by citation identity; the mixed run displayed three unique papers rather than four rows containing one duplicate.
- Diagnostics now distinguish provider failures from requests skipped by shared admission throttling.

## Performance interpretation

- PubMed latency decreased by about **37%** (69.6 s to 43.6 s).
- arXiv latency decreased by about **11%** (48.5 s to 43.1 s).
- The mixed run increased by about **32%** (65.3 s to 86.3 s). One sample cannot establish a stable latency regression, but the larger number of Qwen calls and mixed-source work make this a follow-up target.

## Quality gate status

The broad 12-question comparison was deliberately not started. The PubMed result retained only one paper after bounded relevance screening, so evidence coverage needs checking. A live Ragas rescore was attempted three times and exposed an integration incompatibility between the installed Ragas Google Instructor adapter and the Google client: the adapter constructor requires the synchronous client, while asynchronous metric scoring rejects it. No fallback or invented Ragas scores are reported.

The native Google-client mismatch was replaced with Gemini's OpenAI-compatible async endpoint. Gemini reasoning is disabled for the Ragas judge so that thinking tokens do not exhaust structured output, and the incompatible Ragas Google embedding wrapper was replaced with a modern adapter around the current Google GenAI API. All three saved runs were subsequently scored without repeating research or mining. See [`qwen-fast-smoke-ragas-report-2026-10-30.md`](qwen-fast-smoke-ragas-report-2026-10-30.md).

Raw smoke output is retained locally and is not committed to the repository.
