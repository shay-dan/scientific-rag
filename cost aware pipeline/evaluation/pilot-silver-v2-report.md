# Silver benchmark v2 — independent source and claim audit

This dataset remains **Silver**: it was automatically generated and independently audited, but not human-reviewed.

## Outcome

- Cases audited: **12**
- Eligible repaired cases: **9**
- Rejected cases: **3**
- Atomic claims audited: **51**
- Supported: **40**
- Contradicted: **1**
- Insufficient: **10**

## Source verification

- Original sources fetched: **31 / 31**
- Original excerpts checked: **51**
- Verbatim matches: **36**
- Near-verbatim matches: **1**
- Not found verbatim: **14**

## Cost and method

- Independent judge: `gemini:gemini-2.5-flash`
- Model calls across initial audit and targeted retry: **14**
- Combined preflight estimates: **$0.0405**
- Recorded cumulative model cost: **$0.0541**
- Historical estimate error: **$+0.0136**; future runs reserve 1,800 output tokens per case.
- Each atomic claim received a separate structured decision, although claims from one case were batched into one call to reduce cost.
- Repaired answers are deterministic joins of supported claims; the audit model did not write replacement scientific claims.

## Case status

| Case | Status | Supported | Contradicted | Insufficient |
|---|---|---:|---:|---:|
| pm-01 | accepted_repaired | 5 | 0 | 0 |
| pm-02 | rejected_insufficient_supported_claims | 0 | 0 | 5 |
| pm-03 | rejected_insufficient_supported_claims | 1 | 1 | 2 |
| pm-04 | accepted_repaired | 3 | 0 | 1 |
| ax-01 | accepted_repaired | 4 | 0 | 0 |
| ax-02 | accepted_repaired | 5 | 0 | 0 |
| ax-04 | accepted_repaired | 4 | 0 | 0 |
| mx-01 | accepted_repaired | 3 | 0 | 0 |
| mx-02 | accepted_repaired | 5 | 0 | 0 |
| mx-03 | accepted_repaired | 4 | 0 | 0 |
| mx-04 | rejected_insufficient_supported_claims | 2 | 0 | 2 |
| ax-03 | accepted_repaired | 4 | 0 | 0 |
