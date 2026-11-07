# pilot-silver-v3 — independent source and claim audit

This dataset remains **Silver**: it was automatically generated and independently audited, but not human-reviewed.

## Outcome

- Cases audited: **12**
- Eligible repaired cases: **12**
- Rejected cases: **0**
- Atomic claims audited: **51**
- Supported: **50**
- Contradicted: **0**
- Insufficient: **1**

## Source verification

- Original sources fetched: **31 / 31**
- Original excerpts checked: **51**
- Verbatim matches: **36**
- Near-verbatim matches: **1**
- Not found verbatim: **14**

### Targeted repair source verification

- Repair excerpts checked: **13**
- Repair excerpts verified verbatim: **13**
- Repair excerpts not found: **0**

## Cost and method

- Independent judge: `gemini:gemini-2.5-flash`
- Model calls across initial audit and targeted retry: **20**
- Combined preflight estimates: **$0.0580**
- Recorded cumulative model cost: **$0.0670**
- Historical estimate error: **$+0.0090**; future runs reserve 1,800 output tokens per case.
- Each atomic claim received a separate structured decision, although claims from one case were batched into one call to reduce cost.
- Repaired answers are deterministic joins of supported claims; the audit model did not write replacement scientific claims.

## Case status

| Case | Status | Supported | Contradicted | Insufficient |
|---|---|---:|---:|---:|
| pm-01 | accepted_repaired | 5 | 0 | 0 |
| pm-02 | accepted_repaired | 3 | 0 | 0 |
| pm-03 | accepted_repaired | 5 | 0 | 0 |
| pm-04 | accepted_repaired | 3 | 0 | 1 |
| ax-01 | accepted_repaired | 4 | 0 | 0 |
| ax-02 | accepted_repaired | 5 | 0 | 0 |
| ax-04 | accepted_repaired | 4 | 0 | 0 |
| mx-01 | accepted_repaired | 3 | 0 | 0 |
| mx-02 | accepted_repaired | 5 | 0 | 0 |
| mx-03 | accepted_repaired | 4 | 0 | 0 |
| mx-04 | accepted_repaired | 5 | 0 | 0 |
| ax-03 | accepted_repaired | 4 | 0 | 0 |
