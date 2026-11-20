# R7D-D — live configuration acceptance (synthetic)

**Status:** in review. No live run has been made or is authorized by this stage.

## Purpose and matrix

This stage selects the completion ceiling for the forthcoming R7D benchmark from a small,
fixed matrix. All arms use Groq `openai/gpt-oss-120b`, temperature 0, low reasoning,
semantic verification off, strict structural validation, at most one repair, and no
automatic retries or fallback.

| Configuration | Reasoning | Completion ceiling |
|---|---|---|
| A | low | 2,000 |
| B | low | 3,000 |

The same settings are sent with every initial generation and every repair. The 14 frozen
operations run in this order:

| # | Operation | Config | Prompt |
|---:|---|---|---|
| 1–4 | A001 arms | A, B, B, A | v15, v16, v15, v16 |
| 5 | dedicated repair RF1 (A001) | B | v16 |
| 6–9 | A002 arms | B, A, A, B | v15, v16, v15, v16 |
| 10 | dedicated repair RF2 (A002) | A | v15 |
| 11–14 | A003 arms | A, B, B, A | v16, v15, v16, v15 |

- **Balance:** every scenario gives each configuration one v15 arm and one v16 arm.
- **Dedicated repairs:** they start from a deliberately invalid fixture answer (strict
  schema-valid, evidence-invalid). They never depend on natural failures and send no
  initial request.
- **Request ceiling:** 12 initial generations, plus up to 12 conditional repairs, plus 2
  dedicated repairs gives **at most 26 physical requests**.
- **Commitments:** the schedule and the predeclared selection rules are hashed into the
  dry-run plan, and the authorization binds both.

## Synthetic dataset and references

`evaluation/r7d-d/synthetic-acceptance-v1.json` has SHA-256 `e6fbe08a…2362`. It holds fictional
interventions, trials and numbers, and is not derived from the frozen R7D benchmark.

| Scenario | Tests |
|---|---|
| A001 KESTREL-3 / velarozan | absolute and relative effects, confidence intervals and denominators, a cited registry that must not be attributed to the trial, an unreported diastolic result that must be omitted |
| A002 TWILIGHT / melzorin | a randomized trial against an observational cohort, an inconclusive trial estimate, confounding, a hypothesis-generating subgroup, no head-to-head evidence |
| A003 ORCHID / tralvekimab | five sources and many question parts, an uncontrolled as-observed extension, adverse events, limitations, an unsupported cardiovascular-effect part; a long answer that exercises completion headroom |

`evaluation/r7d-d/synthetic-acceptance-references-v1.json` (`dd269788…bffae`) holds
deterministic predicates:
- expected facts;
- forbidden claims, marked critical;
- required uncertainty;
- required abstention;
- within-claim conditions.

These are **synthetic acceptance references, not independent human-reviewed gold labels**.
They are validated separately, and they are never read by the runner or included in any
prompt. Changing them does not change any prompt or the plan hash.

## Reuse and boundaries

The benchmark manifest contract is exactly 20 frozen development scenarios with one
configuration per campaign. A small adapter (`app/benchmark/acceptance_evaluation.py`)
therefore carries its own dataset, config and authorization contracts. It reuses, unchanged:
- `_evaluate_arm`: production prompts, schemas, validators, the single repair and typed
  failures;
- `GuardedExecution`, extracted unchanged from the benchmark runner so both runners share
  it: admission, bounded token-reset waits, reservations, ledger marks before transmission
  and after settlement, stop precedence, settlement and reasoning-token telemetry;
- `AtomicCheckpoint` and the program ledger: one authoritative checkpoint, a per-campaign
  lock, cumulative accounting.

**Accounting** uses a **separate acceptance program ledger** (`r7d-d-acceptance`, 26 requests /
$0.15):
- the scientific benchmark ledger is not created or modified;
- for total provider-consumption visibility, the closed acceptance campaign can later be
  recorded in the benchmark ledger with `ledger-record-diagnostic`, using the acceptance
  ledger file as the hashed source;
- unknown usage keeps its reservation;
- interrupted and unexecuted operations stay in the denominators.

**Costs** (dry run, zero calls):

| Item | Amount |
|---|---:|
| Exact reservations for the 12 initial and 2 dedicated-repair requests | $0.0326 |
| Typical estimate for all 12 conditional repairs (initial prompt + 4× the ceiling) | about $0.046 |
| Conservative planning bound for the 12 conditional repairs (70,000 extra characters each) | $0.154 |
| **Proposed diagnostic ceiling** | **$0.15** |

Every actual repair is reserved from its real prompt and refused before transmission if it
would not fit. All figures are estimates and reservations, not provider charges and not
guaranteed upper bounds.

**Groq Free capacity:**
- the first request needs about 6.3K tokens uncalibrated, which fits 8,000 per minute;
- later requests are calibrated from provider-reported usage;
- `--wait-for-token-reset` waits at most 65 seconds for a known short reset;
- unknown or long resets stop before transmission, and `--resume-after-rate-limit`
  continues without repeating anything;
- the total actual usage expected (roughly 120K tokens or less) is below the published
  200K tokens per day, but daily usage is not reported in the response headers.

## Metrics (per configuration and prompt version)

- **Reliability:**
  - initial structural success;
  - completion-budget failures;
  - provider structured-generation failures;
  - Pydantic failures;
  - structural failures;
  - repairs attempted and succeeded;
  - final valid;
  - interrupted and unexecuted operations (by number).
- **Scientific (synthetic predicates):**
  - passes per dimension: numerical fidelity, attribution, study design,
    causality/uncertainty, coverage, abstention;
  - critical violations.

  Structural validity is not scientific correctness.
- **Efficiency:**
  - input, completion and reasoning tokens (when reported), and non-reasoning completion
    tokens;
  - minimum completion headroom;
  - latency;
  - known estimated cost and uncertain reservations;
  - physical requests and quota continuations.

## Predeclared selection rules (`r7d-d-selection/v1`)

**Gates** (each configuration must pass all of them):
1. Initial and dedicated repair schemas both parsed on the actual gateway.
2. No execution-integrity stop (model drift, retry, output limit, reservation or ceiling
   overrun, internal or ledger-persistence error).
3. No accounting defect: no unreported usage and no unresolved attempt.
4. Every operation completed, or the missing ones are listed. If any are missing, the
   result is inconclusive.
5. Final structural validity in at least 5 of 6 arms.
6. No material scientific regression: not at least 2 more critical violations, and not at
   least 2 more failed predicates, than the other configuration.

**Choice:**
- If both pass, prefer **2,000**. Choose 3,000 only for a meaningful advantage:
  - fewer completion-budget failures;
  - at least one more final-valid arm;
  - at least 3 more passed predicates;
  - minimum headroom of at least 10% where 2,000 had less than 10%.
- If only one passes, choose that one.
- Otherwise the result is **inconclusive**, with the reasons. Then propose a narrowly
  targeted follow-up; never expand the sample automatically.

The choice is never based on shorter answers, and no statistical superiority is claimed.

## Windows operator runbook (Docker Desktop, Linux containers)

This uses PowerShell. Every private file lives in a new directory outside the repository.
`$PIN` is the **merged** R7D-D commit.

```powershell
$PIN  = "<merged R7D-D commit>"
$Repo = "C:\src\scientific-rag-$($PIN.Substring(0,7))"
$Priv = "C:\r7d-d-private"                      # new; never inside the repository
git -c core.autocrlf=false clone https://github.com/shay-dan/scientific-rag.git $Repo
git -C $Repo -c advice.detachedHead=false checkout $PIN
if ((git -C $Repo rev-parse HEAD) -ne $PIN -or (git -C $Repo status --porcelain)) { throw "wrong or dirty checkout" }
docker build --build-arg "GIT_SHA=$PIN" -t "scientific-rag:$PIN" "$Repo\cost-aware-pipeline"
docker run --rm --network none "scientific-rag:$PIN" python -c "import os,sys; sys.exit(os.environ['SCIENTIFIC_RAG_GIT_SHA']!=sys.argv[1])" $PIN
if ($LASTEXITCODE) { throw "image provenance mismatch" }

New-Item -ItemType Directory $Priv | Out-Null
icacls $Priv /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F" | Out-Null
(Get-Content "$Repo\cost-aware-pipeline\evaluation\r7d-d\acceptance-config.template.json" -Raw).Replace(
  "REPLACE_WITH_MERGED_R7D_D_COMMIT_SHA", $PIN) | Set-Content "$Priv\acceptance-config.json" -Encoding ascii
$off = @("run","--rm","--network","none","--mount","type=bind,src=$Priv,dst=/private","scientific-rag:$PIN","python")

# 1. Separate acceptance ledger and zero-call dry run
docker @off scripts/generation_evaluation_r7d.py ledger-init --ledger /private/acceptance-ledger.json --program-id r7d-d-acceptance --max-calls 26 --max-cost-usd 0.15
docker @off scripts/acceptance_r7d_d.py dry-run --config /private/acceptance-config.json --out /private/acceptance-plan.json
docker @off scripts/generation_evaluation_r7d.py ledger-summary --ledger /private/acceptance-ledger.json   # baseline_sha256

# 2. API key (never echoed, never on a command line), then the non-inference preflight
$s = Read-Host "GROQ_API_KEY" -AsSecureString
$k = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))
Set-Content "$Priv\.env.groq" "GROQ_API_KEY=$k`nLLM_PROVIDER_ORDER_EXPERT=groq`nGROQ_MODEL_EXPERT=openai/gpt-oss-120b" -Encoding ascii -NoNewline
Remove-Variable k, s
$live = @("run","--rm","--env-file","$Priv\.env.groq","--mount","type=bind,src=$Priv,dst=/private","scientific-rag:$PIN","python")
docker @live scripts/generation_evaluation_r7d.py auth-preflight --config /private/acceptance-config.json   # 1 models-list request; no inference

# 3. Authorization: copy evaluation/r7d-d/acceptance-authorization.template.json to $Priv\acceptance-authorization.json
#    and fill it with the dry-run hashes and the ledger baseline ONLY after explicit owner approval.

# 4. Live run (separately authorized): at most 26 requests, bounded waits, no retries
docker @live scripts/acceptance_r7d_d.py run --config /private/acceptance-config.json --authorization /private/acceptance-authorization.json --private-checkpoint /private/acceptance-checkpoint.json --program-ledger /private/acceptance-ledger.json --wait-for-token-reset
#    After a pre-transmission capacity stop, rerun the same command with --resume-after-rate-limit.

# 5. Read-only results export, then close the acceptance campaign (keeps its accounting)
docker @off scripts/acceptance_r7d_d.py report --config /private/acceptance-config.json --checkpoint /private/acceptance-checkpoint.json --out /private/acceptance-report.json
docker @off scripts/generation_evaluation_r7d.py ledger-close --ledger /private/acceptance-ledger.json --checkpoint /private/acceptance-checkpoint.json
```

`auth-preflight` exits non-zero unless the key authenticates and the model is listed. It
never prints the key. The report contains metrics only: no prompts, evidence or answer text.

## Limitations

- Three synthetic scenarios cannot establish statistical superiority, and they do not cover
  every failure mode of the real benchmark.
- The predicates are regex-based synthetic references. They can miss paraphrases or accept
  wording that is technically correct but weak, and they are not human-reviewed gold labels.
- The order is balanced but executed once, so time and quota effects are reduced, not
  removed.
- Costs are estimates and reservations. Daily token use is not visible in the response
  headers.
- Configuration selection is kept separate from the frozen benchmark. Nothing here tunes
  against benchmark cases.
