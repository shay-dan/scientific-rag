# R7D-D — live configuration acceptance (synthetic)

**Status:** in review. No live run has been made or is authorized by this stage.

## Purpose and matrix

This stage selects the completion ceiling for the forthcoming R7D benchmark from a small,
fixed matrix. All arms use Groq `openai/gpt-oss-120b`, temperature 0, low reasoning,
semantic verification and claim audit off, strict claim recovery, at most one repair, and no
automatic retries or fallback.

| Configuration | Reasoning | Completion ceiling |
|---|---|---|
| A | low | 2,000 |
| B | low | 3,000 |

The validator enforces exactly this matrix:
- provider, model, temperature, prompt versions, modes, completion-budget policy
  (`score_as_failure`), the two configurations and 26 requests are all fixed;
- a smaller diagnostic budget is accepted, but nothing above **$0.15**;
- anything else needs a separately reviewed contract.

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
  schema-valid, evidence-invalid). They never depend on natural failures and send no initial
  request.
- **Request ceiling:** 12 initial generations, plus up to 12 conditional repairs, plus 2
  dedicated repairs gives **at most 26 physical requests**.

## Synthetic dataset and frozen references

`evaluation/r7d-d/synthetic-acceptance-v1.json` (`e6fbe08a…2362`) contains fictional
interventions, trials and numbers. It is not derived from the frozen R7D benchmark.

| Scenario | Tests |
|---|---|
| A001 KESTREL-3 / velarozan | absolute and relative effects with their arms, intervals and denominators; a cited registry that must not be attributed to the trial; an unreported diastolic result that must be declined |
| A002 TWILIGHT / melzorin | a randomized trial against an observational cohort; an inconclusive trial estimate; confounding; a hypothesis-generating subgroup; no head-to-head evidence |
| A003 ORCHID / tralvekimab | five sources and many question parts; an uncontrolled extension; adverse events; limitations; an unanswerable cardiovascular part; a long answer that tests completion headroom |

`evaluation/r7d-d/synthetic-acceptance-references-v1.json` (`0cae5ef4…f3e8`) contains
deterministic predicates:
- expected facts, where the main numeric facts must sit beside their own arm and not the
  comparator;
- forbidden claims, which are critical;
- required uncertainty;
- required abstention, which must be stated explicitly;
- within-claim conditions.

These are **synthetic acceptance references, not independent human-reviewed gold labels**.
They are diagnostic evidence, not a semantic verifier.

**Freezing and independence:**
- **Frozen before the run.** The dry-run plan hash commits to the reference hash and the
  selection rules. The authorization binds `references_sha256`, and the checkpoint records
  it.
- **Never read during generation.** The runner binds only the hash; it never reads reference
  contents, and they never reach a prompt. Changing a predicate leaves every prompt hash
  unchanged but changes the plan hash, so a post hoc edit cannot match the authorization.
- **Checked when reporting.** Reporting recomputes the plan from the supplied dataset, config
  and references, and refuses anything that differs from the checkpoint's commitments.

**Scoring rules:**
- An arm without a final answer earns no credit. Forbidden-claim and conditional checks
  never pass vacuously.
- Mentioning an unsupported endpoint is not abstention.
- An interrupted arm is not an observation and is never scored.

## Reuse and continuation safety

A small adapter (`app/benchmark/acceptance_evaluation.py`) carries its own contracts. This is
needed because the benchmark manifest is exactly 20 frozen scenarios with one configuration
per campaign. It reuses, unchanged:
- `_evaluate_arm`: prompts, schemas, validators, the single repair, typed failures;
- `GuardedExecution`, shared with the benchmark runner: admission, reservations, ledger
  marks, stop precedence, settlement;
- `AtomicCheckpoint`, the program ledger, and the benchmark's `_resolved_attempt` and
  `_resume_state` restoration.

On every continuation, before anything is sent, the checkpoint must satisfy all of these:
- its records are exactly a prefix of the frozen schedule (no duplicates, nothing out of
  order, each matching its operation, configuration, prompt version and prompt hash);
- every settled call belongs, in order, to one recorded or in-progress attempt, and the
  budget reconciles with the campaign baseline;
- the saved in-progress operation is the next scheduled one, with the right phase, attempt
  sequence and fingerprints, and every attempt is a resolved, fully accounted response.

Settled responses are then restored, never re-sent:
- an initial generation (valid, or invalid before its repair was recorded);
- a conditional repair;
- a dedicated repair (its initial answer is the fixture, so its only attempt is the repair).

Continuation is refused, and nothing is retransmitted, in these cases:
- an attempt was reserved or transmitted without a recorded result;
- a saved output fails its schema;
- any identity or fingerprint has drifted;
- the run was cancelled after transmission. Cancellation is a terminal `cancelled` stop;
  only a pause during the opt-in pre-transmission wait stays resumable.

## Accounting and costs

A **separate acceptance program ledger** (`r7d-d-acceptance`, 26 requests, $0.15) is used.
The scientific benchmark ledger is not created or modified. For total provider-consumption
visibility, the closed acceptance campaign can later be recorded there with
`ledger-record-diagnostic`.

There are two cases where usage is uncertain, and they are treated differently:
- **Accounted truncation.** This is a predeclared, scored completion-budget failure whose
  usage the provider did not report. It is a model outcome: its full reservation stays
  charged as uncertain spend and is reported separately (`accounting.accounted_truncations`).
  It is not an accounting defect and does not block selection.
- **Accounting defect.** This is a `billing_unknown` stop, or an unresolved, possibly
  transmitted attempt. It stops execution and blocks selection.

**Costs** (dry run, zero calls; unchanged by these corrections):

| Item | Amount |
|---|---:|
| Exact reservations for the 12 initial and 2 dedicated-repair requests | $0.0326 |
| Typical estimate for all 12 conditional repairs (initial prompt + 4× the ceiling) | about $0.046 |
| Conservative planning bound for the 12 conditional repairs (70,000 extra characters each) | $0.154 |
| **Proposed diagnostic ceiling** | **$0.15** |

Every repair is reserved from its real prompt and refused before transmission if it would
not fit. These figures are estimates and reservations, not charges and not guaranteed
upper bounds.

**Groq Free capacity:**
- the first request needs about 6.3K tokens uncalibrated, within 8,000 per minute;
- `--wait-for-token-reset` waits at most 65 seconds for a known short reset;
- unknown or long resets stop before transmission;
- daily usage is not reported in the response headers.

## Metrics (per configuration and prompt version)

- **Reliability:**
  - initial structural success;
  - completion-budget failures (accounted truncations separately);
  - provider structured-generation failures;
  - Pydantic and structural failures;
  - repairs attempted and succeeded;
  - final valid;
  - interrupted and unexecuted operations.
- **Scientific (synthetic predicates):**
  - passes per dimension;
  - critical violations, counted only when an answer exists;
  - arms with no answer.

  Structural validity is not scientific correctness.
- **Efficiency:**
  - input, completion and reasoning tokens (when reported), and non-reasoning completion
    tokens;
  - minimum headroom;
  - latency;
  - known cost and uncertain reservations;
  - requests and quota continuations.

## Predeclared selection rules (`r7d-d-selection/v2`)

**Gates** (each configuration must pass all of them):
1. Initial and dedicated repair schemas parsed on the actual gateway.
2. No recorded execution stop of any kind: integrity, provider, quota, cancellation or
   accounting.
3. No accounting defect, as defined above.
4. Every operation recorded and none interrupted. Otherwise the experiment is
   **incomplete**: a record for every operation is not enough when one was interrupted.
5. Final structural validity in at least 5 of 6 arms.
6. No material scientific regression: not at least 2 more critical violations, and not at
   least 2 more failed predicates, than the other configuration.

**Choice:**
- If both pass, prefer **2,000**. Choose 3,000 only for a meaningful advantage:
  - fewer completion-budget failures;
  - at least one more final-valid arm;
  - at least 3 more passed predicates;
  - at least 10% headroom where 2,000 had less than 10%.
- If only one passes, choose that one.
- Otherwise the result is **inconclusive**. Then propose a narrow follow-up; never expand
  the sample automatically.

**The result is a recommendation only.** Before a configuration is frozen, the owner reviews
the 12 generated answers scientifically, using the private answers export below. The choice
is never based on shorter answers, and no statistical superiority is claimed.

## Windows operator runbook (PowerShell, Docker Desktop with Linux containers)

Private state (ledger, checkpoint, plan, config, authorization) lives in a Docker **named
volume**. A Windows bind mount cannot provide the Linux owner-only permissions that the
checkpoint and ledger require; a named volume can. The permissions are verified inside the
container.

The Groq key is entered hidden and held only in a temporary process environment variable.
That variable is removed in a `finally` block, and the key is never written to a file or a
command line. Every native command's exit code is checked, and the first failure stops
everything.

`$PIN` is the **merged** R7D-D commit.

```powershell
$ErrorActionPreference = "Stop"
function Assert-Native([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE); stopping" } }

$PIN   = "<merged R7D-D commit>"
$Repo  = "C:\src\scientific-rag-$($PIN.Substring(0,7))"
$Ops   = "C:\r7d-d-operator"          # non-secret operator files and the private exports; outside the repository
$Image = "scientific-rag:$PIN"
$Vol   = "r7d-d-private"              # Docker named volume for private state
$OutputEncoding = [Text.UTF8Encoding]::new($false)

# 0. Pinned build and provenance
git -c core.autocrlf=false clone https://github.com/shay-dan/scientific-rag.git $Repo; Assert-Native "git clone"
git -C $Repo -c advice.detachedHead=false checkout $PIN; Assert-Native "git checkout"
if ((git -C $Repo rev-parse HEAD) -ne $PIN -or (git -C $Repo status --porcelain)) { throw "wrong or dirty checkout" }
docker build --build-arg "GIT_SHA=$PIN" -t $Image "$Repo\cost-aware-pipeline"; Assert-Native "docker build"
docker run --rm --network none $Image python -c "import sys; from app.evaluation_report import resolve_git_provenance as r; sys.exit(0 if r() == {'sha': sys.argv[1], 'source': 'build_arg'} else 1)" $PIN
Assert-Native "image provenance check"

# Private volume (owner-only for the container user) and operator folder
docker volume create $Vol | Out-Null; Assert-Native "docker volume create"
docker run --rm --network none --user root --mount "type=volume,src=$Vol,dst=/private" $Image sh -c "chown 1000:1000 /private && chmod 700 /private"
Assert-Native "private volume ownership"
New-Item -ItemType Directory -Force $Ops | Out-Null
icacls $Ops /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F" | Out-Null; Assert-Native "icacls"

$priv = @("--mount", "type=volume,src=$Vol,dst=/private")
$off  = @("run", "--rm", "--network", "none") + $priv + @($Image)
function Copy-ToPrivate([string]$Source, [string]$Name) {
  Get-Content $Source -Raw | docker run --rm -i --network none @priv $Image sh -c "umask 077 && cat > /private/$Name"
  Assert-Native "copy $Name into the private volume"
}
function Assert-PrivateModes {
  docker @off python -c "import os,stat,sys; p=['/private']+[os.path.join('/private',n) for n in os.listdir('/private')]; bad=[x for x in p if stat.S_IMODE(os.stat(x).st_mode) & 0o077]; print('private modes ok' if not bad else bad); sys.exit(1 if bad else 0)"
  Assert-Native "private file permissions"
}
function Invoke-WithGroqKey([scriptblock]$Body) {
  $secure = Read-Host "GROQ_API_KEY (input hidden)" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try { $env:GROQ_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
  finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr); $secure.Dispose() }
  try { & $Body } finally { Remove-Item Env:GROQ_API_KEY -ErrorAction SilentlyContinue }
}
# --env NAME without a value passes the variable from this process; the key is never on a command line.
$live = @("run", "--rm", "--env", "GROQ_API_KEY", "--env", "LLM_PROVIDER_ORDER_EXPERT=groq",
          "--env", "GROQ_MODEL_EXPERT=openai/gpt-oss-120b") + $priv + @($Image)

# 1. Config, separate acceptance ledger, zero-call dry run (no network)
(Get-Content "$Repo\cost-aware-pipeline\evaluation\r7d-d\acceptance-config.template.json" -Raw).Replace(
  "REPLACE_WITH_MERGED_R7D_D_COMMIT_SHA", $PIN) | Set-Content "$Ops\acceptance-config.json" -Encoding ascii
Copy-ToPrivate "$Ops\acceptance-config.json" "acceptance-config.json"
docker @off python scripts/generation_evaluation_r7d.py ledger-init --ledger /private/acceptance-ledger.json --program-id r7d-d-acceptance --max-calls 26 --max-cost-usd 0.15
Assert-Native "ledger-init"
docker @off python scripts/acceptance_r7d_d.py dry-run --config /private/acceptance-config.json --out /private/acceptance-plan.json
Assert-Native "dry run"                 # prints config/schedule/plan/selection-rule/reference hashes
docker @off python scripts/generation_evaluation_r7d.py ledger-summary --ledger /private/acceptance-ledger.json
Assert-Native "ledger-summary"          # prints the baseline_sha256 for the authorization
Assert-PrivateModes

# 2. Authenticated, non-inference preflight (1 models-list request). A failure throws: stop here.
Invoke-WithGroqKey {
  docker @live python scripts/generation_evaluation_r7d.py auth-preflight --config /private/acceptance-config.json
  Assert-Native "authenticated preflight"
}

# 3. ONLY after explicit owner approval: fill $Ops\acceptance-authorization.json from
#    evaluation/r7d-d/acceptance-authorization.template.json with the dry-run hashes (including
#    references_sha256) and the ledger baseline_sha256, then copy it into the private volume.
Copy-ToPrivate "$Ops\acceptance-authorization.json" "acceptance-authorization.json"

# 4. Live run (separately authorized): at most 26 requests, bounded waits, no retries.
Invoke-WithGroqKey {
  docker @live python scripts/acceptance_r7d_d.py run --config /private/acceptance-config.json --authorization /private/acceptance-authorization.json --private-checkpoint /private/acceptance-checkpoint.json --program-ledger /private/acceptance-ledger.json --wait-for-token-reset
  Assert-Native "acceptance run"
}
Assert-PrivateModes
#    Read the printed stop_code. Only after a pre-transmission capacity stop (rate_limit), rerun
#    step 4 with --resume-after-rate-limit added. On any other stop or refusal: do not rerun, do
#    not delete or edit the checkpoint; report the printed summary.

# 5. Read-only export: the private volume is mounted read-only; outputs are copied out.
$name = "r7dd-report-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
docker create --name $name --network none --mount "type=volume,src=$Vol,dst=/private,readonly" $Image python scripts/acceptance_r7d_d.py report --config /private/acceptance-config.json --checkpoint /private/acceptance-checkpoint.json --out /tmp/acceptance-report.json --answers-out /tmp/acceptance-answers.json | Out-Null
Assert-Native "create report container"
try {
  docker start -a $name; Assert-Native "report"
  docker cp "${name}:/tmp/acceptance-report.json" "$Ops\acceptance-report.json"; Assert-Native "export report"
  docker cp "${name}:/tmp/acceptance-answers.json" "$Ops\acceptance-answers.json"; Assert-Native "export answers"
} finally { docker rm -f $name | Out-Null }
#    acceptance-report.json: metrics and the selection recommendation (no prompt, evidence or answer text).
#    acceptance-answers.json: the generated answers, for the owner's scientific review. Keep private.

# 6. After review, close the acceptance campaign (keeps its accounting; it never runs again).
docker @off python scripts/generation_evaluation_r7d.py ledger-close --ledger /private/acceptance-ledger.json --checkpoint /private/acceptance-checkpoint.json
Assert-Native "ledger-close"
```

- `auth-preflight` exits non-zero unless the key authenticates and the model is listed. It
  never prints the key.
- The key exists in plaintext only inside this PowerShell process, for the duration of each
  `Invoke-WithGroqKey` block.

## Limitations

- **Small sample.** Three synthetic scenarios cannot establish statistical superiority, and
  they do not cover every failure mode of the real benchmark.
- **Regex predicates.** These are diagnostic only. Arm association uses comparison segments
  ("A … versus B …"). It can reject correct but unusually phrased answers, or accept weak
  ones. This is why the owner reviews the 12 answers.
- **Single execution.** The order is balanced but runs once, so time and quota effects are
  reduced, not removed.
- **Fail-closed continuation.** A transmitted attempt that cannot be restored is refused
  rather than re-sent. Such a campaign is closed from its ledger mark and reported as
  incomplete.
- **Cost visibility.** Costs are estimates and reservations. Daily token use is not visible
  in the response headers.
