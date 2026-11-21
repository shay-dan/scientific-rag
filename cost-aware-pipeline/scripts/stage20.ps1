# Stage 20G production E2E: one command per operator action (Windows PowerShell + Docker Desktop).
#
#   .\stage20.ps1 -Mode Test                      verify commit and image, credentials (names only),
#                                                 manifest, offline tests, non-inference preflight;
#                                                 prints the authorization fingerprint. No inference.
#   .\stage20.ps1 -Mode Run -Authorization <fp>   the separately approved live campaign: isolated
#                                                 database, Redis and results; refuses reruns.
#   .\stage20.ps1 -Mode Report [-Authorization <fp>]   export sanitized + private review results.
#
# Never modifies the Stage 20F v1 campaign's volumes, journal or authorization marker. Secrets are
# passed to containers with --env-file only and are never printed.
param(
  [ValidateSet("Test", "Run", "Report")][string]$Mode = "Test",
  [string]$Authorization = "",
  [string]$Operator = (Join-Path $env:USERPROFILE "Desktop\stage20-operator"),
  [string]$Credentials = (Join-Path $HOME "stage20f-provider.env"),
  [switch]$NoFallback
)
$ErrorActionPreference = "Stop"
$OutputEncoding = [Text.UTF8Encoding]::new($false)

function Assert-Native([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE); stopping" } }
function Step([string]$Text) { Write-Host "==> $Text" -ForegroundColor Cyan }
function Ok([string]$Text) { Write-Host "    ok  $Text" -ForegroundColor Green }
function Json($Lines) { return (($Lines | Out-String).Trim() | ConvertFrom-Json) }
# Exit code of a probe whose output is discarded. Windows PowerShell 5.1 turns redirected native
# stderr into a terminating error under "Stop", so probes run with "Continue".
function Probe([scriptblock]$Command) {
  $previous = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  try { & $Command *> $null } finally { $ErrorActionPreference = $previous }
  return $LASTEXITCODE
}

$App = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Repo = (Resolve-Path (Join-Path $App "..")).Path
$FallbackArgs = @(); if ($NoFallback) { $FallbackArgs = @("--no-fallback") }
$Variant = if ($NoFallback) { "groq-only" } else { "groq+openrouter-fallback" }

# --- 1. Tools, commit, image -----------------------------------------------------------------
Step "Checking tools and source commit"
foreach ($tool in "git", "docker") { if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH" } }
docker info --format "{{.ServerVersion}}" | Out-Null; Assert-Native "docker engine (is Docker Desktop running?)"
$Sha = (git -C $Repo rev-parse HEAD).Trim(); Assert-Native "git rev-parse"
if (git -C $Repo status --porcelain) { throw "the checkout at $Repo has local changes; refusing (commit or discard them)" }
Ok "commit $Sha (clean)"

$Image = "scientific-rag:$Sha"
if ((Probe { docker image inspect $Image }) -ne 0) {
  Step "Building image $Image (first run for this commit only)"
  docker build --build-arg "GIT_SHA=$Sha" -t $Image $App; Assert-Native "docker build"
}
docker run --rm --network none $Image python -c "import sys; from app.evaluation_report import resolve_git_provenance as r; sys.exit(0 if r() == {'sha': sys.argv[1], 'source': 'build_arg'} else 1)" $Sha
Assert-Native "image provenance check"
Ok "image provenance $Sha"

# --- 2. Credentials (names only) and manifest ------------------------------------------------
Step "Checking credentials file (values are never read into output)"
if (-not (Test-Path $Credentials)) { throw "credentials file not found: $Credentials" }
$present = @{}
foreach ($line in Get-Content $Credentials) {
  if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
    $value = $Matches[2].Trim()
    if ($value.StartsWith('"') -or $value.StartsWith("'")) { throw "$($Matches[1]) is quoted; docker --env-file passes quotes literally. Remove the quotes." }
    $present[$Matches[1]] = ($value.Length -gt 0)
  }
}
$required = @("GROQ_API_KEY", "ENTREZ_EMAIL"); if (-not $NoFallback) { $required += "OPENROUTER_API_KEY" }
$missing = @($required | Where-Object { -not $present[$_] })
if (-not ($present["GOOGLE_API_KEY"] -or $present["GEMINI_API_KEY"])) { $missing += "GOOGLE_API_KEY or GEMINI_API_KEY" }
foreach ($proxy in "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") { if ($present[$proxy]) { throw "$proxy is set in the credentials file; a proxy could bypass request accounting" } }
if ($missing.Count) {
  $hint = if ((-not $NoFallback) -and ($missing -contains "OPENROUTER_API_KEY")) { " (or rerun with -NoFallback for the Groq-only protocol)" } else { "" }
  throw "credentials file lacks: $($missing -join ', ')$hint"
}
Ok "required names present: $($required -join ', '), Gemini key"

Step "Locating the private five-question manifest by its frozen hash"
if (-not (Test-Path $Operator)) { throw "operator folder not found: $Operator" }
$OperatorMount = "type=bind,src=$Operator,dst=/operator,readonly"
$located = Json (docker run --rm --network none --mount $OperatorMount $Image python scripts/stage20_e2e.py locate-manifest --dir /operator)
if ($located.matches.Count -ne 1) { throw "expected exactly one manifest with SHA-256 $($located.expected_manifest_sha256) in $Operator; found $($located.matches.Count)" }
$ManifestName = $located.matches[0]
Ok "manifest $ManifestName"

function Get-Prepared {
  $prepared = Json (docker run --rm --network none --mount $OperatorMount $Image python scripts/stage20_e2e.py prepare --manifest "/operator/$ManifestName" --code-sha $Sha @FallbackArgs)
  Assert-Native "prepare protocol"
  return $prepared
}
function Volume-Exists([string]$Name) { return ((Probe { docker volume inspect $Name }) -eq 0) }

# --- 3. Modes --------------------------------------------------------------------------------
if ($Mode -eq "Test") {
  Step "Running offline tests in the image (no network)"
  $TestsMount = "type=bind,src=$App\tests,dst=/app/tests,readonly"
  docker run --rm --network none --mount $TestsMount $Image python -m unittest tests.test_routing_policy tests.test_stage20g_routing tests.test_stage20g_e2e tests.test_stage20g_runner tests.test_stage20_e2e_budget tests.test_llm_gateway
  Assert-Native "offline tests"
  Ok "offline tests passed"

  Step "Non-inference credential and route preflight (model/key metadata requests only)"
  $preflightLines = docker run --rm --env-file $Credentials $Image python scripts/stage20_e2e.py auth-preflight @FallbackArgs
  $preflightExit = $LASTEXITCODE
  $preflight = Json $preflightLines
  $preflightLines | Write-Host
  if ($preflightExit -ne 0 -or -not $preflight.ready) { throw "preflight not ready; see the checks above (no inference was sent)" }
  Ok "preflight ready"

  $prepared = Get-Prepared
  $Tag = $prepared.fingerprint.Substring(0, 12)
  $Volume = "stage20-e2e-v2-$Tag-results"
  $already = Volume-Exists $Volume
  $report = [ordered]@{
    stage = "20G"; mode = "Test"; commit = $Sha; image = $Image; variant = $Variant
    manifest = $ManifestName; offline_tests = "passed"; preflight_ready = $true
    campaign_id = $prepared.campaign_id; fingerprint = $prepared.fingerprint
    maximum_modeled_reservation_usd = $prepared.maximum_modeled_reservation_usd
    results_volume = $Volume; results_volume_exists = $already; inference_requests = 0
  }
  $reportPath = Join-Path $Operator "stage20g-test-$($Sha.Substring(0,7)).json"
  $report | ConvertTo-Json | Set-Content $reportPath -Encoding utf8
  Write-Host ""
  Write-Host "TEST COMPLETE. No inference request was made. Report: $reportPath" -ForegroundColor Green
  Write-Host "Protocol $Variant, worst-case modeled reservation `$$($prepared.maximum_modeled_reservation_usd)"
  Write-Host "Authorization fingerprint: $($prepared.fingerprint)"
  if ($already) { Write-Host "This campaign already ran (volume $Volume exists); Run will refuse." -ForegroundColor Yellow }
  else {
    $flag = if ($NoFallback) { " -NoFallback" } else { "" }
    Write-Host "After explicit owner approval only:  .\stage20.ps1 -Mode Run$flag -Authorization $($prepared.fingerprint)"
  }
  exit 0
}

if ($Mode -eq "Run") {
  if ($Authorization -notmatch '^[0-9a-f]{64}$') { throw "-Authorization must be the 64-character fingerprint printed by -Mode Test" }
  $prepared = Get-Prepared
  if ($prepared.fingerprint -ne $Authorization) { throw "authorization does not match this commit/manifest/protocol ($($prepared.fingerprint)); refusing" }
  Step "Preflight before the live campaign (no inference)"
  docker run --rm --env-file $Credentials $Image python scripts/stage20_e2e.py auth-preflight @FallbackArgs | Out-Null
  Assert-Native "preflight"

  $Tag = $Authorization.Substring(0, 12)
  $Volume = "stage20-e2e-v2-$Tag-results"
  if (Volume-Exists $Volume) { throw "campaign $Tag already ran (volume $Volume exists); a campaign is never rerun" }
  $Net = "stage20g-$Tag"; $Db = "stage20g-$Tag-db"; $Redis = "stage20g-$Tag-redis"
  foreach ($name in "$Volume", "$Net-db", "$Net-redis") { if (Volume-Exists $name) { throw "volume $name already exists; refusing" } }
  $Password = [guid]::NewGuid().ToString("N")
  $DatabaseUrl = "postgresql+asyncpg://scientific_rag:$Password@$Db`:5432/scientific_rag"
  try {
    Step "Creating isolated network, database, Redis and results volume ($Tag)"
    docker network create $Net | Out-Null; Assert-Native "network"
    foreach ($name in $Volume, "$Net-db", "$Net-redis") { docker volume create $name | Out-Null; Assert-Native "volume $name" }
    docker run -d --name $Db --network $Net -e POSTGRES_DB=scientific_rag -e POSTGRES_USER=scientific_rag -e "POSTGRES_PASSWORD=$Password" --mount "type=volume,src=$Net-db,dst=/var/lib/postgresql/data" pgvector/pgvector:pg16 | Out-Null
    Assert-Native "database container"
    docker run -d --name $Redis --network $Net --mount "type=volume,src=$Net-redis,dst=/data" redis:7-alpine redis-server --save 60 1 --loglevel warning | Out-Null
    Assert-Native "redis container"
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) { if ((Probe { docker exec $Db pg_isready -U scientific_rag -d scientific_rag }) -eq 0) { $ready = $true; break }; Start-Sleep 2 }
    if (-not $ready) { throw "database did not become ready" }
    docker run --rm --user root --mount "type=volume,src=$Volume,dst=/results" $Image sh -c "chown 1000:1000 /results && chmod 700 /results"
    Assert-Native "results volume ownership"
    docker run --rm --network $Net -e "DATABASE_URL=$DatabaseUrl" $Image alembic upgrade head; Assert-Native "migrations"

    Step "LIVE CAMPAIGN $Tag ($Variant): at most 100 chat + 125 embedding requests, worst case `$$($prepared.maximum_modeled_reservation_usd)"
    docker run --rm --name "stage20g-$Tag-runner" --network $Net --env-file $Credentials -e "DATABASE_URL=$DatabaseUrl" -e "REDIS_URL=redis://$Redis`:6379/0" --mount "type=volume,src=$Volume,dst=/results" --mount $OperatorMount $Image python scripts/stage20_e2e.py run --manifest "/operator/$ManifestName" --code-sha $Sha --authorization $Authorization --results /results @FallbackArgs
    $runExit = $LASTEXITCODE
  } finally {
    Step "Stopping campaign database and Redis (volumes are kept)"
    Probe { docker rm -f $Db $Redis } | Out-Null
    Probe { docker network rm $Net } | Out-Null
  }
  if ($runExit -ne 0) { throw "campaign runner exited with $runExit; results (if any) are kept in volume $Volume. Do not rerun." }
  $Mode = "Report"
}

if ($Mode -eq "Report") {
  if ($Authorization) { $Volume = "stage20-e2e-v2-$($Authorization.Substring(0, 12))-results" }
  else {
    $volumes = @(docker volume ls --format "{{.Name}}" | Where-Object { $_ -like "stage20-e2e-v2-*-results" })
    if ($volumes.Count -ne 1) { throw "pass -Authorization <fingerprint>; found $($volumes.Count) Stage 20G result volumes" }
    $Volume = $volumes[0]
  }
  if (-not (Volume-Exists $Volume)) { throw "no results volume $Volume" }
  Step "Exporting results from $Volume (read-only)"
  docker run --rm --network none --mount "type=volume,src=$Volume,dst=/results,readonly" --mount "type=bind,src=$Operator,dst=/out" $Image python scripts/stage20_e2e.py report --results /results --out /out
  Assert-Native "report export"
  Write-Host "Exported to $Operator (the *-review-private.json file contains questions and answers; keep it private)." -ForegroundColor Green
}
