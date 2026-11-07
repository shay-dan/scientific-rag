param(
    [ValidateSet("fast", "standard", "deep")]
    [string]$Mode = "standard",
    [ValidateRange(1, 10)]
    [int]$CaseCount = 1,
    [string]$BaseUrl = "http://localhost:8000",
    [string]$ApiKey = $env:APP_API_KEY
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    throw "Set APP_API_KEY first, for example: `$env:APP_API_KEY = '<the admin key from .env>'"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$benchmarkPath = Join-Path $projectRoot "evaluation\benchmark-v1.json"
$benchmark = Get-Content -Raw $benchmarkPath | ConvertFrom-Json
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$benchmark.cases = @($benchmark.cases | Select-Object -First $CaseCount)
$benchmark.name = "$($benchmark.name)-$CaseCount-cases-$timestamp"
$headers = @{ "X-API-Key" = $ApiKey }

Write-Host "Waiting for API readiness at $BaseUrl/healthz ..."
$ready = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/healthz" -TimeoutSec 3
        if ($health.status -eq "ok") {
            $ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    throw "The API did not become ready within 60 seconds. Run 'docker compose logs api' to see the startup error."
}

$dataset = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/evaluations/datasets" -Headers $headers -ContentType "application/json" -Body ($benchmark | ConvertTo-Json -Depth 10)
$run = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/evaluations/datasets/$($dataset.dataset_id)/runs" -Headers $headers -ContentType "application/json" -Body (@{ mode = $Mode } | ConvertTo-Json)

Write-Host "Benchmark dataset: $($benchmark.name) ($($dataset.case_count) cases, $Mode mode)"
Write-Host "Evaluation run: $($run.evaluation_run_id)"
Write-Host "Live JSON status: $BaseUrl/v1/evaluations/runs/$($run.evaluation_run_id)"
Write-Host "Browser UI: $BaseUrl/evaluation"
