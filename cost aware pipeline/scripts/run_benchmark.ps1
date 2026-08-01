param(
    [ValidateSet("fast", "standard", "deep")]
    [string]$Mode = "standard",
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
$benchmark.name = "$($benchmark.name)-$timestamp"
$headers = @{ "X-API-Key" = $ApiKey }

$dataset = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/evaluations/datasets" -Headers $headers -ContentType "application/json" -Body ($benchmark | ConvertTo-Json -Depth 10)
$run = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/evaluations/datasets/$($dataset.dataset_id)/runs" -Headers $headers -ContentType "application/json" -Body (@{ mode = $Mode } | ConvertTo-Json)

Write-Host "Benchmark dataset: $($benchmark.name) ($($dataset.case_count) cases)"
Write-Host "Evaluation run: $($run.evaluation_run_id)"
Write-Host "Live JSON status: $BaseUrl/v1/evaluations/runs/$($run.evaluation_run_id)"
Write-Host "Browser UI: $BaseUrl/evaluation"
