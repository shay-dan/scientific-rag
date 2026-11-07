param(
    [int]$Requests = 50,
    [int]$Concurrency = 10
)

$ErrorActionPreference = "Stop"
docker compose up -d --build

Write-Host "Waiting for API readiness..."
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/healthz" -TimeoutSec 3
        if ($health.status -eq "ok") { break }
    } catch {}
    Start-Sleep -Seconds 2
}
if ($health.status -ne "ok") { throw "API did not become ready" }

Write-Host "Running pricing, budget-concurrency, and provider failure-injection tests..."
docker compose run --rm --no-deps -v "${PWD}\tests:/app/tests:ro" api `
    sh -lc 'python -m unittest discover -s tests -p "test_cost_control.py" -v && python -m unittest discover -s tests -p "test_llm_gateway.py" -v'

Write-Host "Running live Redis/PostgreSQL/HTTP concurrency probe (no LLM calls)..."
docker compose exec -T api python scripts/reliability_probe.py `
    --base-url http://127.0.0.1:8000 --requests $Requests --concurrency $Concurrency
