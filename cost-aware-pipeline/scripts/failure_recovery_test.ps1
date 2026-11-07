param([int]$WorkerJobs = 20)

$ErrorActionPreference = "Stop"
Write-Host "Preflight: refusing to disrupt services while research is active..."
$active = docker compose exec -T db psql -U scientific_rag -d scientific_rag -tAc "SELECT count(*) FROM research_runs WHERE status IN ('queued','running');"
if ([int]$active -ne 0) { throw "There are $active active research runs. Wait or cancel them before failure injection." }

try {
    Write-Host "Testing four workers with provider-free jobs..."
    docker compose up -d --scale worker=4
    if ($LASTEXITCODE -ne 0) { throw "Could not scale workers to four" }
    docker compose exec -T api python scripts/queue_worker_probes.py --count $WorkerJobs --delay-ms 250 --timeout 60
    if ($LASTEXITCODE -ne 0) { throw "Four-worker queue probe failed" }

    Write-Host "Testing worker outage and recovery..."
    docker compose stop worker
    if ($LASTEXITCODE -ne 0) { throw "Could not stop workers" }
    $outageBatch = "worker-outage-$(Get-Date -Format 'yyyyMMddHHmmss')"
    docker compose exec -T api python scripts/queue_worker_probes.py --count 8 --delay-ms 150 --enqueue-only --batch-id $outageBatch
    if ($LASTEXITCODE -ne 0) { throw "Could not queue work during the worker outage" }
    docker compose start worker
    if ($LASTEXITCODE -ne 0) { throw "Could not restart workers" }
    docker compose exec -T api python scripts/queue_worker_probes.py --count 8 --delay-ms 150 --timeout 60 --batch-id $outageBatch
    if ($LASTEXITCODE -ne 0) { throw "Queued work did not recover after workers restarted" }

    Write-Host "Testing PostgreSQL outage visibility and recovery..."
    docker compose stop db
    try {
        Invoke-RestMethod -Uri "http://localhost:8000/healthz" -TimeoutSec 5 | Out-Null
        throw "Health endpoint unexpectedly succeeded while PostgreSQL was stopped"
    } catch {
        if ($_.Exception.Message -like "Health endpoint unexpectedly*") { throw }
    }
    docker compose start db
    if ($LASTEXITCODE -ne 0) { throw "PostgreSQL did not restart" }
    docker compose up -d api worker
    if ($LASTEXITCODE -ne 0) { throw "API/worker did not recover after PostgreSQL restart" }

    Write-Host "Testing Redis outage and recovery..."
    docker compose stop worker redis
    docker compose exec -T api python -c "import asyncio, os; from arq import create_pool; from arq.connections import RedisSettings; asyncio.run(create_pool(RedisSettings.from_dsn(os.environ['REDIS_URL'])))"
    if ($LASTEXITCODE -eq 0) { throw "Redis connection unexpectedly succeeded while Redis was stopped" }
    docker compose start redis
    if ($LASTEXITCODE -ne 0) { throw "Redis did not restart" }
    docker compose up -d --scale worker=2
    if ($LASTEXITCODE -ne 0) { throw "Workers did not recover after Redis restart" }
} finally {
    docker compose start db redis
    docker compose up -d --scale worker=1 api worker
}

docker compose ps
Write-Host "Failure recovery probe completed; services restored to one worker."
