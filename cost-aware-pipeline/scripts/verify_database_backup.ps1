param([Parameter(Mandatory=$true)][string]$BackupPath)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -LiteralPath $BackupPath).Path
$database = "scientific_rag_restore_test_$(Get-Date -Format 'yyyyMMddHHmmss')"
$containerPath = "/tmp/restore-check.dump"
try {
    docker compose cp $resolved "db:$containerPath"
    docker compose exec -T db createdb -U scientific_rag $database
    docker compose exec -T db pg_restore -U scientific_rag -d $database --no-owner --no-privileges $containerPath
    $tables = docker compose exec -T db psql -U scientific_rag -d $database -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';"
    $vector = docker compose exec -T db psql -U scientific_rag -d $database -tAc "SELECT count(*) FROM pg_extension WHERE extname='vector';"
    if ([int]$tables -lt 1 -or [int]$vector -ne 1) { throw "Restore verification failed: tables=$tables vector=$vector" }
    Write-Host "Restore verified in temporary database $database ($tables public tables, pgvector present)."
} finally {
    docker compose exec -T db dropdb -U scientific_rag --if-exists $database
    docker compose exec -T db rm -f $containerPath
}
