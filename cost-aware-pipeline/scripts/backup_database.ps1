param([string]$OutputDirectory = ".\backups")

$ErrorActionPreference = "Stop"
$resolvedRoot = [System.IO.Path]::GetFullPath((Join-Path $PWD $OutputDirectory))
New-Item -ItemType Directory -Path $resolvedRoot -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$name = "scientific-rag-$stamp.dump"
$containerPath = "/tmp/$name"
$localPath = Join-Path $resolvedRoot $name

docker compose exec -T db pg_dump -U scientific_rag -d scientific_rag -Fc -f $containerPath
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed" }
docker compose cp "db:$containerPath" $localPath
if ($LASTEXITCODE -ne 0) { throw "Could not copy the backup from PostgreSQL" }
docker compose exec -T db rm -f $containerPath
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localPath).Hash
Write-Host "Backup: $localPath"
Write-Host "SHA256: $hash"
