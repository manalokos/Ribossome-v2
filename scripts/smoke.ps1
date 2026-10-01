# Teste rápido: corre a build indicada N segundos com o profiler e termina
# SÓ o processo que lançou (nunca outras sessões abertas do Filipe).
# Uso: powershell -File scripts/smoke.ps1 debug|release [segundos]
param([string]$Build = "release", [int]$Seconds = 12)
$exe = Join-Path $PSScriptRoot "..\target\$Build\ribossome.exe"
$log = Join-Path $env:TEMP "ribo_smoke_$Build.log"
$env:RIBO_PROFILE = "1"
$p = Start-Process -FilePath $exe -PassThru -RedirectStandardError $log -WindowStyle Minimized
Start-Sleep -Seconds $Seconds
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force } else { Write-Output "terminou sozinho com código $($p.ExitCode)" }
Get-Content $log | Where-Object { $_ -notmatch 'Loader Message|objects:|registry|bandicam' } | Select-Object -Last 3
