
param(
  [int]$Tail = 50
)
# Tails the newest trial's heartbeat.ndjson once. Re-run after you start a new study.
$root = (Get-Location).Path
if ($root -like "*\.venv-gpu") { $root = Split-Path $root -Parent }
$reports = Join-Path $root "reports"
if (!(Test-Path $reports)) { Write-Host "No 'reports' directory under $root"; exit 1 }

$study = Get-ChildItem $reports -Filter "study_*" -Directory -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Desc | Select-Object -First 1
if (-not $study) { Write-Host "No study_* folder found."; exit 1 }

$trial = Get-ChildItem (Join-Path $study.FullName "t*") -Directory -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Desc | Select-Object -First 1
if (-not $trial) { Write-Host "Study exists but no t* trial folders yet (training just started?)"; exit 1 }

$hb = Join-Path $trial.FullName "heartbeat.ndjson"
if (!(Test-Path $hb)) { Write-Host "No heartbeat yet at $hb"; exit 1 }

Write-Host "Tailing: $hb"
Get-Content $hb -Tail $Tail -Wait
