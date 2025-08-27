
param(
  [int]$PlvWindow = 128,
  [string]$Device = "cpu",
  [string]$Precision = "fp32"
)
# Produces rc_precise.json/md for the newest trial's tokens.ndjson.
$root = (Get-Location).Path
if ($root -like "*\.venv-gpu") { $root = Split-Path $root -Parent }
$reports = Join-Path $root "reports"
if (!(Test-Path $reports)) { Write-Host "No 'reports' directory under $root"; exit 1 }

$study = Get-ChildItem $reports -Filter "study_*" -Directory -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Desc | Select-Object -First 1
if (-not $study) { Write-Host "No study_* folder found."; exit 1 }

$trial = Get-ChildItem (Join-Path $study.FullName "t*") -Directory -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Desc | Select-Object -First 1
if (-not $trial) { Write-Host "Study exists but no t* trial folders yet"; exit 1 }

$tokens = Join-Path $trial.FullName "tokens.ndjson"
if (!(Test-Path $tokens)) { Write-Host "No tokens.ndjson at $tokens"; exit 1 }

$out_json = Join-Path $trial.FullName "rc_precise.json"
$out_md   = Join-Path $trial.FullName "rc_precise.md"
$cmd = "raincatcher --log_file `"$tokens`" --plv_window $PlvWindow --device $Device --precision $Precision --out_json `"$out_json`" --out_md `"$out_md`""
Write-Host "Running: $cmd"
Invoke-Expression $cmd

Write-Host "`n--- rc_precise.json ---"
Get-Content $out_json
Write-Host "`nSaved:"
Write-Host "  $out_json"
Write-Host "  $out_md"
