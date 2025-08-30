param()
$ErrorActionPreference = "Stop"

# 1) Block deleting protected docs paths
$protected = @('docs/howto/','PITFALLS.md')
$diff = (git diff --cached --name-status) -split "`n"
$bad = @()
foreach ($line in $diff) {
  if ($line -match '^\s*D\s+(.+)$') {
    $path = $Matches[1].Replace('\','/')
    foreach ($p in $protected) { if ($path -like "$p*") { $bad += $path } }
  }
}
if ($bad.Count -gt 0) {
  Write-Host "❌ Pre-commit blocked. You tried to delete protected files:" -ForegroundColor Red
  $bad | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
  exit 1
}

# 2) Block pasted 'Copy code' lines in staged Markdown
$md = (git diff --cached --name-only -- '*.md')
$hit = @()
foreach ($f in $md) {
  if (Test-Path $f) {
    $raw = Get-Content -Raw -LiteralPath $f
    if ($raw -match "(?m)^\s*Copy code\s*$") { $hit += $f }
  }
}
if ($hit.Count -gt 0) {
  Write-Host "❌ Pre-commit blocked. Remove stray 'Copy code' lines from these Markdown files:" -ForegroundColor Red
  $hit | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
  exit 1
}

exit 0
# === ABBR expansion check (docs/howto, docs/manual, README.md) ===
$enforcePaths = @('docs/howto/','docs/manual/','README.md')
$abbrs = @('NAS','CLI','RC','PLV','SSM','RoPE','iRoPE','MHLA','BF16','FP16','FP32','CUDA','GPU','NDJSON','JSON','YAML','TOML')
$viol = @()
$mdfiles = (git diff --cached --name-only -- '*.md') | ForEach-Object { $_.Trim() }

foreach ($f in $mdfiles) {
  $rel = $f.Replace('\','/')
  if (-not ($enforcePaths | Where-Object { $rel -like "$_*" -or $rel -eq $_ })) { continue }
  if (-not (Test-Path $f)) { continue }
  $raw  = Get-Content -Raw -LiteralPath $f
  # strip fenced code blocks to avoid false positives
  $nocode = [regex]::Replace($raw, '(?s)```.*?```', '')
  foreach ($abbr in $abbrs) {
    if ($nocode -match "(?<![A-Za-z])$abbr(?![A-Za-z])") {
      if ($nocode -notmatch "(?<![A-Za-z])$abbr\s*\(") {
        $viol += "$rel — add first‑use expansion for '$abbr' (e.g., $abbr (<definition>))"
      }
    }
  }
}

if ($viol.Count -gt 0) {
  Write-Host "❌ Pre-commit blocked. Missing ABBR→Definition first-use expansions:" -ForegroundColor Red
  $viol | Sort-Object -Unique | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
  Write-Host "Hint: See docs/indices/GLOSSARY.md" -ForegroundColor Yellow
  exit 1
}
