# HOWTO — Pre‑commit hook (protect docs)

**ABBRs used:** PR (Pull Request), CLI (Command‑Line Interface)

**Goal:** prevent accidental deletion of `docs/howto/*` and block stray “Copy code” lines in Markdown.

### Install (repo root)
```
git config core.hooksPath .githooks
New-Item -Force -ItemType Directory .githooks | Out-Null

# pre-commit (shell) — tiny launcher
@'
#!/bin/sh
powershell -NoProfile -ExecutionPolicy Bypass -File "$PWD/.githooks/pre-commit.ps1"
exit $?
'@ | Set-Content -Encoding Ascii .githooks/pre-commit

# pre-commit.ps1 — actual checks
@'
param()
$ErrorActionPreference = "Stop"
$protected = @(''docs/howto/'',''PITFALLS.md'')
$diff = (git diff --cached --name-status) -split "`n"
$bad = @()
foreach ($line in $diff) {
  if ($line -match ''^\s*D\s+(.+)$'') {
    $path = $Matches[1].Replace(''\'',''/'')
    foreach ($p in $protected) { if ($path -like "$p*") { $bad += $path } }
  }
}
if ($bad.Count -gt 0) { Write-Host "Blocked deleting protected files:"; $bad; exit 1 }

$md = (git diff --cached --name-only -- ''*.md'')
$hit = @()
foreach ($f in $md) {
  if (Test-Path $f) {
    $raw = Get-Content -Raw -LiteralPath $f
    if ($raw -match "(?m)^\s*Copy code\s*$") { $hit += $f }
  }
}
if ($hit.Count -gt 0) { Write-Host "Remove stray 'Copy code' lines:"; $hit; exit 1 }
exit 0
'@ | Set-Content -Encoding UTF8 .githooks/pre-commit.ps1
```

### Disable (if needed)
```
git config --unset core.hooksPath
```