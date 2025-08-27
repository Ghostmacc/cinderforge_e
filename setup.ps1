<#
  Safe setup script for CinderForge-E
  - No file generation, no repo mutation
  - Does NOT create a new venv unless missing or -RecreateVenv is passed
  - Installs deps from requirements.txt (or requirements.lock.txt with -UseLock)
  - Optionally prefers CPU Torch with -CPU
#>

[CmdletBinding()]
param(
  [switch]$RecreateVenv = $false,
  [string]$Python = "python",
  [switch]$CPU = $false,
  [switch]$UseLock = $false
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$VenvPath = Join-Path $Root ".venv"

if (Test-Path $VenvPath -and $RecreateVenv) {
  Write-Host "Removing existing venv at $VenvPath" -ForegroundColor Yellow
  Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
  Write-Host "Creating venv at $VenvPath" -ForegroundColor Cyan
  & $Python -m venv $VenvPath
}

& (Join-Path $VenvPath "Scripts/Activate.ps1")

Write-Host "Upgrading pip/setuptools/wheel" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# Base requirements
$ReqFile = if ($UseLock -and (Test-Path (Join-Path $Root "requirements.lock.txt"))) {
  Join-Path $Root "requirements.lock.txt"
} else {
  Join-Path $Root "requirements.txt"
}

if (Test-Path $ReqFile) {
  Write-Host "Installing dependencies from $ReqFile" -ForegroundColor Cyan
  pip install -r $ReqFile
} else {
  Write-Warning "Requirements file not found: $ReqFile"
}

# Torch choice (GPU CUDA 12.4 if possible, else CPU)
if ($CPU) {
  Write-Host "Installing Torch (CPU)" -ForegroundColor Cyan
  pip install --upgrade torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
} else {
  Write-Host "Installing Torch (CUDA 12.4) or fallback to CPU" -ForegroundColor Cyan
  try {
    pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
  } catch {
    Write-Warning "CUDA wheel failed. Falling back to CPU build."
    pip install --upgrade torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  }
}

Write-Host "Installing local package in editable mode (-e .)" -ForegroundColor Cyan
pip install -e .

Write-Host "\nDone. To activate later:" -ForegroundColor Green
Write-Host "& '$VenvPath\Scripts\Activate.ps1'"
Write-Host "Quick smoke: python -m cinderforge_e.validate.cli"
