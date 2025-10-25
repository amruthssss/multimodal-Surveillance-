<#
Setup script for Windows PowerShell.
Usage:
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 [-Python 3.11] [-Gpu]
#>
[CmdletBinding()]
param(
  [string]$PythonVersion = "3.11",
  [switch]$Gpu
)

Write-Host "[1/5] Checking Python..." -ForegroundColor Cyan
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Error "Python not found in PATH. Install Python $PythonVersion first."; exit 1
}

$pyVersionOut = (& python -c "import sys;print('.'.join(map(str, sys.version_info[:3])))" 2>$null)
if (-not $pyVersionOut) { $pyVersionOut = 'unknown' }
Write-Host "Using Python $pyVersionOut"

Write-Host "[2/5] Creating virtual environment (.venv)" -ForegroundColor Cyan
if (Test-Path .venv) { Write-Host "Virtualenv already exists" } else { python -m venv .venv }

Write-Host "[3/5] Activating venv" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "[4/5] Upgrading pip" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "[5/5] Installing requirements" -ForegroundColor Cyan
pip install -r requirements.txt

if ($Gpu) {
  Write-Host "Installing GPU torch (override CPU)" -ForegroundColor Yellow
  # Example for CUDA 12.1 wheels (adjust if different CUDA):
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade --force-reinstall
}

Write-Host "Done. Activate with:`n  . .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Green
