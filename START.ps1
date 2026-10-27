# Run all project services after completing the README setup.
Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requiredPaths = @(
    (Join-Path $projectRoot '.env'),
    (Join-Path $projectRoot 'venv\Scripts\python.exe'),
    (Join-Path $projectRoot 'surveillance-backend\node_modules'),
    (Join-Path $projectRoot 'surveillance-frontend\node_modules')
)

foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path $requiredPath)) {
        throw "Missing $requiredPath. Complete the setup steps in README.md first."
    }
}

$pythonPath = Join-Path $projectRoot 'venv\Scripts\python.exe'
& $pythonPath -c 'import cv2, flask, flask_socketio, librosa, numpy, onnxruntime, requests, sqlalchemy, torch' *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Python dependencies are missing. Activate venv and run: python -m pip install -r requirements.txt'
}

Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$projectRoot'; .\venv\Scripts\python.exe app.py"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$projectRoot\surveillance-backend'; npm start"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$projectRoot\surveillance-frontend'; npm run dev"
Start-Sleep -Seconds 5
Start-Process 'http://127.0.0.1:4300'
