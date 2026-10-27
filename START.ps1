# Run all project services after completing the README setup.
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

Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$projectRoot'; .\venv\Scripts\python.exe app.py"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$projectRoot\surveillance-backend'; npm start"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$projectRoot\surveillance-frontend'; npm run dev"
Start-Sleep -Seconds 5
Start-Process 'http://localhost:3000'
