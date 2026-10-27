@echo off
setlocal
cd /d "%~dp0"

if not exist ".env" (
  echo Missing .env. Copy .env.example to .env and set JWT_SECRET first.
  pause
  exit /b 1
)
if not exist "venv\Scripts\python.exe" (
  echo Missing Python environment. Create venv and install requirements.txt first.
  pause
  exit /b 1
)
if not exist "surveillance-backend\node_modules" (
  echo Missing Node dependencies. Run npm install in surveillance-backend first.
  pause
  exit /b 1
)
if not exist "surveillance-frontend\node_modules" (
  echo Missing frontend dependencies. Run npm install in surveillance-frontend first.
  pause
  exit /b 1
)

start "Surveillance AI" cmd /k "cd /d ""%~dp0"" && venv\Scripts\python.exe app.py"
start "Surveillance API" cmd /k "cd /d ""%~dp0surveillance-backend"" && npm start"
start "Surveillance Frontend" cmd /k "cd /d ""%~dp0surveillance-frontend"" && npm run dev"

timeout /t 5 /nobreak >nul
start "" http://localhost:3000
echo AI service: http://localhost:5000/health
echo API:        http://localhost:5001/api/health
echo Frontend:   http://localhost:3000
pause
