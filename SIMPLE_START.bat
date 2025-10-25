@echo off
cls
echo ================================================
echo  Next-Gen AI Surveillance System - Launcher
echo ================================================
echo.

echo [Step 1/4] Checking Node.js installation...
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed!
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo Download the LTS version and install it.
    echo.
    pause
    start https://nodejs.org/
    exit /b 1
)
node --version
echo OK - Node.js found!
echo.

echo [Step 2/4] Checking if backend dependencies are installed...
if not exist "surveillance-backend\node_modules\" (
    echo Installing backend dependencies...
    cd surveillance-backend
    call npm install
    cd ..
    echo.
)

echo [Step 3/4] Checking if frontend dependencies are installed...
if not exist "surveillance-frontend\node_modules\" (
    echo Installing frontend dependencies...
    cd surveillance-frontend
    call npm install
    cd ..
    echo.
)

echo [Step 4/4] Starting servers...
echo.
echo WARNING: Make sure you have configured MongoDB in:
echo   surveillance-backend\.env
echo.
echo Options:
echo   1. MongoDB Atlas (Cloud - Free): https://mongodb.com/cloud/atlas
echo   2. Local MongoDB: Install from https://mongodb.com/try/download/community
echo.
set /p continue="Press ENTER to continue or Ctrl+C to configure MongoDB first..."

echo.
echo Starting Backend Server (http://localhost:5001)...
start "Backend Server" cmd /k "cd surveillance-backend && npm start"

timeout /t 5

echo Starting Frontend Server (http://localhost:3000)...
start "Frontend Server" cmd /k "cd surveillance-frontend && npm start"

echo.
echo ================================================
echo  Servers are starting!
echo ================================================
echo.
echo Backend:  http://localhost:5001
echo Frontend: http://localhost:3000
echo.
echo The browser will open automatically in a moment...
echo.
echo To configure MongoDB Atlas:
echo 1. Go to: https://mongodb.com/cloud/atlas
echo 2. Create free account and cluster
echo 3. Get connection string
echo 4. Update surveillance-backend\.env
echo.

timeout /t 10
start http://localhost:3000

echo.
echo Press any key to close this window...
pause >nul
