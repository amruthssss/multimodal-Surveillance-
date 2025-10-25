@echo off
echo ================================================
echo  Next-Gen AI Surveillance System - Quick Start
echo ================================================
echo.

echo [1/4] Installing Backend Dependencies...
cd surveillance-backend
call npm install
if errorlevel 1 (
    echo ERROR: Backend installation failed!
    pause
    exit /b 1
)

echo.
echo [2/4] Installing Frontend Dependencies...
cd ..\surveillance-frontend
call npm install
if errorlevel 1 (
    echo ERROR: Frontend installation failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Starting Backend Server...
cd ..\surveillance-backend
start cmd /k "npm start"
timeout /t 3

echo.
echo [4/4] Starting Frontend Server...
cd ..\surveillance-frontend
start cmd /k "npm start"

echo.
echo ================================================
echo  Setup Complete!
echo ================================================
echo.
echo  Backend running on: http://localhost:5001
echo  Frontend running on: http://localhost:3000
echo.
echo  Opening browser in 5 seconds...
timeout /t 5
start http://localhost:3000

echo.
pause
