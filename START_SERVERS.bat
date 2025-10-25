@echo off
echo ================================================
echo  Starting Surveillance System Servers
echo ================================================
echo.

echo Starting Backend Server...
start cmd /k "cd surveillance-backend && npm start"

timeout /t 2

echo Starting Frontend Server...
start cmd /k "cd surveillance-frontend && npm start"

echo.
echo Servers are starting...
echo Backend: http://localhost:5001
echo Frontend: http://localhost:3000
echo.

timeout /t 5
start http://localhost:3000

pause
