@echo off
cls
echo ================================================
echo  SURVEILLANCE SYSTEM - QUICK START
echo ================================================
echo.

echo [Step 1/3] Starting Backend Server (MongoDB + Express + JWT)...
echo.
start cmd /k "cd /d %~dp0surveillance-backend && echo Backend Server Starting... && npm start"
timeout /t 3 /nobreak >nul

echo [Step 2/3] Starting Frontend Server (React App)...
echo.
start cmd /k "cd /d %~dp0surveillance-frontend && echo Frontend Server Starting... && npm start"
timeout /t 3 /nobreak >nul

echo [Step 3/3] Opening Browser...
echo.
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo ================================================
echo  SERVERS STARTED!
echo ================================================
echo.
echo Backend:  http://localhost:5001/api
echo Frontend: http://localhost:3000
echo.
echo Tailwind Landing: http://localhost:3000/landing-tailwind.html
echo React Landing:    http://localhost:3000
echo.
echo INSTRUCTIONS:
echo 1. Wait for both servers to finish starting (15-30 seconds)
echo 2. Browser will open automatically to http://localhost:3000
echo 3. Click "GET STARTED" or "SIGN UP" button
echo 4. Register new account (username, email, mobile, password)
echo 5. Check backend console window for 6-digit OTP code
echo 6. Enter OTP to verify account
echo 7. Login and access Dashboard
echo 8. Configure camera (try Built-in Camera option!)
echo.
echo To stop servers: Close both command windows or press Ctrl+C
echo.
pause
