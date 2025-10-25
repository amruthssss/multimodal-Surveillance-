@echo off
echo ================================================
echo  EASY SETUP - Next-Gen AI Surveillance System
echo ================================================
echo.

echo Choose your database option:
echo.
echo [1] MongoDB Atlas (Cloud - Recommended, No installation needed)
echo [2] Install MongoDB locally
echo [3] Skip MongoDB setup (I'll configure it later)
echo.
set /p choice="Enter choice (1, 2, or 3): "

if "%choice%"=="1" goto atlas
if "%choice%"=="2" goto local
if "%choice%"=="3" goto skip

:atlas
echo.
echo ================================================
echo  Setting up with MongoDB Atlas (Cloud)
echo ================================================
echo.
echo Steps:
echo 1. Go to: https://www.mongodb.com/cloud/atlas
echo 2. Sign up for FREE account
echo 3. Create a FREE cluster (M0 tier)
echo 4. Click "Connect" and get your connection string
echo 5. Copy the connection string
echo.
echo Your connection string looks like:
echo mongodb+srv://username:password@cluster.mongodb.net/surveillance
echo.
pause
echo.
echo Opening MongoDB Atlas in browser...
start https://www.mongodb.com/cloud/atlas/register
echo.
echo After getting your connection string:
echo 1. Open: surveillance-backend\.env
echo 2. Replace MONGODB_URI with your connection string
echo 3. Save the file
echo 4. Run: .\START_SERVERS.bat
echo.
pause
exit

:local
echo.
echo ================================================
echo  Installing MongoDB Locally
echo ================================================
echo.
echo Opening MongoDB download page...
start https://www.mongodb.com/try/download/community
echo.
echo After installation:
echo 1. MongoDB should start automatically
echo 2. Run: .\START_SERVERS.bat
echo.
pause
exit

:skip
echo.
echo ================================================
echo  Skipping MongoDB Setup
echo ================================================
echo.
echo You can configure MongoDB later by editing:
echo   surveillance-backend\.env
echo.
echo Then run: .\START_SERVERS.bat
echo.
pause
exit
