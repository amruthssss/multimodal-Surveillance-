@echo off
echo ================================================
echo  MongoDB Atlas Connection Setup
echo ================================================
echo.
echo Your connection string has been added to:
echo   surveillance-backend\.env
echo.
echo IMPORTANT: You need to replace ^<db_password^> with your actual password!
echo.
echo Steps:
echo 1. Open: surveillance-backend\.env
echo 2. Find line: MONGODB_URI=mongodb+srv://Amruth:^<db_password^>@cluster0...
echo 3. Replace ^<db_password^> with your MongoDB Atlas password
echo 4. Save the file
echo.
echo Example:
echo   Before: mongodb+srv://Amruth:^<db_password^>@cluster0...
echo   After:  mongodb+srv://Amruth:MyPassword123@cluster0...
echo.
echo After updating the password, press any key to start the servers...
pause

echo.
echo Starting servers...
call .\SIMPLE_START.bat
