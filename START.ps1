# Surveillance System - PowerShell Startup Script
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host " SURVEILLANCE SYSTEM - QUICK START" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# Check if Node.js is installed
Write-Host "[1/4] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✓ Node.js $nodeVersion detected" -ForegroundColor Green
} catch {
    Write-Host "✗ Node.js not found! Please install from https://nodejs.org" -ForegroundColor Red
    pause
    exit
}

# Check if dependencies are installed
Write-Host "`n[2/4] Checking dependencies..." -ForegroundColor Yellow
if (!(Test-Path "surveillance-backend\node_modules")) {
    Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
    Set-Location surveillance-backend
    npm install
    Set-Location ..
}
if (!(Test-Path "surveillance-frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location surveillance-frontend
    npm install
    Set-Location ..
}
Write-Host "✓ Dependencies ready" -ForegroundColor Green

# Start Backend Server
Write-Host "`n[3/4] Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\surveillance-backend'; Write-Host '🚀 Backend Server Starting...' -ForegroundColor Cyan; npm start"
Start-Sleep -Seconds 2

# Start Frontend Server
Write-Host "[4/4] Starting Frontend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\surveillance-frontend'; Write-Host '⚛️ React App Starting...' -ForegroundColor Cyan; npm start"
Start-Sleep -Seconds 3

Write-Host "`n================================================" -ForegroundColor Green
Write-Host " SERVERS STARTING!" -ForegroundColor Green
Write-Host "================================================`n" -ForegroundColor Green

Write-Host "Backend:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5001/api" -ForegroundColor Cyan
Write-Host "Frontend: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:3000" -ForegroundColor Cyan
Write-Host "`nTailwind: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:3000/landing-tailwind.html" -ForegroundColor Yellow

Write-Host "`n📋 QUICK START GUIDE:" -ForegroundColor Magenta
Write-Host "   1. Wait 15-30 seconds for servers to start" -ForegroundColor Gray
Write-Host "   2. Browser opens automatically to Landing Page" -ForegroundColor Gray
Write-Host "   3. Click 'GET STARTED' button" -ForegroundColor Gray
Write-Host "   4. Register (username, email, mobile, password)" -ForegroundColor Gray
Write-Host "   5. Backend console shows 6-digit OTP" -ForegroundColor Gray
Write-Host "   6. Enter OTP to verify account" -ForegroundColor Gray
Write-Host "   7. Login and access Dashboard" -ForegroundColor Gray
Write-Host "   8. Configure Built-in Camera! 📹" -ForegroundColor Gray

Write-Host "`n⏳ Opening browser in 5 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
Start-Process "http://localhost:3000"

Write-Host "`n✅ Setup Complete! Servers are running in separate windows." -ForegroundColor Green
Write-Host "To stop: Close the PowerShell windows or press Ctrl+C" -ForegroundColor Gray
Write-Host "`nPress any key to exit this window..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
