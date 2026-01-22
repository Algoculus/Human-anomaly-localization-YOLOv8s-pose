# Fall Detection System - Quick Start (Windows)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fall Detection System - Quick Start" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python found" -ForegroundColor Green

# Check Node.js
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Node.js found" -ForegroundColor Green

# Check pnpm
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  pnpm not found. Installing..." -ForegroundColor Yellow
    npm install -g pnpm
}
Write-Host "✅ pnpm found" -ForegroundColor Green
Write-Host ""

# Setup backend
Write-Host "🔧 Setting up backend..." -ForegroundColor Yellow
Set-Location backend

# Create virtual environment if not exists
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Create artifacts directory
New-Item -ItemType Directory -Force -Path "artifacts\videos" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts\snapshots" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts\reports" | Out-Null

Write-Host "✅ Backend setup complete" -ForegroundColor Green
Set-Location ..
Write-Host ""

# Setup frontend
Write-Host "🔧 Setting up frontend..." -ForegroundColor Yellow
Set-Location frontend

# Install dependencies
Write-Host "📦 Installing Node dependencies..." -ForegroundColor Yellow
pnpm install --silent

Write-Host "✅ Frontend setup complete" -ForegroundColor Green
Set-Location ..
Write-Host ""

# Start backend
Write-Host "🚀 Starting backend server..." -ForegroundColor Yellow
Set-Location backend
& .\venv\Scripts\Activate.ps1

Start-Process -FilePath "cmd" -ArgumentList "/c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 > ..\backend.log 2>&1" -WindowStyle Hidden
Start-Sleep -Seconds 3

Write-Host "✅ Backend started" -ForegroundColor Green
Write-Host "   Logs: backend.log" -ForegroundColor Gray
Write-Host "   API: http://localhost:8000" -ForegroundColor Gray
Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor Gray
Set-Location ..
Write-Host ""

# Start frontend
Write-Host "🚀 Starting frontend dev server..." -ForegroundColor Yellow
Set-Location frontend

Start-Process -FilePath "cmd" -ArgumentList "/c", "pnpm dev > ..\frontend.log 2>&1" -WindowStyle Hidden
Start-Sleep -Seconds 2

Write-Host "✅ Frontend started" -ForegroundColor Green
Write-Host "   Logs: frontend.log" -ForegroundColor Gray
Write-Host "   URL: http://localhost:5173" -ForegroundColor Gray
Set-Location ..
Write-Host ""

# Show status
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✨ System is running!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "   - Camera Simulator: http://localhost:5173/camera" -ForegroundColor Gray
Write-Host "   - Alert Receiver: http://localhost:5173/receiver" -ForegroundColor Gray
Write-Host "   - Video Upload: http://localhost:5173/upload" -ForegroundColor Gray
Write-Host ""
Write-Host "🔌 Backend: http://localhost:8000" -ForegroundColor Green
Write-Host "   - API Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "   - Health: http://localhost:8000/health" -ForegroundColor Gray
Write-Host "   - WebSocket: ws://localhost:8000/ws" -ForegroundColor Gray
Write-Host ""
Write-Host "📝 Logs:" -ForegroundColor Yellow
Write-Host "   - Backend: Get-Content -Path backend.log -Wait" -ForegroundColor Gray
Write-Host "   - Frontend: Get-Content -Path frontend.log -Wait" -ForegroundColor Gray
Write-Host ""
Write-Host "🛑 To stop: Run .\stop.ps1" -ForegroundColor Yellow
Write-Host ""
