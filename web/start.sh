#!/bin/bash

# Fall Detection System - Quick Start Script
# This script sets up and runs both backend and frontend locally

set -e  # Exit on error

echo "=========================================="
echo "Fall Detection System - Quick Start"
echo "=========================================="
echo ""

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not found. Please install Python 3.11+"
        exit 1
    fi
    echo "✅ Python 3 found"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js not found. Please install Node.js 18+"
        exit 1
    fi
    echo "✅ Node.js found"
    
    # Check pnpm
    if ! command -v pnpm &> /dev/null; then
        echo "⚠️  pnpm not found. Installing..."
        npm install -g pnpm
    fi
    echo "✅ pnpm found"
    
    echo ""
}

# Setup backend
setup_backend() {
    echo "🔧 Setting up backend..."
    cd backend
    
    # Create virtual environment if not exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    
    # Install dependencies
    echo "📦 Installing Python dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    # Create artifacts directory
    mkdir -p artifacts/{videos,snapshots,reports}
    
    echo "✅ Backend setup complete"
    cd ..
    echo ""
}

# Setup frontend
setup_frontend() {
    echo "🔧 Setting up frontend..."
    cd frontend
    
    # Install dependencies
    echo "📦 Installing Node dependencies..."
    pnpm install --silent
    
    echo "✅ Frontend setup complete"
    cd ..
    echo ""
}

# Start backend
start_backend() {
    echo "🚀 Starting backend server..."
    cd backend
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    
    # Start in background
    uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../backend.pid
    
    echo "✅ Backend started (PID: $BACKEND_PID)"
    echo "   Logs: backend.log"
    echo "   API: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    cd ..
    echo ""
}

# Start frontend
start_frontend() {
    echo "🚀 Starting frontend dev server..."
    cd frontend
    
    # Start in background
    pnpm dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../frontend.pid
    
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
    echo "   Logs: frontend.log"
    echo "   URL: http://localhost:5173"
    cd ..
    echo ""
}

# Show status
show_status() {
    echo "=========================================="
    echo "✨ System is running!"
    echo "=========================================="
    echo ""
    echo "🌐 Frontend: http://localhost:5173"
    echo "   - Camera Simulator: http://localhost:5173/camera"
    echo "   - Alert Receiver: http://localhost:5173/receiver"
    echo "   - Video Upload: http://localhost:5173/upload"
    echo ""
    echo "🔌 Backend: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo "   - Health: http://localhost:8000/health"
    echo "   - WebSocket: ws://localhost:8000/ws"
    echo ""
    echo "📝 Logs:"
    echo "   - Backend: tail -f backend.log"
    echo "   - Frontend: tail -f frontend.log"
    echo ""
    echo "🛑 To stop:"
    echo "   ./stop.sh"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    setup_backend
    setup_frontend
    
    echo "=========================================="
    echo "🚀 Starting services..."
    echo "=========================================="
    echo ""
    
    start_backend
    sleep 3  # Wait for backend to start
    
    start_frontend
    sleep 2  # Wait for frontend to start
    
    show_status
    
    # Keep script running
    echo "Press Ctrl+C to stop all services..."
    wait
}

# Cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    
    if [ -f "backend.pid" ]; then
        kill $(cat backend.pid) 2>/dev/null || true
        rm backend.pid
    fi
    
    if [ -f "frontend.pid" ]; then
        kill $(cat frontend.pid) 2>/dev/null || true
        rm frontend.pid
    fi
    
    echo "✅ Services stopped"
}

trap cleanup EXIT INT TERM

main
