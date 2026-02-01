"""
FastAPI Main Application
Production-grade fall detection service with WebSocket support.
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add core_ai to path (symlink to urfd_fall_yolo_pose)
CORE_AI_PATH = Path(__file__).parent.parent.parent.parent / "urfd_fall_yolo_pose"
sys.path.insert(0, str(CORE_AI_PATH))

from app.api import health, upload, artifacts
from app.ws import router as ws_router
from app.core.config import get_settings

settings = get_settings()

# Create artifacts directory before FastAPI initialization
Path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print(f"[STARTUP] Starting {settings.app_name} v{settings.version}")
    print(f"[CORE_AI] Core AI path: {CORE_AI_PATH}")
    print(f"[ENV] Environment: {settings.app_env}")
    print(f"[CORS] Allowed origins: {settings.allowed_origins}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down gracefully...")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
app.include_router(ws_router, tags=["websocket"])

# Static files for artifacts
app.mount("/static", StaticFiles(directory=settings.artifact_dir), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=4611,
        reload=True if settings.app_env == "dev" else False
    )
