"""
Health Check Endpoints
"""
from fastapi import APIRouter
from app.ws.connection_manager import manager
from app.core.config import get_settings
import sys
import torch

router = APIRouter()
settings = get_settings()

@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns server status and active connections count.
    """
    total_connections = sum(
        len(room["cameras"]) + len(room["receivers"])
        for room in manager.rooms.values()
    )
    
    return {
        "status": "ok",
        "activeConnections": total_connections,
        "activeRooms": len(manager.rooms),
        "activeSessions": len(manager.session_manager.sessions)
    }

@router.get("/version")
async def version_info():
    """
    System version information.
    Returns API version, Python version, PyTorch version, and config details.
    """
    return {
        "version": "1.0.0",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "config": {
            "maxFps": settings.ws_max_fps,
            "frameRateLimit": settings.frame_rate_limit,
            "maxFrameSize": settings.max_frame_size,
            "artifactDir": str(settings.artifact_dir)
        }
    }

@router.get("/rooms")
async def list_rooms():
    """
    List all active rooms with connection counts.
    """
    return {
        "rooms": [
            {
                "roomId": room_id,
                "cameras": len(conns["cameras"]),
                "receivers": len(conns["receivers"])
            }
            for room_id, conns in manager.rooms.items()
        ]
    }
