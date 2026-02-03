# Artifacts Serving (Processed Videos, Snapshots)
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import get_settings
from pathlib import Path

router = APIRouter()
settings = get_settings()

@router.get("/{artifact_type}/{filename}")
async def get_artifact(artifact_type: str, filename: str):
    # Serve processed artifacts (videos, snapshots, reports)
    
    # Validate artifact type
    allowed_types = ["videos", "snapshots", "reports"]
    if artifact_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artifact type. Must be one of: {allowed_types}"
        )
    
    # Build file path
    artifact_path = settings.artifact_dir / artifact_type / filename
    
    # Security: prevent directory traversal
    try:
        artifact_path = artifact_path.resolve()
        if not str(artifact_path).startswith(str(settings.artifact_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")
    
    # Check if file exists
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact not found: {artifact_type}/{filename}"
        )
    
    # Determine media type
    suffix = artifact_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
        ".txt": "text/plain"
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=filename
    )

@router.delete("/{artifact_type}/{filename}")
async def delete_artifact(artifact_type: str, filename: str):
    # Delete an artifact file
    
    allowed_types = ["videos", "snapshots", "reports"]
    if artifact_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artifact type. Must be one of: {allowed_types}"
        )
    
    artifact_path = settings.artifact_dir / artifact_type / filename
    
    # Security check
    try:
        artifact_path = artifact_path.resolve()
        if not str(artifact_path).startswith(str(settings.artifact_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")
    
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    try:
        artifact_path.unlink()
        return {"status": "deleted", "file": f"{artifact_type}/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
