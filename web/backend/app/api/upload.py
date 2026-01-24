# Video Upload & Offline Processing
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from app.core.config import get_settings
from pathlib import Path
import uuid
import shutil
import subprocess
import json
import cv2

router = APIRouter()
settings = get_settings()

@router.post("/")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    # Upload video for offline fall detection processing
    # Returns: jobId, status="queued", message
    # Client polls GET /upload/{jobId} to check status
    
    # Validate file type
    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Create job directory
    job_dir = settings.artifact_dir / "videos" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded video
    input_path = job_dir / f"input{file_ext}"
    
    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    finally:
        file.file.close()
    
    # Create job metadata
    job_meta = {
        "jobId": job_id,
        "status": "queued",
        "filename": file.filename,
        "inputPath": str(input_path),
        "outputPath": str(job_dir / "output.mp4"),
        "metricsPath": str(job_dir / "metrics.json")
    }
    
    meta_path = job_dir / "job.json"
    with meta_path.open("w") as f:
        json.dump(job_meta, f, indent=2)
    
    # Queue background processing
    if background_tasks:
        background_tasks.add_task(process_video_job, job_id, input_path, job_dir)
    
    return {
        "jobId": job_id,
        "status": "queued",
        "message": f"Video uploaded successfully. Processing started.",
        "pollUrl": f"/upload/{job_id}"
    }

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """
    Get processing status for an upload job.
    
    Returns:
    - status: "queued" | "processing" | "completed" | "failed"
    - progress: 0-100 (percentage)
    - outputUrl: URL to download processed video (if completed)
    - metricsUrl: URL to download metrics JSON (if completed)
    - error: Error message (if failed)
    """
    
    job_dir = settings.artifact_dir / "videos" / job_id
    meta_path = job_dir / "job.json"
    
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    with meta_path.open("r") as f:
        job_meta = json.load(f)
    
    # Add download URLs if completed
    if job_meta["status"] == "completed":
        job_meta["outputUrl"] = f"/artifacts/videos/{job_id}/output.mp4"
        job_meta["metricsUrl"] = f"/artifacts/videos/{job_id}/metrics.json"
    
    return job_meta

async def process_video_job(job_id: str, input_path: Path, job_dir: Path):
    """
    Background task to process uploaded video.
    
    Steps:
    1. Update status to "processing"
    2. Run infer_sequence.py with core AI
    3. Generate overlay video
    4. Compute metrics
    5. Update status to "completed" or "failed"
    """
    
    meta_path = job_dir / "job.json"
    
    def update_status(status: str, progress: int = 0, error: str = None):
        with meta_path.open("r") as f:
            meta = json.load(f)
        meta["status"] = status
        meta["progress"] = progress
        if error:
            meta["error"] = error
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
    
    try:
        update_status("processing", 10)
        
        # Run inference using core AI
        output_path = job_dir / "output.mp4"
        metrics_path = job_dir / "metrics.json"
        
        # Build command to run infer_sequence.py
        cmd = [
            "python",
            str(settings.core_scripts_dir / "infer_sequence.py"),
            "--input", str(input_path),
            "--output", str(output_path),
            "--config", str(settings.core_config_path),
            "--save-metrics", str(metrics_path)
        ]
        
        update_status("processing", 30)
        
        # Execute inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.returncode != 0:
            raise Exception(f"Inference failed: {result.stderr}")
        
        update_status("processing", 80)
        
        # Verify outputs exist
        if not output_path.exists():
            raise Exception("Output video not generated")
        if not metrics_path.exists():
            raise Exception("Metrics not generated")
        
        # Get video duration for metadata
        cap = cv2.VideoCapture(str(output_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Load metrics
        with metrics_path.open("r") as f:
            metrics = json.load(f)
        
        # Update final status
        with meta_path.open("r") as f:
            meta = json.load(f)
        
        meta.update({
            "status": "completed",
            "progress": 100,
            "duration": duration,
            "fps": fps,
            "frameCount": frame_count,
            "metrics": metrics
        })
        
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
    
    except Exception as e:
        update_status("failed", 0, str(e))
