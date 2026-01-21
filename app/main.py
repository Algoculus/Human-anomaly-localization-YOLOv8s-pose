"""
Fall Detection Demo API

FastAPI application to demonstrate the system.
Endpoints:
- /infer/{sequence_id}: Run inference on a sequence
- /results/{sequence_id}: Get JSON results
- /video/{sequence_id}: Get output video
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
from loguru import logger
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import get_config
from src.data.urfall_loader import load_sequence
from scripts.run_infer import process_sequence

app = FastAPI(title="Fall Detection System", version="1.0")
config = get_config()

# Output directory for demo
DEMO_OUTPUT = Path("output/demo")
DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)

# Mount static files for video playback
app.mount("/videos", StaticFiles(directory=str(DEMO_OUTPUT)), name="videos")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fall Detection API Ready"}

@app.post("/infer/{sequence_id}")
async def run_inference(sequence_id: str, background_tasks: BackgroundTasks):
    """
    Trigger inference for a sequence.
    """
    # Verify sequence exists
    seq = load_sequence(sequence_id)
    if not seq:
        raise HTTPException(status_code=404, detail="Sequence not found")
        
    # Run in background
    background_tasks.add_task(process_sequence_task, seq)
    
    return {"message": "Inference started", "sequence_id": sequence_id}

def process_sequence_task(seq):
    try:
        process_sequence(seq, config, DEMO_OUTPUT, save_video=True)
        logger.info(f"Demo inference complete for {seq.sequence_id}")
    except Exception as e:
        logger.error(f"Demo inference failed: {e}")

@app.get("/results/{sequence_id}")
def get_results(sequence_id: str):
    """
    Get generic results statistics.
    """
    csv_path = DEMO_OUTPUT / f"{sequence_id}_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Results not found. Run inference first.")
        
    df = pd.read_csv(csv_path)
    
    # Check for fall
    has_fall = (df['pred_state'] == 'LYING').any()
    max_score = df['fall_score'].max()
    
    return {
        "sequence_id": sequence_id,
        "frames_processed": len(df),
        "detected_fall": bool(has_fall),
        "max_risk_score": float(max_score)
    }

@app.get("/files/video/{sequence_id}")
def get_video(sequence_id: str):
    video_path = DEMO_OUTPUT / f"{sequence_id}_output.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
