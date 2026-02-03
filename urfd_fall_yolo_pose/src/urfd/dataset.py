import cv2
from pathlib import Path

def load_sequence_frames(seq_path):
    # Load RGB frames from a URFD sequence folder, returns (frames list, frame_paths list)
    seq_path = Path(seq_path)
    seq_name = seq_path.name
    
    # Priority list of candidate directories
    candidates = [
        seq_path / seq_name,      # Nested folder with same name
        seq_path,                  # Current folder
        seq_path / "cam0" / "rgb", # Standard URFD structure
        seq_path / "cam0-rgb",     # Alternative naming
        seq_path / "rgb"           # Simple rgb folder
    ]
    
    # Find first valid frame directory
    frame_dir = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            frame_files = list(cand.glob("*.png")) + list(cand.glob("*.jpg"))
            if len(frame_files) > 0:
                frame_dir = cand
                break
    
    if frame_dir is None:
        return [], []
    
    # Sort frames numerically (not alphabetically)
    # This ensures 1, 2, 10 instead of 1, 10, 2
    frame_files = sorted(
        list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg")),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )
    
    # Load frames
    frames = []
    frame_paths = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
            frame_paths.append(str(fp))
    
    return frames, frame_paths
