import cv2
from pathlib import Path

def load_sequence_frames(seq_path):
    """Load RGB frames from a URFD sequence folder.
    
    Searches for frames in:
    1) seq_path/seq_name/ (nested folder with same name)
    2) seq_path/ (directly in the folder)
    3) seq_path/cam0/rgb/
    4) seq_path/cam0-rgb/
    5) seq_path/rgb/
    
    Args:
        seq_path: Path to sequence folder
    
    Returns:
        frames: List of BGR images (numpy arrays)
        frame_paths: List of frame file paths
    """
    seq_path = Path(seq_path)
    seq_name = seq_path.name
    
    # Check candidate directories (including nested structure)
    candidates = [
        seq_path / seq_name,  # Nested folder (URFD actual structure)
        seq_path,              # Direct folder
        seq_path / "cam0" / "rgb",
        seq_path / "cam0-rgb",
        seq_path / "rgb"
    ]
    
    frame_dir = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            frame_files = list(cand.glob("*.png")) + list(cand.glob("*.jpg"))
            if len(frame_files) > 0:
                frame_dir = cand
                break
    
    if frame_dir is None:
        return [], []
    
    # Get all frame files and sort by numeric order
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
