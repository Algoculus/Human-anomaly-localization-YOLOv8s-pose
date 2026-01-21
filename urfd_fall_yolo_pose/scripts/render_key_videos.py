"""Render videos for key sequences showing important changes."""
import subprocess
from pathlib import Path
import json

# Load comparison results
with open("runs/fixed/predictions.csv") as f:
    lines = f.readlines()[1:]  # Skip header

# Define key sequences to render
key_sequences = {
    # Fixed FP → TN
    "adl-17": "Fixed: Was false positive, now correctly classified as ADL",
    
    # Fixed FN → TP  
    "fall-23": "Fixed: Was false negative, now correctly detected as Fall",
    "fall-03": "Fixed: Was false negative, now correctly detected as Fall",
    "fall-27": "Fixed: Was false negative, now correctly detected as Fall",
    
    # Remaining FN (still problematic)
    "fall-19": "Still FN: Detection failure or insufficient motion",
    
    # Regressed TN → FP
    "adl-21": "Regressed: Was correct, now false positive",
    
    # Remaining FP
    "adl-34": "Still FP: Complex ADL motion triggering fall detection",
    "adl-35": "Still FP: Complex ADL motion triggering fall detection"
}

# Base command
base_cmd = 'python scripts/infer_sequence.py --config configs/default.yaml --seq'

print("Rendering videos for key sequences...")
print("="*80)

for seq_name, description in key_sequences.items():
    # Find video path
    if seq_name.startswith("adl"):
        video_path = f"../../data/raw/UR_Fall_Detection_Dataset/data/{seq_name.replace('adl-', 'adl-')}-cam0-rgb.avi"
    else:
        video_path = f"../../data/raw/UR_Fall_Detection_Dataset/data/{seq_name.replace('fall-', 'fall-')}-cam0-rgb.avi"
    
    print(f"\n[{seq_name}] {description}")
    print(f"Command: {base_cmd} {video_path}")
    
    try:
        result = subprocess.run(
            f'{base_cmd} "{video_path}"',
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        print(f"✓ Success: Video saved to outputs/{seq_name}/")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr[:200]}")
        continue

print("\n" + "="*80)
print("Video rendering complete!")
print("Check outputs/{ADL,FALL}/ for rendered videos")
print("="*80)
