import os
import sys
import argparse
import zipfile
from pathlib import Path
import pandas as pd

def find_frames_dir(seq_path):
    # Find the directory containing RGB frames for cam0
    seq_path = Path(seq_path)
    seq_name = seq_path.name
    
    candidates = [
        seq_path / seq_name,
        seq_path,
        seq_path / "cam0" / "rgb",
        seq_path / "cam0-rgb",
        seq_path / "rgb"
    ]
    
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            frames = sorted(cand.glob("*.png")) + sorted(cand.glob("*.jpg"))
            if len(frames) > 0:
                return cand, len(frames)
    
    for zip_file in seq_path.glob("*.zip"):
        zip_name = zip_file.name.lower()
        if "cam0" in zip_name and "rgb" in zip_name:
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(seq_path)
            
            for cand in candidates:
                if cand.exists() and cand.is_dir():
                    frames = sorted(cand.glob("*.png")) + sorted(cand.glob("*.jpg"))
                    if len(frames) > 0:
                        return cand, len(frames)
    
    return None, 0

def prepare_urfd(root_dir, output_index):
    # Prepare URFD dataset index
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist.")
        sys.exit(1)
    
    sequences = []
    seq_folders = []
    
    fall_dir = root_dir / "Fall_sequences"
    adl_dir = root_dir / "Activities_of _Daily_Living_sequences"
    
    if fall_dir.exists() and adl_dir.exists():
        print("Detected Layout 2: Fall_sequences and Activities_of_Daily_Living_sequences")
        seq_folders.extend([d for d in fall_dir.iterdir() if d.is_dir() and 'cam0-rgb' in d.name])
        seq_folders.extend([d for d in adl_dir.iterdir() if d.is_dir() and 'cam0-rgb' in d.name])
    else:
        print("Detected Layout 1: adl-XX and fall-XX folders directly in root")
        seq_folders = [d for d in root_dir.iterdir() if d.is_dir() and (d.name.startswith('adl-') or d.name.startswith('fall-'))]
    
    seq_folders = sorted(seq_folders)
    
    for seq_folder in seq_folders:
        seq_name = seq_folder.name
        base_name = seq_name.replace('-cam0-rgb', '')
        
        if base_name.startswith("fall-"):
            gt_label = 1
        elif base_name.startswith("adl-"):
            gt_label = 0
        else:
            print(f"Skipping {seq_name}: unknown prefix (expected fall-* or adl-*)")
            continue
        
        frame_dir, num_frames = find_frames_dir(seq_folder)
        
        if frame_dir is None or num_frames == 0:
            print(f"Warning: No frames found for {seq_name}, skipping.")
            continue
        
        sequences.append({
            "seq_name": base_name,
            "seq_path": str(seq_folder.absolute()),
            "gt_label": gt_label,
            "frame_dir": str(frame_dir.absolute()),
            "num_frames": num_frames
        })
        
        print(f"Found {base_name}: {num_frames} frames, label={gt_label}")
    
    df = pd.DataFrame(sequences)
    output_index = Path(output_index)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_index, index=False)
    
    print(f"\nPrepared {len(sequences)} sequences.")
    print(f"Index saved to {output_index}")
    print(f"  - ADL sequences: {(df['gt_label'] == 0).sum()}")
    print(f"  - Fall sequences: {(df['gt_label'] == 1).sum()}")

def main():
    parser = argparse.ArgumentParser(description="Prepare URFD dataset index")
    parser.add_argument("--root", type=str, required=True, help="Path to URFD raw data directory")
    parser.add_argument("--out_index", type=str, required=True, help="Path to output index CSV file")
    
    args = parser.parse_args()
    prepare_urfd(args.root, args.out_index)

if __name__ == "__main__":
    main()
