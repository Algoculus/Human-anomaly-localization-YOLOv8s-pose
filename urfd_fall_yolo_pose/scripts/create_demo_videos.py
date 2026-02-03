import argparse
import sys
from pathlib import Path
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.utils import load_config
from src.urfd.yolo_pose import YOLOPoseDetector
from scripts.eval_all import process_sequence

def create_demo_videos(root_dir, index_path, config_path, num_samples=5):
    # Generate demo videos for a subset of sequences
    config = load_config(config_path)
    df_index = pd.read_csv(index_path)
    
    adl_samples = df_index[df_index["gt_label"] == 0].sample(
        n=min(num_samples, len(df_index[df_index["gt_label"] == 0])), random_state=42)
    fall_samples = df_index[df_index["gt_label"] == 1].sample(
        n=min(num_samples, len(df_index[df_index["gt_label"] == 1])), random_state=42)
    
    samples = pd.concat([adl_samples, fall_samples])
    
    print(f"Generating demo videos for {len(samples)} sequences...")
    
    print("Initializing YOLOv8s-pose detector...")
    detector = YOLOPoseDetector(
        model_path=config["yolo_model"],
        imgsz=config["imgsz"],
        conf_thres=config["conf_thres"],
        iou_thres=config["iou_thres"]
    )
    
    for idx, row in samples.iterrows():
        print(f"Processing {row['seq_name']}...")
        process_sequence(row, detector, config, save_video=True)
        
    print(f"\nDemo videos saved to {Path(config['output_dir']) / 'videos'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per class")
    
    args = parser.parse_args()
    create_demo_videos(args.root, args.index, args.config, args.samples)
