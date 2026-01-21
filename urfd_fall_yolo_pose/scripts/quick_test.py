"""Quick test eval on a few sequences."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_all import process_sequence
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.utils import load_config, set_seed

# Load config
config = load_config("configs/default.yaml")
set_seed(config["seed"])

# Load index
df_index = pd.read_csv("d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/urfd_index.csv")

# Test on first 5 sequences
df_test = df_index.head(5)

# Initialize detector
detector = YOLOPoseDetector(
    model_path=config["yolo_model"],
    imgsz=config["imgsz"],
    conf_thres=config["conf_thres"],
    iou_thres=config["iou_thres"],
    preprocess_lowlight=config.get("preprocess_lowlight", False),
    gamma=config.get("gamma", 1.3),
    clahe_clip=config.get("clahe_clip", 2.0),
    clahe_grid=config.get("clahe_grid", 8)
)

results = []
for idx, row in df_test.iterrows():
    print(f"Processing: {row['seq_name']}")
    result = process_sequence(row, detector, config, save_video=False)
    if result:
        results.append(result)
        print(f"  GT={result['gt_label']}, Pred={result['pred_label']}")

print(f"\nProcessed {len(results)} sequences successfully")
