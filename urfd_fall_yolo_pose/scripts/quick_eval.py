"""Quick partial eval on first N sequences."""
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_all import process_sequence
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.utils import load_config, set_seed
from src.urfd.eval import compute_metrics

# Config
N = 70  # All sequences
config = load_config("configs/default.yaml")
set_seed(config["seed"])

# Load index
df_index = pd.read_csv("d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/urfd_index.csv")
df_test = df_index.head(N)

print(f"Evaluating {len(df_test)} sequences...")
print()

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
    print(f"Processing {idx+1}/{len(df_test)}: {row['seq_name']}")
    result = process_sequence(row, detector, config, save_video=False)
    if result:
        results.append(result)

print(f"\nProcessed {len(results)} sequences")

# Save predictions
df_results = pd.DataFrame(results)
Path("outputs").mkdir(exist_ok=True)
df_results.to_csv("outputs/predictions.csv", index=False)
print("Saved predictions.csv")

# Compute metrics
gt_labels = df_results["gt_label"].values
pred_labels = df_results["pred_label"].values
metrics = compute_metrics(gt_labels, pred_labels)

# Save metrics
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics.json")

# Display
print("\n" + "="*80)
print("METRICS:")
print("="*80)
cm = metrics["confusion_matrix"]
print(f"TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f} {'<-- TARGET >= 0.85' if metrics['recall'] < 0.85 else '<-- TARGET MET!'}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"F1-Score:    {metrics['f1_score']:.4f}")
print("="*80)
