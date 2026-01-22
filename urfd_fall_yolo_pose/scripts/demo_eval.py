import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_all import process_sequence
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.utils import load_config, set_seed
from src.urfd.eval import compute_metrics

N = 10
config = load_config("configs/default.yaml")
set_seed(config["seed"])

df_index = pd.read_csv("d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/urfd_index.csv")
df_test = df_index.head(N)

print(f"[DEMO_EVAL] Evaluating {len(df_test)} sequences (quick demo)...")
print(f"[CONFIG] dy_peak_thres: {config['dy_peak_thres']}")
print(f"[CONFIG] confirm_angle_thres: {config['confirm_angle_thres']}")
print(f"[CONFIG] confirm_ar_thres: {config['confirm_ar_thres']}")
print(f"[CONFIG] min_confirm_duration_frames: {config['min_confirm_duration_frames']}")
print()

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
    print(f"[PROCESSING] {idx+1}/{len(df_test)}: {row['seq_name']}")
    result = process_sequence(row, detector, config, save_video=True)
    if result:
        results.append(result)

print(f"\n[RESULTS] Processed {len(results)} sequences")

df_results = pd.DataFrame(results)
Path("outputs").mkdir(exist_ok=True)
df_results.to_csv("outputs/predictions_demo.csv", index=False)

metrics = compute_metrics(
    y_true=df_results["label"].map({"FALL": 1, "ADL": 0}).tolist(),
    y_pred=df_results["pred"].map({"FALL": 1, "ADL": 0}).tolist()
)

with open("outputs/metrics_demo.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[METRICS] Accuracy: {metrics['accuracy']:.3f}")
print(f"[METRICS] Precision: {metrics['precision']:.3f}")
print(f"[METRICS] Recall: {metrics['recall']:.3f}")
print(f"[METRICS] F1: {metrics['f1']:.3f}")
print(f"\n[CONFUSION_MATRIX]")
print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}")
print(f"  FP: {metrics['fp']}, FN: {metrics['fn']}")
print(f"\n[OUTPUT] Saved to outputs/predictions_demo.csv and outputs/metrics_demo.json")
