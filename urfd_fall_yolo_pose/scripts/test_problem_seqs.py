"""Quick test on problematic sequences only."""
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

# Test only problematic sequences
problem_seqs = ["adl-17", "adl-34", "adl-35", "fall-19", "fall-23", "adl-21"]
df_test = df_index[df_index["seq_name"].isin(problem_seqs)]

print(f"Testing {len(df_test)} problematic sequences: {problem_seqs}")
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
    print(f"Processing: {row['seq_name']} (GT={row['gt_label']})")
    result = process_sequence(row, detector, config, save_video=False)
    if result:
        results.append(result)
        correct = "✓" if result['gt_label'] == result['pred_label'] else "✗"
        print(f"  Pred={result['pred_label']}, Score={result['pred_score']:.3f} {correct}")
        if result['first_confirm_frame'] >= 0:
            print(f"  First confirm at frame {result['first_confirm_frame']}")
    print()

print("="*60)
print("SUMMARY:")
print("="*60)
for r in results:
    gt = r['gt_label']
    pred = r['pred_label']
    status = "CORRECT" if gt == pred else "WRONG"
    
    if gt == 0 and pred == 1:
        error_type = "(FP)"
    elif gt == 1 and pred == 0:
        error_type = "(FN)"
    else:
        error_type = ""
    
    print(f"{r['seq_name']:<12} GT={gt} Pred={pred}  {status:<10} {error_type}")

# Count errors
fp_count = sum(1 for r in results if r['gt_label'] == 0 and r['pred_label'] == 1)
fn_count = sum(1 for r in results if r['gt_label'] == 1 and r['pred_label'] == 0)
correct_count = sum(1 for r in results if r['gt_label'] == r['pred_label'])

print()
print(f"Correct: {correct_count}/{len(results)}")
print(f"FP (ADL→Fall): {fp_count}")
print(f"FN (Fall→ADL): {fn_count}")
