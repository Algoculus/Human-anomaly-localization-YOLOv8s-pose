"""Run full evaluation and save results."""
import sys
import os
from pathlib import Path

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("RUNNING FULL EVALUATION")
print("="*80)
print()

try:
    from scripts.eval_all import eval_all
    
    eval_all(
        root_dir="d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/raw/UR_Fall_Detection_Dataset/data",
        index_path="d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/urfd_index.csv",
        config_path="configs/default.yaml",
        no_videos=True
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    
    # Copy to runs/fixed
    from shutil import copy2
    Path("runs/fixed").mkdir(parents=True, exist_ok=True)
    copy2("outputs/predictions.csv", "runs/fixed/predictions.csv")
    copy2("outputs/metrics.json", "runs/fixed/metrics.json")
    copy2("outputs/eval_summary.json", "runs/fixed/eval_summary.json")
    
    print("Results saved to runs/fixed/")
    print()
    
    # Load and display metrics
    import json
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    
    print("FINAL METRICS:")
    print("-"*80)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print()
    cm = metrics['confusion_matrix']
    print(f"TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")
    print("-"*80)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
