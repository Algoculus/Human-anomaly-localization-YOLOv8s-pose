"""Automated eval script that handles all I/O."""
import os
import sys
import json
from pathlib import Path
import pandas as pd

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_all import eval_all

# Run eval
print("Starting evaluation...")
try:
    eval_all(
        root_dir="d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/raw/UR_Fall_Detection_Dataset/data",
        index_path="d:/Workspace for Learning/My Projects/UTH Projects/Human-anomaly-localization-YOLOv8s-pose/data/urfd_index.csv",
        config_path="configs/default.yaml",
        no_videos=True
    )
    print("\nEvaluation completed successfully!")
    
    # Copy outputs to runs/fixed
    from shutil import copy2
    Path("runs/fixed").mkdir(parents=True, exist_ok=True)
    copy2("outputs/predictions.csv", "runs/fixed/predictions.csv")
    copy2("outputs/metrics.json", "runs/fixed/metrics.json")
    copy2("outputs/eval_summary.json", "runs/fixed/eval_summary.json")
    print("Results copied to runs/fixed/")
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
