
import argparse
import sys
import pandas as pd
import json
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_all import process_sequence, compute_metrics, create_evaluation_plots, save_evaluation_summary, save_metrics
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.utils import load_config, set_seed

def eval_subset(root_dir, index_path, config_path, sample_size=20):
    """Evaluate on a subset of sequences."""
    print(f"Starting subset evaluation (N={sample_size})...")
    
    # Load config
    config = load_config(config_path)
    # Ensure determinstic behavior
    set_seed(42)  
    
    # Load index
    df_index = pd.read_csv(index_path)
    
    # Stratified sample
    adl_df = df_index[df_index["gt_label"] == 0]
    fall_df = df_index[df_index["gt_label"] == 1]
    
    n_adl = min(len(adl_df), sample_size // 2)
    n_fall = min(len(fall_df), sample_size - n_adl)
    
    # Sample
    adl_sample = adl_df.sample(n=n_adl, random_state=42)
    fall_sample = fall_df.sample(n=n_fall, random_state=42)
    
    df_subset = pd.concat([adl_sample, fall_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Selected {len(df_subset)} sequences ({len(adl_sample)} ADL, {len(fall_sample)} Fall)")
    
    # Init detector
    detector = YOLOPoseDetector(
        model_path=config["yolo_model"],
        imgsz=config["imgsz"],
        conf_thres=config["conf_thres"],
        iou_thres=config["iou_thres"]
    )
    
    results = []
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in df_subset.iterrows():
        print(f"Processing {idx+1}/{len(df_subset)}: {row['seq_name']}")
        res = process_sequence(row, detector, config, save_video=False)
        if res:
            results.append(res)
            print(f"  GT={res['gt_label']}, Pred={res['pred_label']}, Score={res['pred_score']:.3f}")

    # Save outputs
    df_results = pd.DataFrame(results)
    pred_path = output_dir / "predictions.csv"
    df_results.to_csv(pred_path, index=False)
    
    # Metrics
    gt_labels = df_results["gt_label"].values
    pred_labels = df_results["pred_label"].values
    metrics = compute_metrics(gt_labels, pred_labels)
    
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    
    # Plots
    print("Creating plots...")
    plot_paths = create_evaluation_plots(metrics, output_dir)
    
    # Summary
    save_evaluation_summary(metrics, config, plot_paths, output_dir / "eval_summary.json")
    
    print("\n" + "="*50)
    print("PARTIAL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--size", type=int, default=20)
    
    args = parser.parse_args()
    eval_subset(args.root, args.index, args.config, args.size)
