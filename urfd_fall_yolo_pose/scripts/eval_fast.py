
import argparse
import sys
import multiprocessing
import pandas as pd
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urfd.utils import load_config, set_seed
from src.urfd.yolo_pose import YOLOPoseDetector
from scripts.eval_all import process_sequence, compute_metrics, save_metrics, create_evaluation_plots, save_evaluation_summary

def worker(args):
    """Worker function to process a chunk of sequences."""
    index_chunk, config_path, gpu_id = args
    
    # Load config (each worker loads its own)
    config = load_config(config_path)
    # Potentially set GPU device if needed, but for now assuming single GPU or CPU
    # If standard torch/yolo, it handles it. 
    
    # Init detector locally to avoid pickling issues
    detector = YOLOPoseDetector(
        model_path=config["yolo_model"],
        imgsz=config["imgsz"],
        conf_thres=config["conf_thres"],
        iou_thres=config["iou_thres"]
    )
    
    results = []
    for idx, row in index_chunk.iterrows():
        # print(f"  Worker processing {row['seq_name']}")
        res = process_sequence(row, detector, config, save_video=False)
        if res is not None:
            results.append(res)
            
    return results

def eval_fast(root_dir, index_path, config_path, num_workers=4):
    """Run parallel evaluation."""
    start_time = time.time()
    
    # Load inputs
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_index = pd.read_csv(index_path)
    print(f"Loaded {len(df_index)} sequences. Splitting into {num_workers} workers.")
    
    # Clean plots/results only (videos are separate now)
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        for p in plots_dir.glob("*.png"):
            p.unlink()
            
    # Split data
    chunks = np_split(df_index, num_workers)
    
    # Prepare args
    pool_args = [(chk, config_path, 0) for chk in chunks]
    
    print("Starting pool...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        all_results_lists = pool.map(worker, pool_args)
        
    # Merge results
    final_results = []
    for r_list in all_results_lists:
        final_results.extend(r_list)
        
    print(f"Processed {len(final_results)} sequences in {time.time() - start_time:.2f}s")
    
    # Save predictions
    df_results = pd.DataFrame(final_results)
    pred_path = output_dir / "predictions.csv"
    df_results.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    # Compute Metrics & Plots
    gt_labels = df_results["gt_label"].values
    pred_labels = df_results["pred_label"].values
    
    metrics = compute_metrics(gt_labels, pred_labels)
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    
    print("Creating plots...")
    plot_paths = create_evaluation_plots(metrics, output_dir)
    
    summary_path = output_dir / "eval_summary.json"
    save_evaluation_summary(metrics, config, plot_paths, summary_path)
    
    print("\n" + "="*50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print("="*50)

def np_split(df, n):
    """Split dataframe into n chunks."""
    import numpy as np
    return np.array_split(df, n)

if __name__ == "__main__":
    # Windows support
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    eval_fast(args.root, args.index, args.config, args.workers)
