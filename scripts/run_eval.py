"""
Evaluation Script

Runs the full evaluation pipeline:
1. Runs inference on the test set (or full dataset)
2. Computes Frame-level metrics (Accuracy, F1, AUC)
3. Computes Event-level metrics (TP, FP, Delay)
4. Generates performance plots
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import get_config
from src.data.urfall_loader import get_train_val_test_split, load_all_sequences
from src.eval.metrics import Evaluator
from src.eval.plots import Plotter
from scripts.run_infer import process_sequence

def evaluate_set(name: str, 
                 sequences: list, 
                 config, 
                 output_dir: Path, 
                 generate_video: bool = False):
    """
    Evaluate a specific set (Train/Val/Test).
    """
    logger.info(f"Evaluating {name} set ({len(sequences)} sequences)...")
    set_dir = output_dir / name
    set_dir.mkdir(parents=True, exist_ok=True)
    
    results_list = []
    
    # 1. Run inference
    for seq in tqdm(sequences, desc=f"Infer {name}"):
        try:
            # We reuse process_sequence from run_infer script
            # It returns a DataFrame of frame-level results
            df = process_sequence(seq, config, set_dir, save_video=generate_video)
            if df is not None:
                results_list.append(df)
        except Exception as e:
            logger.error(f"Error processing {seq.sequence_id}: {e}")
            
    if not results_list:
        logger.warning(f"No results for {name} set")
        return
        
    all_results = pd.concat(results_list, ignore_index=True)
    csv_path = set_dir / f"{name}_results.csv"
    all_results.to_csv(csv_path, index=False)
    
    # 2. Compute Metrics
    evaluator = Evaluator(config.dataset.fps)
    
    # Frame-level
    cls_metrics = evaluator.classification_metrics(
        all_results['gt_label'].values,
        all_results['fall_score'].values,
        (all_results['pred_state'] == 'LYING').astype(int).values # Binary pred
    )
    
    # Event-level
    evt_metrics = evaluator.event_metrics(
        all_results, 
        [s.sequence_id for s in sequences]
    )
    
    # 3. Generate Plots (Save to centralized output/plots)
    config.paths.output_plots.mkdir(parents=True, exist_ok=True)
    plotter = Plotter(config.paths.output_plots)

    # Calculate Specificity
    tn, fp, fn, tp = cls_metrics.conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 2b. Plot Evaluation Metrics Bar Chart
    metrics_dict = {
        'Accuracy': cls_metrics.accuracy,
        'Precision': cls_metrics.precision,
        'Recall': cls_metrics.recall,
        'Specificity': specificity,
        'F1-Score': cls_metrics.f1
    }
    plotter.plot_evaluation_metrics(metrics_dict, title=f"Evaluation Metrics - {name}")

    # 3. Generate Plots
    # ROC/PR
    plotter.plot_roc_curve(all_results['gt_label'].values, 
                           all_results['fall_score'].values, 
                           cls_metrics.roc_auc, title=f"ROC - {name}")
    plotter.plot_pr_curve(all_results['gt_label'].values, 
                          all_results['fall_score'].values, 
                          cls_metrics.pr_auc, title=f"PR - {name}")
    plotter.plot_confusion_matrix(cls_metrics.conf_matrix)
    
    # Plot timelines for True Positives (Falls)
    fall_seqs = [s for s in sequences if s.sequence_type == 'fall']
    for seq in fall_seqs[:5]: # Plot first 5 falls
        seq_df = all_results[all_results['sequence'] == seq.sequence_id]
        if not seq_df.empty:
            plotter.plot_timeline(seq_df, seq.sequence_id)
            
    # 4. Log Summary
    summary = (
        f"\n--- {name} Results ---\n"
        f"Frame-level:\n"
        f"  Accuracy: {cls_metrics.accuracy:.4f}\n"
        f"  F1 Score: {cls_metrics.f1:.4f}\n"
        f"  ROC AUC:  {cls_metrics.roc_auc:.4f}\n"
        f"  Specs (Sp): {specificity:.4f}\n"
        f"  Conf Matrix: {cls_metrics.conf_matrix.tolist()} (TN, FP | FN, TP)\n"
        f"Event-level:\n"
        f"  Detected Falls (TP): {evt_metrics.tp_events}/{evt_metrics.tp_events + evt_metrics.fn_events}\n"
        f"  False Alarms (FP):   {evt_metrics.fp_events}\n"
        f"  Recall:              {evt_metrics.recall:.4f}\n"
        f"  Avg Delay:           {evt_metrics.avg_delay_seconds:.3f}s\n"
        f"  False Alarm Rate:    {evt_metrics.false_alarm_rate_per_hour:.2f} alarms/hour\n"
        f"  Total Duration:      {evt_metrics.total_test_duration_hours * 60:.1f} minutes\n"
    )
    logger.info(summary)
    
    # Save text report
    with open(set_dir / "report.txt", "w") as f:
        f.write(summary)
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Use Train/Val/Test split")
    parser.add_argument("--test-only", action="store_true", help="Run only on Test set")
    parser.add_argument("--full", action="store_true", help="Run on full dataset (no split)")
    parser.add_argument("--video", action="store_true", help="Generate videos")
    parser.add_argument("--output", default="output/eval", help="Output directory")
    args = parser.parse_args()
    
    config = get_config()
    output_dir = Path(args.output)
    
    if args.full:
        # Evaluate on everything as one block
        dataset = load_all_sequences(config)
        evaluate_set("FULL", dataset.all_sequences, config, output_dir, args.video)
        
    else:
        # Standard Split
        train, val, test = get_train_val_test_split(config, random_seed=42)
        
        if not args.test_only:
            # Maybe skip train evaluation to save time?
            # evaluate_set("train", train, config, output_dir, args.video)
            evaluate_set("val", val, config, output_dir, args.video)
            
        evaluate_set("test", test, config, output_dir, args.video)

if __name__ == "__main__":
    main()
