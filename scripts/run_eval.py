"""
Evaluation Script - Fixed for Enhanced Metrics

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
            df = process_sequence(seq, config, set_dir, save_video=generate_video)
            if df is not None:
                results_list.append(df)
        except Exception as e:
            logger.error(f"Error processing {seq.sequence_id}: {e}")
            import traceback
            traceback.print_exc()
            
    if not results_list:
        logger.warning(f"No results for {name} set")
        return
        
    all_results = pd.concat(results_list, ignore_index=True)
    csv_path = set_dir / f"{name}_results.csv"
    all_results.to_csv(csv_path, index=False)
    
    # 2. Compute Metrics
    evaluator = Evaluator(config.dataset.fps)
    
    # Frame-level
    y_true = all_results['gt_label'].values
    y_score = all_results['fall_score'].values
    y_pred_binary = (all_results['pred_state'] == 'LYING').astype(int).values
    
    cls_metrics = evaluator.classification_metrics(y_true, y_score, y_pred_binary)
    
    # Event-level
    sequence_ids = [s.sequence_id for s in sequences]
    evt_metrics = evaluator.event_metrics(all_results, sequence_ids)
    
    # 3. Generate Plots
    config.paths.output_plots.mkdir(parents=True, exist_ok=True)
    plotter = Plotter(config.paths.output_plots)
    
    # Metrics bar chart
    metrics_dict = {
        'Accuracy': cls_metrics.accuracy,
        'Precision': cls_metrics.precision,
        'Recall': cls_metrics.recall,
        'Specificity': cls_metrics.specificity,
        'F1-Score': cls_metrics.f1
    }
    plotter.plot_evaluation_metrics(metrics_dict, title=f"Evaluation Metrics - {name}")
    
    # ROC/PR curves
    plotter.plot_roc_curve(y_true, y_score, cls_metrics.roc_auc, title=f"ROC - {name}")
    plotter.plot_pr_curve(y_true, y_score, cls_metrics.pr_auc, title=f"PR - {name}")
    plotter.plot_confusion_matrix(cls_metrics.conf_matrix)
    
    # Plot timelines for detected falls (first 5)
    fall_seqs = [s for s in sequences if s.sequence_type == 'fall']
    detected_falls = all_results[all_results['pred_state'] == 'LYING']['sequence'].unique()
    
    for seq in fall_seqs[:5]:
        if seq.sequence_id in detected_falls:
            seq_df = all_results[all_results['sequence'] == seq.sequence_id]
            if not seq_df.empty:
                plotter.plot_timeline(seq_df, seq.sequence_id)
    
    # 4. Generate and save report
    report = evaluator.generate_report(cls_metrics, evt_metrics, name)
    logger.info(report)
    
    with open(set_dir / "evaluation_report.txt", "w") as f:
        f.write(report)
    
    # 5. Additional analysis - Activity breakdown
    if 'activity_type' in all_results.columns:
        activity_stats = all_results.groupby('activity_type').agg({
            'fall_score': ['mean', 'std', 'max'],
            'pred_state': lambda x: (x == 'LYING').sum()
        }).round(3)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Activity Type Breakdown:")
        logger.info(f"{'='*60}")
        logger.info(f"\n{activity_stats}")
        
        activity_stats.to_csv(set_dir / "activity_breakdown.csv")
    
    # 6. Save detailed metrics as JSON
    import json
    
    metrics_summary = {
        'frame_level': {
            'accuracy': float(cls_metrics.accuracy),
            'precision': float(cls_metrics.precision),
            'recall': float(cls_metrics.recall),
            'specificity': float(cls_metrics.specificity),
            'f1': float(cls_metrics.f1),
            'balanced_accuracy': float(cls_metrics.balanced_accuracy),
            'roc_auc': float(cls_metrics.roc_auc),
            'pr_auc': float(cls_metrics.pr_auc),
            'fpr': float(cls_metrics.false_positive_rate),
            'fnr': float(cls_metrics.false_negative_rate)
        },
        'event_level': {
            'tp': int(evt_metrics.tp_events),
            'fp': int(evt_metrics.fp_events),
            'fn': int(evt_metrics.fn_events),
            'tn': int(evt_metrics.tn_events),
            'precision': float(evt_metrics.precision),
            'recall': float(evt_metrics.recall),
            'specificity': float(evt_metrics.specificity),
            'f1': float(evt_metrics.f1),
            'avg_delay_sec': float(evt_metrics.avg_detection_delay_seconds),
            'median_delay_sec': float(evt_metrics.median_detection_delay_seconds),
            'max_delay_sec': float(evt_metrics.max_detection_delay_seconds),
            'false_alarm_rate': float(evt_metrics.false_alarm_rate_per_hour),
            'test_duration_hours': float(evt_metrics.total_test_duration_hours)
        }
    }
    
    if evt_metrics.activity_breakdown:
        metrics_summary['activity_breakdown'] = evt_metrics.activity_breakdown
    
    with open(set_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"\nDetailed metrics saved to: {set_dir / 'metrics_summary.json'}")
    
    return cls_metrics, evt_metrics

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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.full:
        # Evaluate on everything as one block
        dataset = load_all_sequences(config)
        evaluate_set("FULL", dataset.all_sequences, config, output_dir, args.video)
        
    else:
        # Standard Split
        train, val, test = get_train_val_test_split(config, random_seed=42)
        
        if not args.test_only:
            logger.info("Evaluating validation set...")
            evaluate_set("val", val, config, output_dir, args.video)
        
        logger.info("Evaluating test set...")
        test_metrics = evaluate_set("test", test, config, output_dir, args.video)
        
        if test_metrics:
            cls_metrics, evt_metrics = test_metrics
            
            # Print summary to console
            logger.info(f"\n{'='*70}")
            logger.info(f"FINAL TEST SET PERFORMANCE")
            logger.info(f"{'='*70}")
            logger.info(f"Frame-level Accuracy: {cls_metrics.accuracy:.1%}")
            logger.info(f"Event-level Recall:   {evt_metrics.recall:.1%}")
            logger.info(f"Event-level Precision: {evt_metrics.precision:.1%}")
            logger.info(f"Detection Delay:      {evt_metrics.avg_detection_delay_seconds:.2f}s")
            logger.info(f"False Alarms/hour:    {evt_metrics.false_alarm_rate_per_hour:.2f}")
            logger.info(f"{'='*70}")

if __name__ == "__main__":
    main()