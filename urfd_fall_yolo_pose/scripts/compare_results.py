"""Compare baseline vs fixed results."""
import json
import pandas as pd
from pathlib import Path

def compare_results():
    """Compare baseline and fixed evaluation results."""
    
    # Paths
    baseline_pred = Path("runs/baseline/predictions.csv")
    baseline_metrics = Path("runs/baseline/metrics.json")
    fixed_pred = Path("outputs/predictions.csv")  # or runs/fixed/predictions.csv
    fixed_metrics = Path("outputs/metrics.json")  # or runs/fixed/metrics.json
    
    # Check if files exist
    if not baseline_pred.exists():
        print("ERROR: Baseline predictions not found!")
        return
    
    if not fixed_pred.exists():
        print("ERROR: Fixed predictions not found!")
        print("Please run eval_all.py first")
        return
    
    # Load predictions
    df_baseline = pd.read_csv(baseline_pred)
    df_fixed = pd.read_csv(fixed_pred)
    
    # Load metrics
    with open(baseline_metrics) as f:
        metrics_baseline = json.load(f)
    
    with open(fixed_metrics) as f:
        metrics_fixed = json.load(f)
    
    # Print comparison
    print("="*80)
    print("EVALUATION COMPARISON: BASELINE vs FIXED")
    print("="*80)
    print()
    
    # Metrics comparison
    print("OVERALL METRICS:")
    print("-"*80)
    print(f"{'Metric':<20} {'Baseline':<15} {'Fixed':<15} {'Change':<15}")
    print("-"*80)
    
    for metric in ["accuracy", "precision", "recall", "specificity", "f1_score"]:
        base_val = metrics_baseline[metric]
        fixed_val = metrics_fixed[metric]
        change = fixed_val - base_val
        change_pct = (change / base_val * 100) if base_val > 0 else 0
        print(f"{metric.title():<20} {base_val:<15.4f} {fixed_val:<15.4f} {change:+.4f} ({change_pct:+.1f}%)")
    
    print()
    print("CONFUSION MATRIX:")
    print("-"*80)
    print(f"{'Component':<20} {'Baseline':<15} {'Fixed':<15} {'Change':<15}")
    print("-"*80)
    
    cm_baseline = metrics_baseline["confusion_matrix"]
    cm_fixed = metrics_fixed["confusion_matrix"]
    
    for key in ["TP", "TN", "FP", "FN"]:
        base_val = cm_baseline[key]
        fixed_val = cm_fixed[key]
        change = fixed_val - base_val
        print(f"{key:<20} {base_val:<15} {fixed_val:<15} {change:+d}")
    
    print()
    print("="*80)
    print("SEQUENCE-LEVEL CHANGES:")
    print("="*80)
    print()
    
    # Merge predictions
    df_compare = df_baseline.merge(df_fixed, on=["seq_name", "gt_label"], suffixes=("_baseline", "_fixed"))
    
    # Find changes
    df_changes = df_compare[df_compare["pred_label_baseline"] != df_compare["pred_label_fixed"]]
    
    if len(df_changes) == 0:
        print("No prediction changes between baseline and fixed.")
    else:
        print(f"Found {len(df_changes)} sequences with prediction changes:\n")
        
        # FP → TN (Fixed false positives)
        fp_fixed = df_changes[(df_changes["gt_label"] == 0) & 
                              (df_changes["pred_label_baseline"] == 1) & 
                              (df_changes["pred_label_fixed"] == 0)]
        if len(fp_fixed) > 0:
            print(f"FP → TN (False Positives FIXED): {len(fp_fixed)}")
            for _, row in fp_fixed.iterrows():
                print(f"  - {row['seq_name']}: Score {row['pred_score_baseline']:.3f} → {row['pred_score_fixed']:.3f}")
            print()
        
        # TN → FP (New false positives)
        new_fp = df_changes[(df_changes["gt_label"] == 0) & 
                            (df_changes["pred_label_baseline"] == 0) & 
                            (df_changes["pred_label_fixed"] == 1)]
        if len(new_fp) > 0:
            print(f"TN → FP (New False Positives): {len(new_fp)}")
            for _, row in new_fp.iterrows():
                print(f"  - {row['seq_name']}: Score {row['pred_score_baseline']:.3f} → {row['pred_score_fixed']:.3f}")
            print()
        
        # FN → TP (False negatives FIXED)
        fn_fixed = df_changes[(df_changes["gt_label"] == 1) & 
                              (df_changes["pred_label_baseline"] == 0) & 
                              (df_changes["pred_label_fixed"] == 1)]
        if len(fn_fixed) > 0:
            print(f"FN → TP (False Negatives FIXED): {len(fn_fixed)}")
            for _, row in fn_fixed.iterrows():
                print(f"  - {row['seq_name']}: Score {row['pred_score_baseline']:.3f} → {row['pred_score_fixed']:.3f}")
            print()
        
        # TP → FN (New false negatives)
        new_fn = df_changes[(df_changes["gt_label"] == 1) & 
                            (df_changes["pred_label_baseline"] == 1) & 
                            (df_changes["pred_label_fixed"] == 0)]
        if len(new_fn) > 0:
            print(f"TP → FN (New False Negatives): {len(new_fn)}")
            for _, row in new_fn.iterrows():
                print(f"  - {row['seq_name']}: Score {row['pred_score_baseline']:.3f} → {row['pred_score_fixed']:.3f}")
            print()
    
    print("="*80)
    print("PROBLEMATIC SEQUENCES STATUS:")
    print("="*80)
    print()
    
    # Check specific problematic sequences
    problem_seqs = ["adl-17", "adl-34", "adl-35", "fall-19", "fall-23", "adl-21"]
    
    for seq in problem_seqs:
        row = df_compare[df_compare["seq_name"] == seq]
        if len(row) == 0:
            continue
        
        row = row.iloc[0]
        gt = row["gt_label"]
        pred_base = row["pred_label_baseline"]
        pred_fixed = row["pred_label_fixed"]
        
        status_base = "✓" if gt == pred_base else "✗"
        status_fixed = "✓" if gt == pred_fixed else "✗"
        
        change_marker = "→" if pred_base != pred_fixed else " "
        
        print(f"{seq:<15} GT={gt}  Baseline: {pred_base} {status_base}  {change_marker}  Fixed: {pred_fixed} {status_fixed}")
    
    print()
    print("="*80)

if __name__ == "__main__":
    compare_results()
