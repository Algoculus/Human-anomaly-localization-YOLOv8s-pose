import pandas as pd
import json

df = pd.read_csv("runs/fixed/predictions.csv")

print("[ANALYSIS] False Positive Sequences (ADL predicted as FALL):")
fp_sequences = df[(df['gt_label'] == 0) & (df['pred_label'] == 1)]
print(fp_sequences[['seq_name', 'pred_score', 'first_confirm_frame']])
print(f"\nTotal FP: {len(fp_sequences)}")

print("\n[ANALYSIS] False Negative Sequences (FALL predicted as ADL):")
fn_sequences = df[(df['gt_label'] == 1) & (df['pred_label'] == 0)]
print(fn_sequences[['seq_name', 'pred_score', 'first_confirm_frame']])
print(f"\nTotal FN: {len(fn_sequences)}")

with open("runs/fixed/metrics.json") as f:
    metrics = json.load(f)
    
print(f"\n[METRICS] Confusion Matrix:")
print(f"  TP: {metrics['confusion_matrix']['TP']}")
print(f"  TN: {metrics['confusion_matrix']['TN']}")
print(f"  FP: {metrics['confusion_matrix']['FP']}")
print(f"  FN: {metrics['confusion_matrix']['FN']}")
print(f"\n  Accuracy: {metrics['accuracy']:.3f}")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall: {metrics['recall']:.3f}")
print(f"  F1: {metrics['f1_score']:.3f}")
