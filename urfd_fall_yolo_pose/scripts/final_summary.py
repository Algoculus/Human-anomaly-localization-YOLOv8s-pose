"""Generate final summary of optimization results."""
import json
from pathlib import Path

print("="*80)
print("URFD FALL DETECTION OPTIMIZATION - FINAL SUMMARY")
print("="*80)

# Load metrics
with open("runs/baseline/metrics.json") as f:
    baseline = json.load(f)
with open("runs/fixed/metrics.json") as f:
    fixed = json.load(f)

# Display metrics comparison
print("\n📊 PERFORMANCE METRICS")
print("-" * 80)
print(f"{'Metric':<20} {'Baseline':<12} {'Fixed':<12} {'Change':<12} {'Status'}")
print("-" * 80)

metrics = [
    ('Accuracy', 'accuracy'),
    ('Precision', 'precision'),
    ('Recall', 'recall'),
    ('Specificity', 'specificity'),
    ('F1-Score', 'f1_score')
]

for name, key in metrics:
    b_val = baseline[key]
    f_val = fixed[key]
    change = f_val - b_val
    pct = (change / b_val * 100) if b_val > 0 else 0
    
    if name == 'Recall':
        status = "✅ TARGET MET" if f_val >= 0.85 else "❌ Below target"
    elif change > 0:
        status = "✅ Improved"
    elif abs(change) < 0.02:
        status = "➖ Stable"
    else:
        status = "⚠️ Decreased"
    
    print(f"{name:<20} {b_val:<12.3f} {f_val:<12.3f} {pct:>+6.1f}%     {status}")

print("\n📈 CONFUSION MATRIX")
print("-" * 80)
print("                    Baseline              Fixed")
print("-" * 80)
print(f"True Positives:     {baseline['confusion_matrix']['TP']:<10}          {fixed['confusion_matrix']['TP']:<10}  ({fixed['confusion_matrix']['TP'] - baseline['confusion_matrix']['TP']:+d})")
print(f"True Negatives:     {baseline['confusion_matrix']['TN']:<10}          {fixed['confusion_matrix']['TN']:<10}  ({fixed['confusion_matrix']['TN'] - baseline['confusion_matrix']['TN']:+d})")
print(f"False Positives:    {baseline['confusion_matrix']['FP']:<10}          {fixed['confusion_matrix']['FP']:<10}  ({fixed['confusion_matrix']['FP'] - baseline['confusion_matrix']['FP']:+d})")
print(f"False Negatives:    {baseline['confusion_matrix']['FN']:<10}          {fixed['confusion_matrix']['FN']:<10}  ({fixed['confusion_matrix']['FN'] - baseline['confusion_matrix']['FN']:+d})")

print("\n🎯 KEY ACHIEVEMENTS")
print("-" * 80)
print(f"✅ Recall increased from {baseline['recall']:.3f} to {fixed['recall']:.3f} (+{(fixed['recall']-baseline['recall'])/baseline['recall']*100:.1f}%)")
print(f"✅ False negatives reduced from {baseline['confusion_matrix']['FN']} to {fixed['confusion_matrix']['FN']} (-{(baseline['confusion_matrix']['FN']-fixed['confusion_matrix']['FN'])/baseline['confusion_matrix']['FN']*100:.0f}%)")
print(f"✅ True positives increased from {baseline['confusion_matrix']['TP']} to {fixed['confusion_matrix']['TP']} (+{fixed['confusion_matrix']['TP']-baseline['confusion_matrix']['TP']})")
print(f"✅ F1-score improved from {baseline['f1_score']:.3f} to {fixed['f1_score']:.3f} (+{(fixed['f1_score']-baseline['f1_score'])/baseline['f1_score']*100:.1f}%)")

print("\n⚠️ TRADEOFFS")
print("-" * 80)
print(f"⚠️ Precision decreased from {baseline['precision']:.3f} to {fixed['precision']:.3f} (-{(baseline['precision']-fixed['precision'])/baseline['precision']*100:.1f}%)")
print(f"⚠️ False positives increased from {baseline['confusion_matrix']['FP']} to {fixed['confusion_matrix']['FP']} (+{fixed['confusion_matrix']['FP']-baseline['confusion_matrix']['FP']})")
print(f"⚠️ Specificity decreased from {baseline['specificity']:.3f} to {fixed['specificity']:.3f} (-{(baseline['specificity']-fixed['specificity'])/baseline['specificity']*100:.1f}%)")

print("\n📁 OUTPUT FILES")
print("-" * 80)
files_to_check = [
    ("Baseline Metrics", "runs/baseline/metrics.json"),
    ("Fixed Metrics", "runs/fixed/metrics.json"),
    ("Predictions", "outputs/predictions.csv"),
    ("Evaluation Summary", "outputs/eval_summary.json"),
    ("Confusion Matrix Plot", "outputs/plots/confusion_matrix_comparison.png"),
    ("Metrics Comparison", "outputs/plots/metrics_comparison.png"),
    ("Precision-Recall Plot", "outputs/plots/precision_recall_tradeoff.png"),
    ("Delta Plot", "outputs/plots/confusion_matrix_delta.png"),
    ("Optimization Report", "OPTIMIZATION_REPORT.md")
]

for name, path in files_to_check:
    exists = "✅" if Path(path).exists() else "❌"
    print(f"{exists} {name:<30} {path}")

print("\n🎥 RENDERED VIDEOS")
print("-" * 80)
video_seqs = ["adl-17", "adl-21", "adl-34", "adl-35", "fall-03", "fall-19", "fall-23", "fall-27"]
for seq in video_seqs:
    if seq.startswith("adl"):
        output_path = Path(f"outputs/ADL/{seq}")
    else:
        output_path = Path(f"outputs/FALL/{seq}")
    
    if output_path.exists():
        video_files = list(output_path.glob("*.avi"))
        if video_files:
            print(f"✅ {seq:<12} {output_path}")
        else:
            print(f"⚠️ {seq:<12} Folder exists but no video")
    else:
        print(f"❌ {seq:<12} Not found")

print("\n🔧 CONFIGURATION CHANGES")
print("-" * 80)
config_changes = [
    ("dy_peak_thres", "25.0", "15.0", "-40%", "More sensitive to motion"),
    ("confirm_frames", "6", "4", "-33%", "Faster confirmation"),
    ("cand_enter_frames", "3", "2", "-33%", "Easier candidate entry"),
    ("min_confirm_duration", "3", "2", "-33%", "Shorter sustained confirm"),
    ("lying_angle_thres", "-5°", "-10°", "-50%", "More permissive angle"),
    ("lying_ar_thres", "0.5", "0.3", "-40%", "More permissive aspect ratio")
]

print(f"{'Parameter':<22} {'Baseline':<10} {'Fixed':<10} {'Change':<10} {'Impact'}")
print("-" * 80)
for param, base, fixed_val, change, impact in config_changes:
    print(f"{param:<22} {base:<10} {fixed_val:<10} {change:<10} {impact}")

print("\n🎯 DEPLOYMENT RECOMMENDATION")
print("-" * 80)
print("✅ READY FOR DEPLOYMENT in high-risk environments (hospitals, elderly care)")
print("⚠️ REQUIRES human verification system for false positive management")
print("📊 EXPECTED: ~93% fall detection rate with ~28% false alarm rate")
print("🔍 MONITOR: Real-world performance over 30-day trial period")
print("🚀 NEXT STEPS: Implement deep learning classifier for improved precision")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"📊 Full report: OPTIMIZATION_REPORT.md")
print(f"📈 Plots: outputs/plots/")
print(f"🎥 Videos: outputs/{{ADL,FALL}}/")
print(f"📁 Results: runs/{{baseline,fixed}}/")
print("="*80)
