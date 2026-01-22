"""Final workflow: Generate plots and display summary after evaluation."""
import subprocess
import sys
from pathlib import Path
import json

print("="*80)
print("POST-EVALUATION WORKFLOW")
print("="*80)

# Check if metrics exist
metrics_path = Path("runs/fixed/metrics.json")
if not metrics_path.exists():
    print("\n[ERROR] Metrics file not found!")
    print("Please run: python scripts/run_full_eval.py")
    sys.exit(1)

# Load metrics
with open(metrics_path) as f:
    metrics = json.load(f)

print("\n[STEP 1] Generate Plots")
print("-" * 80)
try:
    subprocess.run([sys.executable, "scripts/create_plots.py"], check=True)
    print("[SUCCESS] Plots generated in outputs/plots/")
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Plot generation failed: {e}")
    sys.exit(1)

print("\n[STEP 2] Display Summary")
print("-" * 80)
try:
    subprocess.run([sys.executable, "scripts/final_summary.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Summary generation failed: {e}")

print("\n[STEP 3] Analysis")
print("-" * 80)
print(f"\nMetrics Summary:")
print(f"  Accuracy:    {metrics['accuracy']:.3f} {'✅' if metrics['accuracy'] >= 0.85 else '❌'}")
print(f"  Precision:   {metrics['precision']:.3f} {'✅' if metrics['precision'] >= 0.85 else '❌'}")
print(f"  Recall:      {metrics['recall']:.3f} {'✅' if metrics['recall'] >= 0.85 else '❌'}")
print(f"  Specificity: {metrics['specificity']:.3f} {'✅' if metrics['specificity'] >= 0.85 else '❌'}")
print(f"  F1-Score:    {metrics['f1_score']:.3f} {'✅' if metrics['f1_score'] >= 0.85 else '❌'}")

cm = metrics['confusion_matrix']
print(f"\nConfusion Matrix:")
print(f"  TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")

# Check if all metrics meet target
all_pass = all([
    metrics['accuracy'] >= 0.85,
    metrics['precision'] >= 0.85,
    metrics['recall'] >= 0.85,
    metrics['specificity'] >= 0.85,
    metrics['f1_score'] >= 0.85
])

print("\n" + "="*80)
if all_pass:
    print("🎉 ALL METRICS >= 0.85 TARGET MET!")
    print("System ready for deployment.")
else:
    print("⚠️ Some metrics below 0.85 target")
    
    # Provide recommendations
    if metrics['recall'] < 0.85:
        print("\n[RECOMMENDATION] Recall too low:")
        print("  - Decrease dy_peak_thres (more sensitive to motion)")
        print("  - Increase confirm_angle_thres (more permissive)")
        print("  - Decrease min_confirm_duration_frames (quicker detection)")
    
    if metrics['precision'] < 0.85:
        print("\n[RECOMMENDATION] Precision too low:")
        print("  - Increase dy_peak_thres (less sensitive)")
        print("  - Decrease confirm_angle_thres (stricter)")
        print("  - Increase min_confirm_duration_frames (sustained detection)")
    
    if metrics['f1_score'] < 0.85:
        print("\n[RECOMMENDATION] F1-Score too low:")
        print("  - Balance recall and precision")
        print("  - Consider using auto_tune.py for optimization")

print("="*80)

print("\n[OUTPUT_FILES]")
print("  📊 Confusion Matrix:  outputs/plots/confusion_matrix.png")
print("  📊 Metrics Chart:     outputs/plots/metrics_bar_chart.png")
print("  📄 Metrics JSON:      runs/fixed/metrics.json")
print("  📄 Predictions CSV:   runs/fixed/predictions.csv")
print("  📖 Project README:    ../README.md")
