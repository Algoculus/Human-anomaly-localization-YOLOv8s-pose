"""Create visualization plots for latest evaluation results only."""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

with open("runs/fixed/metrics.json") as f:
    metrics = json.load(f)

output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Confusion Matrix (Latest)
fig, ax = plt.subplots(figsize=(10, 8))

cm = [[metrics['confusion_matrix']['TN'], metrics['confusion_matrix']['FP']],
      [metrics['confusion_matrix']['FN'], metrics['confusion_matrix']['TP']]]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted ADL', 'Predicted FALL'],
            yticklabels=['Actual ADL', 'Actual FALL'],
            cbar_kws={'label': 'Count'}, annot_kws={'size': 16})

ax.set_title(f'Confusion Matrix\nAccuracy: {metrics["accuracy"]:.3f} | Precision: {metrics["precision"]:.3f} | Recall: {metrics["recall"]:.3f} | F1: {metrics["f1_score"]:.3f}',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
ax.set_xlabel('Prediction', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"[PLOT] Saved {output_dir / 'confusion_matrix.png'}")
plt.close()

# 2. Metrics Bar Chart (Latest)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_vals = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                metrics['specificity'], metrics['f1_score']]

colors = ['green' if v >= 0.85 else 'orange' if v >= 0.80 else 'red' for v in metrics_vals]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(metrics_names, metrics_vals, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target Threshold (≥0.85)')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11)

for bar, val in zip(bars, metrics_vals):
    height = bar.get_height()
    status = '✓' if val >= 0.85 else '✗'
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}\n{status}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'metrics_bar_chart.png', dpi=150, bbox_inches='tight')
print(f"[PLOT] Saved {output_dir / 'metrics_bar_chart.png'}")
plt.close()

print("\n" + "="*80)
print("[SUCCESS] All plots saved to outputs/plots/")
print("="*80)
