"""Create visualization plots for evaluation results."""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Load metrics
with open("runs/baseline/metrics.json") as f:
    baseline = json.load(f)
with open("runs/fixed/metrics.json") as f:
    fixed = json.load(f)

# Create output directory
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Confusion Matrix Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

cm_baseline = [[baseline['confusion_matrix']['TN'], baseline['confusion_matrix']['FP']],
               [baseline['confusion_matrix']['FN'], baseline['confusion_matrix']['TP']]]
cm_fixed = [[fixed['confusion_matrix']['TN'], fixed['confusion_matrix']['FP']],
            [fixed['confusion_matrix']['FN'], fixed['confusion_matrix']['TP']]]

# Baseline
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Pred ADL', 'Pred Fall'], 
            yticklabels=['True ADL', 'True Fall'],
            cbar_kws={'label': 'Count'})
ax1.set_title(f'Baseline\nAccuracy={baseline["accuracy"]:.3f}, Recall={baseline["recall"]:.3f}')
ax1.set_ylabel('Ground Truth')
ax1.set_xlabel('Prediction')

# Fixed
sns.heatmap(cm_fixed, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['Pred ADL', 'Pred Fall'],
            yticklabels=['True ADL', 'True Fall'],
            cbar_kws={'label': 'Count'})
ax2.set_title(f'Fixed (High Recall)\nAccuracy={fixed["accuracy"]:.3f}, Recall={fixed["recall"]:.3f}')
ax2.set_ylabel('Ground Truth')
ax2.set_xlabel('Prediction')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved {output_dir / 'confusion_matrix_comparison.png'}")
plt.close()

# 2. Metrics Comparison Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
baseline_vals = [baseline['accuracy'], baseline['precision'], baseline['recall'], 
                 baseline['specificity'], baseline['f1_score']]
fixed_vals = [fixed['accuracy'], fixed['precision'], fixed['recall'], 
              fixed['specificity'], fixed['f1_score']]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, fixed_vals, width, label='Fixed (High Recall)', color='lightgreen', edgecolor='black')

ax.set_ylabel('Score')
ax.set_title('Performance Metrics: Baseline vs Fixed')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0, 1.0)
ax.axhline(y=0.85, color='red', linestyle='--', linewidth=1, label='Recall Target (0.85)')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved {output_dir / 'metrics_comparison.png'}")
plt.close()

# 3. Precision-Recall Tradeoff
fig, ax = plt.subplots(figsize=(8, 8))

# Plot points
ax.scatter(baseline['recall'], baseline['precision'], s=200, c='blue', 
           marker='o', label='Baseline', edgecolors='black', linewidths=2)
ax.scatter(fixed['recall'], fixed['precision'], s=200, c='green',
           marker='s', label='Fixed (High Recall)', edgecolors='black', linewidths=2)

# Arrow showing transition
ax.annotate('', xy=(fixed['recall'], fixed['precision']), 
            xytext=(baseline['recall'], baseline['precision']),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Annotations
ax.text(baseline['recall'], baseline['precision'] + 0.02, 
        f'Baseline\nR={baseline["recall"]:.3f}\nP={baseline["precision"]:.3f}',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(fixed['recall'], fixed['precision'] - 0.05,
        f'Fixed\nR={fixed["recall"]:.3f}\nP={fixed["precision"]:.3f}',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Target lines
ax.axvline(x=0.85, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Recall Target')
ax.axhline(y=0.70, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Min Precision')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
ax.legend(loc='lower left')
ax.set_xlim(0.6, 1.0)
ax.set_ylim(0.6, 1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=150, bbox_inches='tight')
print(f"Saved {output_dir / 'precision_recall_tradeoff.png'}")
plt.close()

# 4. Confusion Matrix Delta
cm_delta = np.array([[fixed['confusion_matrix']['TN'] - baseline['confusion_matrix']['TN'],
                      fixed['confusion_matrix']['FP'] - baseline['confusion_matrix']['FP']],
                     [fixed['confusion_matrix']['FN'] - baseline['confusion_matrix']['FN'],
                      fixed['confusion_matrix']['TP'] - baseline['confusion_matrix']['TP']]])

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm_delta, annot=True, fmt='d', cmap='RdYlGn', center=0, ax=ax,
            xticklabels=['Pred ADL', 'Pred Fall'],
            yticklabels=['True ADL', 'True Fall'],
            cbar_kws={'label': 'Change (Fixed - Baseline)'})
ax.set_title('Change in Confusion Matrix (Fixed - Baseline)\nGreen = Improvement, Red = Degradation')
ax.set_ylabel('Ground Truth')
ax.set_xlabel('Prediction')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_delta.png', dpi=150, bbox_inches='tight')
print(f"Saved {output_dir / 'confusion_matrix_delta.png'}")
plt.close()

print("\n" + "="*80)
print("All plots saved to outputs/plots/")
print("="*80)
