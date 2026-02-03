import numpy as np
import json
from pathlib import Path
from datetime import datetime

def compute_metrics(gt_labels, pred_labels):
    # Compute evaluation metrics: confusion matrix, accuracy, precision, recall, specificity, F1-score
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    
    # Compute confusion matrix components using boolean indexing
    TP = np.sum((gt_labels == 1) & (pred_labels == 1))  # True Positive
    TN = np.sum((gt_labels == 0) & (pred_labels == 0))  # True Negative
    FP = np.sum((gt_labels == 0) & (pred_labels == 1))  # False Positive
    FN = np.sum((gt_labels == 1) & (pred_labels == 0))  # False Negative
    
    total = len(gt_labels)
    
    # Compute metrics with division-by-zero protection
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "confusion_matrix": {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score)
    }
    
    return metrics

def save_metrics(metrics, output_path):
    # Save metrics to JSON file
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_confusion_matrix(metrics, output_path):
    # Plot confusion matrix as heatmap and save to PNG
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Fall', 'Predicted ADL'],
                yticklabels=['Actual Fall', 'Actual ADL'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_bars(metrics, output_path):
    # Plot metrics as bar chart (Accuracy, Precision, Recall, Specificity, F1-Score)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    metric_values = [
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['specificity'], metrics['f1_score']
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_evaluation_plots(metrics, output_dir):
    # Create all evaluation plots and return dict of plot paths
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate confusion matrix plot
    cm_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics, cm_path)
    
    # Generate metrics bar chart
    bars_path = plots_dir / "metrics_bars.png"
    plot_metrics_bars(metrics, bars_path)
    
    return {"confusion_matrix": str(cm_path), "metrics_bars": str(bars_path)}

def save_evaluation_summary(metrics, config, plot_paths, output_path):
    # Save comprehensive evaluation summary to JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "plots": plot_paths
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
