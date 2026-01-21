"""
Plotting Module

Generates visualization plots for evaluation analysis:
1. ROC and PR Curves
2. Confusion Matrix Heatmap
3. Timeline analysis of Fall Score vs Ground Truth
"""

from pathlib import Path
from typing import Optional, List
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

class Plotter:
    """
    Generates and saves evaluation plots.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set style
        sns.set_theme(style="whitegrid")
        
    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                       auc_score: float, title: str = "ROC Curve"):
        """Plot Receiver Operating Characteristic curve."""
        # Clean data (remove transitions 0)
        valid = y_true != 0
        y_true = (y_true[valid] == 1).astype(int)
        y_score = y_score[valid]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        save_path = self.output_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def plot_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                      ap_score: float, title: str = "Precision-Recall Curve"):
        """Plot Precision-Recall curve."""
        valid = y_true != 0
        y_true = (y_true[valid] == 1).astype(int)
        y_score = y_score[valid]
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        
        save_path = self.output_dir / "pr_curve.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def plot_confusion_matrix(self, cm: np.ndarray, 
                              class_names: List[str] = ['Normal', 'Fall']):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix (Frame-level)')
        
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def plot_timeline(self, result_df: pd.DataFrame, sequence_id: str):
        """
        Plot timeline of Fall Score, State, and GT for a detected sequence.
        """
        plt.figure(figsize=(12, 6))
        
        # Time axis
        t = result_df['time_ms'] / 1000.0
        
        # 1. Fall Score
        plt.plot(t, result_df['fall_score'], label='Fall Risk Score', 
                 color='red', linewidth=2)
        
        # 2. GT Label (scaled to display)
        # Map -1 to 0, 1 to 1.1 (slightly offset)
        gt = result_df['gt_label'].copy()
        gt_disp = gt.map({-1: 0.0, 1: 1.05, 0: -0.1})
        plt.plot(t, gt_disp, label='Ground Truth (Lying)', 
                 color='green', linestyle='--', alpha=0.7)
        
        # 3. Predicted State (colored regions)
        # We can't plot state string easily, but we can highlight regions
        # falling_mask = result_df['pred_state'] == 'FALLING'
        # lying_mask = result_df['pred_state'] == 'LYING'
        
        # Threshold line
        plt.axhline(y=0.7, color='grey', linestyle=':', label='Alert Threshold')
        
        plt.ylim([-0.2, 1.2])
        plt.xlabel('Time (s)')
        plt.ylabel('Score / Label')
        plt.title(f'Detection Timeline - {sequence_id}')
        plt.legend(loc='upper left')
        
        save_path = self.output_dir / f"timeline_{sequence_id}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_evaluation_metrics(self, metrics: dict, title: str = "Evaluation Metrics"):
        """
        Plot evaluation metrics as a bar chart.
        """
        plt.figure(figsize=(10, 6))
        
        # Filter metrics for plotting
        keys = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
        values = [metrics.get(k, 0.0) for k in keys]
        colors = ['#5cafe6', '#5ce691', '#eb7063', '#f5b542', '#a87ccf']
        
        bars = plt.bar(keys, values, color=colors, edgecolor='black', alpha=0.9)
        
        plt.ylim([0.0, 1.1])
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        save_path = self.output_dir / "evaluation_metrics.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    # Test
    plotter = Plotter(Path("output/test_plots"))
    y_true = np.array([-1]*50 + [1]*50)
    y_score = np.linspace(0, 1, 100)
    plotter.plot_roc_curve(y_true, y_score, 0.95)
    
    metrics = {
        'Accuracy': 0.814, 'Precision': 0.730, 'Recall': 0.900, 
        'Specificity': 0.750, 'F1-Score': 0.806
    }
    plotter.plot_evaluation_metrics(metrics)
    print("Test plots saved.")
