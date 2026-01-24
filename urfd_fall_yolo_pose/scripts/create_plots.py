import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def create_plots(output_dir):
    """
    Create visualization plots for evaluation results.
    
    Args:
        output_dir: Directory containing metrics.json and to save plots
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Metrics file not found at {metrics_path}")
        return
        
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    cm = [[metrics['confusion_matrix']['TN'], metrics['confusion_matrix']['FP']],
          [metrics['confusion_matrix']['FN'], metrics['confusion_matrix']['TP']]]
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred ADL', 'Pred Fall'], 
                yticklabels=['True ADL', 'True Fall'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\nAccuracy={metrics["accuracy"]:.3f}, Recall={metrics["recall"]:.3f}')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    
    plt.tight_layout()
    cm_path = plots_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Saved {cm_path}")
    plt.close()
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    vals = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
             metrics['specificity'], metrics['f1_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.5
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, vals, width, color='skyblue', edgecolor='black')
    
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.0)
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=1, label='Target (0.8)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    bar_path = plots_dir / 'metrics_bar.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"Saved {bar_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    create_plots(args.output_dir)
