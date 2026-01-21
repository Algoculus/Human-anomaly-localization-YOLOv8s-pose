"""
Metrics Module

Calculates comprehensive performance metrics for Fall Detection.
1. Frame-level Classification (Accuracy, F1, ROC-AUC)
2. Score-based Optimization (Optimal Thresholds)
3. Event-level Performance (Detection Delay, False Alarms)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from loguru import logger

from src.rules.state_machine import State

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    conf_matrix: np.ndarray # [[TN, FP], [FN, TP]]

@dataclass
class EventMetrics:
    tp_events: int # Correctly detected falls
    fp_events: int # False alarms (ADL detected as fall)
    fn_events: int # Missed falls
    precision: float
    recall: float
    f1: float
    avg_delay_seconds: float # Mean time from GT start to Detection
    false_alarm_rate_per_hour: float = 0.0
    total_test_duration_hours: float = 0.0

class Evaluator:
    """
    Evaluates predictions against ground truth.
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.positive_label = 1 # Lying
        self.negative_label = -1 # Not lying
        
    def classification_metrics(self, 
                               y_true: np.ndarray, 
                               y_score: np.ndarray, 
                               y_pred_binary: np.ndarray) -> ClassificationMetrics:
        """
        Compute frame-level classification metrics.
        
        Args:
            y_true: Ground truth labels (1 or -1)
            y_score: Continuous fall score (0.0 to 1.0)
            y_pred_binary: Binary predictions (1 for Fall/Lying, 0 for Normal)
        """
        # Ensure Valid inputs (remove transitions if any passed inadvertently, though loader handles this)
        valid_mask = y_true != 0
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
        y_pred_binary = y_pred_binary[valid_mask]
        
        # Remap -1 to 0 for sklearn binary metrics
        y_true_bin = (y_true == 1).astype(int)
        
        # 1. Metrics
        acc = accuracy_score(y_true_bin, y_pred_binary)
        prec = precision_score(y_true_bin, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_binary, zero_division=0)
        
        # 2. AUCs (using continuous score)
        try:
            roc = roc_auc_score(y_true_bin, y_score)
            pr = average_precision_score(y_true_bin, y_score)
        except ValueError:
            roc = 0.0
            pr = 0.0
            
        # 3. Confusion Matrix
        cm = confusion_matrix(y_true_bin, y_pred_binary, labels=[0, 1])
        
        return ClassificationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc,
            pr_auc=pr,
            conf_matrix=cm
        )
        
    def find_optimal_threshold(self, 
                               y_true: np.ndarray, 
                               y_score: np.ndarray) -> Tuple[float, float]:
        """
        Find optimal threshold using Youden's J statistic (Sensitivity + Specificity - 1).
        
        Returns:
            (best_threshold, best_j_score)
        """
        valid_mask = y_true != 0
        y_true_bin = (y_true[valid_mask] == 1).astype(int)
        y_score_valid = y_score[valid_mask]
        
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_score_valid)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        return thresholds[best_idx], j_scores[best_idx]

    def event_metrics(self, 
                      results_df: pd.DataFrame, 
                      sequences: List[str]) -> EventMetrics:
        """
        Compute event-level metrics (TP/FP/FN sequences).
        
        Assumptions:
        - A 'Fall' sequence (name starts with 'fall') contains exactly one fall context.
        - An 'ADL' sequence contains zero falls.
        - A detected event is defined as entering LYING state continuously for > 1 second (approx).
          (However, our State Machine has 'LYING' as a latched state, so any occurrence of LYING state
           implies a detected fall event).
        """
        tp = 0
        fp = 0
        fn = 0
        delays = []
        
        unique_seqs = results_df['sequence'].unique()
        
        for seq_id in unique_seqs:
            seq_df = results_df[results_df['sequence'] == seq_id]
            is_fall_seq = seq_id.startswith('fall')
            
            # Check if prediction ever reached LYING state
            # We map string 'State.LYING' to detection
            detected = (seq_df['pred_state'] == 'LYING').any()
            
            if is_fall_seq:
                if detected:
                    tp += 1
                    # Calculate Delay: First Pred Frame - First GT Frame (label=1)
                    # Note: GT label=1 starts when person is on ground.
                    # Pred LYING might start slightly after.
                    
                    gt_lying = seq_df[seq_df['gt_label'] == 1]
                    pred_lying = seq_df[seq_df['pred_state'] == 'LYING']
                    
                    if not gt_lying.empty and not pred_lying.empty:
                        start_gt = gt_lying.iloc[0]['time_ms']
                        start_pred = pred_lying.iloc[0]['time_ms']
                        delay = (start_pred - start_gt) / 1000.0 # Seconds
                        delays.append(delay)
                else:
                    fn += 1 # Missed fall
            else:
                # ADL sequence
                if detected:
                    fp += 1 # False Alarm
                # Else TN (implicitly)
                
        # Calculate derived metrics
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        avg_delay = np.mean(delays) if delays else 0.0
        
        # Calculate Total Duration of Test Set for False Alarm Rate
        total_duration_ms = results_df['time_ms'].max() - results_df['time_ms'].min()
        # Sum of durations of all sequences
        total_duration_ms = 0
        for seq_id in unique_seqs:
            seq_df = results_df[results_df['sequence'] == seq_id]
            if not seq_df.empty:
                total_duration_ms += seq_df['time_ms'].max() - seq_df['time_ms'].min()
        
        total_hours = total_duration_ms / (1000.0 * 3600.0)
        false_alarm_rate = fp / total_hours if total_hours > 0 else 0.0
        
        return EventMetrics(
            tp_events=tp,
            fp_events=fp,
            fn_events=fn,
            precision=prec,
            recall=rec,
            f1=f1,
            avg_delay_seconds=avg_delay,
            false_alarm_rate_per_hour=false_alarm_rate,
            total_test_duration_hours=total_hours
        )

if __name__ == "__main__":
    print("Testing Evaluator...")
    evaluator = Evaluator()
    
    # Mock Data
    y_true = np.array([-1, -1, 0, 1, 1, 1, -1, -1])
    y_score = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.7, 0.1, 0.3])
    y_pred = np.array([ 0,   0,   0,   1,   1,   1,   0,   0])
    
    metrics = evaluator.classification_metrics(y_true, y_score, y_pred)
    print("\nClassification Metrics:")
    print(f"  Accuracy: {metrics.accuracy:.2f}")
    print(f"  F1: {metrics.f1:.2f}")
    print(f"  AUC: {metrics.roc_auc:.2f}")
    print(f"  Confusion Matrix: {metrics.conf_matrix.tolist()}")
    
    thr, j_val = evaluator.find_optimal_threshold(y_true, y_score)
    print(f"  Optimal Threshold: {thr:.2f}")
    
    # Mock Event Data
    mock_df = pd.DataFrame({
        'sequence': ['fall-01']*5 + ['adl-01']*5,
        'gt_label': [-1, -1, 1, 1, 1, -1, -1, -1, -1, -1],
        'pred_state': ['NORMAL', 'NORMAL', 'LYING', 'LYING', 'LYING', 
                       'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL'],
        'time_ms': [0, 33, 66, 99, 132, 0, 33, 66, 99, 132]
    })
    
    evt_metrics = evaluator.event_metrics(mock_df, ['fall-01', 'adl-01'])
    print("\nEvent Metrics:")
    print(f"  TP: {evt_metrics.tp_events}, FP: {evt_metrics.fp_events}")
    print(f"  Recall: {evt_metrics.recall:.2f}")
    print(f"  Avg Delay: {evt_metrics.avg_delay_seconds:.3f} sec")
