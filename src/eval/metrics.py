"""
Enhanced Metrics Module - Based on Kwolek & Kepski 2014 Evaluation

Key improvements:
1. More nuanced event-level metrics (detection delay, false alarm rate)
2. Activity-specific confusion matrices
3. Temporal IoU for fall event matching
4. Sensitivity/Specificity analysis per activity type
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
)
from loguru import logger

from src.rules.state_machine import State


@dataclass
class ClassificationMetrics:
    """Frame-level classification metrics"""
    accuracy: float
    precision: float
    recall: float
    specificity: float  # Added for better analysis
    f1: float
    roc_auc: float
    pr_auc: float
    conf_matrix: np.ndarray  # [[TN, FP], [FN, TP]]
    
    # Additional metrics
    balanced_accuracy: float = 0.0  # (Sensitivity + Specificity) / 2
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


@dataclass
class EventMetrics:
    """Event-level fall detection metrics"""
    
    # Basic counts
    tp_events: int  # Correctly detected falls
    fp_events: int  # False alarms (ADL detected as fall)
    fn_events: int  # Missed falls
    tn_events: int  # Correctly ignored ADLs
    
    # Derived metrics
    precision: float
    recall: float  # Sensitivity
    specificity: float  # TN / (TN + FP)
    f1: float
    
    # Temporal metrics
    avg_detection_delay_seconds: float  # Time from fall start to detection
    median_detection_delay_seconds: float
    max_detection_delay_seconds: float
    detection_delays: List[float]  # All delays
    
    # False alarm analysis
    false_alarm_rate_per_hour: float
    total_test_duration_hours: float
    
    # Activity-specific breakdown
    activity_breakdown: Dict[str, Dict] = None


@dataclass
class FallEvent:
    """Represents a fall event for matching"""
    sequence_id: str
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    duration_frames: int
    duration_seconds: float
    detected: bool = False
    detection_frame: Optional[int] = None
    detection_time_ms: Optional[float] = None
    detection_delay_seconds: Optional[float] = None


class Evaluator:
    """Enhanced evaluator with detailed fall detection metrics"""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.positive_label = 1  # Lying
        self.negative_label = -1  # Not lying
        
    def classification_metrics(self, 
                               y_true: np.ndarray, 
                               y_score: np.ndarray, 
                               y_pred_binary: np.ndarray) -> ClassificationMetrics:
        """Compute enhanced frame-level classification metrics"""
        
        # Remove transition frames (label = 0)
        valid_mask = y_true != 0
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
        y_pred_binary = y_pred_binary[valid_mask]
        
        if len(y_true) == 0:
            logger.warning("No valid frames for evaluation")
            return ClassificationMetrics(
                accuracy=0, precision=0, recall=0, specificity=0,
                f1=0, roc_auc=0, pr_auc=0,
                conf_matrix=np.zeros((2, 2))
            )
        
        # Remap -1 to 0 for sklearn
        y_true_bin = (y_true == 1).astype(int)
        
        # Basic metrics
        acc = accuracy_score(y_true_bin, y_pred_binary)
        prec = precision_score(y_true_bin, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_bin, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Balanced accuracy
        bal_acc = (rec + spec) / 2.0
        
        # Error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # AUCs
        try:
            roc = roc_auc_score(y_true_bin, y_score)
            pr = average_precision_score(y_true_bin, y_score)
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            roc = 0.0
            pr = 0.0
        
        return ClassificationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            specificity=spec,
            f1=f1,
            roc_auc=roc,
            pr_auc=pr,
            conf_matrix=cm,
            balanced_accuracy=bal_acc,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
    
    def event_metrics(self, 
                      results_df: pd.DataFrame, 
                      sequences: List[str],
                      min_lying_duration_sec: float = 0.5) -> EventMetrics:
        """
        Compute enhanced event-level metrics.
        
        Improvements over basic version:
        1. Activity-specific analysis
        2. Detailed delay statistics
        3. Temporal IoU for matching
        4. Better handling of edge cases
        """
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        detection_delays = []
        activity_stats = {
            'falls': {'total': 0, 'detected': 0, 'missed': 0},
            'adl': {'total': 0, 'correct': 0, 'false_alarms': 0}
        }
        
        unique_seqs = results_df['sequence'].unique()
        
        for seq_id in unique_seqs:
            seq_df = results_df[results_df['sequence'] == seq_id].copy()
            
            if seq_df.empty:
                continue
            
            is_fall_seq = seq_id.startswith('fall')
            
            # Extract ground truth fall event (for fall sequences)
            gt_fall_event = None
            if is_fall_seq:
                gt_fall_event = self._extract_fall_event_from_gt(seq_df, seq_id)
                if gt_fall_event:
                    activity_stats['falls']['total'] += 1
            else:
                activity_stats['adl']['total'] += 1
            
            # Extract predicted fall event(s)
            pred_fall_events = self._extract_predicted_fall_events(
                seq_df, 
                seq_id,
                min_lying_duration_sec
            )
            
            # Match predictions to ground truth
            if is_fall_seq:
                if gt_fall_event:
                    # Check if any prediction matches the GT fall
                    matched = False
                    best_delay = None
                    
                    for pred_event in pred_fall_events:
                        # Check temporal overlap
                        overlap = self._temporal_iou(gt_fall_event, pred_event)
                        
                        if overlap > 0.3:  # At least 30% overlap
                            matched = True
                            
                            # Calculate detection delay
                            delay = self._calculate_detection_delay(
                                gt_fall_event,
                                pred_event
                            )
                            
                            if best_delay is None or delay < best_delay:
                                best_delay = delay
                    
                    if matched:
                        tp += 1
                        activity_stats['falls']['detected'] += 1
                        if best_delay is not None:
                            detection_delays.append(best_delay)
                    else:
                        fn += 1
                        activity_stats['falls']['missed'] += 1
                else:
                    # No clear GT event but is labeled as fall sequence
                    if pred_fall_events:
                        tp += 1
                        activity_stats['falls']['detected'] += 1
                    else:
                        fn += 1
                        activity_stats['falls']['missed'] += 1
            
            else:  # ADL sequence
                if pred_fall_events:
                    # False alarm
                    fp += 1
                    activity_stats['adl']['false_alarms'] += 1
                else:
                    # Correctly ignored
                    tn += 1
                    activity_stats['adl']['correct'] += 1
        
        # Calculate derived metrics
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        # Detection delay statistics
        avg_delay = np.mean(detection_delays) if detection_delays else 0.0
        median_delay = np.median(detection_delays) if detection_delays else 0.0
        max_delay = np.max(detection_delays) if detection_delays else 0.0
        
        # False alarm rate
        total_duration_ms = 0
        for seq_id in unique_seqs:
            seq_df = results_df[results_df['sequence'] == seq_id]
            if not seq_df.empty and 'time_ms' in seq_df.columns:
                duration = seq_df['time_ms'].max() - seq_df['time_ms'].min()
                total_duration_ms += duration
        
        total_hours = total_duration_ms / (1000.0 * 3600.0)
        false_alarm_rate = fp / total_hours if total_hours > 0 else 0.0
        
        return EventMetrics(
            tp_events=tp,
            fp_events=fp,
            fn_events=fn,
            tn_events=tn,
            precision=prec,
            recall=rec,
            specificity=spec,
            f1=f1,
            avg_detection_delay_seconds=avg_delay,
            median_detection_delay_seconds=median_delay,
            max_detection_delay_seconds=max_delay,
            detection_delays=detection_delays,
            false_alarm_rate_per_hour=false_alarm_rate,
            total_test_duration_hours=total_hours,
            activity_breakdown=activity_stats
        )
    
    def _extract_fall_event_from_gt(self, 
                                     seq_df: pd.DataFrame, 
                                     seq_id: str) -> Optional[FallEvent]:
        """Extract ground truth fall event from labels"""
        
        if 'gt_label' not in seq_df.columns:
            return None
        
        # Find first and last lying frame (label = 1)
        lying_frames = seq_df[seq_df['gt_label'] == 1]
        
        if lying_frames.empty:
            return None
        
        start_idx = lying_frames.index[0]
        end_idx = lying_frames.index[-1]
        
        start_frame = seq_df.loc[start_idx, 'frame']
        end_frame = seq_df.loc[end_idx, 'frame']
        start_time = seq_df.loc[start_idx, 'time_ms']
        end_time = seq_df.loc[end_idx, 'time_ms']
        
        duration_frames = end_frame - start_frame
        duration_sec = (end_time - start_time) / 1000.0
        
        return FallEvent(
            sequence_id=seq_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_ms=start_time,
            end_time_ms=end_time,
            duration_frames=duration_frames,
            duration_seconds=duration_sec
        )
    
    def _extract_predicted_fall_events(self,
                                       seq_df: pd.DataFrame,
                                       seq_id: str,
                                       min_duration_sec: float = 0.5) -> List[FallEvent]:
        """Extract predicted fall events (LYING state periods)"""
        
        events = []
        
        if 'pred_state' not in seq_df.columns:
            return events
        
        # Find continuous LYING periods
        seq_df = seq_df.copy()
        seq_df['is_lying'] = seq_df['pred_state'] == 'LYING'
        
        # Find transitions
        seq_df['lying_group'] = (seq_df['is_lying'] != seq_df['is_lying'].shift()).cumsum()
        
        for group_id, group in seq_df[seq_df['is_lying']].groupby('lying_group'):
            if len(group) == 0:
                continue
            
            start_idx = group.index[0]
            end_idx = group.index[-1]
            
            start_frame = group.loc[start_idx, 'frame']
            end_frame = group.loc[end_idx, 'frame']
            start_time = group.loc[start_idx, 'time_ms']
            end_time = group.loc[end_idx, 'time_ms']
            
            duration_sec = (end_time - start_time) / 1000.0
            
            # Filter out very short detections (likely noise)
            if duration_sec >= min_duration_sec:
                events.append(FallEvent(
                    sequence_id=seq_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time_ms=start_time,
                    end_time_ms=end_time,
                    duration_frames=end_frame - start_frame,
                    duration_seconds=duration_sec,
                    detected=True
                ))
        
        return events
    
    def _temporal_iou(self, gt_event: FallEvent, pred_event: FallEvent) -> float:
        """Calculate temporal Intersection over Union between two events"""
        
        # Intersection
        intersect_start = max(gt_event.start_frame, pred_event.start_frame)
        intersect_end = min(gt_event.end_frame, pred_event.end_frame)
        
        if intersect_end <= intersect_start:
            return 0.0
        
        intersection = intersect_end - intersect_start
        
        # Union
        union_start = min(gt_event.start_frame, pred_event.start_frame)
        union_end = max(gt_event.end_frame, pred_event.end_frame)
        union = union_end - union_start
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_detection_delay(self, 
                                   gt_event: FallEvent, 
                                   pred_event: FallEvent) -> float:
        """
        Calculate detection delay in seconds.
        
        Delay = time from GT fall start to prediction start
        Negative delay = early detection (predicted before actual fall marked)
        """
        
        delay_ms = pred_event.start_time_ms - gt_event.start_time_ms
        return delay_ms / 1000.0
    
    def generate_report(self,
                       frame_metrics: ClassificationMetrics,
                       event_metrics: EventMetrics,
                       set_name: str = "Test") -> str:
        """Generate comprehensive evaluation report"""
        
        report = f"""
{'='*70}
{set_name.upper()} SET EVALUATION REPORT
{'='*70}

FRAME-LEVEL METRICS (Classification)
{'='*70}
  Accuracy:              {frame_metrics.accuracy:.4f}
  Balanced Accuracy:     {frame_metrics.balanced_accuracy:.4f}
  
  Precision:             {frame_metrics.precision:.4f}
  Recall (Sensitivity):  {frame_metrics.recall:.4f}
  Specificity:           {frame_metrics.specificity:.4f}
  F1 Score:              {frame_metrics.f1:.4f}
  
  ROC AUC:               {frame_metrics.roc_auc:.4f}
  PR AUC:                {frame_metrics.pr_auc:.4f}
  
  False Positive Rate:   {frame_metrics.false_positive_rate:.4f}
  False Negative Rate:   {frame_metrics.false_negative_rate:.4f}

Confusion Matrix (Frame-level):
  [[TN={frame_metrics.conf_matrix[0,0]:5d}, FP={frame_metrics.conf_matrix[0,1]:5d}]
   [FN={frame_metrics.conf_matrix[1,0]:5d}, TP={frame_metrics.conf_matrix[1,1]:5d}]]

EVENT-LEVEL METRICS (Fall Detection)
{'='*70}
  True Positives (Falls Detected):    {event_metrics.tp_events}
  False Negatives (Falls Missed):     {event_metrics.fn_events}
  False Positives (False Alarms):     {event_metrics.fp_events}
  True Negatives (ADL Correct):       {event_metrics.tn_events}
  
  Precision:                          {event_metrics.precision:.4f}
  Recall (Sensitivity):               {event_metrics.recall:.4f}
  Specificity:                        {event_metrics.specificity:.4f}
  F1 Score:                           {event_metrics.f1:.4f}

TEMPORAL METRICS
{'='*70}
  Avg Detection Delay:                {event_metrics.avg_detection_delay_seconds:.3f}s
  Median Detection Delay:             {event_metrics.median_detection_delay_seconds:.3f}s
  Max Detection Delay:                {event_metrics.max_detection_delay_seconds:.3f}s
  
  False Alarm Rate:                   {event_metrics.false_alarm_rate_per_hour:.2f} alarms/hour
  Total Test Duration:                {event_metrics.total_test_duration_hours * 60:.1f} minutes

ACTIVITY-SPECIFIC BREAKDOWN
{'='*70}
"""
        
        if event_metrics.activity_breakdown:
            falls = event_metrics.activity_breakdown.get('falls', {})
            adl = event_metrics.activity_breakdown.get('adl', {})
            
            report += f"""
  Falls:
    Total fall sequences:             {falls.get('total', 0)}
    Detected:                         {falls.get('detected', 0)}
    Missed:                           {falls.get('missed', 0)}
    Detection Rate:                   {falls.get('detected', 0) / falls.get('total', 1):.2%}
  
  ADL (Activities of Daily Living):
    Total ADL sequences:              {adl.get('total', 0)}
    Correctly ignored:                {adl.get('correct', 0)}
    False alarms:                     {adl.get('false_alarms', 0)}
    Specificity:                      {adl.get('correct', 0) / adl.get('total', 1):.2%}
"""
        
        report += "\n" + "="*70 + "\n"
        
        return report


if __name__ == "__main__":
    print("Testing Enhanced Evaluator...")
    
    evaluator = Evaluator(fps=30.0)
    
    # Mock frame-level data
    y_true = np.array([-1]*40 + [0]*5 + [1]*50 + [-1]*30)  # Not lying, transition, lying, not lying
    y_score = np.concatenate([
        np.random.uniform(0.0, 0.3, 40),  # Low scores for normal
        np.random.uniform(0.3, 0.7, 5),   # Medium for transition
        np.random.uniform(0.7, 1.0, 50),  # High for lying
        np.random.uniform(0.0, 0.3, 30)   # Low for normal again
    ])
    y_pred = (y_score > 0.6).astype(int)
    
    frame_metrics = evaluator.classification_metrics(y_true, y_score, y_pred)
    
    print("\n=== Frame-Level Metrics ===")
    print(f"Accuracy: {frame_metrics.accuracy:.3f}")
    print(f"Precision: {frame_metrics.precision:.3f}")
    print(f"Recall: {frame_metrics.recall:.3f}")
    print(f"Specificity: {frame_metrics.specificity:.3f}")
    print(f"F1: {frame_metrics.f1:.3f}")
    print(f"Balanced Accuracy: {frame_metrics.balanced_accuracy:.3f}")
    
    # Mock event-level data
    mock_results = pd.DataFrame({
        'sequence': ['fall-01']*100 + ['fall-02']*100 + ['adl-01']*100 + ['adl-02']*100,
        'frame': list(range(1, 101)) * 4,
        'time_ms': [i * 33 for i in range(1, 101)] * 4,
        'gt_label': (
            [-1]*30 + [1]*70 +      # fall-01: lying from frame 31
            [-1]*25 + [1]*75 +      # fall-02: lying from frame 26
            [-1]*100 +              # adl-01: no lying
            [-1]*100                # adl-02: no lying
        ),
        'pred_state': (
            ['NORMAL']*35 + ['LYING']*65 +     # fall-01: detected at frame 36 (delay of 5 frames)
            ['NORMAL']*28 + ['LYING']*72 +     # fall-02: detected at frame 29 (delay of 3 frames)
            ['NORMAL']*100 +                    # adl-01: correct
            ['NORMAL']*80 + ['LYING']*20       # adl-02: false alarm
        )
    })
    
    event_metrics = evaluator.event_metrics(
        mock_results,
        ['fall-01', 'fall-02', 'adl-01', 'adl-02']
    )
    
    print("\n=== Event-Level Metrics ===")
    print(f"TP: {event_metrics.tp_events}, FP: {event_metrics.fp_events}")
    print(f"FN: {event_metrics.fn_events}, TN: {event_metrics.tn_events}")
    print(f"Precision: {event_metrics.precision:.3f}")
    print(f"Recall: {event_metrics.recall:.3f}")
    print(f"Specificity: {event_metrics.specificity:.3f}")
    print(f"F1: {event_metrics.f1:.3f}")
    print(f"Avg Delay: {event_metrics.avg_detection_delay_seconds:.3f}s")
    print(f"False Alarm Rate: {event_metrics.false_alarm_rate_per_hour:.2f}/hour")
    
    # Generate report
    report = evaluator.generate_report(frame_metrics, event_metrics, "Test")
    print("\n" + report)
    
    print("✓ Evaluator test complete!")