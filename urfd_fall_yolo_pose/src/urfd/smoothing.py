"""
Fall Detection State Machine

Implements dual-tier Hypothesis→Verification architecture:
- NORMAL: No fall indicators
- HYPOTHESIS: Potential fall detected (Tier 1 trigger)
- VERIFYING: Confirming fall in realtime mode (Tier 2)
- FALL: Confirmed fall

Outputs FallDecision with:
- Label: FALL, NORMAL, UNCERTAIN
- Confidence: 0..1
- Top features contributing to decision
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

from .rules import FallScorer, FallScore, check_lying_posture
from .features import FeatureBuffer, FrameFeatures


class FallState(Enum):
    """State machine states."""
    NORMAL = "NORMAL"
    HYPOTHESIS = "HYPOTHESIS"
    VERIFYING = "VERIFYING"
    FALL = "FALL_CONFIRMED"


class FallLabel(Enum):
    """Output labels."""
    FALL = "FALL"
    NORMAL = "NORMAL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class FallDecision:
    """
    Complete fall detection decision with explainability.
    
    Attributes:
        label: FALL, NORMAL, or UNCERTAIN
        confidence: Overall confidence score (0..1)
        top_features: List of (feature_name, value, contribution)
        fsm_state: Current FSM state string
        frame_id: Frame number
        track_id: Person track ID
        reason: Human-readable explanation
    """
    label: str = "NORMAL"
    confidence: float = 0.0
    top_features: List[Tuple[str, float, float]] = field(default_factory=list)
    fsm_state: str = "NORMAL"
    frame_id: int = 0
    track_id: int = -1
    reason: str = ""
    
    def __str__(self) -> str:
        features_str = ", ".join([f"{n}={v:.2f}" for n, v, _ in self.top_features[:3]])
        return f"[{self.label}] conf={self.confidence:.2f} state={self.fsm_state} | {features_str}"


class FallStateMachine:
    """
    Dual-tier state machine for fall detection.
    
    Architecture:
    - Tier 1 (Hypothesis): Fast trigger for short clips (0.2-0.5s)
    - Tier 2 (Verification): Confirmation for realtime (0.5-2s)
    
    States:
        NORMAL → HYPOTHESIS: Score above early_thres
        HYPOTHESIS → FALL: High confidence early decision (demo mode)
        HYPOTHESIS → VERIFYING: Moderate confidence (realtime mode)
        HYPOTHESIS → NORMAL: No impact detected (false alarm)
        VERIFYING → FALL: Sustained lying posture confirmed
        VERIFYING → NORMAL: Timeout or recovery
        FALL → NORMAL: Recovery detected
    """
    
    def __init__(self, config: Dict):
        """
        Initialize state machine with config.
        
        Args:
            config: Configuration dict with thresholds
        """
        self.config = config
        self.scorer = FallScorer(config)
        
        # State
        self.state = FallState.NORMAL
        self.score_accumulator = 0.0
        self.frame_count = 0
        
        # Mode
        self.mode = config.get("detection_mode", "realtime")  # 'demo' or 'realtime'
        
        # Hypothesis tracking
        hypothesis_cfg = config.get("hypothesis", {})
        self.early_frames_thres = hypothesis_cfg.get("early_frames", 3)
        self.early_score_high = hypothesis_cfg.get("early_score_high", 0.65)
        self.hypothesis_frame_count = 0
        self.hypothesis_scores = []
        
        # Verification tracking
        verify_cfg = config.get("verification", {})
        self.confirm_frames = verify_cfg.get("confirm_frames", 8)
        self.confirm_window = verify_cfg.get("confirm_window", 30)
        self.post_motion_thres = verify_cfg.get("post_motion_thres", 3.0)
        self.verify_frame_count = 0
        self.verify_lying_count = 0
        
        # Recovery tracking
        self.recovery_window = config.get("recovery_window", 13)
        self.recovery_upright_angle = config.get("recovery_upright_angle_thres", 48.0)
        self.recovery_ar = config.get("recovery_ar_thres", 1.12)
        self.recovery_history = []
        
        # Height tracking
        self.height_history = []
        self.height_window = config.get("height_window", 16)
        
        # Missing streak
        self.missing_streak = 0
        self.max_missing = config.get("max_missing_streak", 5)
        
        # Score parameters
        self.score_decay = config.get("score_decay", 0.85)
        
        # Legacy thresholds
        self.angle_thres = config.get("angle_thres", 47.0)
        self.ar_thres = config.get("ar_thres", 1.08)
        self.height_drop_thres = config.get("height_drop_thres", 0.13)
        self.dy_fall_thres = config.get("dy_fall_thres", 7.0)
        self.confirm_angle_thres = config.get("confirm_angle_thres", 42.0)
        self.confirm_ar_thres = config.get("confirm_ar_thres", 1.25)
    
    def reset(self):
        """Reset state machine to initial state."""
        self.state = FallState.NORMAL
        self.score_accumulator = 0.0
        self.frame_count = 0
        self.hypothesis_frame_count = 0
        self.hypothesis_scores = []
        self.verify_frame_count = 0
        self.verify_lying_count = 0
        self.recovery_history = []
        self.height_history = []
        self.missing_streak = 0
    
    def _compute_height_drop(self, current_height: float) -> float:
        """Compute normalized height drop from baseline."""
        if current_height is None or current_height <= 0:
            return 0.0
        
        self.height_history.append(current_height)
        if len(self.height_history) > self.height_window:
            self.height_history = self.height_history[-self.height_window:]
        
        if len(self.height_history) < 2:
            return 0.0
        
        baseline_heights = self.height_history[:-1]
        median_height = np.median(baseline_heights)
        
        if median_height <= 0:
            return 0.0
        
        drop = (median_height - current_height) / median_height
        return np.clip(drop, 0.0, 1.0)
    
    def _check_recovery(self, features: Dict) -> bool:
        """Check if person has recovered (standing up).
        
        Improved for realtime: faster recovery detection when person stands up.
        """
        # Check upright via keypoints
        is_upright_kp = False
        if features.get("body_angle_deg") is not None:
            is_upright_kp = features["body_angle_deg"] < self.recovery_upright_angle
        
        # Check upright via bbox AR (wider threshold for realtime)
        ar = features.get("bbox_aspect_ratio", 1.5)
        hw_ratio = features.get("hw_ratio", 0.5)
        
        # AR < 1.2 means height > width (standing/upright)
        is_upright_ar = ar < 1.2
        
        # hw_ratio < 0.7 also indicates upright (width < 70% of height)
        is_upright_hw = hw_ratio < 0.7
        
        is_upright = is_upright_kp or is_upright_ar or is_upright_hw
        
        # Track recovery history
        self.recovery_history.append(1 if is_upright else 0)
        if len(self.recovery_history) > self.recovery_window:
            self.recovery_history.pop(0)
        
        # Fast recovery: if current frame is strongly upright (AR < 0.9), reset immediately
        if ar < 0.9 and hw_ratio < 0.55:
            return True
        
        # Check height recovery (relaxed for realtime)
        height_recovered = True  # Default to true for faster response
        if len(self.height_history) >= 5:
            current_h = features.get("height", 0)
            recent_max = max(self.height_history[-5:])
            if recent_max > 0 and current_h > 0:
                # If current height is at least 60% of recent max, consider recovered
                height_recovered = (current_h / recent_max) > 0.6
        
        # Need just 2-3 consistent upright frames (faster than before)
        if len(self.recovery_history) >= 2:
            recent_upright = sum(self.recovery_history[-3:]) if len(self.recovery_history) >= 3 else sum(self.recovery_history[-2:])
            required = 2 if len(self.recovery_history) >= 3 else 1
            if recent_upright >= required and height_recovered:
                return True
        
        return False
    
    def _get_label_from_state(self, confidence: float) -> str:
        """Convert state + confidence to output label."""
        if self.state == FallState.FALL:
            return FallLabel.FALL.value
        elif self.state == FallState.NORMAL:
            return FallLabel.NORMAL.value
        elif self.state in [FallState.HYPOTHESIS, FallState.VERIFYING]:
            # Uncertain during transition states
            if confidence >= 0.55:
                return FallLabel.UNCERTAIN.value
            else:
                return FallLabel.NORMAL.value
        return FallLabel.NORMAL.value
    
    def update(self, features: Dict) -> Tuple[str, float]:
        """
        Update state machine with new frame features.
        
        Legacy API: Returns (state_string, score).
        For new API with FallDecision, use update_v2().
        """
        decision = self.update_v2(features)
        # Map FALL label to FALL_CONFIRMED state for backward compatibility
        state_str = decision.fsm_state
        return state_str, decision.confidence
    
    def update_v2(self, features, track_id: int = -1, frame_idx: int = 0) -> FallDecision:
        """
        Process new frame features and return decision.
        
        Args:
            features: FrameFeatures object or dict
            track_id: Track ID
            frame_idx: Frame index
        
        Returns:
            FallDecision with label, confidence, and metadata
        """
        # Convert FrameFeatures object to dict if needed
        if hasattr(features, 'to_dict'):
            features = features.to_dict()
        
        self.frame_count += 1
        
        decision = FallDecision(
            frame_id=frame_idx,
            track_id=track_id,
            fsm_state=self.state.value
        )
        
        # =========================================================
        # HANDLE MISSING DETECTION
        # =========================================================
        if features.get("bbox") is None:
            self.missing_streak += 1
            if self.missing_streak > self.max_missing:
                self.score_accumulator *= 0.5
            decision.label = self._get_label_from_state(self.score_accumulator)
            decision.confidence = self.score_accumulator
            decision.reason = "No detection"
            return decision
        else:
            self.missing_streak = 0
        
        # =========================================================
        # COMPUTE SCORES
        # =========================================================
        fall_score = self.scorer.compute_fall_confidence(features)
        height_drop = self._compute_height_drop(features.get("height"))
        
        # Update score accumulator with EMA
        frame_score = fall_score.confidence
        self.score_accumulator = self.score_decay * self.score_accumulator + (1 - self.score_decay) * frame_score
        
        # Simplified check using FallScorer only (removed legacy check_fall_candidate)
        is_candidate = self.scorer.is_hypothesis_trigger(features, fall_score)
        
        is_lying = check_lying_posture(features)
        
        # =========================================================
        # STATE MACHINE TRANSITIONS
        # =========================================================
        if self.state == FallState.NORMAL:
            # Check for hypothesis trigger
            if self.scorer.is_hypothesis_trigger(features, fall_score) or is_candidate:
                self.state = FallState.HYPOTHESIS
                self.hypothesis_frame_count = 1
                self.hypothesis_scores = [fall_score.confidence]
                decision.reason = "Hypothesis triggered"
        
        elif self.state == FallState.HYPOTHESIS:
            self.hypothesis_frame_count += 1
            self.hypothesis_scores.append(fall_score.confidence)
            
            # Quick recovery check - if person is clearly upright, reset immediately
            ar = features.get("bbox_aspect_ratio", 1.5)
            hw_ratio = features.get("hw_ratio", 0.5)
            if ar < 0.9 and hw_ratio < 0.55:
                self.state = FallState.NORMAL
                self.hypothesis_scores = []
                self.score_accumulator *= 0.2
                decision.reason = "Quick recovery - upright posture detected"
                # Skip rest of hypothesis processing
            elif self.mode == "demo":
                # Check for early high-confidence decision (demo mode)
                max_score = max(self.hypothesis_scores)
                if max_score >= self.early_score_high and self.hypothesis_frame_count <= self.early_frames_thres:
                    self.state = FallState.FALL
                    self.score_accumulator = 1.0
                    decision.reason = "Early high-confidence fall (demo mode)"
            
            # Check for transition to VERIFYING (realtime mode)
            if self.state == FallState.HYPOTHESIS:
                avg_score = np.mean(self.hypothesis_scores[-5:]) if len(self.hypothesis_scores) >= 5 else np.mean(self.hypothesis_scores)
                
                if self.scorer.is_verification_confirm(features, fall_score) or is_lying:
                    self.state = FallState.VERIFYING
                    self.verify_frame_count = 1
                    self.verify_lying_count = 1 if is_lying else 0
                    decision.reason = "Entered verification"
                elif self.hypothesis_frame_count > 10 and avg_score < 0.35:
                    # False alarm - not enough evidence (faster timeout)
                    self.state = FallState.NORMAL
                    self.hypothesis_scores = []
                    self.score_accumulator *= 0.3
                    decision.reason = "Hypothesis timeout, no fall"
        
        elif self.state == FallState.VERIFYING:
            self.verify_frame_count += 1
            
            # Check if condition persists (Smoothed Score > Threshold OR Lying Posture)
            # Strict Motion Gating removed to improve Recall (catch slow/slump falls)
            
            threshold = 0.40 # Aggressive threshold (Config is 0.45, we loose it slightly here)
            is_confirmed_frame = (self.score_accumulator >= threshold) or is_lying
            
            if is_confirmed_frame:
                self.verify_lying_count += 1
            
            # Check for confirmation
            if self.verify_lying_count >= self.confirm_frames:
                # Check post-motion settled
                post_motion = features.get("post_motion", 0.0)
                motion_settled = post_motion < self.post_motion_thres
                
                if motion_settled or self.verify_lying_count >= self.confirm_frames + 3:
                    self.state = FallState.FALL
                    self.score_accumulator = 1.0
                    decision.reason = f"Fall confirmed (lying={self.verify_lying_count})"
            
            # Check for timeout
            if self.verify_frame_count > self.confirm_window:
                if self.verify_lying_count < self.confirm_frames // 2:
                    self.state = FallState.NORMAL
                    self.score_accumulator *= 0.2
                    decision.reason = "Verification timeout, no fall"
            
            # Check for recovery
            if self._check_recovery(features):
                self.state = FallState.NORMAL
                self.score_accumulator = 0.0
                decision.reason = "Recovery detected during verification"
        
        elif self.state == FallState.FALL:
            # Stay in FALL, check for recovery
            if self._check_recovery(features):
                self.state = FallState.NORMAL
                self.score_accumulator = 0.0
                self.reset()
                decision.reason = "Recovery from fall"
            else:
                self.score_accumulator = 1.0
        
        # =========================================================
        # BUILD DECISION
        # =========================================================
        decision.fsm_state = self.state.value
        decision.confidence = np.clip(self.score_accumulator, 0.0, 1.0)
        decision.label = self._get_label_from_state(decision.confidence)
        decision.top_features = fall_score.get_top_features(3)
        
        if not decision.reason:
            decision.reason = f"State={self.state.value}"
        
        return decision


# =========================================================
# LOGGING HELPER
# =========================================================

def log_decision(decision: FallDecision, verbose: bool = False) -> str:
    """
    Format decision for logging.
    
    Args:
        decision: FallDecision object
        verbose: Include all details
        
    Returns:
        Log string
    """
    base = f"F{decision.frame_id:04d} T{decision.track_id} [{decision.label:8s}] conf={decision.confidence:.3f} state={decision.fsm_state}"
    
    if verbose and decision.top_features:
        features_str = " | ".join([f"{n}={v:.2f}({s:.2f})" for n, v, s in decision.top_features])
        base += f" | {features_str}"
    
    if decision.reason:
        base += f" | {decision.reason}"
    
    return base
