import numpy as np
from .rules import check_fall_candidate, check_lying_posture

class FallStateMachine:
    """State machine for temporal smoothing of fall detection with tolerance.
    
    States:
    - NORMAL: No fall detected
    - CANDIDATE: Potential fall detected, awaiting confirmation
    - FALL_CONFIRMED: Fall confirmed
    """
    
    def __init__(self, config):
        """Initialize state machine.
        
        Args:
            config: Configuration dict with thresholds
        """
        self.config = config
        self.state = "NORMAL"
        
        # Counters for state transitions with tolerance
        self.candidate_history = []
        self.confirm_history = []
        self.missing_streak = 0
        
        # Height history for height drop computation
        self.height_history = []
        
        # Score tracking
        self.score_accumulator = 0.0
        self.fall_confirmed = False
    
    def _compute_height_drop(self, current_height):
        """Compute normalized height drop.
        
        Args:
            current_height: Current bbox height
        
        Returns:
            height_drop: Normalized height drop in [0, 1]
        """
        if current_height is None:
            return 0.0
        
        height_window = self.config["height_window"]
        
        # Add current height to history
        self.height_history.append(current_height)
        
        # Keep only last height_window samples
        if len(self.height_history) > height_window:
            self.height_history = self.height_history[-height_window:]
        
        # Compute median of history
        if len(self.height_history) < 2:
            return 0.0
        
        # Use all but the last sample for median baseline
        baseline_heights = self.height_history[:-1]
        median_height = np.median(baseline_heights)
        
        if median_height <= 0:
            return 0.0
        
        # Compute normalized drop
        drop = (median_height - current_height) / median_height
        
        # Clamp to [0, 1]
        drop = np.clip(drop, 0.0, 1.0)
        
        return drop
    
    def _compute_frame_score(self, features, height_drop):
        """Compute a deterministic fall score for current frame.
        
        Args:
            features: Frame features
            height_drop: Normalized height drop
        
        Returns:
            score: Fall score in [0, 1]
        """
        if features["bbox"] is None:
            return 0.0
        
        score = 0.0
        
        # Component 1: Angle contribution (normalized to [0, 1])
        if features["feature_valid"] and features["body_angle_deg"] is not None:
            angle_normalized = min(features["body_angle_deg"] / 90.0, 1.0)
            score += 0.3 * angle_normalized
        
        # Component 2: Aspect ratio contribution
        ar_normalized = min(features["bbox_aspect_ratio"] / 2.0, 1.0)
        score += 0.25 * ar_normalized
        
        # Component 3: Height drop contribution
        score += 0.25 * height_drop
        
        # Component 4: dy velocity contribution (fast fall)
        if features["dy"] > 0:
            dy_normalized = min(features["dy"] / 50.0, 1.0)
            score += 0.2 * dy_normalized
        
        return np.clip(score, 0.0, 1.0)
    
    def update(self, features):
        """Update state machine with new frame features.
        
        Uses tolerant counting: allows gaps in confirmation windows.
        
        Args:
            features: Frame features dict
        
        Returns:
            state: Current state string
            score: Current fall score
        """
        # Handle missing detection
        if features["bbox"] is None:
            self.missing_streak += 1
            if self.missing_streak > self.config["max_missing_streak"]:
                # Too many missing frames, decay score
                self.score_accumulator *= 0.5
            # Keep current state during missing streak
            return self.state, self.score_accumulator
        else:
            self.missing_streak = 0
        
        # Compute height drop
        height_drop = self._compute_height_drop(features["height"])
        
        # Compute frame score
        frame_score = self._compute_frame_score(features, height_drop)
        
        # Update accumulator with decay
        decay = self.config["score_decay"]
        self.score_accumulator = decay * self.score_accumulator + (1 - decay) * frame_score
        
        # Check fall candidate
        is_candidate = check_fall_candidate(
            features,
            self.config["angle_thres"],
            self.config["ar_thres"],
            height_drop,
            self.config["height_drop_thres"],
            self.config["dy_fall_thres"]
        )
        
        # Check lying posture
        is_lying = check_lying_posture(
            features,
            self.config["confirm_angle_thres"],
            self.config["confirm_ar_thres"]
        )
        
        # State machine logic with tolerance
        if self.state == "NORMAL":
            # Track candidate frames in window
            self.candidate_history.append(1 if is_candidate else 0)
            if len(self.candidate_history) > self.config["cand_enter_frames"] + self.config["cand_tolerance"]:
                self.candidate_history.pop(0)
            
            # Check if enough candidate frames in window
            candidate_count = sum(self.candidate_history)
            if candidate_count >= self.config["cand_enter_frames"]:
                self.state = "CANDIDATE"
                self.confirm_history = []
                self.score_accumulator += self.config["score_boost_candidate"]
        
        elif self.state == "CANDIDATE":
            # Track lying frames in confirmation window
            self.confirm_history.append(1 if is_lying else 0)
            if len(self.confirm_history) > self.config["confirm_window"]:
                self.confirm_history.pop(0)
            
            # Check if enough lying frames in window
            lying_count = sum(self.confirm_history)
            if lying_count >= self.config["confirm_frames"]:
                self.state = "FALL_CONFIRMED"
                self.fall_confirmed = True
                self.score_accumulator = self.config["score_boost_confirmed"]
            elif not is_candidate and len(self.confirm_history) >= self.config["confirm_window"]:
                # No longer candidate and window expired, go back to NORMAL
                if lying_count < self.config["confirm_frames"] - self.config["confirm_tolerance"]:
                    self.state = "NORMAL"
                    self.candidate_history = []
        
        elif self.state == "FALL_CONFIRMED":
            # Once confirmed, stay confirmed and keep score at 1.0
            self.score_accumulator = 1.0
        
        # Clamp score
        self.score_accumulator = np.clip(self.score_accumulator, 0.0, 1.0)
        
        return self.state, self.score_accumulator
