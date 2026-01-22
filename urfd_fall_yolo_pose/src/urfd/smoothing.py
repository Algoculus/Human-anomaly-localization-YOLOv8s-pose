import numpy as np
from .rules import check_fall_candidate, check_lying_posture

class FallStateMachine:
    # State machine: NORMAL -> CANDIDATE -> FALL_CONFIRMED (with recovery)
    
    def __init__(self, config):
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
        
        # Temporal discriminator tracking
        self.recovery_history = []  # Track upright recovery after lying
        self.dy_peak_history = []   # Track dy peaks for slow transition detection
        self.motion_settled = False  # Flag for settled motion after lying
    
    def _compute_height_drop(self, current_height):
        # Compute normalized height drop [0, 1]
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
    
    def _check_slow_transition(self, features):
        """Check if current lying posture is from slow transition (not a fall).
        
        Slow transitions (e.g., lie down to sleep) have lower dy_peak values.
        
        Args:
            features: Current frame features
        
        Returns:
            is_slow: True if transition appears slow
        """
        dy_peak_thres = self.config["dy_peak_thres"]
        slow_lie_max_dy_peak = self.config["slow_lie_max_dy_peak"]
        
        # Add current dy_peak to history
        self.dy_peak_history.append(features["dy_peak"])
        dy_long_window = self.config["dy_long_window"]
        if len(self.dy_peak_history) > dy_long_window:
            self.dy_peak_history.pop(0)
        
        # Check peak dy over recent window
        if len(self.dy_peak_history) >= self.config["dy_short_window"]:
            max_dy_peak = max(self.dy_peak_history[-self.config["dy_short_window"]:])
            # If peak is below slow threshold, it's a slow transition
            return max_dy_peak < slow_lie_max_dy_peak
        
        return False
    
    def _check_recovery(self, features):
        """Check if person has recovered from lying (false positive suppression).
        
        If person briefly lies down then returns upright, it's likely not a fall.
        Works with both keypoint-based and bbox-only detection.
        
        Args:
            features: Current frame features
        
        Returns:
            recovered: True if recovery detected
        """
        recovery_window = self.config["recovery_window"]
        recovery_upright_angle_thres = self.config["recovery_upright_angle_thres"]
        recovery_ar_thres = self.config["recovery_ar_thres"]
        recovery_height_recover_ratio = self.config["recovery_height_recover_ratio"]
        
        # Check if currently upright (bbox-only version for fallback compatibility)
        # Upright if: angle < threshold (if available) OR (AR < threshold AND height recovered)
        is_upright_keypoint = (
            features["body_angle_deg"] is not None and
            features["body_angle_deg"] < recovery_upright_angle_thres
        )
        
        is_upright_bbox = (
            features["bbox_aspect_ratio"] is not None and
            features["bbox_aspect_ratio"] < recovery_ar_thres
        )
        
        # Accept either keypoint or bbox evidence
        is_upright = is_upright_keypoint or is_upright_bbox
        
        # Track recovery history
        self.recovery_history.append(1 if is_upright else 0)
        if len(self.recovery_history) > recovery_window:
            self.recovery_history.pop(0)
        
        # Check if height has recovered (returned to baseline)
        height_recovered = False
        if len(self.height_history) >= recovery_window:
            recent_max_height = max(self.height_history[-recovery_window:])
            baseline_height = np.median(self.height_history[:min(30, len(self.height_history))])
            if baseline_height > 0:
                height_recovered = (recent_max_height / baseline_height) > recovery_height_recover_ratio
        
        # IMPROVED: More responsive recovery detection
        # For FALL_CONFIRMED state: Require only 3-4 consecutive upright frames (0.2-0.3s at 15fps)
        # For CANDIDATE state: Require more sustained upright (original logic)
        if self.state == "FALL_CONFIRMED":
            # Quick recovery check: 3-4 consecutive upright frames
            if len(self.recovery_history) >= 4:
                recent_upright = self.recovery_history[-4:]
                if sum(recent_upright) >= 3 and height_recovered:
                    return True
        else:
            # Original logic for CANDIDATE state
            if len(self.recovery_history) >= recovery_window // 2:
                upright_count = sum(self.recovery_history[-recovery_window // 2:])
                if upright_count >= recovery_window // 3 and height_recovered:
                    return True
        
        return False
    
    def _check_motion_settled(self, features):
        """Check if motion has settled after lying down.
        
        True falls have rapid motion that settles. Slow transitions have minimal motion.
        
        Args:
            features: Current frame features
        
        Returns:
            settled: True if motion appears settled
        """
        motion_settle_window = self.config["motion_settle_window"]
        settle_dy_abs_thres = self.config["settle_dy_abs_thres"]
        
        # Check if recent dy values are small (settled)
        if len(self.dy_peak_history) >= motion_settle_window:
            recent_dy_peaks = self.dy_peak_history[-motion_settle_window:]
            avg_recent_dy = np.mean(recent_dy_peaks)
            return avg_recent_dy < settle_dy_abs_thres
        
        return False
    
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
            
            # Temporal discriminator: Recovery detection
            has_recovered = self._check_recovery(features)
            
            # Priority: Recovery gate (strong evidence of NOT a fall)
            if has_recovered:
                # Person recovered upright after lying, clearly not a fall
                self.state = "NORMAL"
                self.candidate_history = []
                self.score_accumulator *= 0.2  # Strong suppression
                return self.state, self.score_accumulator
            
            # Confirmation with dy_peak gate (optimized for high recall)
            if lying_count >= self.config["confirm_frames"]:
                # Multiple paths to confirmation for maximum recall
                dy_peak = features.get("dy_peak", 0.0)
                dy_peak_thres = self.config["dy_peak_thres"]
                height_drop_strong = self.config.get("height_drop_thres_strong", 0.3)
                height_drop_moderate = 0.22
                
                # Relax dy threshold moderately when height drop present
                effective_dy_thres = dy_peak_thres
                if height_drop >= height_drop_moderate:
                    effective_dy_thres *= 0.70  # Moderate relaxation for balance
                
                # Three-path confirmation for balanced recall+precision
                # Path 1: Fast motion with moderate threshold
                fast_motion = dy_peak >= effective_dy_thres
                
                # Path 2: Strong height drop + sustained lying
                strong_height = (height_drop >= height_drop_strong and 
                                lying_count >= self.config["confirm_frames"])
                
                # Path 3: Moderate height drop + very sustained lying (stricter)
                moderate_height = (height_drop >= height_drop_moderate and 
                                  lying_count >= self.config["confirm_frames"] + 3)
                
                can_confirm = fast_motion or strong_height or moderate_height
                
                if can_confirm:
                    self.state = "FALL_CONFIRMED"
                    self.fall_confirmed = True
                    self.score_accumulator = self.config["score_boost_confirmed"]
                # else: stay in CANDIDATE
            elif not is_candidate and len(self.confirm_history) >= self.config["confirm_window"]:
                # No longer candidate and window expired, go back to NORMAL
                if lying_count < self.config["confirm_frames"] - self.config["confirm_tolerance"]:
                    self.state = "NORMAL"
                    self.candidate_history = []
        
        elif self.state == "FALL_CONFIRMED":
            # FIX-4: Check for recovery (person stands up after brief fall alarm)
            has_recovered = self._check_recovery(features)
            
            if has_recovered:
                # Person recovered, cancel fall alarm immediately
                self.state = "NORMAL"
                self.fall_confirmed = False
                self.score_accumulator = 0.0  # Complete reset
                self.candidate_history = []
                self.confirm_history = []
                self.recovery_history = []  # Clear recovery history
                print("[RECOVERY] Person stood up, resetting to NORMAL state")
            else:
                # Stay confirmed and keep score at 1.0
                self.score_accumulator = 1.0
        
        # Clamp score
        self.score_accumulator = np.clip(self.score_accumulator, 0.0, 1.0)
        
        return self.state, self.score_accumulator
