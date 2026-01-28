import numpy as np
from .rules import check_fall_candidate, check_lying_posture

class FallStateMachine:
    """
    State machine for temporal smoothing of fall detection with tolerance.
    
    States:
        - NORMAL: No fall detected
        - CANDIDATE: Potential fall detected, awaiting confirmation
        - FALL_CONFIRMED: Fall confirmed
    """
    
    def __init__(self, config):
        """
        Initialize state machine.
        
        Args:
            config: Configuration dict with thresholds
        """
        self.config = config
        self.state = "NORMAL"
        
        # =========================================================
        # STATE TRANSITION TRACKING
        # =========================================================
        self.candidate_history = []  # Binary history of candidate frames
        self.confirm_history = []    # Binary history of lying posture frames
        self.missing_streak = 0      # Consecutive frames with no detection
        
        # =========================================================
        # HEIGHT & SCORE TRACKING
        # =========================================================
        self.height_history = []     # Bbox heights for drop computation
        self.score_accumulator = 0.0 # EMA (Exponential Moving Average) score
        self.fall_confirmed = False
        self.confirmed_frame_count = 0  # Track how long we've been in CONFIRMED state
        
        # =========================================================
        # TEMPORAL DISCRIMINATORS
        # =========================================================
        self.recovery_history = []   # Track upright recovery frames
        self.dy_peak_history = []    # Track dy peaks for slow transition detection
        self.motion_settled = False
        self.post_confirm_lying_count = 0  # Track lying frames after confirmation
    
    def _compute_height_drop(self, current_height):
        """
        Compute normalized height drop.
        
        Height drop = (median_baseline - current) / median_baseline
        This detects when person collapses from standing to lying.
        
        Args:
            current_height: Current bbox height
        
        Returns:
            height_drop: Normalized height drop in [0, 1]
        """
        if current_height is None:
            return 0.0
        
        height_window = self.config["height_window"]
        
        # Add to rolling history
        self.height_history.append(current_height)
        
        # Keep only recent samples
        if len(self.height_history) > height_window:
            self.height_history = self.height_history[-height_window:]
        
        if len(self.height_history) < 2:
            return 0.0
        
        # Use median of history (excluding current) as baseline
        baseline_heights = self.height_history[:-1]
        median_height = np.median(baseline_heights)
        
        if median_height <= 0:
            return 0.0
        
        # Compute normalized drop, clamp to [0, 1]
        drop = (median_height - current_height) / median_height
        drop = np.clip(drop, 0.0, 1.0)
        
        return drop
    
    def _compute_frame_score(self, features, height_drop):
        """
        Compute a deterministic fall score for current frame.
        
        Score combines multiple indicators:
        - Body angle (30% weight)
        - Aspect ratio (25% weight)
        - Height drop (25% weight)
        - Vertical velocity (20% weight)
        
        Args:
            features: Frame features dict
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
        """
        Check if current lying posture is from slow transition (not a fall).
        
        Slow transitions (yoga, sleeping) have lower dy_peak values.
        
        Args:
            features: Current frame features
        
        Returns:
            is_slow: True if transition appears slow
        """
        dy_peak_thres = self.config["dy_peak_thres"]
        slow_lie_max_dy_peak = self.config["slow_lie_max_dy_peak"]
        
        # Track dy peaks over time
        self.dy_peak_history.append(features["dy_peak"])
        dy_long_window = self.config["dy_long_window"]
        if len(self.dy_peak_history) > dy_long_window:
            self.dy_peak_history.pop(0)
        
        # Check maximum dy_peak in short window
        if len(self.dy_peak_history) >= self.config["dy_short_window"]:
            max_dy_peak = max(self.dy_peak_history[-self.config["dy_short_window"]:])
            # If peak is below slow threshold, it's controlled motion
            return max_dy_peak < slow_lie_max_dy_peak
        
        return False
    
    def _is_controlled_lying(self, features, height_drop):
        """
        Detect if person is lying down in a controlled manner (not a fall).
        
        Controlled lying characteristics:
        - Slow, gradual descent (low dy_peak)
        - Maintained posture control (consistent AR changes)
        - No sudden impact markers
        
        Args:
            features: Current frame features
            height_drop: Current normalized height drop
            
        Returns:
            is_controlled: True if lying appears intentional/controlled
        """
        # Need enough history to assess motion pattern
        if len(self.dy_peak_history) < 5:
            return False
        
        recent_peaks = self.dy_peak_history[-5:]
        max_recent_peak = max(recent_peaks)
        avg_recent_peak = np.mean(recent_peaks)
        
        # Controlled lying: consistently low motion, no spikes
        # Falls typically have at least one high dy_peak (impact)
        # Tightened thresholds for better discrimination
        very_slow_descent = max_recent_peak < 10.0 and avg_recent_peak < 6.0
        
        # Also check if height drop is gradual (not sudden)
        gradual_drop = height_drop < 0.22
        
        # Additional check: if we have longer history, look for any spike
        had_impact_recently = False
        if len(self.dy_peak_history) >= 8:
            # Check if there was any significant peak in recent history
            for peak in self.dy_peak_history[-8:]:
                if peak > 15.0:  # Significant impact indicator
                    had_impact_recently = True
                    break
        
        return very_slow_descent and gradual_drop and not had_impact_recently
    
    def _check_recovery(self, features):
        """
        Check if person has recovered from lying (false positive suppression).
        
        If person briefly lies down then returns upright, it's likely not a fall.
        
        Args:
            features: Current frame features
        
        Returns:
            recovered: True if recovery detected
        """
        recovery_window = self.config["recovery_window"]
        recovery_upright_angle_thres = self.config["recovery_upright_angle_thres"]
        recovery_ar_thres = self.config["recovery_ar_thres"]
        recovery_height_recover_ratio = self.config["recovery_height_recover_ratio"]
        
        # Check upright posture via keypoints
        is_upright_keypoint = (
            features["body_angle_deg"] is not None and
            features["body_angle_deg"] < recovery_upright_angle_thres
        )
        
        # Check upright posture via bbox (fallback)
        is_upright_bbox = (
            features["bbox_aspect_ratio"] is not None and
            features["bbox_aspect_ratio"] < recovery_ar_thres
        )
        
        is_upright = is_upright_keypoint or is_upright_bbox
        
        # Track recovery history
        self.recovery_history.append(1 if is_upright else 0)
        if len(self.recovery_history) > recovery_window:
            self.recovery_history.pop(0)
        
        # Check if height has recovered to baseline
        height_recovered = False
        if len(self.height_history) >= recovery_window:
            recent_max_height = max(self.height_history[-recovery_window:])
            baseline_height = np.median(self.height_history[:min(30, len(self.height_history))])
            if baseline_height > 0:
                height_recovered = (recent_max_height / baseline_height) > recovery_height_recover_ratio
        
        # Different recovery criteria based on current state
        if self.state == "FALL_CONFIRMED":
            # Quick recovery: 3-4 consecutive upright frames
            if len(self.recovery_history) >= 4:
                recent_upright = self.recovery_history[-4:]
                if sum(recent_upright) >= 3 and height_recovered:
                    return True
        else:
            # Standard recovery for CANDIDATE state
            if len(self.recovery_history) >= recovery_window // 2:
                upright_count = sum(self.recovery_history[-recovery_window // 2:])
                if upright_count >= recovery_window // 3 and height_recovered:
                    return True
        
        return False
    
    def _check_motion_settled(self, features):
        """
        Check if motion has settled after lying down.
        
        True falls have rapid motion that settles.
        
        Args:
            features: Current frame features
        
        Returns:
            settled: True if motion appears settled
        """
        motion_settle_window = self.config["motion_settle_window"]
        settle_dy_abs_thres = self.config["settle_dy_abs_thres"]
        
        if len(self.dy_peak_history) >= motion_settle_window:
            recent_dy_peaks = self.dy_peak_history[-motion_settle_window:]
            avg_recent_dy = np.mean(recent_dy_peaks)
            return avg_recent_dy < settle_dy_abs_thres
        
        return False
    
    def update(self, features):
        """
        Update state machine with new frame features.
        
        State transitions:
        - NORMAL -> CANDIDATE: When fall indicators detected
        - CANDIDATE -> FALL_CONFIRMED: When lying posture sustained with impact
        - FALL_CONFIRMED -> NORMAL: When recovery detected
        
        Args:
            features: Frame features dict from compute_frame_features
        
        Returns:
            state: Current state string (NORMAL, CANDIDATE, FALL_CONFIRMED)
            score: Current fall score in [0, 1]
        """
        # =========================================================
        # HANDLE MISSING DETECTIONS
        # =========================================================
        if features["bbox"] is None:
            self.missing_streak += 1
            if self.missing_streak > self.config["max_missing_streak"]:
                # Too many missing frames - decay the score
                self.score_accumulator *= 0.5
            return self.state, self.score_accumulator
        else:
            self.missing_streak = 0
        
        # =========================================================
        # COMPUTE FRAME-LEVEL METRICS
        # =========================================================
        height_drop = self._compute_height_drop(features["height"])
        frame_score = self._compute_frame_score(features, height_drop)
        
        # Update score using exponential moving average
        decay = self.config["score_decay"]
        self.score_accumulator = decay * self.score_accumulator + (1 - decay) * frame_score
        
        # =========================================================
        # CHECK FALL CANDIDATE STATUS
        # =========================================================
        is_candidate = check_fall_candidate(
            features,
            self.config["angle_thres"],
            self.config["ar_thres"],
            height_drop,
            self.config["height_drop_thres"],
            self.config["dy_fall_thres"],
            self.config.get("impact_dy_thres", 4.0),
            self.config.get("high_angle_thres", 60.0),
            self.config.get("sustained_lying_thres", 50.0)
        )
        
        # Check lying posture (relaxed thresholds for confirmation)
        is_lying = check_lying_posture(
            features,
            self.config["confirm_angle_thres"],
            self.config["confirm_ar_thres"]
        )
        
        # =========================================================
        # STATE MACHINE TRANSITIONS
        # =========================================================
        if self.state == "NORMAL":
            # Track candidate frames in sliding window
            self.candidate_history.append(1 if is_candidate else 0)
            if len(self.candidate_history) > self.config["cand_enter_frames"] + self.config["cand_tolerance"]:
                self.candidate_history.pop(0)
            
            # Transition to CANDIDATE if enough candidate frames
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
            
            lying_count = sum(self.confirm_history)
            
            # Check for recovery (person stood up)
            has_recovered = self._check_recovery(features)
            
            # Check for controlled lying (intentional lying down, not a fall)
            is_controlled = self._is_controlled_lying(features, height_drop)
            
            if has_recovered:
                # Recovery detected - reset to NORMAL
                self.state = "NORMAL"
                self.candidate_history = []
                self.score_accumulator *= 0.2
                return self.state, self.score_accumulator
            
            # =========================================================
            # MULTI-PATH CONFIRMATION LOGIC
            # More paths = better recall for different fall types
            # =========================================================
            if lying_count >= self.config["confirm_frames"]:
                dy_peak = features.get("dy_peak", 0.0)
                dy_peak_thres = self.config["dy_peak_thres"]
                height_drop_strong = self.config.get("height_drop_thres_strong", 0.25)
                height_drop_moderate = 0.18
                height_drop_light = 0.12
                
                # Relax dy threshold based on height drop level
                effective_dy_thres = dy_peak_thres
                if height_drop >= height_drop_moderate:
                    effective_dy_thres *= 0.60  # More relaxed
                elif height_drop >= height_drop_light:
                    effective_dy_thres *= 0.75
                
                # PATH 1: Fast motion with impact
                fast_motion = dy_peak >= effective_dy_thres
                
                # PATH 2: Strong height drop + sustained lying
                strong_height = (height_drop >= height_drop_strong and 
                                lying_count >= self.config["confirm_frames"])
                
                # PATH 3: Moderate height drop + sustained lying
                moderate_height = (height_drop >= height_drop_moderate and 
                                  lying_count >= self.config["confirm_frames"] + 1)
                
                # PATH 4: Light height drop + very sustained lying
                light_height = (height_drop >= height_drop_light and
                               lying_count >= self.config["confirm_frames"] + 3)
                
                # PATH 5: Posture-only confirmation (high angle + high AR sustained)
                posture_only = False
                if features["feature_valid"] and features["body_angle_deg"] is not None:
                    if (features["body_angle_deg"] >= 55.0 and 
                        features["bbox_aspect_ratio"] >= 1.4 and
                        lying_count >= self.config["confirm_frames"] + 2):
                        posture_only = True
                
                # PATH 6: AR-dominant confirmation (very wide bbox sustained)
                ar_dominant = (features["bbox_aspect_ratio"] >= 1.7 and
                              lying_count >= self.config["confirm_frames"] + 1)
                
                can_confirm = (fast_motion or strong_height or moderate_height or 
                              light_height or posture_only or ar_dominant)
                
                # VETO: Block confirmation if this looks like controlled lying
                # But only if there's no strong evidence of a fall
                # Stricter veto: apply unless there's very strong evidence
                if is_controlled and not (fast_motion or (strong_height and dy_peak >= 4.0)):
                    can_confirm = False
                
                if can_confirm:
                    self.state = "FALL_CONFIRMED"
                    self.fall_confirmed = True
                    self.score_accumulator = self.config["score_boost_confirmed"]
            
            elif not is_candidate and len(self.confirm_history) >= self.config["confirm_window"]:
                # Confirmation window expired without confirmation
                if lying_count < self.config["confirm_frames"] - self.config["confirm_tolerance"]:
                    self.state = "NORMAL"
                    self.candidate_history = []
        
        elif self.state == "FALL_CONFIRMED":
            self.confirmed_frame_count += 1
            
            # Check if person is still lying
            if is_lying:
                self.post_confirm_lying_count += 1
            
            # Check for recovery (person stands up after fall alarm)
            has_recovered = self._check_recovery(features)
            
            # Early recovery check: if person stands up very quickly after confirmation,
            # it might have been a false positive (controlled lie-down-stand-up)
            early_recovery = False
            if self.confirmed_frame_count <= 20:  # Within first 20 frames of confirmation (relaxed from 15)
                # If not lying in most of these frames, likely a quick stand-up
                # Stricter: require lying in at least 60% of frames
                if self.post_confirm_lying_count < self.confirmed_frame_count * 0.6:
                    early_recovery = has_recovered
            
            if has_recovered or early_recovery:
                # Recovery - reset everything
                self.state = "NORMAL"
                self.fall_confirmed = False
                self.score_accumulator = 0.0
                self.candidate_history = []
                self.confirm_history = []
                self.recovery_history = []
                self.confirmed_frame_count = 0
                self.post_confirm_lying_count = 0
                print("RECOVERY DETECTED: Person stood up, resetting to NORMAL state")
            else:
                # Maintain confirmed state
                self.score_accumulator = 1.0
        
        # Clamp score to valid range
        self.score_accumulator = np.clip(self.score_accumulator, 0.0, 1.0)
        
        return self.state, self.score_accumulator
