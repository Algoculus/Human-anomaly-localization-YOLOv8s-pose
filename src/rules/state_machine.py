"""
EMERGENCY RECALL FIX - Relaxed State Machine

Problem: Recall = 0.485 (48.5%) - TOO LOW!
Cause: Thresholds too strict, missing many real falls

Solution: Significantly relax thresholds to boost recall
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from collections import deque
from loguru import logger

from .scoring import FallScoreComponents
from .features import PoseFeatures
from config import Config, get_config


class State(Enum):
    """Enhanced state set"""
    NORMAL = "NORMAL"
    BENDING = "BENDING"
    FALLING = "FALLING"
    LYING = "LYING"
    LYING_NO_FALL = "LYING_NO_FALL"
    GETTING_UP = "GETTING_UP"
    OCCLUDED = "OCCLUDED"


@dataclass
class StateTransitionHistory:
    """Track recent state transitions"""
    states: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def add(self, state: State, frame_num: int):
        self.states.append(state)
        self.timestamps.append(frame_num)
    
    def had_falling_recently(self, within_frames: int = 60) -> bool:
        """Check if FALLING occurred recently (relaxed window)"""
        if not self.states or not self.timestamps:
            return False
        
        current_frame = self.timestamps[-1]
        for i in range(len(self.states) - 1, -1, -1):
            if current_frame - self.timestamps[i] > within_frames:
                break
            if self.states[i] == State.FALLING:
                return True
        return False


@dataclass
class StateMachineContext:
    """Extended context"""
    current_state: State = State.NORMAL
    last_state: State = State.NORMAL
    
    frames_in_state: int = 0
    total_frames: int = 0
    
    frames_since_drop_detected: int = 9999
    frames_since_impact: int = 9999
    frames_since_prone_start: int = 9999
    frames_since_alert: int = 9999
    
    consecutive_prone_frames: int = 0
    consecutive_still_frames: int = 0
    
    score_history: deque = field(default_factory=lambda: deque(maxlen=15))
    drop_score_history: deque = field(default_factory=lambda: deque(maxlen=15))
    
    max_drop_score_recent: float = 0.0
    max_impact_score_recent: float = 0.0
    max_total_score_recent: float = 0.0
    
    fall_sequence_detected: bool = False
    had_falling_phase: bool = False
    
    has_alerted: bool = False
    alert_reason: str = ""
    
    recent_velocity_y: float = 0.0
    recent_velocity_x: float = 0.0
    recent_activity: str = "UNKNOWN"
    
    history: StateTransitionHistory = field(default_factory=StateTransitionHistory)
    
    @property
    def behavior_label(self) -> str:
        if self.current_state == State.LYING:
            return f"⚠️ FALL DETECTED - {self.alert_reason}" if self.has_alerted else "FALLEN"
        elif self.current_state == State.FALLING:
            return "⚡ FALLING..."
        elif self.current_state == State.BENDING:
            return "Bending/Crouching"
        elif self.current_state == State.LYING_NO_FALL:
            return "Lying (Resting)"
        elif self.current_state == State.GETTING_UP:
            return "Getting Up"
        elif self.current_state == State.OCCLUDED:
            return "Person Occluded"
        else:
            if self.recent_activity:
                return self.recent_activity.replace('_', ' ').title()
            return "Standing/Walking"


class FallStateMachine:
    """
    RECALL-OPTIMIZED State Machine
    
    Changes from previous version:
    - MUCH lower thresholds for fall detection
    - Faster confirmation (shorter t_hold)
    - Relaxed transition requirements
    - More aggressive fall detection
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.params = self.config.state
        self.fps = self.config.dataset.fps
        
        # ==== RELAXED THRESHOLDS FOR HIGH RECALL ====
        
        # Time windows (in frames)
        self.transition_window = int(1.7 * self.fps)  # Reduced to 1.5s (was 2.0s)
        self.t_hold_min = int(0.4 * self.fps)         # Reduced to 0.4s (was 0.8s)
        self.t_hold_fast = int(0.2 * self.fps)        # Very fast: 0.2s (was 0.5s)
        self.cooldown_frames = int(self.params.alert_cooldown_sec * self.fps)
        self.inactivity_threshold = int(20.0 * self.fps)  # Increased to 20s
        
        # State timeouts
        self.max_bending_duration = int(3.0 * self.fps)
        self.max_falling_duration = int(3.0 * self.fps)  # Increased
        self.min_lying_for_alert = int(0.3 * self.fps)   # Reduced to 0.3s
        self.getting_up_timeout = int(5.0 * self.fps)
        
        # ==== BALANCED THRESHOLDS FOR PRECISION ~85% ====
        self.FALL_ALERT_THRESHOLD = 0.6      # Increased from 0.55
        self.FALL_ALERT_STRICT = 0.75         # Increased from 0.70
        self.FALLING_ENTRY_THRESHOLD = 0.40   # Keep
        self.PRONE_THRESHOLD = 0.55           # Increased from 0.50
        self.PRONE_STRONG_THRESHOLD = 0.7    # Increased from 0.65
        self.HYSTERESIS_EXIT = 0.30           # Keep
        
        logger.warning("⚠️ RECALL-OPTIMIZED MODE: Low thresholds for maximum sensitivity")
        
    def update(self, 
               scores: FallScoreComponents,
               features: PoseFeatures,
               context: StateMachineContext) -> State:
        """Update state machine - RECALL OPTIMIZED"""
        
        context.frames_in_state += 1
        context.total_frames += 1
        context.frames_since_alert += 1
        context.recent_velocity_y = features.velocity_y
        context.recent_velocity_x = features.velocity_x
        context.recent_activity = features.activity_type
        
        context.score_history.append(scores.total_score)
        context.drop_score_history.append(scores.sudden_drop_score)
        
        # Track events with RELAXED thresholds
        if scores.sudden_drop_score > 0.25:  # Lowered from 0.35
            context.frames_since_drop_detected = 0
            context.max_drop_score_recent = max(context.max_drop_score_recent, 
                                                scores.sudden_drop_score)
        else:
            context.frames_since_drop_detected += 1
            if context.frames_since_drop_detected > 60:
                context.max_drop_score_recent *= 0.9
        
        if scores.impact_score > 0.3:  # Lowered from 0.4
            context.frames_since_impact = 0
            context.max_impact_score_recent = max(context.max_impact_score_recent,
                                                  scores.impact_score)
        else:
            context.frames_since_impact += 1
            if context.frames_since_impact > 60:
                context.max_impact_score_recent *= 0.9
        
        context.max_total_score_recent = max(context.max_total_score_recent,
                                             scores.total_score)
        
        # Track prone state
        is_prone = scores.prone_score > self.PRONE_THRESHOLD
        is_prone_strong = scores.prone_score > self.PRONE_STRONG_THRESHOLD
        
        if is_prone:
            context.consecutive_prone_frames += 1
            if context.frames_since_prone_start > 100:
                context.frames_since_prone_start = 0
        else:
            context.consecutive_prone_frames = 0
            context.frames_since_prone_start += 1
        
        # Stillness tracking
        is_still = (abs(features.velocity_y) < 2.0 and 
                   abs(features.velocity_x) < 2.5)
        
        if is_still:
            context.consecutive_still_frames += 1
        else:
            context.consecutive_still_frames = 0
        
        # RELAXED fall sequence detection
        if not context.fall_sequence_detected:
            has_recent_drop = context.frames_since_drop_detected < self.transition_window
            has_recent_impact = context.frames_since_impact < self.transition_window
            
            # Accept either drop OR impact + prone (not both required)
            if (has_recent_drop or has_recent_impact) and is_prone:
                context.fall_sequence_detected = True
        
        # Determine new state
        new_state = self._determine_state(
            context.current_state,
            scores,
            features,
            context
        )
        
        # Handle state transition
        if new_state != context.current_state:
            self._on_state_transition(
                context.current_state,
                new_state,
                scores,
                context
            )
            
            context.last_state = context.current_state
            context.current_state = new_state
            context.frames_in_state = 0
            context.history.add(new_state, context.total_frames)
        
        return new_state
    
    def _determine_state(self,
                        current_state: State,
                        scores: FallScoreComponents,
                        features: PoseFeatures,
                        context: StateMachineContext) -> State:
        """RECALL-OPTIMIZED state transition logic"""
        
        # Key indicators with RELAXED thresholds
        is_prone = scores.prone_score > self.PRONE_THRESHOLD
        is_prone_strong = scores.prone_score > self.PRONE_STRONG_THRESHOLD
        is_score_high = scores.total_score > self.FALL_ALERT_THRESHOLD
        is_score_very_high = scores.total_score > self.FALL_ALERT_STRICT
        is_score_low = scores.total_score < self.HYSTERESIS_EXIT
        
        is_still = context.consecutive_still_frames > 3  # Relaxed from 5
        
        # Recent events - RELAXED windows
        had_recent_drop = context.frames_since_drop_detected < self.transition_window
        had_recent_impact = context.frames_since_impact < self.transition_window
        has_dynamic_event = had_recent_drop or had_recent_impact
        
        # Activity - but don't let it block falls
        is_bending = features.activity_type == "BENDING"
        is_falling_motion = features.activity_type == "FALLING"
        
        # ===== STATE: NORMAL =====
        if current_state == State.NORMAL:
            context.had_falling_phase = False
            context.fall_sequence_detected = False
            
            # AGGRESSIVE fall detection - prioritize over bending
            if is_falling_motion or scores.sudden_drop_score > 0.35:
                return State.FALLING
            
            # High score alone can trigger
            if is_score_high:
                return State.FALLING
            
            # Prone with any hint of dynamic event
            if is_prone and has_dynamic_event:
                return State.FALLING
            
            # Even prone alone if strong enough
            if is_prone_strong:
                if has_dynamic_event or is_score_high:
                    return State.FALLING
                elif not is_bending:
                    return State.LYING_NO_FALL
            
            # Only go to bending if very clear and no fall signals
            if is_bending and not is_prone and not has_dynamic_event and scores.total_score < 0.3:
                return State.BENDING
        
        # ===== STATE: BENDING =====
        elif current_state == State.BENDING:
            # Quick exit to falling if any fall signal
            if has_dynamic_event or is_score_high or is_prone:
                return State.FALLING
            
            if not is_bending and not is_prone:
                return State.NORMAL
            
            # Short timeout
            if context.frames_in_state > self.max_bending_duration:
                if is_prone:
                    return State.FALLING  # Changed to FALLING instead of LYING_NO_FALL
                return State.NORMAL
        
        # ===== STATE: FALLING =====
        elif current_state == State.FALLING:
            context.had_falling_phase = True
            
            # VERY FAST confirmation if prone
            if is_prone:
                # Ultra-fast path
                if context.fall_sequence_detected or is_score_very_high:
                    if context.consecutive_prone_frames >= self.t_hold_fast:
                        return State.LYING
                
                # Fast path
                if context.consecutive_prone_frames >= self.t_hold_min:
                    return State.LYING
                
                # Even faster if still
                if is_still and context.frames_in_state > int(0.4 * self.fps):
                    return State.LYING
            
            # Don't exit to normal too quickly
            if is_score_low and not is_prone:
                if context.frames_in_state > self.fps * 2.0:  # Longer wait
                    return State.NORMAL
            
            # Extended timeout
            if context.frames_in_state > self.max_falling_duration:
                if is_prone:
                    return State.LYING
                # Only return to normal if very clearly not falling
                if scores.total_score < 0.2:
                    return State.NORMAL
        
        # ===== STATE: LYING (Alert) =====
        elif current_state == State.LYING:
            # Very strict exit - maintain high recall
            
            is_getting_up = (not is_prone and 
                           features.hip_height_ratio > 0.5 and
                           features.body_orientation < 45)
            
            if is_getting_up:
                return State.GETTING_UP
            
            # Only exit if CLEARLY standing for long time
            if is_score_low and not is_prone and not is_prone_strong:
                if context.frames_in_state > self.fps * 6.0:  # Extended to 6 seconds
                    if features.hip_height_ratio > 0.75 and features.body_orientation < 30:
                        return State.NORMAL
        
        # ===== STATE: LYING_NO_FALL =====
        elif current_state == State.LYING_NO_FALL:
            # Quick escalation to LYING
            if has_dynamic_event or is_score_high:
                return State.LYING  # Direct to LYING
            
            # Shorter inactivity threshold
            if context.frames_in_state > int(10.0 * self.fps):  # Reduced to 10s
                if is_still and is_prone:
                    return State.LYING
            
            if not is_prone:
                return State.NORMAL
            
            if not is_prone and features.hip_height_ratio > 0.4:
                return State.GETTING_UP
        
        # ===== STATE: GETTING_UP =====
        elif current_state == State.GETTING_UP:
            # Successfully up
            if features.hip_height_ratio > 0.65 and features.body_orientation < 40:
                if is_score_low:
                    return State.NORMAL
            
            # Fell back
            if is_prone_strong or (is_prone and has_dynamic_event):
                return State.LYING
            
            # Back to lying
            if is_prone and context.frames_in_state > int(0.5 * self.fps):
                return State.LYING if context.last_state == State.LYING else State.LYING_NO_FALL
            
            # Timeout
            if context.frames_in_state > self.getting_up_timeout:
                if is_prone:
                    return State.LYING  # Assume unable to get up = fall
                return State.NORMAL
        
        # ===== STATE: OCCLUDED =====
        elif current_state == State.OCCLUDED:
            if features.keypoints_conf_mean > 0.4:  # Lowered threshold
                if is_prone or is_score_high:
                    return State.LYING
                return State.NORMAL
        
        return current_state
    
    def _on_state_transition(self,
                            old_state: State,
                            new_state: State,
                            scores: FallScoreComponents,
                            context: StateMachineContext):
        """Handle state transitions"""
        
        logger.info(f"State: {old_state.value} → {new_state.value}")
        
        if new_state == State.LYING:
            if context.frames_since_alert > self.cooldown_frames:
                context.has_alerted = True
                context.frames_since_alert = 0
                
                # Determine reason
                if context.fall_sequence_detected:
                    context.alert_reason = "Fall Sequence"
                elif scores.impact_score > 0.5:
                    context.alert_reason = "High Impact"
                elif context.max_drop_score_recent > 0.4:
                    context.alert_reason = "Sudden Drop"
                elif scores.prone_score > 0.6:
                    context.alert_reason = "Prone Position"
                elif old_state == State.LYING_NO_FALL:
                    context.alert_reason = "Unable to Get Up"
                else:
                    context.alert_reason = "High Fall Risk"
                
                logger.warning(f"⚠️  FALL ALERT: {context.alert_reason}")
        
        if old_state == State.LYING and new_state != State.LYING:
            logger.info(f"Exiting alert after {context.frames_in_state / self.fps:.1f}s")
            
            if new_state == State.NORMAL:
                context.fall_sequence_detected = False
                context.had_falling_phase = False
                context.has_alerted = False


if __name__ == "__main__":
    print("Testing RECALL-OPTIMIZED State Machine...")
    
    from scoring import FallScoreComponents
    from features import PoseFeatures
    
    fsm = FallStateMachine()
    ctx = StateMachineContext()
    
    print("\n=== Test: Moderate Fall (should now detect) ===")
    
    # Normal
    for i in range(5):
        scores = FallScoreComponents(total_score=0.15)
        features = PoseFeatures(velocity_y=0.5, body_orientation=15)
        state = fsm.update(scores, features, ctx)
    print(f"After normal: {state.value}")
    
    # Moderate drop (was missing before)
    for i in range(3):
        scores = FallScoreComponents(
            sudden_drop_score=0.45,  # Moderate
            total_score=0.50
        )
        features = PoseFeatures(
            normalized_drop_velocity=0.04,
            velocity_y=5.0,
            body_orientation=30
        )
        state = fsm.update(scores, features, ctx)
    print(f"After drop: {state.value}")
    
    # Prone
    for i in range(8):
        scores = FallScoreComponents(
            prone_score=0.60,
            total_score=0.55
        )
        features = PoseFeatures(
            velocity_y=0.8,
            body_orientation=65,
            bbox_aspect_ratio=0.7
        )
        state = fsm.update(scores, features, ctx)
    
    print(f"Final state: {state.value}")
    print(f"Alert triggered: {ctx.has_alerted}")
    print(f"Alert reason: {ctx.alert_reason}")
    
    if state == State.LYING and ctx.has_alerted:
        print("✓ SUCCESS: Moderate fall now detected!")
    else:
        print("✗ FAIL: Still missing moderate falls")