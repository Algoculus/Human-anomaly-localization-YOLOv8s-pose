"""
State Machine Module

Determines the semantic state of the person (NORMAL, FALLING, LYING, etc.)
and manages fall alerts based on temporal transitions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from loguru import logger

from .scoring import FallScoreComponents
from config import Config, get_config

class State(Enum):
    NORMAL = "NORMAL"
    FALLING = "FALLING"
    LYING = "LYING" # Confirmed fall
    LYING_NO_FALL = "LYING_NO_FALL" # Resting/sleeping
    OCCLUDED = "OCCLUDED"

@dataclass
class StateMachineContext:
    """Historical context for state transitions."""
    current_state: State = State.NORMAL
    last_state: State = State.NORMAL
    
    # Timers (frames)
    frames_in_state: int = 0
    frames_since_drop: int = 9999
    frames_since_impact: int = 9999
    frames_prone: int = 0  # Stillness counter for T_hold requirement
    frames_since_alert: int = 9999  # Cooldown timer
    
    # Transition flags
    had_falling_transition: bool = False  # True if FALLING occurred before current prone
    
    # Flags
    has_alerted: bool = False
    
    # Buffer for recent max scores
    max_drop_score_window: float = 0.0
    max_impact_score_window: float = 0.0
    
    # Movement tracking
    recent_velocity: float = 0.0
    
    @property
    def behavior_label(self) -> str:
        """Get human-readable behavior label based on current state and context."""
        if self.current_state == State.LYING:
            return "FALLEN - ALERT!"
        elif self.current_state == State.FALLING:
            return "Falling..."
        elif self.current_state == State.LYING_NO_FALL:
            return "Resting/Sitting"
        elif self.current_state == State.OCCLUDED:
            return "Person Occluded"
        else:  # NORMAL
            if abs(self.recent_velocity) > 3.0:
                return "Walking"
            else:
                return "Standing"

class FallStateMachine:
    """
    Finite State Machine for Fall Detection.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.params = self.config.state
        self.fps = self.config.dataset.fps
        
        # Thresholds conversion to frames
        self.transition_window_frames = int(self.params.transition_window_sec * self.fps)
        self.t_hold_frames = int(self.params.t_hold_sec * self.fps)
        self.cooldown_frames = int(self.params.alert_cooldown_sec * self.fps)
        self.inactivity_alert_frames = int(self.params.inactivity_alert_sec * self.fps)
    
    def update(self, 
               scores: FallScoreComponents, 
               context: StateMachineContext) -> State:
        """
        Update state based on current scores and context.
        Implements: transition requirement, hysteresis, stillness window, cooldown.
        Returns the new state.
        """
        context.frames_in_state += 1
        context.frames_since_alert += 1
        
        # 1. Update Context Buffers (drop/impact tracking)
        if scores.sudden_drop_score > 0.35:
            context.frames_since_drop = 0
            context.max_drop_score_window = max(context.max_drop_score_window, scores.sudden_drop_score)
        else:
            context.frames_since_drop += 1
            
        if scores.impact_score > 0.4:
            context.frames_since_impact = 0
            context.max_impact_score_window = max(context.max_impact_score_window, scores.impact_score)
        else:
            context.frames_since_impact += 1
        
        # 2. Core indicators (Lowered for better frame-level recall)
        is_prone = scores.prone_score > 0.5  # Lowered from 0.6
        is_prone_strong = scores.prone_score > 0.6  # Lowered from 0.7
        is_score_high = scores.total_score > self.params.fall_alert_threshold
        is_score_low = scores.total_score < self.params.hysteresis_exit_threshold  # Hysteresis
        
        # Recent transition indicators
        was_recent_drop = context.frames_since_drop < self.transition_window_frames
        was_recent_impact = context.frames_since_impact < self.transition_window_frames
        has_transition_evidence = was_recent_drop or was_recent_impact
        
        # Stillness tracking
        is_still = abs(context.recent_velocity) < 2.0
        if is_prone and is_still:
            context.frames_prone += 1
        else:
            context.frames_prone = 0
        
        # 3. State Transitions
        new_state = context.current_state
        
        # --- NORMAL ---
        if context.current_state == State.NORMAL:
            # Reset transition flag when normal
            context.had_falling_transition = False
            
            # Check for fall indicators
            if scores.sudden_drop_score > 0.3 or scores.impact_score > 0.4:
                new_state = State.FALLING
                context.had_falling_transition = True
                
            elif is_score_high:
                if has_transition_evidence:
                    new_state = State.FALLING
                    context.had_falling_transition = True
                elif is_prone:
                    # Slow lie-down without transition -> Resting
                    new_state = State.LYING_NO_FALL
            
            elif is_prone_strong:
                if has_transition_evidence:
                    new_state = State.FALLING
                    context.had_falling_transition = True
                else:
                    new_state = State.LYING_NO_FALL
            
        # --- FALLING ---
        elif context.current_state == State.FALLING:
            context.had_falling_transition = True  # Mark that we had falling phase
            
            if is_prone:
                # TRANSITION REQUIREMENT: Only LYING (alert) if came from FALLING
                # Faster entry for better frame recall
                if context.frames_prone >= self.t_hold_frames // 3:  # Faster stillness check
                    new_state = State.LYING
                elif context.frames_in_state > self.fps * 0.5:  # Faster: 0.5s
                    new_state = State.LYING
                    
            elif is_score_low and context.frames_in_state > self.fps * 1.5:
                # Recovered from falling
                new_state = State.NORMAL
                context.had_falling_transition = False
        
        # --- LYING (Alert State) - Stay in this state longer for better recall ---
        elif context.current_state == State.LYING:
            # Stricter exit: require BOTH low score AND not prone for longer time
            if is_score_low and not is_prone and not is_prone_strong:
                # Require longer time standing up to confirm recovery
                if context.frames_in_state > self.fps * 2.0:  # 2 seconds to exit
                    new_state = State.NORMAL
                    context.had_falling_transition = False
        
        # --- LYING_NO_FALL (Resting) ---
        elif context.current_state == State.LYING_NO_FALL:
            # Escalate to LYING if sudden drop/impact detected while resting
            if has_transition_evidence:
                new_state = State.LYING
                context.had_falling_transition = True
            
            # INACTIVITY TIMER: If lying too long without getting up, escalate to alert
            # This catches "slow falls" where person can't get up
            elif context.frames_in_state > self.inactivity_alert_frames:
                new_state = State.LYING
                context.had_falling_transition = True  # Treat as fall (can't get up)
            
            elif not is_prone:
                new_state = State.NORMAL

        # 4. Transition Logic Clean-up
        if new_state != context.current_state:
            context.last_state = context.current_state
            context.current_state = new_state
            context.frames_in_state = 0
            
            # Alert logic with cooldown
            if new_state == State.LYING:
                # Only alert if: 1) had transition, 2) cooldown passed
                if context.had_falling_transition and context.frames_since_alert > self.cooldown_frames:
                    context.has_alerted = True
                    context.frames_since_alert = 0
        
        return new_state

if __name__ == "__main__":
    print("Testing State Machine...")
    fsm = FallStateMachine()
    ctx = StateMachineContext()
    
    # 1. Simulate Normal
    s_normal = FallScoreComponents(total_score=0.1)
    state = fsm.update(s_normal, ctx)
    print(f"Frame 1 (0.1): {state.value}")
    
    # 2. Simulate Sudden Drop (Falling)
    s_drop = FallScoreComponents(sudden_drop_score=0.8, total_score=0.6)
    state = fsm.update(s_drop, ctx)
    print(f"Frame 2 (Drop): {state.value}")
    
    # 3. Simulate Lying after Drop (Crash)
    s_lie = FallScoreComponents(prone_score=0.9, total_score=0.9)
    state = fsm.update(s_lie, ctx)
    print(f"Frame 3 (Lie): {state.value} - Alert: {ctx.has_alerted}")
    
    # 4. Simulate Recovery
    s_rec = FallScoreComponents(total_score=0.1, prone_score=0.1)
    # Fast forward
    for _ in range(30): fsm.update(s_rec, ctx)
    print(f"Frame 40 (Recovered): {ctx.current_state.value}")
