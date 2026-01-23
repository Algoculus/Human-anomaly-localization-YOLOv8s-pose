"""
RECALL-OPTIMIZED Fall Scoring

Changes:
- Lower thresholds for drop/impact detection
- Reduced activity penalties
- More generous prone scoring
- Prioritize recall over precision
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from loguru import logger

from .features import PoseFeatures
from config import Config, get_config, FallScoreConfig


@dataclass
class FallScoreComponents:
    """Score components"""
    sudden_drop_score: float = 0.0
    prone_score: float = 0.0
    impact_score: float = 0.0
    sustained_lying_score: float = 0.0
    activity_penalty: float = 0.0
    confidence_factor: float = 1.0
    total_score: float = 0.0
    triggering_factors: List[str] = None
    
    def __post_init__(self):
        if self.triggering_factors is None:
            self.triggering_factors = []


def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


class FallScorer:
    """RECALL-OPTIMIZED Fall Scorer"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.params: FallScoreConfig = self.config.fall_score
        
        # ==== BALANCED THRESHOLDS FOR PRECISION ~85% ====
        self.FAST_DROP_THRESHOLD = 0.045      # Keep
        self.MEDIUM_DROP_THRESHOLD = 0.025    # Keep
        self.SLOW_DROP_THRESHOLD = 0.015      # Keep
        
        self.HIGH_IMPACT_THRESHOLD = 2.5      # Keep
        self.MEDIUM_IMPACT_THRESHOLD = 2.0    # Keep
        
        self.LYING_ORIENTATION = 60           # Increased from 58
        self.PRONE_ASPECT_RATIO = 0.80        # Decreased from 0.82
        
        logger.warning("⚠️ RECALL MODE: Using relaxed thresholds")
        
    def calculate(self, features: PoseFeatures) -> FallScoreComponents:
        """Calculate fall scores - RECALL OPTIMIZED"""
        
        components = FallScoreComponents()
        
        # Assess data quality
        has_accelerometer = features.acc_sv_total is not None
        has_depth = features.depth_hw_ratio is not None
        has_good_pose = features.keypoints_conf_mean > 0.4  # Lowered from 0.5
        
        components.confidence_factor = self._calculate_confidence(
            features.keypoints_conf_mean,
            has_accelerometer,
            has_depth
        )
        
        # === Score Components ===
        components.sudden_drop_score = self._score_sudden_drop(features)
        components.impact_score = self._score_impact(features)
        components.prone_score = self._score_prone_position(features)
        components.sustained_lying_score = self._score_sustained_lying(features)
        
        # REDUCED activity penalty for recall
        components.activity_penalty = self._calculate_activity_penalty(features) * 0.5
        
        # === Combine with recall-friendly weights ===
        components.total_score = self._combine_scores(
            components,
            features,
            has_accelerometer,
            has_depth,
            has_good_pose
        )
        
        components.triggering_factors = self._identify_triggers(components, features)
        
        return components
    
    def _calculate_confidence(self, pose_conf: float, has_acc: bool, has_depth: bool) -> float:
        """Calculate confidence - more lenient"""
        confidence = max(pose_conf, 0.5)  # Minimum 0.5
        
        if has_acc:
            confidence = min(1.0, confidence * 1.1)
        if has_depth:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence
    
    def _score_sudden_drop(self, features: PoseFeatures) -> float:
        """Score drop - MORE SENSITIVE"""
        
        score = 0.0
        
        if features.normalized_drop_velocity > 0:
            vel = features.normalized_drop_velocity
            acc = features.acceleration_y if features.acceleration_y > 0 else 0
            
            # More generous thresholds
            if vel >= self.FAST_DROP_THRESHOLD:
                score = min(1.0, sigmoid(vel, k=40.0, x0=0.035))
                if acc > 1.2:
                    score = min(1.0, score * 1.25)
                    
            elif vel >= self.MEDIUM_DROP_THRESHOLD:
                score = sigmoid(vel, k=25.0, x0=0.025)
                if acc > 0.8:
                    score *= 1.2
                elif acc < 0.2:
                    score *= 0.8  # Less penalty
                    
            elif vel >= self.SLOW_DROP_THRESHOLD:
                score = sigmoid(vel, k=18.0, x0=0.015) * 0.6  # Increased from 0.5
            else:
                # Still give some score for very slow drops
                score = vel / self.SLOW_DROP_THRESHOLD * 0.15
        
        # Height drop check
        if features.normalized_height < 0.35 and features.velocity_y > 2.5:
            score = max(score, 0.5)
        
        return min(score, 1.0)
    
    def _score_impact(self, features: PoseFeatures) -> float:
        """Score impact - MORE SENSITIVE"""
        
        if features.acc_sv_total is None:
            return 0.0
        
        sv = features.acc_sv_total
        
        # Lowered thresholds
        if sv >= self.HIGH_IMPACT_THRESHOLD:
            score = min(1.0, sigmoid(sv, k=3.0, x0=2.0))
        elif sv >= self.MEDIUM_IMPACT_THRESHOLD:
            score = sigmoid(sv, k=5.0, x0=1.9)
            if features.is_lying_pose:
                score *= 1.3
        elif sv >= 1.3:  # Lowered from 1.5
            score = 0.35 * sigmoid(sv, k=4.0, x0=1.5)
        else:
            score = 0.0
        
        # Delta bonus
        if features.acc_delta is not None and features.acc_delta > 0.8:
            score = min(1.0, score * 1.2)
        
        return min(score, 1.0)
    
    def _score_prone_position(self, features: PoseFeatures) -> float:
        """Score prone - MORE GENEROUS"""
        
        signals = []
        
        # === Depth signals ===
        if features.depth_hw_ratio is not None:
            if features.depth_hw_ratio < 0.75:  # Relaxed from 0.65
                s = 1.0 - sigmoid(features.depth_hw_ratio, k=6.0, x0=0.85)
                signals.append(('depth_hw', s, 0.45, True))
            else:
                s = 0.35 * (1.0 - sigmoid(features.depth_hw_ratio, k=5.0, x0=1.1))
                signals.append(('depth_hw', s, 0.30, True))
        
        if features.p40 is not None:
            if features.p40 > 0.25:  # Lowered from 0.30
                s = sigmoid(features.p40, k=7.0, x0=0.30)
                signals.append(('floor_proximity', s, 0.40, True))
        
        if features.dist_to_floor_mm is not None:
            if features.dist_to_floor_mm < 550:  # Increased from 500
                s = 1.0 - sigmoid(features.dist_to_floor_mm, k=0.008, x0=400)
                signals.append(('floor_distance', s, 0.40, True))
            elif features.dist_to_floor_mm > 650:  # Increased threshold
                signals.append(('furniture_penalty', -0.3, 1.0, True))
        
        # === Pose signals - MORE GENEROUS ===
        if features.keypoints_conf_mean > 0.4:  # Lowered from 0.5
            # Orientation
            if features.body_orientation > 45:  # Lowered from 50
                s = sigmoid(features.body_orientation, k=0.10, x0=50)
                signals.append(('orientation', s, 0.35, False))
            else:
                s = max(0, 0.25 - features.body_orientation / 120)
                signals.append(('orientation', s, 0.20, False))
            
            # Aspect ratio
            ar = features.bbox_aspect_ratio
            if ar < 0.95:  # Relaxed from 0.85
                s = 1.0 - sigmoid(ar, k=6.0, x0=1.0)
                signals.append(('aspect_ratio', s, 0.30, False))
            
            # Hip height
            if features.hip_height_ratio < 0.40:  # Increased from 0.35
                s = 1.0 - sigmoid(features.hip_height_ratio, k=12.0, x0=0.30)
                signals.append(('hip_height', s, 0.35, False))
            elif features.hip_height_ratio < 0.60 and features.body_orientation < 40:
                signals.append(('bending_penalty', -0.2, 0.4, False))  # Reduced penalty
        
        # === Combine - favor highest signals ===
        if not signals:
            return 0.0
        
        positive_signals = [(name, s, w) for name, s, w, _ in signals if s > 0]
        penalties = sum([s for name, s, w, _ in signals if s < 0])
        
        if not positive_signals:
            return max(0.0, penalties)
        
        # Take top 3 signals instead of top 2
        positive_signals.sort(key=lambda x: x[1] * x[2], reverse=True)
        top_signals = positive_signals[:3]
        total_weight = sum(w for _, _, w in top_signals)
        score = sum(s * w for _, s, w in top_signals) / total_weight if total_weight > 0 else 0
        
        # Reduced penalty impact
        final_score = max(0.0, score + penalties * 0.7)
        
        return min(final_score, 1.0)
    
    def _score_sustained_lying(self, features: PoseFeatures) -> float:
        """Score sustained lying - RELAXED"""
        
        score = 0.0
        
        if features.is_lying_pose or features.body_orientation > 50:
            # Relaxed stillness check
            is_still_vertical = abs(features.velocity_y) < 2.2
            is_still_horizontal = abs(features.velocity_x) < 2.5
            is_orientation_stable = abs(features.orientation_velocity) < 3.5
            
            stillness_score = (
                0.4 * (1.0 if is_still_vertical else 0.4) +
                0.3 * (1.0 if is_still_horizontal else 0.4) +
                0.3 * (1.0 if is_orientation_stable else 0.3)
            )
            
            pose_stable = features.pose_stability > 0.65  # Lowered from 0.75
            
            if stillness_score > 0.65 and pose_stable:
                score = 1.0
            elif stillness_score > 0.50:
                score = 0.75
            elif stillness_score > 0.35:
                score = 0.5
            else:
                score = 0.3
        
        return score
    
    def _calculate_activity_penalty(self, features: PoseFeatures) -> float:
        """Activity penalty - MUCH REDUCED for recall"""
        
        penalty = 0.0
        activity = features.activity_type
        
        # Only penalize very clear bending
        if activity == "BENDING":
            if features.knee_bend_angle < 110 and abs(features.velocity_y) < 1.5:
                penalty = -0.25  # Reduced from -0.4
        
        # Very slow intentional lying
        elif activity == "LYING_STILL":
            if abs(features.velocity_y) < 1.0 and abs(features.orientation_velocity) < 3.0:
                penalty = -0.15  # Reduced from -0.25
        
        # Clear walking only
        elif activity == "WALKING":
            if not features.is_lying_pose and features.hip_height_ratio > 0.6:
                penalty = -0.3  # Reduced from -0.5
        
        # Sitting - reduced penalty
        if 0.35 < features.hip_height_ratio < 0.65 and features.pose_stability > 0.85:
            if abs(features.velocity_y) < 1.0:
                penalty = min(penalty, -0.2)  # Reduced from -0.3
        
        return penalty
    
    def _combine_scores(self, 
                       components: FallScoreComponents,
                       features: PoseFeatures,
                       has_acc: bool,
                       has_depth: bool,
                       has_good_pose: bool) -> float:
        """Combine scores - RECALL OPTIMIZED"""
        
        w_drop = self.params.weight_sudden_drop
        w_prone = self.params.weight_prone
        w_impact = self.params.weight_impact
        w_sustained = self.params.weight_sustained_lying
        
        # === Redistribute weights for recall ===
        if not has_acc:
            w_prone += w_impact * 0.6
            w_sustained += w_impact * 0.25
            w_drop += w_impact * 0.15
            w_impact = 0.0
        
        # Boost prone weight more aggressively
        if components.impact_score > 0.5:
            w_prone = 0.60
            w_impact = 0.25
            w_sustained = 0.10
            w_drop = 0.05
            
            if components.prone_score > 0.35:  # Lowered from 0.5
                components.prone_score = min(1.0, components.prone_score * 1.25)
        
        # Boost if sequence detected
        has_drop = components.sudden_drop_score > 0.30  # Lowered from 0.4
        has_impact = components.impact_score > 0.40     # Lowered from 0.5
        has_prone = components.prone_score > 0.45       # Lowered from 0.6
        
        if (has_drop or has_impact) and has_prone:
            components.sudden_drop_score = min(1.0, components.sudden_drop_score * 1.10)
            components.prone_score = min(1.0, components.prone_score * 1.10)
            if has_impact:
                components.impact_score = min(1.0, components.impact_score * 1.10)
        
        # Normalize
        total_weight = w_drop + w_prone + w_impact + w_sustained
        if total_weight == 0:
            return 0.0
        
        # Weighted sum
        score = (
            (w_drop / total_weight) * components.sudden_drop_score +
            (w_prone / total_weight) * components.prone_score +
            (w_impact / total_weight) * components.impact_score +
            (w_sustained / total_weight) * components.sustained_lying_score
        )
        
        # Apply reduced penalty
        score = max(0.0, score + components.activity_penalty)
        
        # Less harsh confidence penalty
        if components.confidence_factor < 0.6:
            score *= (0.8 + 0.4 * components.confidence_factor)
        
        return min(max(score, 0.0), 1.0)
    
    def _identify_triggers(self, components: FallScoreComponents, features: PoseFeatures) -> List[str]:
        """Identify triggers"""
        triggers = []
        
        if components.sudden_drop_score > 0.35:
            triggers.append(f"Drop ({features.normalized_drop_velocity:.3f})")
        
        if components.impact_score > 0.40:
            triggers.append(f"Impact ({features.acc_sv_total:.2f}g)")
        
        if components.prone_score > 0.45:
            triggers.append(f"Prone ({features.body_orientation:.0f}°)")
        
        if components.sustained_lying_score > 0.60:
            triggers.append("Sustained")
        
        if abs(components.activity_penalty) > 0.15:
            triggers.append(f"Activity: {features.activity_type}")
        
        return triggers


if __name__ == "__main__":
    print("Testing RECALL-OPTIMIZED Scorer...")
    
    from features import PoseFeatures
    scorer = FallScorer()
    
    # Test moderate fall
    print("\n=== Moderate Fall (should score higher now) ===")
    moderate_fall = PoseFeatures(
        normalized_drop_velocity=0.035,  # Moderate
        acceleration_y=1.5,
        body_orientation=60,
        bbox_aspect_ratio=0.75,
        hip_height_ratio=0.25,
        velocity_y=4.0,
        pose_stability=0.85,
        keypoints_conf_mean=0.75
    )
    
    result = scorer.calculate(moderate_fall)
    print(f"Total Score: {result.total_score:.3f}")
    print(f"  Drop: {result.sudden_drop_score:.3f}")
    print(f"  Prone: {result.prone_score:.3f}")
    print(f"  Sustained: {result.sustained_lying_score:.3f}")
    print(f"Triggers: {', '.join(result.triggering_factors)}")
    
    if result.total_score > 0.45:
        print("✓ Moderate falls now detected!")
    else:
        print(f"✗ Still too low: {result.total_score:.3f}")