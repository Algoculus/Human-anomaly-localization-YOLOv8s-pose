"""
Fall Scoring Module

Calculates the risk of a fall based on extracted features.
Implements the weighted sum of score components with adaptive weights:
FALL_SCORE(t) = w_drop * S_drop + w_prone * S_prone + w_impact * S_impact + w_lie * S_lie
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger

from .features import PoseFeatures
from config import Config, get_config, FallScoreConfig

@dataclass
class FallScoreComponents:
    """Individual components of the full fall score."""
    sudden_drop_score: float = 0.0
    prone_score: float = 0.0
    impact_score: float = 0.0
    sustained_lying_score: float = 0.0
    total_score: float = 0.0
    confidence: float = 1.0  # Overall confidence in the score

def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Sigmoid function for soft thresholding."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

class FallScorer:
    """
    Computes fall risk scores with adaptive weighting.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.params: FallScoreConfig = self.config.fall_score
        
    def calculate(self, features: PoseFeatures) -> FallScoreComponents:
        """
        Compute fall score components from input features.
        Uses adaptive weights when certain sensors are unavailable.
        """
        components = FallScoreComponents()
        
        # Track which sensors are available for adaptive weighting
        has_accelerometer = features.acc_sv_total is not None
        has_depth = features.depth_hw_ratio is not None or features.p40 is not None
        has_good_pose = features.keypoints_conf_mean > 0.5
        
        # Set confidence based on data quality
        components.confidence = features.keypoints_conf_mean if has_good_pose else 0.3
        
        # ========== 1. Sudden Drop Score ==========
        if features.normalized_drop_velocity > 0:  # Downward movement
            velocity = features.normalized_drop_velocity
            
            # Three-tier velocity classification
            if velocity < 0.015:
                # Very slow - intentional lie-down
                components.sudden_drop_score = 0.0
            elif velocity > 0.05:
                # Fast drop - definite fall indicator
                components.sudden_drop_score = min(sigmoid(velocity, k=40.0, x0=0.04), 1.0)
            else:
                # Medium velocity - gradual scoring
                components.sudden_drop_score = sigmoid(velocity, k=25.0, x0=0.03)
        else:
            components.sudden_drop_score = 0.0
            
        # ========== 2. Prone Score (Multi-source fusion) ==========
        prone_signals = []
        
        # 2a. Orientation signal (from spine angle)
        if has_good_pose:
            s_orient = sigmoid(features.body_orientation, k=0.15, x0=40.0)
            prone_signals.append(('orient', s_orient, 0.4))
        
        # 2b. Aspect Ratio signal (prefer depth if available)
        ar = features.depth_hw_ratio if has_depth else features.bbox_aspect_ratio
        if ar is not None:
            # H/W < 1.0 means lying (wider than tall)
            s_ar = 1.0 - sigmoid(ar, k=6.0, x0=1.0)
            prone_signals.append(('aspect', s_ar, 0.3))
        
        # 2c. Floor proximity signals (depth sensors)
        if features.p40 is not None and features.p40 > 0.2:
            s_floor = sigmoid(features.p40, k=6.0, x0=0.35)
            prone_signals.append(('p40', s_floor, 0.5))  # High weight - reliable
            
        if features.dist_to_floor_mm is not None:
            if features.dist_to_floor_mm < 450:
                s_dist = 1.0 - sigmoid(features.dist_to_floor_mm, k=0.008, x0=300)
                prone_signals.append(('dist', s_dist, 0.5))
            elif features.dist_to_floor_mm > 550:
                # On furniture - apply penalty
                prone_signals.append(('furniture_penalty', -0.3, 1.0))
        
        # Combine prone signals (weighted max with penalty)
        if prone_signals:
            positive_signals = [(s, w) for name, s, w in prone_signals if s > 0]
            penalties = sum([s for name, s, w in prone_signals if s < 0])
            
            if positive_signals:
                # Take weighted average of top 2 signals
                positive_signals.sort(key=lambda x: x[0] * x[1], reverse=True)
                top_signals = positive_signals[:2]
                total_weight = sum(w for s, w in top_signals)
                components.prone_score = sum(s * w for s, w in top_signals) / total_weight if total_weight > 0 else 0
                components.prone_score = max(0.0, components.prone_score + penalties)
            else:
                components.prone_score = 0.0
        else:
            components.prone_score = 0.0

        # ========== 3. Impact Score (Accelerometer) ==========
        if has_accelerometer:
            # Normal walking: 1.0-1.5g, Fall impact: > 2.0g
            components.impact_score = sigmoid(
                features.acc_sv_total, 
                k=self.params.impact_sigmoid_k, 
                x0=self.params.impact_sv_threshold
            )
        else:
            components.impact_score = 0.0

        # ========== 4. Sustained Lying Score ==========
        is_still = abs(features.velocity_y) < 2.5 and abs(features.orientation_velocity) < 3.0
        
        if components.prone_score > 0.6:
            if is_still:
                components.sustained_lying_score = 1.0
            else:
                # Moving while prone - might be trying to get up
                components.sustained_lying_score = 0.4
        elif components.prone_score > 0.4 and is_still:
            components.sustained_lying_score = 0.5
        else:
            components.sustained_lying_score = 0.0

        # ========== Adaptive Weighted Sum ==========
        # Adjust weights based on sensor availability and triggers
        w_drop = self.params.weight_sudden_drop
        w_prone = self.params.weight_prone
        w_impact = self.params.weight_impact
        w_sustained = self.params.weight_sustained_lying
        
        # Kwolek & Kepski Logic: Impact triggers specific pose checks
        # If High Impact detected, prioritize Prone Score verification
        if components.impact_score > 0.6:
            w_prone = 0.6  # Boost prone weight significantly
            w_drop = 0.1   # Drop is less relevant after impact
            w_sustained = 0.1
            w_impact = 0.2
            
            # Boost prone sensitivity if impact occurred
            if components.prone_score > 0.4:
                components.prone_score = min(components.prone_score * 1.3, 1.0)
                
        # If no accelerometer, redistribute impact weight
        elif not has_accelerometer:
            w_prone += w_impact * 0.6
            w_sustained += w_impact * 0.4
            w_impact = 0.0
            
        # If poor pose quality, boost depth-based prone weight
        if not has_good_pose and has_depth:
            w_prone *= 1.2
            w_drop *= 0.5
        
        # Normalize weights
        total_weight = w_drop + w_prone + w_impact + w_sustained
        
        total = (
            (w_drop / total_weight) * components.sudden_drop_score +
            (w_prone / total_weight) * components.prone_score +
            (w_impact / total_weight) * components.impact_score +
            (w_sustained / total_weight) * components.sustained_lying_score
        )
        
        # Apply confidence penalty for low-quality poses
        if components.confidence < 0.5:
            total *= 0.8 + 0.4 * components.confidence  # Scale down uncertain scores
        
        components.total_score = min(max(total, 0.0), 1.0)  # Clip 0-1
        
        return components
