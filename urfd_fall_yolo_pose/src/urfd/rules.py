"""
Fall Detection Rules Module - Simplified

Clean implementation with single unified scoring strategy.
Uses 4-5 features from YOLO + Paper's pre-extracted features.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FeatureContribution:
    """Single feature's contribution to fall score."""
    name: str
    value: float
    weight: float
    score: float


@dataclass
class FallScore:
    """Fall confidence score with explainability."""
    confidence: float = 0.0
    contributions: List[FeatureContribution] = field(default_factory=list)
    
    def get_top_features(self, n: int = 3) -> List[tuple]:
        """Get top N contributing features."""
        sorted_contribs = sorted(self.contributions, key=lambda x: abs(x.score), reverse=True)
        return [(c.name, c.value, c.score) for c in sorted_contribs[:n]]


class FallScorer:
    """
    Simplified fall detection scorer.
    
    Uses linear combination of 4 core features:
    1. hw_ratio: Width/Height (>1 = lying posture)
    2. height_ratio: Current height / Baseline (<1 = fallen)
    3. floor_distance: Vertical position (>0.7 = near floor)
    4. position_variance: Motion indicator (high = active movement)
    
    Optional:
    5. MaxStdXZ: Paper's key feature (if available from pre-extracted)
    """
    
    # Default SVM-learned weights (tuned for balance)
    # Floor exercises have high floor_distance but low hw_ratio
    # Falls have both high floor_distance AND high hw_ratio
    DEFAULT_WEIGHTS = {
        'hw_ratio': 2.5,        # Increased: prioritize lying posture
        'height_ratio': -6.0,   # Reduced penalty for height change
        'floor_distance': 8.0,  # Reduced: floor exercises also have high floor_dist
        'variance_boost': 2.5,
        'max_std_xz': 0.5,
        'intercept': -10.0      # Increased: harder to trigger (reduces FPs)
    }
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dict with thresholds and weights
        """
        self.config = config
        
        # Load weights from config or use defaults
        weights_cfg = config.get('svm_weights', {})
        self.weights = {
            'hw_ratio': weights_cfg.get('hw_ratio', self.DEFAULT_WEIGHTS['hw_ratio']),
            'height_ratio': weights_cfg.get('height_ratio', self.DEFAULT_WEIGHTS['height_ratio']),
            'floor_distance': weights_cfg.get('floor_distance', self.DEFAULT_WEIGHTS['floor_distance']),
            'variance_boost': weights_cfg.get('variance_boost', self.DEFAULT_WEIGHTS['variance_boost']),
            'max_std_xz': weights_cfg.get('max_std_xz', self.DEFAULT_WEIGHTS['max_std_xz']),
            'intercept': weights_cfg.get('intercept', self.DEFAULT_WEIGHTS['intercept'])
        }
        
        # Thresholds
        thresholds = config.get('thresholds', {})
        self.variance_threshold = thresholds.get('variance_threshold', 10.0)
        self.depth_threshold = thresholds.get('depth_threshold', 53.0)
        
        # Hypothesis/Verification thresholds
        hypothesis = config.get('hypothesis', {})
        self.hypothesis_threshold = hypothesis.get('early_score_thres', 0.35)
        
        verification = config.get('verification', {})
        self.verification_threshold = verification.get('verify_score_thres', 0.45)
    
    def _sigmoid(self, x: float, scale: float = 1.0) -> float:
        """Apply sigmoid to map score to [0, 1]."""
        return 1.0 / (1.0 + np.exp(-scale * x))
    
    def compute_fall_confidence(self, features: Dict) -> FallScore:
        """
        Compute fall confidence from features.
        
        Args:
            features: Dict with keys: hw_ratio, height_ratio, floor_distance,
                      position_variance, depth_mean, max_std_xz (optional)
                      
        Returns:
            FallScore with confidence [0, 1] and feature contributions
        """
        result = FallScore()
        contributions = []
        
        # Extract features
        hw_ratio = features.get('hw_ratio', 0.0)
        height_ratio = features.get('height_ratio', 1.0)
        floor_distance = features.get('floor_distance', 0.0)
        variance = features.get('position_variance', 0.0)
        depth_mean = features.get('depth_mean', 0.0)
        max_std_xz = features.get('max_std_xz', 0.0)
        
        # Compute linear score
        score = self.weights['intercept']
        
        # Feature 1: HW Ratio (>1 indicates lying)
        hw_contrib = self.weights['hw_ratio'] * hw_ratio
        score += hw_contrib
        contributions.append(FeatureContribution('hw_ratio', hw_ratio, self.weights['hw_ratio'], hw_contrib))
        
        # Feature 2: Height Ratio (< 1 indicates fallen)
        hr_contrib = self.weights['height_ratio'] * height_ratio
        score += hr_contrib
        contributions.append(FeatureContribution('height_ratio', height_ratio, self.weights['height_ratio'], hr_contrib))
        
        # Feature 3: Floor Distance (> 0.7 indicates near floor)
        fd_contrib = self.weights['floor_distance'] * floor_distance
        score += fd_contrib
        contributions.append(FeatureContribution('floor_distance', floor_distance, self.weights['floor_distance'], fd_contrib))
        
        # Feature 4: Variance Boost (high motion = likely fall)
        var_boost = 0.0
        if variance > self.variance_threshold:
            var_boost = self.weights['variance_boost']
            score += var_boost
        contributions.append(FeatureContribution('variance', variance, 1.0, var_boost))
        
        # Feature 5: MaxStdXZ from paper (if available)
        if max_std_xz > 0:
            # Paper uses MaxStdXZ as key fall indicator
            # High values indicate fall motion signature
            msx_contrib = self.weights['max_std_xz'] * max_std_xz
            score += msx_contrib
            contributions.append(FeatureContribution('max_std_xz', max_std_xz, self.weights['max_std_xz'], msx_contrib))
        
        # Depth penalty for elevated surfaces (sofa detection)
        if depth_mean > 0 and depth_mean < self.depth_threshold and variance < self.variance_threshold:
            # Static + Elevated = Likely on furniture, not floor fall
            score -= 5.0
            contributions.append(FeatureContribution('depth_penalty', depth_mean, -5.0, -5.0))
        
        # Convert to probability
        result.confidence = self._sigmoid(score)
        result.contributions = contributions
        
        return result
    
    def is_hypothesis_trigger(self, features: Dict, score: FallScore = None) -> bool:
        """Check if features should trigger hypothesis state."""
        if score is None:
            score = self.compute_fall_confidence(features)
        return score.confidence >= self.hypothesis_threshold
    
    def is_verification_confirm(self, features: Dict, score: FallScore = None) -> bool:
        """Check if features confirm fall in verification state."""
        if score is None:
            score = self.compute_fall_confidence(features)
        return score.confidence >= self.verification_threshold


def check_lying_posture(features: Dict, hw_ratio_thres: float = 1.1, height_ratio_thres: float = 0.65) -> bool:
    """
    Simple check for lying posture.
    
    Args:
        features: Feature dict
        hw_ratio_thres: Width/Height threshold (>1 = lying)
        height_ratio_thres: Height ratio threshold (<0.65 = fallen height)
        
    Returns:
        True if lying posture detected
    """
    hw_ratio = features.get('hw_ratio', 0.0)
    height_ratio = features.get('height_ratio', 1.0)
    
    is_wide = hw_ratio > hw_ratio_thres
    is_short = height_ratio < height_ratio_thres
    
    return is_wide or is_short
