"""
Fall Detection Feature Extraction Module

Implements comprehensive feature extraction for fall detection:
- Impact/Motion features (dy, acceleration, impact signature)
- Shape/BBox dynamics (height drop, width increase, deltas)
- Pose dynamics (hip drop, torso compaction, angle change)
- Quality metrics (keypoint validity, bbox stability, border penalty)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque

# COCO keypoint indices
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12
KEYPOINT_LEFT_KNEE = 13
KEYPOINT_RIGHT_KNEE = 14

# Border margin for edge detection
BORDER_MARGIN = 8


@dataclass
class FrameFeatures:
    """Container for all features extracted from a single frame."""
    # Metadata
    track_id: int = -1
    frame_idx: int = 0
    feature_valid: bool = False
    
    # Raw bbox data
    bbox: Optional[Tuple[float, float, float, float]] = None
    center_x: float = 0.0
    center_y: float = 0.0
    height: float = 0.0
    width: float = 0.0
    bottom_y: float = 0.0
    
    # Shape features
    bbox_aspect_ratio: float = 0.0
    height_drop_norm: float = 0.0  # Normalized height drop vs baseline
    width_increase: float = 0.0   # Width increase vs baseline
    dH: float = 0.0               # Frame-to-frame height delta
    dW: float = 0.0               # Frame-to-frame width delta  
    dAR: float = 0.0              # Frame-to-frame AR delta
    bottom_y_shift: float = 0.0   # Change in bottom edge (floor proxy)
    
    # Motion features (normalized by baseline_h)
    dy: float = 0.0               # Raw vertical velocity
    dy_norm: float = 0.0          # dy / baseline_h
    dy_peak: float = 0.0          # Max dy in window
    dy_acc: float = 0.0           # Acceleration (jerk)
    impact_signature: float = 0.0 # Combined impact pattern
    post_motion: float = 0.0      # Motion after impact
    
    # Pose features
    shoulder_mid: Optional[Tuple[float, float]] = None
    hip_mid: Optional[Tuple[float, float]] = None
    body_angle_deg: Optional[float] = None
    angle_change: float = 0.0      # Delta from recent median
    hip_drop: float = 0.0          # Change in hip_mid.y
    head_drop: float = 0.0         # Change in head.y
    torso_len: float = 0.0         # Distance shoulder_mid to hip_mid
    torso_compaction: float = 0.0  # Reduction in torso_len
    knee_relation: float = 0.0     # hip_drop correlation with knee
    
    # Quality metrics
    keypoint_valid_ratio: float = 0.0  # Fraction of confident keypoints
    upper_kp_ratio: float = 0.0        # Upper body keypoint ratio
    lower_kp_ratio: float = 0.0        # Lower body keypoint ratio
    bbox_stability: float = 1.0        # Low std = stable (inverted)
    border_penalty: float = 0.0        # 0..1 penalty when touching edge
    is_touching_border: bool = False
    
    # Paper-aligned features (KwolekKepski URFD methodology)
    hw_ratio: float = 0.0              # width/height (paper's h/w ratio, >1 = lying)
    height_ratio: float = 1.0          # current_h / baseline_h (paper's h/hmax)
    floor_distance: float = 0.0        # Proxy: 1.0 - normalized center_y (higher = near floor)
    position_variance: float = 0.0     # max(var_x, var_y) from history (movement spread)
    
    # Detection reference
    detection: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "feature_valid": self.feature_valid,
            "bbox": self.bbox,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "height": self.height,
            "width": self.width,
            "bottom_y": self.bottom_y,
            "bbox_aspect_ratio": self.bbox_aspect_ratio,
            "height_drop_norm": self.height_drop_norm,
            "width_increase": self.width_increase,
            "dH": self.dH,
            "dW": self.dW,
            "dAR": self.dAR,
            "bottom_y_shift": self.bottom_y_shift,
            "dy": self.dy,
            "dy_norm": self.dy_norm,
            "dy_peak": self.dy_peak,
            "dy_acc": self.dy_acc,
            "impact_signature": self.impact_signature,
            "post_motion": self.post_motion,
            "shoulder_mid": self.shoulder_mid,
            "hip_mid": self.hip_mid,
            "body_angle_deg": self.body_angle_deg,
            "angle_change": self.angle_change,
            "hip_drop": self.hip_drop,
            "head_drop": self.head_drop,
            "torso_len": self.torso_len,
            "torso_compaction": self.torso_compaction,
            "knee_relation": self.knee_relation,
            "keypoint_valid_ratio": self.keypoint_valid_ratio,
            "upper_kp_ratio": self.upper_kp_ratio,
            "lower_kp_ratio": self.lower_kp_ratio,
            "bbox_stability": self.bbox_stability,
            "border_penalty": self.border_penalty,
            "is_touching_border": self.is_touching_border,
            # Paper-aligned features
            "hw_ratio": self.hw_ratio,
            "height_ratio": self.height_ratio,
            "floor_distance": self.floor_distance,
            "position_variance": self.position_variance,
            "detection": self.detection,
        }


class FeatureBuffer:
    """
    Circular buffer for temporal feature analysis.
    
    Maintains history of features for computing:
    - Dynamic baselines (height, width)
    - Temporal derivatives (velocity, acceleration)
    - Pattern detection (impact signatures)
    """
    
    def __init__(self, max_size: int = 60, stable_motion_thres: float = 2.0):
        """
        Args:
            max_size: Maximum frames to keep (default ~2s at 30fps)
            stable_motion_thres: dy below this = stable frame for baseline
        """
        self.max_size = max_size
        self.stable_motion_thres = stable_motion_thres
        self.buffer: deque = deque(maxlen=max_size)
        
        # Baseline tracking
        self._baseline_h: float = 0.0
        self._baseline_w: float = 0.0
        self._baseline_valid: bool = False
        
    def add(self, features: FrameFeatures) -> None:
        """Add new frame features to buffer."""
        self.buffer.append(features)
        self._update_baseline()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_recent(self, n: int = 1) -> List[FrameFeatures]:
        """Get n most recent features."""
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def get_prev(self, offset: int = 1) -> Optional[FrameFeatures]:
        """Get feature from offset frames ago."""
        if offset >= len(self.buffer):
            return None
        return self.buffer[-offset - 1] if len(self.buffer) > offset else None
    
    def _update_baseline(self) -> None:
        """Update dynamic baseline from stable frames."""
        if len(self.buffer) < 5:
            return
            
        # Collect heights/widths from stable (low motion) frames
        stable_heights = []
        stable_widths = []
        
        for feat in self.buffer:
            if abs(feat.dy) < self.stable_motion_thres and feat.height > 0:
                stable_heights.append(feat.height)
                stable_widths.append(feat.width)
        
        if len(stable_heights) >= 3:
            self._baseline_h = np.median(stable_heights)
            self._baseline_w = np.median(stable_widths)
            self._baseline_valid = True
        elif len(self.buffer) >= 5:
            # Fallback: use median of all frames
            all_heights = [f.height for f in self.buffer if f.height > 0]
            all_widths = [f.width for f in self.buffer if f.width > 0]
            if all_heights:
                self._baseline_h = np.median(all_heights)
                self._baseline_w = np.median(all_widths)
                self._baseline_valid = True
    
    @property
    def baseline_h(self) -> float:
        """Get current baseline height."""
        return self._baseline_h if self._baseline_valid else 0.0
    
    @property
    def baseline_w(self) -> float:
        """Get current baseline width."""
        return self._baseline_w if self._baseline_valid else 0.0
    
    @property
    def baseline_valid(self) -> bool:
        return self._baseline_valid
    
    def get_dy_samples(self, window: int) -> List[float]:
        """Get dy values from recent window."""
        recent = self.get_recent(window)
        return [f.dy for f in recent]
    
    def get_angle_median(self, window: int = 10) -> Optional[float]:
        """Get median body angle from recent window."""
        recent = self.get_recent(window)
        angles = [f.body_angle_deg for f in recent if f.body_angle_deg is not None]
        return np.median(angles) if angles else None
    
    def get_bbox_stability(self, window: int = 5) -> float:
        """Compute bbox stability (inverse of std)."""
        recent = self.get_recent(window)
        if len(recent) < 3:
            return 1.0
        
        cxs = [f.center_x for f in recent if f.bbox is not None]
        cys = [f.center_y for f in recent if f.bbox is not None]
        ws = [f.width for f in recent if f.bbox is not None]
        hs = [f.height for f in recent if f.bbox is not None]
        
        if len(cxs) < 3:
            return 1.0
        
        # Normalize by baseline
        baseline = self.baseline_h if self.baseline_h > 0 else np.mean(hs)
        if baseline <= 0:
            return 1.0
            
        std_sum = (np.std(cxs) + np.std(cys) + np.std(ws) + np.std(hs)) / baseline
        # Invert: low std = high stability
        stability = 1.0 / (1.0 + std_sum)
        return stability
    
    def get_position_variance(self, window: int = 10) -> float:
        """
        Compute position variance (paper's max(σx, σz) feature).
        
        Returns max of normalized variance in x and y positions.
        Higher value = more movement/instability.
        """
        recent = self.get_recent(window)
        if len(recent) < 3:
            return 0.0
        
        cxs = [f.center_x for f in recent if f.bbox is not None]
        cys = [f.center_y for f in recent if f.bbox is not None]
        
        if len(cxs) < 3:
            return 0.0
        
        # Normalize by baseline or mean width for scale-invariance
        baseline = self.baseline_h if self.baseline_h > 0 else 100.0
        
        var_x = np.var(cxs) / (baseline ** 2) if baseline > 0 else 0.0
        var_y = np.var(cys) / (baseline ** 2) if baseline > 0 else 0.0
        
        return max(var_x, var_y)

def compute_keypoint_quality(keypoints, keypoint_conf_thres: float) -> Tuple[float, float, float]:
    """
    Compute keypoint validity ratios.
    
    Returns:
        (overall_ratio, upper_ratio, lower_ratio)
    """
    if keypoints is None:
        return 0.0, 0.0, 0.0
    
    # Upper body: nose, eyes, shoulders (indices 0-6)
    upper_indices = [0, 1, 2, 3, 4, 5, 6]
    # Lower body: hips, knees, ankles (indices 11-16)
    lower_indices = [11, 12, 13, 14, 15, 16]
    
    upper_valid = sum(1 for i in upper_indices if i < len(keypoints) and keypoints[i][2] >= keypoint_conf_thres)
    lower_valid = sum(1 for i in lower_indices if i < len(keypoints) and keypoints[i][2] >= keypoint_conf_thres)
    total_valid = sum(1 for kp in keypoints if kp[2] >= keypoint_conf_thres)
    
    upper_ratio = upper_valid / len(upper_indices) if upper_indices else 0.0
    lower_ratio = lower_valid / len(lower_indices) if lower_indices else 0.0
    overall_ratio = total_valid / len(keypoints) if len(keypoints) > 0 else 0.0
    
    return overall_ratio, upper_ratio, lower_ratio


def compute_frame_features(
    detection,
    keypoint_conf_thres: float,
    dy_window: int,
    history: List,  # Legacy compatibility
    track_id: int = -1,
    frame_idx: int = 0,
    image_size: Optional[Tuple[int, int]] = None,
    feature_buffer: Optional[FeatureBuffer] = None
) -> Dict:
    """
    Compute comprehensive features for a single detection.
    
    This function maintains backward compatibility while using the new
    FeatureBuffer internally when provided.
    
    Args:
        detection: Detection dict with 'bbox', 'keypoints'
        keypoint_conf_thres: Minimum keypoint confidence
        dy_window: Window size for velocity computation
        history: Legacy feature history list
        track_id: Track ID
        frame_idx: Current frame index
        image_size: (width, height) for border check
        feature_buffer: Optional FeatureBuffer for enhanced features
        
    Returns:
        Dictionary of features (backward compatible)
    """
    features = FrameFeatures(track_id=track_id, frame_idx=frame_idx)
    
    # Handle no detection
    if detection is None:
        return features.to_dict()
    
    features.detection = detection
    bbox = detection.get("bbox")
    keypoints = detection.get("keypoints")
    is_fallback = detection.get("is_fallback", False)
    
    if bbox is None:
        return features.to_dict()
    
    # =========================================================
    # BASIC BBOX FEATURES
    # =========================================================
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    features.bbox = tuple(bbox)
    features.center_x = cx
    features.center_y = cy
    features.width = w
    features.height = h
    features.bottom_y = y2
    features.bbox_aspect_ratio = w / h if h > 0 else 0.0
    
    # =========================================================
    # BORDER CHECK
    # =========================================================
    if image_size is not None:
        img_w, img_h = image_size
        if (x1 <= BORDER_MARGIN or y1 <= BORDER_MARGIN or 
            x2 >= img_w - BORDER_MARGIN or y2 >= img_h - BORDER_MARGIN):
            features.is_touching_border = True
            # Compute penalty based on how much is clipped
            clip_amount = max(
                max(0, BORDER_MARGIN - x1) / w if w > 0 else 0,
                max(0, BORDER_MARGIN - y1) / h if h > 0 else 0,
                max(0, x2 - (img_w - BORDER_MARGIN)) / w if w > 0 else 0,
                max(0, y2 - (img_h - BORDER_MARGIN)) / h if h > 0 else 0
            )
            features.border_penalty = min(clip_amount * 2, 1.0)
    
    # =========================================================
    # KEYPOINT QUALITY METRICS
    # =========================================================
    if keypoints is not None and not is_fallback:
        overall, upper, lower = compute_keypoint_quality(keypoints, keypoint_conf_thres)
        features.keypoint_valid_ratio = overall
        features.upper_kp_ratio = upper
        features.lower_kp_ratio = lower
    
    # =========================================================
    # POSE FEATURES (from keypoints)
    # =========================================================
    if not is_fallback and keypoints is not None:
        # Extract torso keypoints
        ls = keypoints[KEYPOINT_LEFT_SHOULDER]
        rs = keypoints[KEYPOINT_RIGHT_SHOULDER]
        lh = keypoints[KEYPOINT_LEFT_HIP]
        rh = keypoints[KEYPOINT_RIGHT_HIP]
        
        ls_conf = ls[2] >= keypoint_conf_thres
        rs_conf = rs[2] >= keypoint_conf_thres
        lh_conf = lh[2] >= keypoint_conf_thres
        rh_conf = rh[2] >= keypoint_conf_thres
        
        # Shoulder midpoint
        if ls_conf and rs_conf:
            features.shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        
        # Hip midpoint
        if lh_conf and rh_conf:
            features.hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        
        # Body angle and torso length
        if features.shoulder_mid is not None and features.hip_mid is not None:
            dx = features.shoulder_mid[0] - features.hip_mid[0]
            dy_torso = features.shoulder_mid[1] - features.hip_mid[1]
            
            # Torso length
            features.torso_len = np.sqrt(dx**2 + dy_torso**2)
            
            # Angle: 0° = upright, 90° = horizontal
            angle_rad = np.arctan2(np.abs(dx), np.abs(dy_torso) + 1e-6)
            features.body_angle_deg = np.degrees(angle_rad)
            features.feature_valid = True
        
        # Head position (for head drop)
        nose = keypoints[KEYPOINT_NOSE]
        if nose[2] >= keypoint_conf_thres:
            features.head_drop = 0.0  # Will compute delta later
    
    # =========================================================
    # TEMPORAL FEATURES (using history or buffer)
    # =========================================================
    prev_features = None
    baseline_h = 0.0
    baseline_w = 0.0
    
    if feature_buffer is not None and len(feature_buffer) > 0:
        prev_features = feature_buffer.get_prev(1)
        baseline_h = feature_buffer.baseline_h
        baseline_w = feature_buffer.baseline_w
        features.bbox_stability = feature_buffer.get_bbox_stability()
        
        # Angle change from median
        angle_median = feature_buffer.get_angle_median()
        if angle_median is not None and features.body_angle_deg is not None:
            features.angle_change = abs(features.body_angle_deg - angle_median)
    elif len(history) > 0:
        # Legacy: use history list
        prev_features_dict = history[-1]
        baseline_h = prev_features_dict.get("height", 0) if len(history) < 10 else np.median([h.get("height", 0) for h in history[-10:]])
        baseline_w = prev_features_dict.get("width", 0) if len(history) < 10 else np.median([h.get("width", 0) for h in history[-10:]])
    
    # Compute deltas from previous frame
    if prev_features is not None:
        # Shape deltas
        features.dH = h - prev_features.height if hasattr(prev_features, 'height') else h - prev_features.get("height", h)
        features.dW = w - prev_features.width if hasattr(prev_features, 'width') else w - prev_features.get("width", w)
        
        prev_ar = prev_features.bbox_aspect_ratio if hasattr(prev_features, 'bbox_aspect_ratio') else prev_features.get("bbox_aspect_ratio", features.bbox_aspect_ratio)
        features.dAR = features.bbox_aspect_ratio - prev_ar
        
        prev_bottom = prev_features.bottom_y if hasattr(prev_features, 'bottom_y') else prev_features.get("bottom_y", y2)
        features.bottom_y_shift = y2 - prev_bottom
        
        # Motion: dy
        prev_cy = prev_features.center_y if hasattr(prev_features, 'center_y') else prev_features.get("center_y", cy)
        features.dy = cy - prev_cy
        
        # Hip drop
        if features.hip_mid is not None:
            prev_hip = prev_features.hip_mid if hasattr(prev_features, 'hip_mid') else prev_features.get("hip_mid")
            if prev_hip is not None:
                features.hip_drop = features.hip_mid[1] - prev_hip[1]
        
        # Torso compaction
        prev_torso = prev_features.torso_len if hasattr(prev_features, 'torso_len') else prev_features.get("torso_len", 0)
        if prev_torso > 0 and features.torso_len > 0:
            features.torso_compaction = (prev_torso - features.torso_len) / prev_torso
    elif len(history) > 0:
        # Legacy compatibility
        prev = history[-1]
        if prev.get("center_y") is not None and cy is not None:
            features.dy = cy - prev["center_y"]
        if prev.get("height") is not None:
            features.dH = h - prev["height"]
        if prev.get("width") is not None:
            features.dW = w - prev["width"]
        if prev.get("bbox_aspect_ratio") is not None:
            features.dAR = features.bbox_aspect_ratio - prev["bbox_aspect_ratio"]
    
    # Normalized vertical velocity
    if baseline_h > 0:
        features.dy_norm = features.dy / baseline_h
        features.height_drop_norm = max(0, (baseline_h - h) / baseline_h)
    
    if baseline_w > 0:
        features.width_increase = max(0, (w - baseline_w) / baseline_w)
    
    # =========================================================
    # VELOCITY DERIVATIVES (dy_peak, dy_acc)
    # =========================================================
    dy_samples = []
    
    if feature_buffer is not None:
        dy_samples = feature_buffer.get_dy_samples(dy_window)
    elif len(history) > 0:
        # Legacy
        for i in range(1, min(dy_window + 1, len(history) + 1)):
            if i <= len(history):
                hist_feat = history[-i]
                dy_val = hist_feat.get("dy", 0.0)
                if dy_val != 0.0:
                    dy_samples.append(abs(dy_val))
    
    dy_samples.append(abs(features.dy))
    
    if len(dy_samples) > 0:
        features.dy_peak = max(dy_samples)
    
    # Acceleration (second derivative)
    if len(dy_samples) >= 2:
        # Compute differences of dy (acceleration)
        dy_diffs = [abs(dy_samples[i] - dy_samples[i-1]) for i in range(1, len(dy_samples))]
        features.dy_acc = max(dy_diffs) if dy_diffs else 0.0
    
    # =========================================================
    # IMPACT SIGNATURE
    # Combined measure of dy_peak + post-settling
    # =========================================================
    if features.dy_peak > 0:
        # Look for settling pattern: high peak followed by low motion
        if len(dy_samples) >= 3:
            recent_motion = np.mean(dy_samples[-2:]) if len(dy_samples) >= 2 else features.dy
            peak_ratio = features.dy_peak / (recent_motion + 1e-6)
            # High ratio = impact-like (spike then settle)
            features.impact_signature = min(peak_ratio / 5.0, 1.0) * min(features.dy_peak / 10.0, 1.0)
            features.post_motion = recent_motion
    
    # =========================================================
    # PAPER-ALIGNED FEATURES (KwolekKepski URFD methodology)
    # =========================================================
    # hw_ratio: width/height (paper's h/w ratio, >1 means lying/horizontal)
    features.hw_ratio = w / h if h > 0 else 0.0
    
    # height_ratio: current_h / baseline_h (paper's h/hmax)
    if baseline_h > 0:
        features.height_ratio = h / baseline_h
    else:
        features.height_ratio = 1.0
    
    # floor_distance: proxy for paper's D (distance to floor plane)
    # Using 1.0 - normalized center_y (higher value = closer to bottom = near floor)
    if image_size is not None:
        img_w, img_h = image_size
        features.floor_distance = cy / img_h  # 0.0 at top, 1.0 at bottom
    else:
        features.floor_distance = 0.0
    
    # position_variance: paper's max(σx, σz) - movement spread
    if feature_buffer is not None:
        features.position_variance = feature_buffer.get_position_variance(10)
    
    # =========================================================
    # RETURN AS DICT (backward compatible)
    # =========================================================
    return features.to_dict()


def compute_frame_features_v2(
    detection,
    config: Dict,
    feature_buffer: FeatureBuffer,
    track_id: int = -1,
    frame_idx: int = 0,
    image_size: Optional[Tuple[int, int]] = None
) -> FrameFeatures:
    """
    Enhanced feature computation using FeatureBuffer.
    
    This is the new API that returns FrameFeatures directly.
    """
    keypoint_conf_thres = config.get("keypoint_conf_thres", 0.5)
    dy_window = config.get("dy_window", 5)
    
    # Compute features
    feat_dict = compute_frame_features(
        detection=detection,
        keypoint_conf_thres=keypoint_conf_thres,
        dy_window=dy_window,
        history=[],  # Not used when buffer provided
        track_id=track_id,
        frame_idx=frame_idx,
        image_size=image_size,
        feature_buffer=feature_buffer
    )
    
    # Convert to FrameFeatures
    features = FrameFeatures(
        track_id=feat_dict.get("track_id", track_id),
        frame_idx=feat_dict.get("frame_idx", frame_idx),
        feature_valid=feat_dict.get("feature_valid", False),
        bbox=feat_dict.get("bbox"),
        center_x=feat_dict.get("center_x", 0.0),
        center_y=feat_dict.get("center_y", 0.0),
        height=feat_dict.get("height", 0.0),
        width=feat_dict.get("width", 0.0),
        bottom_y=feat_dict.get("bottom_y", 0.0),
        bbox_aspect_ratio=feat_dict.get("bbox_aspect_ratio", 0.0),
        height_drop_norm=feat_dict.get("height_drop_norm", 0.0),
        width_increase=feat_dict.get("width_increase", 0.0),
        dH=feat_dict.get("dH", 0.0),
        dW=feat_dict.get("dW", 0.0),
        dAR=feat_dict.get("dAR", 0.0),
        bottom_y_shift=feat_dict.get("bottom_y_shift", 0.0),
        dy=feat_dict.get("dy", 0.0),
        dy_norm=feat_dict.get("dy_norm", 0.0),
        dy_peak=feat_dict.get("dy_peak", 0.0),
        dy_acc=feat_dict.get("dy_acc", 0.0),
        impact_signature=feat_dict.get("impact_signature", 0.0),
        post_motion=feat_dict.get("post_motion", 0.0),
        shoulder_mid=feat_dict.get("shoulder_mid"),
        hip_mid=feat_dict.get("hip_mid"),
        body_angle_deg=feat_dict.get("body_angle_deg"),
        angle_change=feat_dict.get("angle_change", 0.0),
        hip_drop=feat_dict.get("hip_drop", 0.0),
        head_drop=feat_dict.get("head_drop", 0.0),
        torso_len=feat_dict.get("torso_len", 0.0),
        torso_compaction=feat_dict.get("torso_compaction", 0.0),
        keypoint_valid_ratio=feat_dict.get("keypoint_valid_ratio", 0.0),
        upper_kp_ratio=feat_dict.get("upper_kp_ratio", 0.0),
        lower_kp_ratio=feat_dict.get("lower_kp_ratio", 0.0),
        bbox_stability=feat_dict.get("bbox_stability", 1.0),
        border_penalty=feat_dict.get("border_penalty", 0.0),
        is_touching_border=feat_dict.get("is_touching_border", False),
        # Paper-aligned features
        hw_ratio=feat_dict.get("hw_ratio", 0.0),
        height_ratio=feat_dict.get("height_ratio", 1.0),
        floor_distance=feat_dict.get("floor_distance", 0.0),
        position_variance=feat_dict.get("position_variance", 0.0),
        detection=feat_dict.get("detection"),
    )
    
    # Add to buffer
    feature_buffer.add(features)
    
    return features
