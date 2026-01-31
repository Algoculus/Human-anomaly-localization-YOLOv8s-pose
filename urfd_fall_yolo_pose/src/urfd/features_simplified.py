"""
Simplified Fall Detection Features - YOLO-Compatible Only

Computes ONLY features that can be reliably extracted from YOLO bbox:
1. hw_ratio: Width/Height ratio (lying posture indicator)
2. height_ratio: Current height / Baseline height (fallen state)
3. floor_distance: Vertical position in frame (proximity to floor)

Removed incompatible features:
- Pose keypoint angles (unreliable in frontal view)
- MaxStdXZ (requires 3D point cloud from depth camera)
- Exact depth D (requires depth sensor)
- Jerk/acceleration (too noisy from bbox tracking)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from collections import deque

@dataclass
class FrameFeatures:
    """Container for bbox-based features only."""
    track_id: int = -1
    frame_idx: int = 0
    feature_valid: bool = False
    
    # Raw bbox
    bbox: Optional[Tuple[float, float, float, float]] = None
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    # Core 3 features for SVM
    hw_ratio: float = 0.0          # w/h (lying posture)
    height_ratio: float = 1.0      # h/baseline_h (fallen state)
    floor_distance: float = 0.0    # cy/img_h (floor proximity)
    
    # Temporal motion (optional, for variance)
    position_variance: float = 0.0
    
    # Detection reference
    detection: Optional[Dict] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'track_id': self.track_id,
            'frame_idx': self.frame_idx,
            'feature_valid': self.feature_valid,
            'bbox': self.bbox,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'width': self.width,
            'height': self.height,
            'hw_ratio': self.hw_ratio,
            'height_ratio': self.height_ratio,
            'floor_distance': self.floor_distance,
            'position_variance': self.position_variance
        }


class FeatureBuffer:
    """Simplified buffer for baseline and variance computation."""
    
    def __init__(self, max_size: int = 60, stable_motion_thres: float = 2.0):
        """
        Args:
            max_size: Maximum frames to keep (~2s at 30fps)
            stable_motion_thres: dy threshold for stable frames
        """
        self.max_size = max_size
        self.stable_motion_thres = stable_motion_thres
        self.buffer = deque(maxlen=max_size)
        
        # Dynamic baseline (from stable frames)
        self._baseline_h = None
        self._baseline_w = None
        self._baseline_valid = False
    
    def add(self, features: FrameFeatures):
        """Add new frame features."""
        self.buffer.append(features)
        self._update_baseline()
    
    def __len__(self):
        return len(self.buffer)
    
    def get_recent(self, n: int = 1):
        """Get n most recent features."""
        if len(self.buffer) < n:
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def _update_baseline(self):
        """Update baseline from stable frames (standing/normal posture)."""
        if len(self.buffer) < 10:
            return
        
        # Use frames with low vertical motion as baseline
        stable_frames = []
        for i in range(1, len(self.buffer)):
            prev = self.buffer[i-1]
            curr = self.buffer[i]
            dy = abs(curr.center_y - prev.center_y)
            
            if dy < self.stable_motion_thres and curr.feature_valid:
                stable_frames.append(curr)
        
        if len(stable_frames) >= 5:
            heights = [f.height for f in stable_frames]
            widths = [f.width for f in stable_frames]
            
            # Use 75th percentile as baseline (standing height)
            self._baseline_h = np.percentile(heights, 75)
            self._baseline_w = np.percentile(widths, 75)
            self._baseline_valid = True
    
    def baseline_h(self):
        """Get current baseline height."""
        return self._baseline_h if self._baseline_valid else None
    
    def baseline_w(self):
        """Get current baseline width."""
        return self._baseline_w if self._baseline_valid else None
    
    def baseline_valid(self):
        return self._baseline_valid
    
    def get_position_variance(self, window: int = 10):
        """
        Compute position variance (movement indicator).
        
        Returns max of normalized variance in x and y positions.
        Higher value = more movement/instability.
        """
        if len(self.buffer) < window:
            return 0.0
        
        recent = list(self.buffer)[-window:]
        
        # Compute variance of centroid positions
        cx_vals = [f.center_x for f in recent if f.feature_valid]
        cy_vals = [f.center_y for f in recent if f.feature_valid]
        
        if len(cx_vals) < 3:
            return 0.0
        
        var_x = np.var(cx_vals)
        var_y = np.var(cy_vals)
        
        # Return max variance (normalized by image size assumption)
        return max(var_x, var_y)


def compute_frame_features(
    detection,
    config: Dict,
    feature_buffer: FeatureBuffer,
    track_id: int = -1,
    frame_idx: int = 0,
    image_size: Optional[Tuple[int, int]] = None
) -> FrameFeatures:
    """
    Compute simplified bbox-based features.
    
    Args:
        detection: YOLO detection dict with 'bbox', 'conf', 'keypoints'
        config: Configuration dict
        feature_buffer: FeatureBuffer for temporal analysis
        track_id: Track ID
        frame_idx: Frame index
        image_size: (width, height) of image
    
    Returns:
        FrameFeatures object
    """
    features = FrameFeatures(track_id=track_id, frame_idx=frame_idx)
    features.detection = detection
    
    # Extract bbox
    bbox = detection.get('bbox')
    if bbox is None or len(bbox) != 4:
        return features
    
    x1, y1, x2, y2 = bbox
    features.bbox = tuple(bbox)
    
    # Compute bbox properties
    w = x2 - x1
    h = y2 - y1
    
    if w <= 0 or h <= 0:
        return features
    
    features.width = w
    features.height = h
    features.center_x = (x1 + x2) / 2.0
    features.center_y = (y1 + y2) / 2.0
    
    # Image size for normalization
    if image_size:
        img_w, img_h = image_size
    else:
        # Estimate from bbox (assume person is in frame)
        img_w = max(x2 * 1.2, 640)
        img_h = max(y2 * 1.2, 480)
    
    # Feature 1: HW Ratio (w/h)
    features.hw_ratio = w / h
    
    # Feature 2: Height Ratio (h/baseline_h)
    baseline_h = feature_buffer.baseline_h()
    if baseline_h and baseline_h > 0:
        features.height_ratio = h / baseline_h
    else:
        # Default: assume standing height is ~60% of image height
        estimated_baseline = img_h * 0.6
        features.height_ratio = h / estimated_baseline
    
    # Feature 3: Floor Distance (cy/img_h)
    # Higher value = closer to bottom = closer to floor
    features.floor_distance = features.center_y / img_h
    
    # Optional: Position Variance
    variance_window = config.get('paper_features', {}).get('variance_window', 10)
    features.position_variance = feature_buffer.get_position_variance(variance_window)
    
    features.feature_valid = True
    return features
