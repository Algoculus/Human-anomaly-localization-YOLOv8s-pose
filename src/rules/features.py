"""
Feature Extraction Module

Extracts relevant features from:
1. YOLOv8-pose keypoints (orientation, velocity, aspect ratio)
2. Depth CSV data (Height, Distance to floor, P40, etc.)
3. Accelerometer data (Signal Vector Magnitude)

This module standardizes the input for the scoring logic.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config, get_config
from infer_pose import PoseResult
from utils.smoothing import EMAFilter, MedianFilter, VectorEMAFilter


@dataclass
class PoseFeatures:
    """
    Consolidated feature set for a single frame.
    Mix of Pose (YOLO), Depth (CSV), and Accel data.
    """
    # --- From YOLOv8 Pose ---
    keypoints_conf_mean: float = 0.0      # Average confidence of keypoints
    bbox_aspect_ratio: float = 1.0        # Height / Width (Note: Dataset uses H/W, we standardise to H/W)
    body_orientation: float = 0.0         # Angle of spine wrt vertical [0, 90] degrees. 0=upright, 90=lying
    normalized_height: float = 1.0        # BBox Height / Image Height
    
    # Velocities (calculated from tracking history)
    velocity_y: float = 0.0               # Vertical velocity (pixels/frame)
    normalized_drop_velocity: float = 0.0 # Vertical velocity / bbox height (body_heights/frame)
    orientation_velocity: float = 0.0     # Change in orientation (deg/frame)
    
    # --- From Depth CSV (Ground Truth Features) ---
    # These are None if not available (e.g. live webcam or cam1 without depth processing)
    depth_hw_ratio: Optional[float] = None # Height/Width ratio from depth blob
    depth_height_mm: Optional[float] = None # 'H': Actual height in mm
    dist_to_floor_mm: Optional[float] = None # 'D': Distance to floor in mm
    p40: Optional[float] = None           # 'P40': Ratio of points near floor
    max_std_xz: Optional[float] = None    # 'MaxStdXZ': Spread of points
    
    # --- From Accelerometer ---
    acc_sv_total: Optional[float] = None  # Signal Vector Total (g)
    
    @property
    def is_lying_pose(self) -> bool:
        """
        Heuristic check if pose looks like lying based on available features.
        Prioritizes Depth features if available.
        """
        # 1. Check Depth Features (Most Reliable)
        if self.depth_hw_ratio is not None:
             # Typically H/W < 1.0 means width > height -> lying
            if self.depth_hw_ratio < 0.8: 
                return True
        
        # 2. Check YOLO Aspect Ratio
        if self.bbox_aspect_ratio < 0.8:
            return True
            
        # 3. Check Orientation
        if self.body_orientation > 60: # Degrees from vertical
            return True
            
        return False


class FeatureExtractor:
    """
    Stateful feature extractor.
    Maintains history for velocity calculation and smoothing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.fps = self.config.dataset.fps
        
        # Smoothing filters
        self.orientation_filter = MedianFilter(window_size=5)
        self.height_filter = EMAFilter(alpha=0.3)
        self.aspect_ratio_filter = EMAFilter(alpha=0.5)
        self.velocity_filter = EMAFilter(alpha=0.4)  # Smooth velocity for stability
        
        # History for velocity
        self.prev_center_y: Optional[float] = None
        self.prev_orientation: Optional[float] = None
        
    def reset(self):
        """Reset history and filters."""
        self.orientation_filter.reset()
        self.height_filter.reset()
        self.aspect_ratio_filter.reset()
        self.velocity_filter.reset()
        self.prev_center_y = None
        self.prev_orientation = None

    def extract(self, 
                pose: Optional[PoseResult], 
                image_shape: Tuple[int, int],
                depth_features: Optional[Dict] = None,
                acc_data: Optional[Dict] = None) -> PoseFeatures:
        """
        Extract features from current frame data.
        
        Args:
            pose: YOLOv8 PoseResult (can be None if no person detected)
            image_shape: (height, width)
            depth_features: Row from annotation CSV (H, D, P40, etc.)
            acc_data: Accelerometer data {'sv_total': ...}
            
        Returns:
            PoseFeatures object
        """
        img_h, img_w = image_shape
        features = PoseFeatures()
        
        # --- Process External Sensor Data ---
        if depth_features:
            features.depth_hw_ratio = depth_features.get('HeightWidthRatio')
            features.depth_height_mm = depth_features.get('H')
            features.dist_to_floor_mm = depth_features.get('D')
            features.p40 = depth_features.get('P40')
            features.max_std_xz = depth_features.get('MaxStdXZ')
            
        if acc_data:
            features.acc_sv_total = acc_data.get('sv_total')

        # --- Process YOLO Pose Data ---
        if pose:
            # 1. Basic Confidence
            features.keypoints_conf_mean = float(np.mean(pose.keypoints[:, 2]))
            
            # 2. Aspect Ratio (Height / Width)
            # YOLO bbox format: [x1, y1, x2, y2]
            w = pose.bbox[2] - pose.bbox[0]
            h = pose.bbox[3] - pose.bbox[1]
            raw_ar = h / w if w > 0 else 0
            
            # Smooth Aspect Ratio
            features.bbox_aspect_ratio = self.aspect_ratio_filter.update(raw_ar)
            
            # 3. Normalized Height
            features.normalized_height = self.height_filter.update(h / img_h)

            # 4. Body Orientation (Spine Angle)
            # We use midpoint of shoulders and midpoint of hips to define the spine vector
            shoulder_center = pose.shoulder_center
            hip_center = pose.hip_center
            
            # Vector from Hip to Shoulder (pointing up)
            spine_vec = (shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
            
            # Angle with vertical (0, -1) [Remember Y increases downwards in images]
            # Ideally: Upright = (0, -H) -> Angle 0
            # Lying = (W, 0) -> Angle 90
            
            # Use atan2 to get angle
            # dx = spine_x, dy = spine_y. 
            # Vertical vector is (0, -1). 
            
            if np.linalg.norm(spine_vec) > 0:
                # Angle wrt vertical axis
                # abs(arctan(dx/dy)) -> 0 if dx=0 (vertical), 90 if dy=0 (horizontal)
                # Note: dy is negative for upright (shoulder above hip)
                angle_rad = np.arctan2(abs(spine_vec[0]), abs(spine_vec[1])) 
                angle_deg = np.degrees(angle_rad)
                
                # Smooth Orientation
                features.body_orientation = self.orientation_filter.update(angle_deg)
            else:
                # Fallback if keypoints collapsed
                features.body_orientation = self.orientation_filter.value or 0.0

            # 5. Velocity Calculation
            curr_center_y = pose.bbox_center[1]
            
            if self.prev_center_y is not None:
                # Pixels per frame -> downwards is positive
                features.velocity_y = (curr_center_y - self.prev_center_y)
                
                # Calculate Normalized Drop Velocity (Body stats per frame)
                # Normalize by current bbox height to make it scale invariant
                # If bbox height is 0, avoid division by zero
                if h > 0:
                    raw_normalized_vel = features.velocity_y / h
                    # Apply EMA smoothing for stability
                    features.normalized_drop_velocity = self.velocity_filter.update(raw_normalized_vel)
                else:
                    features.normalized_drop_velocity = 0.0
            
            self.prev_center_y = curr_center_y
            
            if self.prev_orientation is not None:
                 # Change in angle per frame
                features.orientation_velocity = abs(features.body_orientation - self.prev_orientation)
            self.prev_orientation = features.body_orientation
            
        else:
            # No pose detected: Decay values or hold last valid
            # For this implementation, we mostly return defaults or filtered values
            features.bbox_aspect_ratio = self.aspect_ratio_filter.value or 1.0
            features.normalized_height = self.height_filter.value or 0.0
            features.body_orientation = self.orientation_filter.value or 0.0
            
            # Reset velocity tracking on loss of tracking to prevent large jumps on re-acquisition
            self.prev_center_y = None
            self.prev_orientation = None

        return features

if __name__ == "__main__":
    print("Testing FeatureExtractor...")
    extractor = FeatureExtractor()
    
    # Mock data
    dummy_pose = PoseResult(
        bbox=np.array([100, 100, 200, 400]), # 100x300 box (Tall)
        confidence=0.9,
        keypoints=np.zeros((17, 3))
    )
    # Set mock keypoints for upright: Shoulders (150, 120), Hips (150, 250)
    dummy_pose.keypoints[5] = [140, 120, 0.9] # L Shoulder
    dummy_pose.keypoints[6] = [160, 120, 0.9] # R Shoulder
    dummy_pose.keypoints[11] = [140, 250, 0.9] # L Hip
    dummy_pose.keypoints[12] = [160, 250, 0.9] # R Hip
    
    dummy_depth = {
        'HeightWidthRatio': 2.5, # Tall
        'H': 1700,
        'D': 850
    }
    
    features = extractor.extract(dummy_pose, (600, 800), depth_features=dummy_depth)
    
    print(f"Aspect Ratio (Target ~3.0): {features.bbox_aspect_ratio:.2f}")
    print(f"Orientation (Target ~0.0): {features.body_orientation:.2f} deg")
    print(f"Depth H/W: {features.depth_hw_ratio}")
    print(f"Is Lying: {features.is_lying_pose}")
    
    # Test Lying Pose
    # Shoulders (120, 300), Hips (250, 300) -> Horizontal
    dummy_pose.keypoints[5] = [120, 290, 0.9]
    dummy_pose.keypoints[6] = [120, 310, 0.9]
    dummy_pose.keypoints[11] = [250, 290, 0.9]
    dummy_pose.keypoints[12] = [250, 310, 0.9]
    # BBox wide: 100, 250, 300, 350 -> 200x100
    dummy_pose.bbox = np.array([100, 250, 300, 350])
    
    dummy_depth['HeightWidthRatio'] = 0.5
    
    features = extractor.extract(dummy_pose, (600, 800), depth_features=dummy_depth)
    print("\n--- Lying Update ---")
    print(f"Aspect Ratio (Target ~0.5): {features.bbox_aspect_ratio:.2f}")
    print(f"Orientation (Target ~90.0): {features.body_orientation:.2f} deg")
    print(f"Is Lying: {features.is_lying_pose}")
    
