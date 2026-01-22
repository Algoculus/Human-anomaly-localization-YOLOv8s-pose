"""
Enhanced Feature Extraction Module - Based on Kwolek & Kepski 2014

Key improvements:
1. Better body orientation calculation using multiple keypoint pairs
2. Enhanced velocity features with acceleration
3. Improved aspect ratio with temporal stability
4. Activity classification features (bending vs falling vs lying)
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
from utils.smoothing import EMAFilter, MedianFilter, VectorEMAFilter, HistoryBuffer


@dataclass
class PoseFeatures:
    """Enhanced feature set for fall detection"""
    
    # === Pose-based Features (YOLO) ===
    keypoints_conf_mean: float = 0.0
    bbox_aspect_ratio: float = 1.0
    body_orientation: float = 0.0  # [0-90] degrees
    normalized_height: float = 1.0
    
    # Velocity & Acceleration
    velocity_y: float = 0.0  # pixels/frame
    velocity_x: float = 0.0
    acceleration_y: float = 0.0  # pixels/frame²
    normalized_drop_velocity: float = 0.0
    orientation_velocity: float = 0.0
    
    # Enhanced geometric features
    verticality_ratio: float = 1.0  # Height of vertical keypoints / total height
    symmetry_score: float = 1.0  # Left-right symmetry
    compactness: float = 0.0  # How compact the pose is
    
    # Activity-specific features
    hip_height_ratio: float = 1.0  # Hip height / image height
    shoulder_hip_distance: float = 0.0  # Normalized
    knee_bend_angle: float = 180.0  # Average knee angle
    
    # === Depth Features (Optional) ===
    depth_hw_ratio: Optional[float] = None
    depth_height_mm: Optional[float] = None
    dist_to_floor_mm: Optional[float] = None
    p40: Optional[float] = None
    max_std_xz: Optional[float] = None
    
    # === Accelerometer Features (Optional) ===
    acc_sv_total: Optional[float] = None
    acc_delta: Optional[float] = None  # Change in SV from previous frame
    
    # === Temporal Context ===
    motion_continuity: float = 1.0  # How smooth is the motion
    pose_stability: float = 1.0  # How stable is the pose
    
    @property
    def is_lying_pose(self) -> bool:
        """Enhanced lying detection using multiple cues"""
        lying_score = 0.0
        
        # 1. Depth features (most reliable)
        if self.depth_hw_ratio is not None:
            if self.depth_hw_ratio < 0.7:
                lying_score += 0.4
            if self.p40 is not None and self.p40 > 0.35:
                lying_score += 0.3
            if self.dist_to_floor_mm is not None and self.dist_to_floor_mm < 400:
                lying_score += 0.3
        else:
            # 2. Pose features (fallback)
            if self.bbox_aspect_ratio < 0.75:
                lying_score += 0.35
            if self.body_orientation > 55:
                lying_score += 0.35
            if self.hip_height_ratio < 0.3:
                lying_score += 0.3
        
        return lying_score >= 0.6
    
    @property
    def is_bending_pose(self) -> bool:
        """Detect bending/crouching vs falling"""
        # Bending: upright orientation but low height
        is_upright = self.body_orientation < 45
        is_low = self.hip_height_ratio < 0.5
        has_knee_bend = self.knee_bend_angle < 140
        is_slow = abs(self.velocity_y) < 3.0
        
        return is_upright and is_low and has_knee_bend and is_slow
    
    @property  
    def activity_type(self) -> str:
        """Classify current activity"""
        if self.is_lying_pose:
            if abs(self.velocity_y) < 1.5 and self.pose_stability > 0.8:
                return "LYING_STILL"
            else:
                return "LYING_MOVING"
        elif self.is_bending_pose:
            return "BENDING"
        elif abs(self.velocity_y) > 5.0 and self.acceleration_y > 2.0:
            return "FALLING"
        elif abs(self.velocity_y) > 2.0 or abs(self.velocity_x) > 2.0:
            return "WALKING"
        else:
            return "STANDING"


class FeatureExtractor:
    """Enhanced stateful feature extractor"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.fps = self.config.dataset.fps
        
        # Smoothing filters
        self.orientation_filter = MedianFilter(window_size=5)
        self.height_filter = EMAFilter(alpha=0.3)
        self.aspect_ratio_filter = EMAFilter(alpha=0.4)
        self.velocity_filter = EMAFilter(alpha=0.35)
        self.hip_height_filter = EMAFilter(alpha=0.4)
        
        # History buffers for acceleration & temporal features
        self.velocity_history = HistoryBuffer(max_length=10)
        self.pose_history = HistoryBuffer(max_length=15)
        self.bbox_history = HistoryBuffer(max_length=10)
        
        # Previous values for derivatives
        self.prev_center_y: Optional[float] = None
        self.prev_center_x: Optional[float] = None
        self.prev_velocity_y: Optional[float] = None
        self.prev_orientation: Optional[float] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        
    def reset(self):
        """Reset all state"""
        self.orientation_filter.reset()
        self.height_filter.reset()
        self.aspect_ratio_filter.reset()
        self.velocity_filter.reset()
        self.hip_height_filter.reset()
        
        self.velocity_history.clear()
        self.pose_history.clear()
        self.bbox_history.clear()
        
        self.prev_center_y = None
        self.prev_center_x = None
        self.prev_velocity_y = None
        self.prev_orientation = None
        self.prev_keypoints = None

    def extract(self, 
                pose: Optional[PoseResult], 
                image_shape: Tuple[int, int],
                depth_features: Optional[Dict] = None,
                acc_data: Optional[Dict] = None) -> PoseFeatures:
        """Extract comprehensive features from pose data"""
        
        img_h, img_w = image_shape
        features = PoseFeatures()
        
        # === Process External Sensors ===
        if depth_features:
            features.depth_hw_ratio = depth_features.get('HeightWidthRatio')
            features.depth_height_mm = depth_features.get('H')
            features.dist_to_floor_mm = depth_features.get('D')
            features.p40 = depth_features.get('P40')
            features.max_std_xz = depth_features.get('MaxStdXZ')
            
        if acc_data:
            features.acc_sv_total = acc_data.get('sv_total')
            # Calculate acceleration change
            if self.velocity_history and len(self.velocity_history) > 0:
                prev_acc = self.velocity_history[-1] if hasattr(self.velocity_history[-1], '__getitem__') else 0
                if features.acc_sv_total and prev_acc:
                    features.acc_delta = abs(features.acc_sv_total - prev_acc)

        # === Process YOLO Pose ===
        if pose and np.mean(pose.keypoints[:, 2]) > 0.3:  # Minimum confidence check
            kpts = pose.keypoints
            bbox = pose.bbox
            
            # 1. Confidence
            features.keypoints_conf_mean = float(np.mean(kpts[:, 2]))
            
            # 2. Bounding Box Features
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            if w > 0 and h > 0:
                raw_ar = h / w
                features.bbox_aspect_ratio = self.aspect_ratio_filter.update(raw_ar)
                features.normalized_height = self.height_filter.update(h / img_h)
                
                # Store bbox for temporal analysis
                self.bbox_history.append(np.array([w, h]))
            
            # 3. Enhanced Body Orientation
            features.body_orientation = self._calculate_body_orientation(kpts)
            
            # 4. Activity-specific geometric features
            features.verticality_ratio = self._calculate_verticality(kpts)
            features.symmetry_score = self._calculate_symmetry(kpts)
            features.compactness = self._calculate_compactness(kpts, bbox)
            
            # 5. Joint-based features
            features.hip_height_ratio = self._calculate_hip_height(kpts, img_h)
            features.shoulder_hip_distance = self._calculate_shoulder_hip_distance(kpts, h)
            features.knee_bend_angle = self._calculate_knee_angle(kpts)
            
            # 6. Velocity & Acceleration
            curr_center_y = (bbox[1] + bbox[3]) / 2
            curr_center_x = (bbox[0] + bbox[2]) / 2
            
            if self.prev_center_y is not None:
                raw_vel_y = curr_center_y - self.prev_center_y
                raw_vel_x = curr_center_x - self.prev_center_x
                
                features.velocity_y = self.velocity_filter.update(raw_vel_y)
                features.velocity_x = raw_vel_x  # No smoothing for x to detect lateral movement
                
                # Normalized drop velocity
                if h > 0:
                    features.normalized_drop_velocity = features.velocity_y / h
                
                # Acceleration
                if self.prev_velocity_y is not None:
                    features.acceleration_y = features.velocity_y - self.prev_velocity_y
                
                # Store velocity for history
                self.velocity_history.append(features.velocity_y)
            
            self.prev_center_y = curr_center_y
            self.prev_center_x = curr_center_x
            self.prev_velocity_y = features.velocity_y
            
            # 7. Orientation velocity
            if self.prev_orientation is not None:
                features.orientation_velocity = abs(features.body_orientation - self.prev_orientation)
            self.prev_orientation = features.body_orientation
            
            # 8. Temporal stability features
            features.motion_continuity = self._calculate_motion_continuity()
            features.pose_stability = self._calculate_pose_stability(kpts)
            
            # Store current pose for next frame
            self.prev_keypoints = kpts.copy()
            self.pose_history.append(kpts.copy())
            
        else:
            # No valid pose detected - use filtered values
            features.bbox_aspect_ratio = self.aspect_ratio_filter.value or 1.0
            features.normalized_height = self.height_filter.value or 0.0
            features.body_orientation = self.orientation_filter.value or 0.0
            
            # Reset tracking on lost pose
            self.prev_center_y = None
            self.prev_center_x = None
            self.prev_velocity_y = None

        return features
    
    def _calculate_body_orientation(self, kpts: np.ndarray) -> float:
        """
        Enhanced orientation using multiple keypoint pairs.
        Returns angle in degrees [0-90] where 0=upright, 90=horizontal
        """
        # Use shoulder-hip line as primary orientation
        shoulder_center = (kpts[5, :2] + kpts[6, :2]) / 2
        hip_center = (kpts[11, :2] + kpts[12, :2]) / 2
        
        spine_conf = min(kpts[5, 2], kpts[6, 2], kpts[11, 2], kpts[12, 2])
        
        if spine_conf > 0.3:
            dx = abs(shoulder_center[0] - hip_center[0])
            dy = abs(shoulder_center[1] - hip_center[1])
            
            if dy > 1:  # Avoid division by zero
                angle_rad = np.arctan2(dx, dy)
                angle_deg = np.degrees(angle_rad)
                return self.orientation_filter.update(angle_deg)
        
        # Fallback: use bbox aspect ratio
        if self.aspect_ratio_filter.value:
            # Convert aspect ratio to rough angle
            ar = self.aspect_ratio_filter.value
            if ar < 1:
                return self.orientation_filter.update(90 - ar * 45)
            else:
                return self.orientation_filter.update(max(0, 90 - ar * 30))
        
        return self.orientation_filter.value or 0.0
    
    def _calculate_verticality(self, kpts: np.ndarray) -> float:
        """Ratio of vertical extent to total extent"""
        valid_kpts = kpts[kpts[:, 2] > 0.3]
        if len(valid_kpts) < 3:
            return 1.0
        
        y_extent = valid_kpts[:, 1].max() - valid_kpts[:, 1].min()
        x_extent = valid_kpts[:, 0].max() - valid_kpts[:, 0].min()
        
        total_extent = np.sqrt(y_extent**2 + x_extent**2)
        if total_extent > 0:
            return y_extent / total_extent
        return 1.0
    
    def _calculate_symmetry(self, kpts: np.ndarray) -> float:
        """Left-right symmetry score"""
        # Compare left/right keypoint pairs
        pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        
        symmetry_scores = []
        for left_idx, right_idx in pairs:
            if kpts[left_idx, 2] > 0.3 and kpts[right_idx, 2] > 0.3:
                left_y = kpts[left_idx, 1]
                right_y = kpts[right_idx, 1]
                
                # Y-coordinate should be similar for symmetric pose
                y_diff = abs(left_y - right_y)
                symmetry_scores.append(1.0 / (1.0 + y_diff / 50.0))
        
        return np.mean(symmetry_scores) if symmetry_scores else 1.0
    
    def _calculate_compactness(self, kpts: np.ndarray, bbox: np.ndarray) -> float:
        """How compact is the pose within the bbox"""
        valid_kpts = kpts[kpts[:, 2] > 0.3]
        if len(valid_kpts) < 3:
            return 0.0
        
        kpt_area = (valid_kpts[:, 1].max() - valid_kpts[:, 1].min()) * \
                   (valid_kpts[:, 0].max() - valid_kpts[:, 0].min())
        
        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        
        if bbox_area > 0:
            return kpt_area / bbox_area
        return 0.0
    
    def _calculate_hip_height(self, kpts: np.ndarray, img_h: float) -> float:
        """Hip center height ratio"""
        if kpts[11, 2] > 0.3 and kpts[12, 2] > 0.3:
            hip_y = (kpts[11, 1] + kpts[12, 1]) / 2
            ratio = 1.0 - (hip_y / img_h)  # Invert so higher = higher hip
            return self.hip_height_filter.update(ratio)
        
        return self.hip_height_filter.value or 0.5
    
    def _calculate_shoulder_hip_distance(self, kpts: np.ndarray, bbox_h: float) -> float:
        """Normalized distance between shoulder and hip centers"""
        shoulder_center = (kpts[5, :2] + kpts[6, :2]) / 2
        hip_center = (kpts[11, :2] + kpts[12, :2]) / 2
        
        conf = min(kpts[5, 2], kpts[6, 2], kpts[11, 2], kpts[12, 2])
        
        if conf > 0.3 and bbox_h > 0:
            dist = np.linalg.norm(shoulder_center - hip_center)
            return dist / bbox_h
        
        return 0.5  # Default
    
    def _calculate_knee_angle(self, kpts: np.ndarray) -> float:
        """Average knee bend angle"""
        angles = []
        
        # Left knee: hip(11), knee(13), ankle(15)
        if all(kpts[i, 2] > 0.3 for i in [11, 13, 15]):
            angle = self._angle_between_points(kpts[11, :2], kpts[13, :2], kpts[15, :2])
            angles.append(angle)
        
        # Right knee: hip(12), knee(14), ankle(16)
        if all(kpts[i, 2] > 0.3 for i in [12, 14, 16]):
            angle = self._angle_between_points(kpts[12, :2], kpts[14, :2], kpts[16, :2])
            angles.append(angle)
        
        return np.mean(angles) if angles else 180.0
    
    def _angle_between_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(cos_angle))
        return angle
    
    def _calculate_motion_continuity(self) -> float:
        """How smooth is the motion over recent frames"""
        if len(self.velocity_history) < 3:
            return 1.0
        
        recent_vels = self.velocity_history.get_last_n(5)
        if not recent_vels:
            return 1.0
        
        # Calculate variance in velocity
        vel_std = np.std(recent_vels)
        # Low variance = high continuity
        continuity = 1.0 / (1.0 + vel_std / 5.0)
        
        return continuity
    
    def _calculate_pose_stability(self, kpts: np.ndarray) -> float:
        """How stable is the pose compared to recent history"""
        if len(self.pose_history) < 3 or self.prev_keypoints is None:
            return 1.0
        
        # Compare current keypoints to recent average
        recent_poses = self.pose_history.get_last_n(5)
        if not recent_poses:
            return 1.0
        
        # Calculate average keypoint displacement
        displacements = []
        for prev_kpts in recent_poses:
            valid_mask = (kpts[:, 2] > 0.3) & (prev_kpts[:, 2] > 0.3)
            if valid_mask.sum() > 0:
                disp = np.linalg.norm(kpts[valid_mask, :2] - prev_kpts[valid_mask, :2], axis=1)
                displacements.extend(disp)
        
        if displacements:
            avg_disp = np.mean(displacements)
            # Low displacement = high stability
            stability = 1.0 / (1.0 + avg_disp / 20.0)
            return stability
        
        return 1.0


if __name__ == "__main__":
    print("Testing Enhanced FeatureExtractor...")
    extractor = FeatureExtractor()
    
    # Mock upright pose
    dummy_pose = PoseResult(
        bbox=np.array([100, 100, 200, 400]),
        confidence=0.9,
        keypoints=np.zeros((17, 3))
    )
    
    # Set realistic keypoints for standing person
    dummy_pose.keypoints[5] = [140, 150, 0.9]  # L Shoulder
    dummy_pose.keypoints[6] = [160, 150, 0.9]  # R Shoulder
    dummy_pose.keypoints[11] = [140, 280, 0.9]  # L Hip
    dummy_pose.keypoints[12] = [160, 280, 0.9]  # R Hip
    dummy_pose.keypoints[13] = [135, 350, 0.85]  # L Knee
    dummy_pose.keypoints[14] = [165, 350, 0.85]  # R Knee
    dummy_pose.keypoints[15] = [130, 390, 0.8]  # L Ankle
    dummy_pose.keypoints[16] = [170, 390, 0.8]  # R Ankle
    
    features = extractor.extract(dummy_pose, (480, 640))
    
    print(f"\n=== Standing Pose ===")
    print(f"Orientation: {features.body_orientation:.1f}° (expect ~0-20°)")
    print(f"Aspect Ratio: {features.bbox_aspect_ratio:.2f} (expect >1.5)")
    print(f"Activity: {features.activity_type}")
    print(f"Hip Height: {features.hip_height_ratio:.2f}")
    print(f"Knee Angle: {features.knee_bend_angle:.1f}°")
    print(f"Verticality: {features.verticality_ratio:.2f}")
    print(f"Is Lying: {features.is_lying_pose}")
    print(f"Is Bending: {features.is_bending_pose}")
    
    print("\n✓ Feature extractor test complete!")