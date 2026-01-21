"""
YOLOv8s-Pose Inference Module

This module provides pose estimation using YOLOv8s-pose pretrained model.
No training is done - we use the pretrained weights directly.

Key features:
- Person detection with bounding boxes
- 17-keypoint pose estimation (COCO format)
- Keypoint confidence scores
- Integration with simple tracker for temporal consistency
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import cv2
from loguru import logger

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Some features will be unavailable.")

import sys
sys.path.append(str(Path(__file__).parent))
from config import Config, get_config, ModelConfig


# COCO keypoint indices for YOLOv8-pose (17 keypoints)
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),           # Nose to eyes
    (1, 3), (2, 4),           # Eyes to ears
    (5, 6),                   # Shoulder to shoulder
    (5, 7), (7, 9),           # Left arm
    (6, 8), (8, 10),          # Right arm
    (5, 11), (6, 12),         # Shoulders to hips
    (11, 12),                 # Hip to hip
    (11, 13), (13, 15),       # Left leg
    (12, 14), (14, 16),       # Right leg
]


@dataclass
class PoseResult:
    """
    Result from pose estimation on a single frame.
    """
    # Detection info
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float  # Detection confidence
    
    # Keypoints: (17, 3) array where each row is [x, y, confidence]
    keypoints: np.ndarray
    
    # Convenience accessors
    @property
    def keypoint_xy(self) -> np.ndarray:
        """Get keypoint coordinates only (17, 2)."""
        return self.keypoints[:, :2]
    
    @property
    def keypoint_conf(self) -> np.ndarray:
        """Get keypoint confidences only (17,)."""
        return self.keypoints[:, 2]
    
    @property
    def num_valid_keypoints(self, threshold: float = 0.5) -> int:
        """Count keypoints with confidence above threshold."""
        return int(np.sum(self.keypoints[:, 2] > threshold))
    
    def get_keypoint(self, idx: int) -> Tuple[float, float, float]:
        """Get keypoint by index: (x, y, conf)."""
        return tuple(self.keypoints[idx])
    
    def get_keypoint_by_name(self, name: str) -> Tuple[float, float, float]:
        """Get keypoint by name: (x, y, conf)."""
        idx = KEYPOINT_NAMES.index(name)
        return self.get_keypoint(idx)
    
    @property
    def left_shoulder(self) -> Tuple[float, float, float]:
        return self.get_keypoint(5)
    
    @property
    def right_shoulder(self) -> Tuple[float, float, float]:
        return self.get_keypoint(6)
    
    @property
    def left_hip(self) -> Tuple[float, float, float]:
        return self.get_keypoint(11)
    
    @property
    def right_hip(self) -> Tuple[float, float, float]:
        return self.get_keypoint(12)
    
    @property
    def shoulder_center(self) -> Tuple[float, float]:
        """Get center point between shoulders."""
        ls = self.keypoints[5]
        rs = self.keypoints[6]
        if ls[2] < 0.3 or rs[2] < 0.3:
            # Low confidence, use available one
            if ls[2] > rs[2]:
                return (ls[0], ls[1])
            else:
                return (rs[0], rs[1])
        return ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    
    @property
    def hip_center(self) -> Tuple[float, float]:
        """Get center point between hips."""
        lh = self.keypoints[11]
        rh = self.keypoints[12]
        if lh[2] < 0.3 or rh[2] < 0.3:
            if lh[2] > rh[2]:
                return (lh[0], lh[1])
            else:
                return (rh[0], rh[1])
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    
    @property
    def bbox_center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        h = self.bbox_height
        if h == 0:
            return 1.0
        return self.bbox_width / h
    
    def is_pose_reliable(self, min_keypoints: int = 8, conf_threshold: float = 0.5) -> bool:
        """
        Check if pose estimation is reliable.
        
        Args:
            min_keypoints: Minimum number of keypoints above threshold
            conf_threshold: Confidence threshold for valid keypoints
            
        Returns:
            True if pose has enough valid keypoints
        """
        valid_count = np.sum(self.keypoints[:, 2] > conf_threshold)
        return valid_count >= min_keypoints


class PoseEstimator:
    """
    YOLOv8s-Pose based pose estimator.
    
    Usage:
        estimator = PoseEstimator()
        results = estimator.predict(image)
        for pose in results:
            print(pose.keypoints)
    """
    
    def __init__(self, config: Optional[Config] = None, model_path: Optional[str] = None):
        """
        Initialize pose estimator.
        
        Args:
            config: Configuration object
            model_path: Optional path to YOLO model. If None, uses config or downloads.
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.config = config or get_config()
        self.model_config = self.config.model
        
        # Determine model path
        if model_path is None:
            model_path = self.model_config.model_name
        
        logger.info(f"Loading YOLOv8 pose model: {model_path}")
        self.model = YOLO(model_path)
        
        # Move to device
        if self.model_config.device == "cuda":
            import torch
            if torch.cuda.is_available():
                logger.info("Using CUDA for inference")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                self.model_config.device = "cpu"
        
        logger.info("Pose estimator initialized successfully")
    
    def predict(self, image: np.ndarray, 
                conf_threshold: Optional[float] = None,
                max_persons: int = 1) -> List[PoseResult]:
        """
        Run pose estimation on an image.
        
        Args:
            image: BGR image (OpenCV format)
            conf_threshold: Optional confidence threshold override
            max_persons: Maximum number of persons to return (sorted by confidence)
            
        Returns:
            List of PoseResult objects
        """
        if conf_threshold is None:
            conf_threshold = self.model_config.conf_threshold
        
        # Run inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=self.model_config.iou_threshold,
            device=self.model_config.device,
            imgsz=self.model_config.imgsz,
            verbose=self.model_config.verbose,
        )
        
        # Parse results
        pose_results = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            result = results[0]
            
            # Get boxes and keypoints
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # Get bounding box
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Get keypoints for this person
                    # keypoints.data shape: (num_persons, 17, 3) where 3 = (x, y, conf)
                    kpts = keypoints.data[i].cpu().numpy()
                    
                    pose_results.append(PoseResult(
                        bbox=bbox,
                        confidence=conf,
                        keypoints=kpts
                    ))
        
        # Sort by confidence and limit
        pose_results.sort(key=lambda x: -x.confidence)
        if max_persons > 0:
            pose_results = pose_results[:max_persons]
        
        return pose_results
    
    def predict_batch(self, images: List[np.ndarray], 
                      conf_threshold: Optional[float] = None) -> List[List[PoseResult]]:
        """
        Run pose estimation on a batch of images.
        
        Args:
            images: List of BGR images
            conf_threshold: Optional confidence threshold override
            
        Returns:
            List of lists of PoseResult objects (one list per image)
        """
        if conf_threshold is None:
            conf_threshold = self.model_config.conf_threshold
        
        # Run batch inference
        results = self.model.predict(
            images,
            conf=conf_threshold,
            iou=self.model_config.iou_threshold,
            device=self.model_config.device,
            imgsz=self.model_config.imgsz,
            verbose=self.model_config.verbose,
        )
        
        all_pose_results = []
        
        for result in results:
            pose_results = []
            
            if result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        kpts = keypoints.data[i].cpu().numpy()
                        
                        pose_results.append(PoseResult(
                            bbox=bbox,
                            confidence=conf,
                            keypoints=kpts
                        ))
            
            all_pose_results.append(pose_results)
        
        return all_pose_results


def draw_pose(image: np.ndarray, pose: PoseResult, 
              color_bbox: Tuple[int, int, int] = (0, 255, 0),
              color_skeleton: Tuple[int, int, int] = (255, 255, 0),
              conf_threshold: float = 0.5,
              draw_bbox: bool = True,
              draw_keypoints: bool = True,
              draw_skeleton: bool = True) -> np.ndarray:
    """
    Draw pose estimation results on an image.
    
    Args:
        image: BGR image to draw on (will be modified in place)
        pose: PoseResult to visualize
        color_bbox: BGR color for bounding box
        color_skeleton: BGR color for skeleton
        conf_threshold: Minimum keypoint confidence to draw
        draw_bbox: Whether to draw bounding box
        draw_keypoints: Whether to draw keypoints
        draw_skeleton: Whether to draw skeleton connections
        
    Returns:
        Image with drawings (same as input, modified in place)
    """
    if draw_bbox:
        x1, y1, x2, y2 = pose.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_bbox, 2)
        
        # Draw confidence
        label = f"{pose.confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 1)
    
    if draw_skeleton:
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start_kp = pose.keypoints[start_idx]
            end_kp = pose.keypoints[end_idx]
            
            if start_kp[2] > conf_threshold and end_kp[2] > conf_threshold:
                start_pt = (int(start_kp[0]), int(start_kp[1]))
                end_pt = (int(end_kp[0]), int(end_kp[1]))
                cv2.line(image, start_pt, end_pt, color_skeleton, 2)
    
    if draw_keypoints:
        for i, kp in enumerate(pose.keypoints):
            if kp[2] > conf_threshold:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    
    return image


class PoseInferencePipeline:
    """
    Complete inference pipeline for fall detection.
    
    Combines pose estimation with tracking for temporal consistency.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize inference pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.estimator = PoseEstimator(self.config)
        
        # Import tracker
        from utils.tracking import SimpleTracker
        self.tracker = SimpleTracker(
            iou_threshold=self.config.tracking.iou_threshold,
            max_age=self.config.tracking.max_age,
            min_hits=self.config.tracking.min_hits
        )
        
        self.frame_count = 0
        logger.info("Pose inference pipeline initialized")
    
    def process_frame(self, image: np.ndarray) -> Dict:
        """
        Process a single frame.
        
        Args:
            image: BGR image
            
        Returns:
            Dict with:
                - 'poses': List of PoseResult
                - 'tracks': List of active tracks
                - 'primary_track': The main tracked person (or None)
                - 'frame_number': Current frame number
        """
        self.frame_count += 1
        
        # Run pose estimation
        poses = self.estimator.predict(image, max_persons=5)
        
        # Prepare detections for tracker
        detections = []
        for pose in poses:
            detections.append({
                'bbox': pose.bbox,
                'confidence': pose.confidence,
                'keypoints': pose.keypoints[:, :2],
                'keypoint_confidences': pose.keypoints[:, 2],
            })
        
        # Update tracker
        active_tracks = self.tracker.update(detections)
        primary_track = self.tracker.get_primary_track()
        
        return {
            'poses': poses,
            'tracks': active_tracks,
            'primary_track': primary_track,
            'frame_number': self.frame_count
        }
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.frame_count = 0


if __name__ == "__main__":
    """Test the pose inference module."""
    from loguru import logger
    import sys
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("Testing Pose Inference Module...")
    print("=" * 50)
    
    # Test with a sample image from the dataset
    from data.urfall_loader import load_sequence
    
    # Load fall-01 sequence
    seq = load_sequence("fall-01")
    if seq and seq.cam0 and seq.cam0.frames:
        frame_data = seq.cam0.frames[50]  # Get a middle frame
        print(f"\nLoading frame {frame_data.frame_number} from {seq.sequence_id}")
        print(f"  Label: {frame_data.label} ({'lying' if frame_data.label == 1 else 'not lying'})")
        
        # Load image
        image = frame_data.load_rgb()
        if image is not None:
            print(f"  Image shape: {image.shape}")
            
            # Run pose estimation
            estimator = PoseEstimator()
            poses = estimator.predict(image)
            
            print(f"\n  Detected {len(poses)} person(s)")
            
            for i, pose in enumerate(poses):
                print(f"\n  Person {i+1}:")
                print(f"    BBox: {pose.bbox.astype(int)}")
                print(f"    Confidence: {pose.confidence:.3f}")
                print(f"    Valid keypoints: {pose.num_valid_keypoints}")
                print(f"    Pose reliable: {pose.is_pose_reliable()}")
                print(f"    Aspect ratio: {pose.aspect_ratio:.3f}")
                
                # Get key body points
                sc = pose.shoulder_center
                hc = pose.hip_center
                print(f"    Shoulder center: ({sc[0]:.1f}, {sc[1]:.1f})")
                print(f"    Hip center: ({hc[0]:.1f}, {hc[1]:.1f})")
            
            # Draw and save result
            if poses:
                output_img = image.copy()
                for pose in poses:
                    draw_pose(output_img, pose)
                
                output_path = Path("output/test_pose.png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), output_img)
                print(f"\n  Saved visualization to: {output_path}")
        
        # Clear cache
        frame_data.clear_cache()
    
    print("\n" + "=" * 50)
    print("Pose inference test complete!")
