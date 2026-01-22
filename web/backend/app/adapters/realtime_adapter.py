"""
Real-time Frame-by-Frame Adapter
Wraps core AI for streaming inference WITHOUT modifying core logic.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

# Import core AI components (ZERO modifications)
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.tracking import PrimaryPersonTracker
from src.urfd.fallback_tracker import FallbackTracker
from src.urfd.features import compute_frame_features
from src.urfd.smoothing import FallStateMachine
from src.urfd.utils import load_config, validate_config, set_seed

class RealtimeSessionState:
    """
    Maintains state for a single real-time camera session.
    Reuses core AI classes without modification.
    """
    
    def __init__(self, config_path: str):
        """Initialize session with core AI components"""
        # Load and validate config
        self.config = load_config(config_path)
        validate_config(self.config)
        set_seed(self.config["seed"])
        
        # Initialize detector (shared across sessions if needed)
        self.detector = YOLOPoseDetector(
            model_path=self.config["yolo_model"],
            imgsz=self.config["imgsz"],
            conf_thres=self.config["conf_thres"],
            iou_thres=self.config["iou_thres"],
            preprocess_lowlight=self.config.get("preprocess_lowlight", False),
            gamma=self.config.get("gamma", 1.3),
            clahe_clip=self.config.get("clahe_clip", 2.0),
            clahe_grid=self.config.get("clahe_grid", 8)
        )
        
        # Initialize trackers
        self.tracker = PrimaryPersonTracker(self.config)
        self.fallback_tracker = FallbackTracker(
            tracker_type=self.config.get("fallback_tracker_type", "kcf"),
            max_gap=self.config.get("fallback_track_max_gap", 6)
        )
        
        # Initialize state machine
        self.state_machine = FallStateMachine(self.config)
        
        # Frame history
        self.all_features: List[Dict] = []
        self.all_states: List[str] = []
        self.all_scores: List[float] = []
        self.primary_indices: List[int] = []
        
        # Frame counter
        self.frame_count = 0
        
        # Alarm tracking
        self.last_alarm_frame = -1
        self.alarm_cooldown = 30  # frames
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through core AI pipeline.
        Returns telemetry + alarm status.
        
        Args:
            frame: RGB numpy array (H, W, 3)
        
        Returns:
            {
                "frameId": int,
                "state": str,
                "score": float,
                "isCandidate": bool,
                "isLying": bool,
                "isFallConfirmed": bool,
                "bbox": [x1, y1, x2, y2] or None,
                "keypoints": [...] or None,
                "features": {...},
                "alarm": bool,  # True if NEW alarm triggered
                "snapshotFrame": np.ndarray or None
            }
        """
        # Run YOLO detection
        detections = self.detector.detect(frame)
        
        # Apply fallback tracker if needed
        fallback_det = self.fallback_tracker.update(frame, detections)
        if fallback_det is not None:
            detections = [fallback_det]
        
        # Initialize fallback when we have valid primary
        if len(detections) > 0 and fallback_det is None:
            if self.tracker.track_bbox is not None:
                self.fallback_tracker.initialize(frame, self.tracker.track_bbox)
        
        # Compute features (reuse core function)
        features = compute_frame_features(
            detections,
            self.config["keypoint_conf_thres"],
            self.config["dy_window"],
            self.all_features,
            tracker=self.tracker,
            frame_idx=self.frame_count
        )
        
        self.all_features.append(features)
        self.primary_indices.append(features["primary_person_idx"])
        
        # Update state machine
        state, score = self.state_machine.update(features)
        self.all_states.append(state)
        self.all_scores.append(score)
        
        # Check for NEW alarm
        alarm = False
        snapshot = None
        if state == "FALL_CONFIRMED":
            # Check if this is a new alarm (not within cooldown)
            if self.frame_count - self.last_alarm_frame >= self.alarm_cooldown:
                alarm = True
                snapshot = frame.copy()  # Capture snapshot
                self.last_alarm_frame = self.frame_count
        
        # Extract bbox and keypoints for telemetry
        bbox = None
        keypoints = None
        if features["bbox"] is not None:
            x1, y1, x2, y2 = features["bbox"]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        # Get keypoints from detections if primary person detected
        if features["primary_person_idx"] >= 0:
            primary_det = features["detections"][features["primary_person_idx"]]
            if primary_det.get("keypoints") is not None:
                keypoints = primary_det["keypoints"].tolist()
        
        self.frame_count += 1
        
        return {
            "frameId": self.frame_count - 1,
            "state": state,
            "score": float(score),
            "isCandidate": state == "CANDIDATE",
            "isLying": features.get("is_lying", False),
            "isFallConfirmed": state == "FALL_CONFIRMED",
            "bbox": bbox,
            "keypoints": keypoints,
            "features": {
                "dy_peak": float(features.get("dy_peak", 0.0)),
                "height_drop": float(features.get("height_drop", 0.0)),
                "body_angle_deg": float(features.get("body_angle_deg")) if features.get("body_angle_deg") is not None else 0.0,
                "bbox_aspect_ratio": float(features.get("bbox_aspect_ratio")) if features.get("bbox_aspect_ratio") is not None else 0.0
            },
            "alarm": alarm,
            "snapshotFrame": snapshot
        }
    
    def get_summary(self) -> Dict:
        """Get session summary statistics"""
        return {
            "totalFrames": self.frame_count,
            "fallConfirmedFrames": sum(1 for s in self.all_states if s == "FALL_CONFIRMED"),
            "candidateFrames": sum(1 for s in self.all_states if s == "CANDIDATE"),
            "maxScore": max(self.all_scores) if self.all_scores else 0.0,
            "avgScore": np.mean(self.all_scores) if self.all_scores else 0.0
        }

class RealtimeSessionManager:
    """Manages multiple real-time sessions (one per camera/room)"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.sessions: Dict[str, RealtimeSessionState] = {}
    
    def get_or_create_session(self, session_id: str) -> RealtimeSessionState:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = RealtimeSessionState(self.config_path)
        return self.sessions[session_id]
    
    def remove_session(self, session_id: str):
        """Remove session and cleanup"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_count(self) -> int:
        """Get active session count"""
        return len(self.sessions)
