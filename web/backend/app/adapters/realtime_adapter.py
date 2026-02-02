# Real-time Frame-by-Frame Adapter
# Wraps core AI for streaming inference WITHOUT modifying core logic.

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

# Import core AI components (ZERO modifications)
from src.urfd.yolo_pose import YOLOPoseDetector
from src.urfd.tracking import MultiPersonTracker
from src.urfd.features import compute_frame_features
from src.urfd.smoothing import FallStateMachine
from src.urfd.utils import load_config, validate_config, set_seed

class RealtimeSessionState:
    # Maintains state for a single real-time camera session.
    # Reuses core AI classes without modification.
    
    def __init__(self, config_path: Union[str, Path]):
        # Initialize session with core AI components
        # Load and validate config (convert Path to str if needed)
        self.config = load_config(str(config_path))
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
        
        # Initialize tracker
        self.tracker = MultiPersonTracker(self.config)
        
        # Per-track state
        self.state_machines = {} # tid -> FallStateMachine
        self.histories = {}      # tid -> list of features
        
        # Frame counter
        self.frame_count = 0
        
        # Alarm tracking
        self.last_alarm_frame = -1
        self.alarm_cooldown = 30  # frames
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        # Process single frame through core AI pipeline.
        # Returns telemetry + alarm status.
        #
        # Args:
        #     frame: RGB numpy array (H, W, 3)
        #
        # Returns:
        #     {
        #         "frameId": int,
        #         "tracks": [
        #             {
        #                 "id": int,
        #                 "state": str,
        #                 "score": float,
        #                 "bbox": [x1, y1, x2, y2],
        #                 "keypoints": [...]
        #             }, ...
        #         ],
        #         "alarm": bool,  # True if NEW alarm triggered
        #         "snapshotFrame": np.ndarray or None
        #     }
        
        # Run YOLO detection
        detections = self.detector.detect(frame)
        
        # Update tracker
        matches = self.tracker.update(detections, self.config["keypoint_conf_thres"], self.frame_count)
        
        current_tracks_data = []
        max_frame_score = 0.0
        alarm_triggered = False
        
        # Process all active tracks
        active_tids = list(self.tracker.tracks.keys())
        
        for tid in active_tids:
            # Get detection if matched
            det = None
            if tid in matches:
                det = detections[matches[tid]]
                
            # Initialize track state if new
            if tid not in self.state_machines:
                self.state_machines[tid] = FallStateMachine(self.config)
                self.histories[tid] = []
                
            # Compute features
            features = compute_frame_features(
                det,
                self.config["keypoint_conf_thres"],
                self.config["dy_window"],
                self.histories[tid],
                track_id=tid,
                frame_idx=self.frame_count
            )
            
            self.histories[tid].append(features)
            
            # Update state machine
            state, score = self.state_machines[tid].update(features)
            
            if score > max_frame_score:
                max_frame_score = score
            
            # Check for alarm
            if state == "FALL_CONFIRMED":
                 alarm_triggered = True
            
            # Prepare track data
            track_data = {
                "id": tid,
                "state": state,
                "score": float(score),
                "bbox": None,
                "keypoints": None,
                "dy_peak": float(features.get("dy_peak", 0.0)),  # For debugging
                "body_angle": float(features.get("body_angle_deg", 0.0)) if features.get("body_angle_deg") else None
            }
            
            if features["bbox"] is not None:
                x1, y1, x2, y2 = features["bbox"]
                track_data["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
                
            if det and det.get("keypoints") is not None:
                track_data["keypoints"] = det["keypoints"].tolist()
            
            current_tracks_data.append(track_data)
        
        # Handle global alarm cooldown
        final_alarm = False
        snapshot = None
        if alarm_triggered:
            if self.frame_count - self.last_alarm_frame >= self.alarm_cooldown:
                final_alarm = True
                snapshot = frame.copy()
                self.last_alarm_frame = self.frame_count
        
        self.frame_count += 1
        
        return {
            "frameId": self.frame_count - 1,
            "tracks": current_tracks_data,
            "alarm": final_alarm,
            "snapshotFrame": snapshot
        }
    
    def get_summary(self) -> Dict:
        # Get session summary statistics
        # Simple summary based on current state machines
        fall_counts = 0
        for sm in self.state_machines.values():
            if sm.state == "FALL_CONFIRMED":
                fall_counts += 1
                
        return {
            "totalFrames": self.frame_count,
            "activeTracks": len(self.tracker.tracks),
            "fallConfirmedTracks": fall_counts
        }

class RealtimeSessionManager:
    # Manages multiple real-time sessions (one per camera/room)
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = config_path
        self.sessions: Dict[str, RealtimeSessionState] = {}
    
    def get_or_create_session(self, session_id: str) -> RealtimeSessionState:
        # Get existing session or create new one
        if session_id not in self.sessions:
            self.sessions[session_id] = RealtimeSessionState(self.config_path)
        return self.sessions[session_id]
    
    def remove_session(self, session_id: str):
        # Remove session and cleanup
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_count(self) -> int:
        # Get active session count
        return len(self.sessions)
