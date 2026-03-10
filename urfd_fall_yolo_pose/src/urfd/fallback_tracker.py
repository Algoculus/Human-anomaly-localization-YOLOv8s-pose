"""
Fallback classical tracker for when YOLO fails to detect in dark scenes.
Uses OpenCV KCF/CSRT tracker to propagate bbox.
"""

import cv2
import numpy as np

class FallbackTracker:
    """
    Classical bbox tracker for propagating detections when YOLO misses.
    
    Use cases:
    - Motion blur causing YOLO to miss
    - Temporary occlusion
    - Low-light conditions
    
    Supported trackers:
    - KCF (Kernelized Correlation Filter) - fast, less accurate
    - CSRT (Channel and Spatial Reliability) - slower, more accurate
    """
    
    def __init__(self, tracker_type="kcf", max_gap=6):
        """
        Initialize fallback tracker.
        
        Args:
            tracker_type: "kcf", "csrt", or "none"
            max_gap: Maximum frames to propagate without YOLO detection
        """
        self.tracker_type = tracker_type.lower()
        self.max_gap = max_gap
        self.tracker = None
        self.missing_count = 0
        self.last_bbox = None
        self.is_active = False
    
    def _create_tracker(self):
        """
        Create OpenCV tracker instance.
        
        Handles different OpenCV versions (legacy vs new API).
        
        Returns:
            OpenCV tracker or None if unavailable
        """
        if self.tracker_type == "kcf":
            # Try new API first, then legacy API
            try:
                return cv2.legacy.TrackerKCF_create()
            except AttributeError:
                try:
                    return cv2.TrackerKCF_create()
                except AttributeError:
                    return None
        elif self.tracker_type == "csrt":
            try:
                return cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                try:
                    return cv2.TrackerCSRT_create()
                except AttributeError:
                    return None
        else:
            return None
    
    def initialize(self, frame, bbox):
        """
        Initialize tracker with a bbox.
        
        Called when YOLO successfully detects a person.
        
        Args:
            frame: Current frame (BGR)
            bbox: [x1, y1, x2, y2]
        """
        if self.tracker_type == "none":
            return
        
        # Convert xyxy to xywh format for OpenCV tracker
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        self.tracker = self._create_tracker()
        if self.tracker is not None:
            self.tracker.init(frame, (x1, y1, w, h))
        self.last_bbox = bbox
        self.is_active = True
        self.missing_count = 0
    
    def update(self, frame, yolo_detections):
        """
        Update tracker and return pseudo-detection if needed.
        
        Logic:
        1. If YOLO detected something: reset and return None
        2. If within max_gap: propagate bbox using classical tracker
        3. If beyond max_gap: give up and return None
        
        Args:
            frame: Current frame (BGR)
            yolo_detections: List of YOLO detections (may be empty)
        
        Returns:
            pseudo_detection: Dict with bbox if propagating, None otherwise
        """
        # If YOLO detected, no need for fallback
        if len(yolo_detections) > 0:
            self.missing_count = 0
            self.is_active = False
            return None
        
        self.missing_count += 1
        
        # Exceeded max gap or not active
        if self.missing_count > self.max_gap or not self.is_active:
            return None
        
        # Fallback: use last known bbox (if no OpenCV tracker available)
        if self.tracker is None:
            if self.last_bbox is not None:
                x1, y1, x2, y2 = self.last_bbox
                pseudo_detection = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": 0.5,  # Reduced confidence for fallback
                    "keypoints": None,
                    "bbox_area": (x2 - x1) * (y2 - y1),
                    "is_fallback": True  # Flag for feature extraction
                }
                return pseudo_detection
            return None
        
        # Update OpenCV tracker
        success, bbox_xywh = self.tracker.update(frame)
        
        if success:
            # Convert xywh back to xyxy
            x, y, w, h = bbox_xywh
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Clamp to frame boundaries
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame))
            y1 = max(0, min(y1, h_frame))
            x2 = max(0, min(x2, w_frame))
            y2 = max(0, min(y2, h_frame))
            
            pseudo_detection = {
                "bbox": [x1, y1, x2, y2],
                "conf": 0.5,
                "keypoints": None,  # No keypoints from classical tracker
                "bbox_area": (x2 - x1) * (y2 - y1),
                "is_fallback": True
            }
            
            self.last_bbox = [x1, y1, x2, y2]
            return pseudo_detection
        else:
            # Tracker lost the target
            self.is_active = False
            return None
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = None
        self.missing_count = 0
        self.last_bbox = None
        self.is_active = False