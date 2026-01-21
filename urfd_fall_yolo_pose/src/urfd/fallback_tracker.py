"""
Fallback classical tracker for when YOLO fails to detect in dark scenes.
Uses OpenCV KCF/CSRT tracker to propagate bbox.
"""

import cv2
import numpy as np

class FallbackTracker:
    """Classical bbox tracker for propagating detections when YOLO misses."""
    
    def __init__(self, tracker_type="kcf", max_gap=6):
        """Initialize fallback tracker.
        
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
        """Create OpenCV tracker instance."""
        if self.tracker_type == "kcf":
            try:
                # Try new API first (OpenCV 4.5.1+)
                return cv2.legacy.TrackerKCF_create()
            except AttributeError:
                try:
                    # Fall back to old API
                    return cv2.TrackerKCF_create()
                except AttributeError:
                    # If neither works, use None (will fall back to simple bbox propagation)
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
        """Initialize tracker with a bbox.
        
        Args:
            frame: Current frame (BGR)
            bbox: [x1, y1, x2, y2]
        """
        if self.tracker_type == "none":
            return
        
        # Convert to (x, y, w, h)
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
        """Update tracker and return pseudo-detection if needed.
        
        Args:
            frame: Current frame (BGR)
            yolo_detections: List of YOLO detections (may be empty)
        
        Returns:
            pseudo_detection: Dict with bbox (and no keypoints) if propagating,
                             None if not using fallback
        """
        if len(yolo_detections) > 0:
            # YOLO detected something, reset fallback
            self.missing_count = 0
            self.is_active = False
            return None
        
        # YOLO returned nothing
        self.missing_count += 1
        
        if self.missing_count > self.max_gap or not self.is_active:
            # Exceeded gap or not initialized
            return None
        
        # Try to propagate bbox with tracker
        if self.tracker is None:
            # No tracker available, use simple bbox propagation (last known bbox)
            if self.last_bbox is not None:
                x1, y1, x2, y2 = self.last_bbox
                pseudo_detection = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": 0.5,
                    "keypoints": None,
                    "bbox_area": (x2 - x1) * (y2 - y1),
                    "is_fallback": True
                }
                return pseudo_detection
            return None
        
        success, bbox_xywh = self.tracker.update(frame)
        
        if success:
            x, y, w, h = bbox_xywh
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Clamp to frame bounds
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame))
            y1 = max(0, min(y1, h_frame))
            x2 = max(0, min(x2, w_frame))
            y2 = max(0, min(y2, h_frame))
            
            # Create pseudo-detection (bbox only, no keypoints)
            pseudo_detection = {
                "bbox": [x1, y1, x2, y2],
                "conf": 0.5,  # Dummy confidence
                "keypoints": None,  # No keypoints available
                "bbox_area": (x2 - x1) * (y2 - y1),
                "is_fallback": True
            }
            
            self.last_bbox = [x1, y1, x2, y2]
            return pseudo_detection
        else:
            # Tracker failed
            self.is_active = False
            return None
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = None
        self.missing_count = 0
        self.last_bbox = None
        self.is_active = False
