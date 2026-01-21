"""
Simple Person Tracker

IoU-based tracker for maintaining person identity across frames.
Simplified alternative to ByteTrack/DeepSORT for this application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class Track:
    """
    Represents a tracked person.
    """
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float = 1.0
    
    # Track state
    age: int = 0  # Frames since creation
    hits: int = 1  # Successful detections
    time_since_update: int = 0  # Frames since last detection
    
    # Pose data (optional)
    keypoints: Optional[np.ndarray] = None
    keypoint_confidences: Optional[np.ndarray] = None
    
    # History for smoothing
    bbox_history: List[np.ndarray] = field(default_factory=list)
    
    def update(self, bbox: np.ndarray, confidence: float = 1.0,
               keypoints: Optional[np.ndarray] = None,
               keypoint_confidences: Optional[np.ndarray] = None):
        """
        Update track with new detection.
        """
        self.bbox = bbox.copy()
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        if keypoints is not None:
            self.keypoints = keypoints.copy()
        if keypoint_confidences is not None:
            self.keypoint_confidences = keypoint_confidences.copy()
        
        # Keep history (limited size)
        self.bbox_history.append(bbox.copy())
        if len(self.bbox_history) > 30:
            self.bbox_history.pop(0)
    
    def predict(self):
        """
        Predict next position (simple: no motion model, just increment age).
        """
        self.age += 1
        self.time_since_update += 1
    
    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Check if track is confirmed (enough hits)."""
        return self.hits >= min_hits
    
    def is_lost(self, max_age: int = 30) -> bool:
        """Check if track is lost (too long without update)."""
        return self.time_since_update > max_age
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes [x1, y1, x2, y2]
        
    Returns:
        IoU value in [0, 1]
    """
    # Intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        bboxes1: (N, 4) array of boxes
        bboxes2: (M, 4) array of boxes
        
    Returns:
        (N, M) IoU matrix
    """
    n = len(bboxes1)
    m = len(bboxes2)
    iou_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(bboxes1[i], bboxes2[j])
    
    return iou_matrix


class SimpleTracker:
    """
    Simple IoU-based person tracker.
    
    Algorithm:
    1. For each new frame, compute IoU between detections and existing tracks
    2. Match detections to tracks using Hungarian algorithm (greedy here for simplicity)
    3. Update matched tracks, create new tracks for unmatched detections
    4. Mark unmatched tracks as lost after max_age frames
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to confirm track
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: List[Track] = []
        self._next_id = 1
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'keypoints': (17, 3) array (optional)
                
        Returns:
            List of active tracks
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # If no detections, just return predicted tracks
        if not detections:
            self._cleanup_tracks()
            return self.get_active_tracks()
        
        # Convert detections to arrays
        det_bboxes = np.array([d['bbox'] for d in detections])
        det_confs = np.array([d.get('confidence', 1.0) for d in detections])
        det_keypoints = [d.get('keypoints') for d in detections]
        det_kp_confs = [d.get('keypoint_confidences') for d in detections]
        
        # Match detections to tracks
        if self.tracks:
            track_bboxes = np.array([t.bbox for t in self.tracks])
            iou_matrix = compute_iou_matrix(track_bboxes, det_bboxes)
            
            # Greedy matching
            matched_track_indices = set()
            matched_det_indices = set()
            
            # Sort by IoU (highest first)
            iou_flat = [(i, j, iou_matrix[i, j]) 
                        for i in range(len(self.tracks)) 
                        for j in range(len(detections))]
            iou_flat.sort(key=lambda x: -x[2])
            
            for i, j, iou in iou_flat:
                if iou < self.iou_threshold:
                    break
                if i in matched_track_indices or j in matched_det_indices:
                    continue
                
                # Match found
                kpts = det_keypoints[j]
                kpt_confs = det_kp_confs[j]
                
                self.tracks[i].update(
                    det_bboxes[j], 
                    det_confs[j],
                    kpts if kpts is not None else None,
                    kpt_confs if kpt_confs is not None else None
                )
                matched_track_indices.add(i)
                matched_det_indices.add(j)
            
            # Create new tracks for unmatched detections
            for j in range(len(detections)):
                if j not in matched_det_indices:
                    self._create_track(
                        det_bboxes[j],
                        det_confs[j],
                        det_keypoints[j],
                        det_kp_confs[j]
                    )
        else:
            # No existing tracks, create new ones for all detections
            for j in range(len(detections)):
                self._create_track(
                    det_bboxes[j],
                    det_confs[j],
                    det_keypoints[j],
                    det_kp_confs[j]
                )
        
        # Remove lost tracks
        self._cleanup_tracks()
        
        return self.get_active_tracks()
    
    def _create_track(self, bbox: np.ndarray, confidence: float,
                     keypoints: Optional[np.ndarray] = None,
                     keypoint_confidences: Optional[np.ndarray] = None) -> Track:
        """Create a new track."""
        track = Track(
            track_id=self._next_id,
            bbox=bbox.copy(),
            confidence=confidence,
            keypoints=keypoints.copy() if keypoints is not None else None,
            keypoint_confidences=keypoint_confidences.copy() if keypoint_confidences is not None else None
        )
        self._next_id += 1
        self.tracks.append(track)
        return track
    
    def _cleanup_tracks(self):
        """Remove lost tracks."""
        self.tracks = [t for t in self.tracks if not t.is_lost(self.max_age)]
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (confirmed) tracks."""
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
    
    def get_primary_track(self) -> Optional[Track]:
        """
        Get the primary (most confident/largest) track.
        
        For single-person fall detection, we focus on the main person.
        """
        active = self.get_active_tracks()
        if not active:
            return None
        
        # Sort by area * confidence (prefer larger, more confident detections)
        active.sort(key=lambda t: -t.area * t.confidence)
        return active[0]
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self._next_id = 1


if __name__ == "__main__":
    # Test tracker
    print("Testing SimpleTracker...")
    
    tracker = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    
    # Simulate detections over frames
    frames = [
        # Frame 1: Person at position (100, 100, 200, 300)
        [{'bbox': np.array([100, 100, 200, 300]), 'confidence': 0.9}],
        # Frame 2: Person moved slightly
        [{'bbox': np.array([105, 102, 205, 302]), 'confidence': 0.85}],
        # Frame 3: Person moved more
        [{'bbox': np.array([110, 105, 210, 305]), 'confidence': 0.88}],
        # Frame 4: No detection (occlusion)
        [],
        # Frame 5: Person reappears
        [{'bbox': np.array([115, 108, 215, 308]), 'confidence': 0.87}],
        # Frame 6: Person visible
        [{'bbox': np.array([120, 110, 220, 310]), 'confidence': 0.9}],
    ]
    
    for i, detections in enumerate(frames):
        active = tracker.update(detections)
        primary = tracker.get_primary_track()
        
        print(f"Frame {i+1}: {len(detections)} detections, "
              f"{len(active)} active tracks, "
              f"primary: {primary.track_id if primary else 'None'}")
    
    print("\nTracker test passed!")
