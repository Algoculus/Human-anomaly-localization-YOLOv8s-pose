import numpy as np

class PrimaryPersonTracker:
    """Temporal tracking of primary person using deterministic track-by-detection."""
    
    def __init__(self, config):
        """Initialize tracker.
        
        Args:
            config: Configuration dict with tracking parameters
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset tracker state."""
        self.track_bbox = None
        self.track_center = None
        self.track_velocity = np.array([0.0, 0.0])
        self.track_last_seen = 0
        self.track_missing_count = 0
        self.track_keypoints = None
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
        
        Returns:
            iou: Intersection over Union
        """
        x1_int = max(bbox1[0], bbox2[0])
        y1_int = max(bbox1[1], bbox2[1])
        x2_int = min(bbox1[2], bbox2[2])
        y2_int = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _compute_center_distance(self, bbox1, bbox2):
        """Compute normalized center distance between two bboxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
        
        Returns:
            dist: Normalized distance
        """
        c1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        c2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        
        h1 = bbox1[3] - bbox1[1]
        w1 = bbox1[2] - bbox1[0]
        normalize_factor = max(h1, w1)
        
        if normalize_factor <= 0:
            return 1e6
        
        return np.linalg.norm(c1 - c2) / normalize_factor
    
    def _compute_fall_likelihood_score(self, bbox, keypoints, keypoint_conf_thres):
        """Compute a fall-likelihood proxy score for re-initialization.
        
        Args:
            bbox: [x1, y1, x2, y2]
            keypoints: (17, 3) array
            keypoint_conf_thres: Threshold for keypoint confidence
        
        Returns:
            score: Higher means more likely to be a falling/fallen person
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if h <= 0:
            return 0.0
        
        ar = w / h
        center_y = (y1 + y2) / 2
        
        # Higher AR = more lying-like
        ar_score = min(ar / 2.0, 1.0)
        
        # Lower center (higher y) = likely on ground
        # Normalize by assuming image height ~480-720
        y_score = min(center_y / 500.0, 1.0)
        
        # Check if keypoints suggest horizontal posture
        angle_score = 0.0
        if keypoints is not None:
            ls = keypoints[5]
            rs = keypoints[6]
            lh = keypoints[11]
            rh = keypoints[12]
            
            if ls[2] >= keypoint_conf_thres and rs[2] >= keypoint_conf_thres and \
               lh[2] >= keypoint_conf_thres and rh[2] >= keypoint_conf_thres:
                shoulder_mid = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
                hip_mid = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
                
                dx = abs(shoulder_mid[0] - hip_mid[0])
                dy = abs(shoulder_mid[1] - hip_mid[1])
                
                if dy > 1e-6:
                    angle_rad = np.arctan2(dx, dy)
                    angle_deg = np.degrees(angle_rad)
                    angle_score = min(angle_deg / 90.0, 1.0)
        
        return 0.4 * ar_score + 0.3 * y_score + 0.3 * angle_score
    
    def update(self, detections, keypoint_conf_thres, frame_idx):
        """Update tracker with new detections.
        
        Args:
            detections: List of detection dicts
            keypoint_conf_thres: Keypoint confidence threshold
            frame_idx: Current frame index
        
        Returns:
            primary_idx: Index of primary person in detections, or -1 if none
        """
        if len(detections) == 0:
            self.track_missing_count += 1
            if self.track_missing_count > self.config["track_max_missing"]:
                self.reset()
            return -1
        
        # First frame or track lost: initialize
        if self.track_bbox is None:
            # Select by fall likelihood
            scores = [self._compute_fall_likelihood_score(
                det["bbox"], det["keypoints"], keypoint_conf_thres
            ) for det in detections]
            
            max_score_idx = np.argmax(scores)
            
            # If all scores are low, fall back to largest area
            if scores[max_score_idx] < 0.1:
                max_score_idx = max(range(len(detections)), key=lambda i: detections[i]["bbox_area"])
            
            self.track_bbox = detections[max_score_idx]["bbox"]
            bbox = self.track_bbox
            self.track_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            self.track_keypoints = detections[max_score_idx]["keypoints"]
            self.track_last_seen = frame_idx
            self.track_missing_count = 0
            
            return max_score_idx
        
        # Match to existing track
        best_idx = -1
        best_cost = 1e9
        
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            
            # Compute IoU
            iou = self._compute_iou(self.track_bbox, bbox)
            
            # Compute center distance
            center_dist = self._compute_center_distance(self.track_bbox, bbox)
            
            # Combined cost (lower is better)
            # Favor high IoU and low distance
            cost = (1.0 - iou) + center_dist
            
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        
        # Check if match is good enough
        if best_idx >= 0:
            iou = self._compute_iou(self.track_bbox, detections[best_idx]["bbox"])
            center_dist = self._compute_center_distance(self.track_bbox, detections[best_idx]["bbox"])
            
            if iou >= self.config["track_iou_thres"] or center_dist <= self.config["track_center_dist_thres"]:
                # Good match, update track
                bbox = detections[best_idx]["bbox"]
                new_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                
                # Update velocity (simple diff)
                self.track_velocity = new_center - self.track_center
                
                self.track_bbox = bbox
                self.track_center = new_center
                self.track_keypoints = detections[best_idx]["keypoints"]
                self.track_last_seen = frame_idx
                self.track_missing_count = 0
                
                return best_idx
        
        # No good match, increment missing
        self.track_missing_count += 1
        
        if self.track_missing_count > self.config["track_max_missing"]:
            # Re-initialize with fall likelihood
            scores = [self._compute_fall_likelihood_score(
                det["bbox"], det["keypoints"], keypoint_conf_thres
            ) for det in detections]
            
            max_score_idx = np.argmax(scores)
            
            if scores[max_score_idx] < 0.1:
                max_score_idx = max(range(len(detections)), key=lambda i: detections[i]["bbox_area"])
            
            self.track_bbox = detections[max_score_idx]["bbox"]
            bbox = self.track_bbox
            self.track_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            self.track_keypoints = detections[max_score_idx]["keypoints"]
            self.track_last_seen = frame_idx
            self.track_missing_count = 0
            
            return max_score_idx
        
        # Return -1 for missing but within tolerance
        return -1
