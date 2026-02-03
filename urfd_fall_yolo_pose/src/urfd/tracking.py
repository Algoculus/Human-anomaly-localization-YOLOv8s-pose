import numpy as np
from src.urfd.utils import compute_iou, compute_center_distance

class MultiPersonTracker:
    # Multi-person tracker using greedy IoU-based assignment with velocity prediction
    
    def __init__(self, config):
        # Initialize tracker with config parameters
        self.config = config
        self.tracks = {}       # {track_id: track_data}
        self.next_track_id = 0
    
    def reset(self):
        # Reset tracker state
        self.tracks = {}
        self.next_track_id = 0
    
    def _compute_fall_likelihood_score(self, bbox, keypoints, keypoint_conf_thres):
        # Compute a fall-likelihood proxy score for re-initialization (higher = more likely fallen)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if h <= 0:
            return 0.0
        
        ar = w / h
        center_y = (y1 + y2) / 2
        
        # Higher AR = more lying-like (40% weight)
        ar_score = min(ar / 2.0, 1.0)
        
        # Lower position (higher y) = likely on ground (30% weight)
        y_score = min(center_y / 500.0, 1.0)
        
        # Check body angle from keypoints (30% weight)
        angle_score = 0.0
        if keypoints is not None:
            ls = keypoints[5]   # Left shoulder
            rs = keypoints[6]   # Right shoulder
            lh = keypoints[11]  # Left hip
            rh = keypoints[12]  # Right hip
            
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
        # Update tracker with new detections, returns dict of {track_id: detection_index}
        # =========================================================
        # HANDLE NO DETECTIONS
        # =========================================================
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["missing_count"] += 1
                # Remove stale tracks
                if self.tracks[tid]["missing_count"] > self.config["track_max_missing"]:
                    del self.tracks[tid]
            return {}
        
        # =========================================================
        # INITIALIZE TRACKS IF EMPTY
        # =========================================================
        if len(self.tracks) == 0:
            assigned_matches = {}
            for i, det in enumerate(detections):
                tid = self.next_track_id
                self.next_track_id += 1
                
                bbox = det["bbox"]
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                
                self.tracks[tid] = {
                    "bbox": bbox,
                    "center": center,
                    "velocity": np.array([0.0, 0.0]),
                    "keypoints": det["keypoints"],
                    "last_seen": frame_idx,
                    "missing_count": 0
                }
                assigned_matches[tid] = i
            return assigned_matches

        # =========================================================
        # COMPUTE COST MATRIX
        # Cost = (1 - IoU) + normalized_center_distance
        # =========================================================
        track_ids = list(self.tracks.keys())
        costs = np.zeros((len(track_ids), len(detections)))
        
        for r, tid in enumerate(track_ids):
            track = self.tracks[tid]
            
            # Predict new position using velocity
            pred_center = track["center"] + track["velocity"]
            w = track["bbox"][2] - track["bbox"][0]
            h = track["bbox"][3] - track["bbox"][1]
            pred_bbox = [pred_center[0] - w/2, pred_center[1] - h/2, 
                         pred_center[0] + w/2, pred_center[1] + h/2]
            
            for c, det in enumerate(detections):
                bbox = det["bbox"]
                iou = compute_iou(pred_bbox, bbox)
                dist = compute_center_distance(pred_bbox, bbox)
                # Lower cost = better match
                costs[r, c] = (1.0 - iou) + dist
        
        # =========================================================
        # GREEDY MATCHING (lowest cost first)
        # =========================================================
        assigned_track_indices = set()
        assigned_det_indices = set()
        matches = {}
        
        # Sort all possible matches by cost
        possible_matches = []
        for r in range(len(track_ids)):
            for c in range(len(detections)):
                possible_matches.append((costs[r, c], r, c))
        
        possible_matches.sort(key=lambda x: x[0])
        
        for cost, r, c in possible_matches:
            if r in assigned_track_indices or c in assigned_det_indices:
                continue
            
            track_id = track_ids[r]
            
            # Cost threshold (could be tightened if needed)
            if cost > 2.0:
                pass  # Still allow, but could reject here
            
            matches[track_id] = c
            assigned_track_indices.add(r)
            assigned_det_indices.add(c)
        
        # =========================================================
        # UPDATE MATCHED TRACKS
        # =========================================================
        final_matches = {}
        for tid, det_idx in matches.items():
            det = detections[det_idx]
            bbox = det["bbox"]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            track = self.tracks[tid]
            prev_center = track["center"]
            
            # Update velocity estimate
            velocity = center - prev_center
            
            self.tracks[tid]["bbox"] = bbox
            self.tracks[tid]["center"] = center
            self.tracks[tid]["velocity"] = velocity
            self.tracks[tid]["keypoints"] = det["keypoints"]
            self.tracks[tid]["last_seen"] = frame_idx
            self.tracks[tid]["missing_count"] = 0
            
            final_matches[tid] = det_idx
        
        # =========================================================
        # HANDLE UNMATCHED TRACKS (missing detections)
        # =========================================================
        for r, tid in enumerate(track_ids):
            if r not in assigned_track_indices:
                self.tracks[tid]["missing_count"] += 1
                if self.tracks[tid]["missing_count"] > self.config["track_max_missing"]:
                    del self.tracks[tid]
        
        # =========================================================
        # CREATE NEW TRACKS FOR UNMATCHED DETECTIONS
        # =========================================================
        for c, det in enumerate(detections):
            if c not in assigned_det_indices:
                tid = self.next_track_id
                self.next_track_id += 1
                
                bbox = det["bbox"]
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                
                self.tracks[tid] = {
                    "bbox": bbox,
                    "center": center,
                    "velocity": np.array([0.0, 0.0]),
                    "keypoints": det["keypoints"],
                    "last_seen": frame_idx,
                    "missing_count": 0
                }
                final_matches[tid] = c
                
        return final_matches