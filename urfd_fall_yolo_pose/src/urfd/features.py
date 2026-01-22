import numpy as np

# COCO keypoint indices
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

def compute_frame_features(detection, keypoint_conf_thres, dy_window, history, track_id=-1, frame_idx=0):
    # Compute features for a single track/detection
    # Args:
    #     detection: Standard detection dict (bbox, keypoints, etc.)
    #     keypoint_conf_thres: Minimum confidence for keypoints
    #     dy_window: Number of frames to look back for dy computation
    #     history: List of previous frame features for THIS track
    #     track_id: ID of the person track
    #     frame_idx: Current frame index
    
    features = {
        "track_id": track_id,
        "bbox": None,
        "center_y": None,
        "height": None,
        "width": None,
        "bbox_aspect_ratio": None,
        "shoulder_mid": None,
        "hip_mid": None,
        "body_angle_deg": None,
        "feature_valid": False,
        "dy": 0.0,
        "dy_velocity": 0.0,
        "dy_peak": 0.0,
        "detection": detection 
    }
    
    if detection is None:
        return features
        
    bbox = detection["bbox"]
    keypoints = detection["keypoints"]
    
    # Check if this is a fallback detection (no keypoints)
    is_fallback = detection.get("is_fallback", False)
    
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    features["bbox"] = bbox
    features["center_y"] = cy
    features["height"] = h
    features["width"] = w
    features["bbox_aspect_ratio"] = w / h if h > 0 else 0.0
    
    # For fallback detections: no keypoints available
    if is_fallback or keypoints is None:
        features["shoulder_mid"] = None
        features["hip_mid"] = None
        features["body_angle_deg"] = None
        features["feature_valid"] = False
        # Skip keypoint processing, go to dy computation
    else:
        # Extract shoulder and hip keypoints
        left_shoulder = keypoints[KEYPOINT_LEFT_SHOULDER]
        right_shoulder = keypoints[KEYPOINT_RIGHT_SHOULDER]
        left_hip = keypoints[KEYPOINT_LEFT_HIP]
        right_hip = keypoints[KEYPOINT_RIGHT_HIP]
        
        # Check if keypoints are confident
        ls_conf = left_shoulder[2] >= keypoint_conf_thres
        rs_conf = right_shoulder[2] >= keypoint_conf_thres
        lh_conf = left_hip[2] >= keypoint_conf_thres
        rh_conf = right_hip[2] >= keypoint_conf_thres
        
        shoulder_mid = None
        hip_mid = None
        
        if ls_conf and rs_conf:
            shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]
            features["shoulder_mid"] = shoulder_mid
        
        if lh_conf and rh_conf:
            hip_mid = [(left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2]
            features["hip_mid"] = hip_mid
        
        # Compute body angle if both shoulder_mid and hip_mid are available
        if shoulder_mid is not None and hip_mid is not None:
            dx = shoulder_mid[0] - hip_mid[0]
            dy_torso = shoulder_mid[1] - hip_mid[1]
            
            # Angle relative to vertical (y-axis)
            angle_rad = np.arctan2(np.abs(dx), np.abs(dy_torso) + 1e-6)
            angle_deg = np.degrees(angle_rad)
            
            features["body_angle_deg"] = angle_deg
            features["feature_valid"] = True
        else:
            features["body_angle_deg"] = None
            features["feature_valid"] = False
    
    # Compute dy (change in center_y), velocity, and peak
    # FIX-1: Changed dy_peak to instantaneous velocity peak (not cumulative)
    if len(history) >= 1:
        prev_feat = history[-1]
        if prev_feat["center_y"] is not None and cy is not None:
            # Instantaneous velocity: frame-to-frame change
            dy_inst = cy - prev_feat["center_y"]
            features["dy"] = dy_inst
            
            # Compute dy_peak as max instantaneous velocity in recent window
            dy_inst_samples = []
            for i in range(1, min(dy_window + 1, len(history) + 1)):
                idx = -i
                if abs(idx) <= len(history):
                    hist_feat = history[idx]
                    # Check if dy was already computed for this history frame
                    if hist_feat.get("dy", 0.0) != 0.0:
                        dy_inst_samples.append(abs(hist_feat["dy"]))
            
            # Add current instantaneous dy
            dy_inst_samples.append(abs(dy_inst))
            
            if len(dy_inst_samples) > 0:
                # dy_peak: maximum instantaneous velocity (not cumulative displacement)
                features["dy_peak"] = max(dy_inst_samples)
                # dy_velocity: average instantaneous velocity
                features["dy_velocity"] = np.mean(dy_inst_samples)
    
    # Fallback for first frame or missing cy in window
    if len(history) >= dy_window and features["dy"] == 0.0:
        prev_feat_win = history[-dy_window]
        if prev_feat_win["center_y"] is not None and cy is not None:
            features["dy"] = cy - prev_feat_win["center_y"]
    
    return features
