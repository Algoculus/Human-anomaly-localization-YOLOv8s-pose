import numpy as np

# COCO keypoint indices
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

def compute_frame_features(detections, keypoint_conf_thres, dy_window, history):
    """Compute features for a single frame.
    
    Args:
        detections: List of person detections from YOLO
        keypoint_conf_thres: Minimum confidence for keypoints
        dy_window: Number of frames to look back for dy computation
        history: List of previous frame features
    
    Returns:
        features: Dict containing:
            - primary_person_idx: index of primary person (-1 if none)
            - bbox: [x1, y1, x2, y2] of primary person
            - center_y: bbox center y
            - height: bbox height
            - width: bbox width
            - bbox_aspect_ratio: width / height
            - shoulder_mid: [x, y] or None
            - hip_mid: [x, y] or None
            - body_angle_deg: angle in degrees or None
            - feature_valid: bool, True if keypoints are reliable
            - dy: change in center_y over last dy_window frames
            - dy_velocity: average dy over window
            - detections: original detections for overlay
    """
    features = {
        "primary_person_idx": -1,
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
        "detections": detections
    }
    
    if len(detections) == 0:
        return features
    
    # Select primary person (largest bbox area)
    primary_idx = max(range(len(detections)), key=lambda i: detections[i]["bbox_area"])
    features["primary_person_idx"] = primary_idx
    
    det = detections[primary_idx]
    bbox = det["bbox"]
    keypoints = det["keypoints"]
    
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
    
    # Compute dy (change in center_y) and velocity
    if len(history) >= dy_window:
        prev_feat = history[-dy_window]
        if prev_feat["center_y"] is not None:
            features["dy"] = cy - prev_feat["center_y"]
            
            # Compute average dy velocity over window
            dy_samples = []
            for i in range(1, min(dy_window + 1, len(history) + 1)):
                if history[-i]["center_y"] is not None:
                    dy_samples.append(cy - history[-i]["center_y"])
            if len(dy_samples) > 0:
                features["dy_velocity"] = np.mean(dy_samples)
    
    return features
