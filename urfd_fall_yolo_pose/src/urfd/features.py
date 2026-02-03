import numpy as np

# COCO keypoint indices for torso landmarks
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

# Border margin (pixels) for detecting clipped bounding boxes
BORDER_MARGIN = 8

def compute_frame_features(detection, keypoint_conf_thres, dy_window, history, 
                           track_id=-1, frame_idx=0, image_size=None):
    # Compute features for a single track/detection (bbox, angle, dy, etc.)
    # Initialize default feature values
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
        "is_touching_border": False,
        "detection": detection 
    }
    
    # Early return if no detection
    if detection is None:
        return features
        
    bbox = detection["bbox"]
    keypoints = detection["keypoints"]
    is_fallback = detection.get("is_fallback", False)
    
    # Extract bbox coordinates and compute dimensions
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2  # Center X
    cy = (y1 + y2) / 2  # Center Y (important for fall detection)
    w = x2 - x1         # Bounding box width
    h = y2 - y1         # Bounding box height
    
    features["bbox"] = bbox
    features["center_y"] = cy
    features["height"] = h
    features["width"] = w
    # Aspect ratio: W/H > 1 means lying down posture
    features["bbox_aspect_ratio"] = w / h if h > 0 else 0.0
    
    # =========================================================
    # BORDER INTEGRITY CHECK
    # When bbox touches image edge, AR becomes unreliable due to clipping
    # =========================================================
    if image_size is not None:
        img_w, img_h = image_size
        if (x1 <= BORDER_MARGIN or y1 <= BORDER_MARGIN or 
            x2 >= img_w - BORDER_MARGIN or y2 >= img_h - BORDER_MARGIN):
            features["is_touching_border"] = True
    
    # =========================================================
    # KEYPOINT-BASED BODY ANGLE COMPUTATION
    # =========================================================
    if is_fallback or keypoints is None:
        # Fallback detections have no keypoints - use bbox-only features
        features["shoulder_mid"] = None
        features["hip_mid"] = None
        features["body_angle_deg"] = None
        features["feature_valid"] = False
    else:
        # Extract shoulder and hip keypoints (COCO format)
        left_shoulder = keypoints[KEYPOINT_LEFT_SHOULDER]
        right_shoulder = keypoints[KEYPOINT_RIGHT_SHOULDER]
        left_hip = keypoints[KEYPOINT_LEFT_HIP]
        right_hip = keypoints[KEYPOINT_RIGHT_HIP]
        
        # Check keypoint confidence thresholds
        ls_conf = left_shoulder[2] >= keypoint_conf_thres
        rs_conf = right_shoulder[2] >= keypoint_conf_thres
        lh_conf = left_hip[2] >= keypoint_conf_thres
        rh_conf = right_hip[2] >= keypoint_conf_thres
        
        shoulder_mid = None
        hip_mid = None
        
        # Compute shoulder midpoint if both shoulders are confident
        if ls_conf and rs_conf:
            shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]
            features["shoulder_mid"] = shoulder_mid
        
        # Compute hip midpoint if both hips are confident
        if lh_conf and rh_conf:
            hip_mid = [(left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2]
            features["hip_mid"] = hip_mid
        
        # Compute body angle from torso vector
        if shoulder_mid is not None and hip_mid is not None:
            dx = shoulder_mid[0] - hip_mid[0]  # Horizontal displacement
            dy_torso = shoulder_mid[1] - hip_mid[1]  # Vertical displacement
            
            # Angle relative to vertical axis (Y)
            # angle = 0° means standing upright
            # angle = 90° means lying horizontally
            angle_rad = np.arctan2(np.abs(dx), np.abs(dy_torso) + 1e-6)
            angle_deg = np.degrees(angle_rad)
            
            features["body_angle_deg"] = angle_deg
            features["feature_valid"] = True
        else:
            features["body_angle_deg"] = None
            features["feature_valid"] = False
    
    # =========================================================
    # VERTICAL VELOCITY (dy) COMPUTATION
    # Critical for detecting rapid downward movement during falls
    # =========================================================
    if len(history) >= 1:
        prev_feat = history[-1]
        if prev_feat["center_y"] is not None and cy is not None:
            # Instantaneous velocity: frame-to-frame change in center_y
            dy_inst = cy - prev_feat["center_y"]
            features["dy"] = dy_inst
            
            # Collect dy samples from recent history for peak detection
            dy_inst_samples = []
            for i in range(1, min(dy_window + 1, len(history) + 1)):
                idx = -i
                if abs(idx) <= len(history):
                    hist_feat = history[idx]
                    if hist_feat.get("dy", 0.0) != 0.0:
                        dy_inst_samples.append(abs(hist_feat["dy"]))
            
            # Include current frame's velocity
            dy_inst_samples.append(abs(dy_inst))
            
            if len(dy_inst_samples) > 0:
                # dy_peak: maximum velocity in window (detects impact moment)
                features["dy_peak"] = max(dy_inst_samples)
                # dy_velocity: average velocity in window
                features["dy_velocity"] = np.mean(dy_inst_samples)
    
    # Fallback for first frame or missing center_y
    if len(history) >= dy_window and features["dy"] == 0.0:
        prev_feat_win = history[-dy_window]
        if prev_feat_win["center_y"] is not None and cy is not None:
            features["dy"] = cy - prev_feat_win["center_y"]
    
    return features