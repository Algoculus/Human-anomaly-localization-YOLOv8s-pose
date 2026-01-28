import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, 
                         height_drop_thres, dy_fall_thres, impact_dy_thres=4.0,
                         high_angle_thres=60.0, sustained_lying_thres=50.0):
    """
    Check if current frame is a fall candidate.
    
    A frame is a fall candidate if ANY of these conditions are met:
    - PATH 1: (body_angle >= angle_thres AND AR >= ar_thres AND dy_peak >= impact_dy_thres)
    - PATH 2: height_drop >= height_drop_thres
    - PATH 3: dy >= dy_fall_thres (fast downward motion)
    - PATH 4: body_angle >= high_angle_thres AND dy_peak >= 2.0 (horizontal posture)
    - PATH 5: sustained_lying (high angle + high AR, no motion required - for slow falls)
    - PATH 6: AR-only path for very wide bbox (> 1.6) with any downward motion
    
    Args:
        features: Frame features dict from compute_frame_features
        angle_thres: Angle threshold in degrees for lying posture (50.0)
        ar_thres: Aspect ratio threshold (W/H) (1.10)
        height_drop: Normalized height drop value
        height_drop_thres: Height drop threshold (0.15)
        dy_fall_thres: dy threshold for fast fall detection (8.0)
        impact_dy_thres: Minimum dy_peak for posture path (4.0)
        high_angle_thres: High angle threshold for Path 4 (60.0)
        sustained_lying_thres: Angle threshold for sustained lying (50.0)
        
    Returns:
        is_candidate: True if frame is a fall candidate
    """
    if features["bbox"] is None:
        return False
    
    # =========================================================
    # BORDER INTEGRITY CHECK
    # When bbox is clipped at image edge, AR becomes unreliable
    # =========================================================
    is_touching_border = features.get("is_touching_border", False)
    
    # =========================================================
    # PATH 1: POSTURE-BASED DETECTION (Angle + AR + Impact)
    # Requires lying posture with some impact velocity
    # DISABLED when border is clipped (AR unreliable)
    # =========================================================
    angle_ar_condition = False
    if not is_touching_border:
        if features["feature_valid"] and features["body_angle_deg"] is not None:
            dy_peak = features.get("dy_peak", 0.0)
            # Requires lying posture AND impact velocity (relaxed)
            if (features["body_angle_deg"] >= angle_thres and 
                features["bbox_aspect_ratio"] >= ar_thres and
                dy_peak >= impact_dy_thres):
                angle_ar_condition = True
    
    # =========================================================
    # PATH 2: HEIGHT DROP DETECTION
    # Detects significant reduction in bbox height (person collapsed)
    # =========================================================
    height_drop_condition = height_drop >= height_drop_thres
    
    # =========================================================
    # PATH 3: FAST MOTION DETECTION
    # Detects rapid downward movement (free fall phase)
    # =========================================================
    dy_condition = features["dy"] >= dy_fall_thres
    
    # =========================================================
    # PATH 4: HIGH-ANGLE DETECTION (horizontal posture with motion)
    # Catches slow falls and frontal falls
    # =========================================================
    high_angle_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        dy_peak = features.get("dy_peak", 0.0)
        # Horizontal posture with any detectable motion
        if features["body_angle_deg"] >= high_angle_thres and dy_peak >= 2.0:
            high_angle_condition = True
    
    # =========================================================
    # PATH 5: SUSTAINED LYING DETECTION (for slow/controlled falls)
    # High angle + high AR without requiring motion
    # Catches cases where person slowly lowers to ground
    # =========================================================
    sustained_lying_condition = False
    if not is_touching_border:
        if features["feature_valid"] and features["body_angle_deg"] is not None:
            # Very horizontal (>50°) with wide bbox (AR > 1.3)
            if (features["body_angle_deg"] >= sustained_lying_thres and 
                features["bbox_aspect_ratio"] >= 1.3):
                sustained_lying_condition = True
    
    # =========================================================
    # PATH 6: WIDE BBOX DETECTION (AR-focused)
    # Very wide bbox (AR > 1.6) indicates lying, with any downward motion
    # =========================================================
    wide_bbox_condition = False
    if not is_touching_border:
        dy = features.get("dy", 0.0)
        # Very wide aspect ratio with some downward movement
        if features["bbox_aspect_ratio"] >= 1.6 and dy > 1.0:
            wide_bbox_condition = True
    
    # Any of the six paths triggers candidate status
    return (angle_ar_condition or height_drop_condition or dy_condition or 
            high_angle_condition or sustained_lying_condition or wide_bbox_condition)

def check_lying_posture(features, confirm_angle_thres, confirm_ar_thres):
    """
    Check if current frame shows lying-like posture.
    
    Uses OR logic with relaxed thresholds for maximum recall.
    
    Args:
        features: Frame features dict from compute_frame_features
        confirm_angle_thres: Angle threshold for confirmation (degrees)
        confirm_ar_thres: Aspect ratio threshold for confirmation
        
    Returns:
        is_lying: True if posture appears to be lying down
    """
    if features["bbox"] is None:
        return False
    
    # =========================================================
    # ANGLE CHECK (keypoint-based)
    # Relaxed by 12 degrees for better recall
    # =========================================================
    angle_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        # Use relaxed threshold (confirm_angle_thres - 12)
        if features["body_angle_deg"] >= (confirm_angle_thres - 12.0):
            angle_condition = True
    
    # =========================================================
    # ASPECT RATIO CHECK (bbox-based fallback)
    # Relaxed by 0.25 for better recall
    # Works even when keypoints are unavailable
    # =========================================================
    ar_condition = features["bbox_aspect_ratio"] >= (confirm_ar_thres - 0.25)
    
    # =========================================================
    # COMBINED CHECK: Moderate angle + moderate AR
    # Catches edge cases where neither alone crosses threshold
    # =========================================================
    combined_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        # Moderate angle (>35°) combined with moderate AR (>1.0)
        if (features["body_angle_deg"] >= (confirm_angle_thres - 18.0) and 
            features["bbox_aspect_ratio"] >= (confirm_ar_thres - 0.35)):
            combined_condition = True
    
    # Any condition triggers lying posture
    return angle_condition or ar_condition or combined_condition
