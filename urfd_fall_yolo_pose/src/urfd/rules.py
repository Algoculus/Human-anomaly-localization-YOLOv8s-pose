import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, 
                         height_drop_thres, dy_fall_thres, impact_dy_thres=10.0,
                         high_angle_thres=65.0, high_angle_dy_peak_thres=8.0,
                         min_candidate_dy_peak=12.0):
    """
    Check if current frame is a fall candidate.
    
    ALL PATHS NOW REQUIRE MINIMUM IMPACT MOTION (dy_peak >= min_candidate_dy_peak)
    This prevents false positives from controlled lying down.
    
    A frame is a fall candidate if ANY of these conditions are met:
    - PATH 1: (body_angle >= angle_thres AND AR >= ar_thres AND dy_peak >= impact_dy_thres)
    - PATH 2: height_drop >= height_drop_thres AND dy_peak >= min_candidate_dy_peak
    - PATH 3: dy >= dy_fall_thres AND dy_peak >= min_candidate_dy_peak
    - PATH 4: body_angle >= high_angle_thres AND dy_peak >= high_angle_dy_peak_thres
    
    Args:
        features: Frame features dict from compute_frame_features
        angle_thres: Angle threshold in degrees for lying posture (55.0)
        ar_thres: Aspect ratio threshold (W/H) (1.15)
        height_drop: Normalized height drop value
        height_drop_thres: Height drop threshold (0.18)
        dy_fall_thres: dy threshold for fast fall detection (10.0)
        impact_dy_thres: Minimum dy_peak for posture path (10.0)
        high_angle_thres: High angle threshold for Path 4 (65.0)
        high_angle_dy_peak_thres: Minimum dy_peak for high angle path (8.0)
        min_candidate_dy_peak: Minimum dy_peak for PATH 2 & 3 (12.0)
        
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
            # Requires lying posture AND impact velocity
            if (features["body_angle_deg"] >= angle_thres and 
                features["bbox_aspect_ratio"] >= ar_thres and
                dy_peak >= impact_dy_thres):
                angle_ar_condition = True
    
    # =========================================================
    # PATH 2: HEIGHT DROP DETECTION
    # Detects significant reduction in bbox height (person collapsed)
    # REQUIRES minimum impact to distinguish from controlled lying
    # =========================================================
    dy_peak = features.get("dy_peak", 0.0)
    height_drop_condition = (height_drop >= height_drop_thres and 
                            dy_peak >= min_candidate_dy_peak)
    
    # =========================================================
    # PATH 3: FAST MOTION DETECTION
    # Detects rapid downward movement (free fall phase)
    # REQUIRES minimum dy_peak to distinguish from controlled movement
    # =========================================================
    dy_condition = (features["dy"] >= dy_fall_thres and 
                   dy_peak >= min_candidate_dy_peak)
    
    # =========================================================
    # PATH 4: HIGH-ANGLE DETECTION (for catching frontal falls)
    # Very horizontal posture with significant impact motion
    # Requires higher dy_peak to distinguish from controlled lying
    # =========================================================
    high_angle_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        dy_peak = features.get("dy_peak", 0.0)
        # Require significant impact to distinguish from normal lying
        if features["body_angle_deg"] >= high_angle_thres and dy_peak >= high_angle_dy_peak_thres:
            high_angle_condition = True
    
    # Any of the four paths triggers candidate status
    return angle_ar_condition or height_drop_condition or dy_condition or high_angle_condition

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
    # Relaxed by 10 degrees for better recall (changed from 8)
    # =========================================================
    angle_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        # Use relaxed threshold (confirm_angle_thres - 10)
        if features["body_angle_deg"] >= (confirm_angle_thres - 10.0):
            angle_condition = True
    
    # =========================================================
    # ASPECT RATIO CHECK (bbox-based fallback)
    # Relaxed by 0.20 for better recall (changed from 0.17)
    # Works even when keypoints are unavailable
    # =========================================================
    ar_condition = features["bbox_aspect_ratio"] >= (confirm_ar_thres - 0.20)
    
    # OR logic: either condition triggers lying posture
    return angle_condition or ar_condition