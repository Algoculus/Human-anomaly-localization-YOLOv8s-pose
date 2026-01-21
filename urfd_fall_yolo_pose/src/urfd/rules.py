import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, height_drop_thres, dy_fall_thres):
    """Check if current frame is a fall candidate.
    
    A frame is a fall candidate if:
    - (body_angle_deg >= angle_thres AND bbox_aspect_ratio >= ar_thres) OR
    - height_drop >= height_drop_thres OR
    - dy >= dy_fall_thres (fast downward motion)
    
    Args:
        features: Frame features dict
        angle_thres: Angle threshold in degrees
        ar_thres: Aspect ratio threshold
        height_drop: Normalized height drop
        height_drop_thres: Height drop threshold
        dy_fall_thres: dy threshold for fast fall detection
    
    Returns:
        is_candidate: bool
    """
    if features["bbox"] is None:
        return False
    
    # Path 1: Angle + AR based (requires valid features)
    angle_ar_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        if features["body_angle_deg"] >= angle_thres and features["bbox_aspect_ratio"] >= ar_thres:
            angle_ar_condition = True
    
    # Path 2: Height drop based (bbox only)
    height_drop_condition = height_drop >= height_drop_thres
    
    # Path 3: Fast downward motion (dy-based)
    dy_condition = features["dy"] >= dy_fall_thres
    
    return angle_ar_condition or height_drop_condition or dy_condition

def check_lying_posture(features, confirm_angle_thres, confirm_ar_thres):
    """Check if current frame shows lying-like posture.
    
    Using OR logic with relaxed thresholds for maximum recall.
    Will accept false positives to ensure we don't miss real falls.
    
    Args:
        features: Frame features dict
        confirm_angle_thres: Angle threshold for confirmation
        confirm_ar_thres: Aspect ratio threshold for confirmation
    
    Returns:
        is_lying: bool
    """
    if features["bbox"] is None:
        return False
    
    # Check angle condition (requires valid features)
    angle_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        # Moderate relaxation (8 degrees) for balanced recall/precision
        if features["body_angle_deg"] >= (confirm_angle_thres - 8.0):
            angle_condition = True
    
    # Check AR condition (always available)
    # Moderate relaxation (0.17) for balanced recall/precision
    ar_condition = features["bbox_aspect_ratio"] >= (confirm_ar_thres - 0.17)
    
    # Use OR logic for better recall while maintaining precision
    return angle_condition or ar_condition
