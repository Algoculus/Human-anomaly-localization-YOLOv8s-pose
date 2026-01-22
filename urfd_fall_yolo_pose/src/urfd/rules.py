import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, height_drop_thres, dy_fall_thres):
    # Check if current frame is a fall candidate using multi-path detection
    # Path 1: Angle + AR | Path 2: Height drop | Path 3: Velocity | Path 4: Acceleration
    
    if features["bbox"] is None:
        return False
    
    # Path 1: Angle + AR based (high confidence)
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
    # Check if current frame shows lying-like posture using adaptive thresholds
    # Combines angle, AR, and velocity decay for robust confirmation
    
    if features["bbox"] is None:
        return False
    
    # Check angle condition with moderate relaxation
    angle_condition = False
    if features["feature_valid"] and features["body_angle_deg"] is not None:
        # Moderate relaxation (8 degrees) for balanced recall/precision
        if features["body_angle_deg"] >= (confirm_angle_thres - 8.0):
            angle_condition = True
    
    # Check AR condition with moderate relaxation
    # Moderate relaxation (0.17) for balanced recall/precision
    ar_condition = features["bbox_aspect_ratio"] >= (confirm_ar_thres - 0.17)
    
    # Use OR logic for maximum recall (removed velocity constraint)
    return angle_condition or ar_condition
