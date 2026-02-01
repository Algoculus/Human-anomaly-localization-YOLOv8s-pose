import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, 
                         height_drop_thres, dy_fall_thres, impact_dy_thres=5.0,
                         high_angle_thres=65.0):
    """
    Check if current frame is a fall candidate.
    
    IMPROVED: Uses temporal features (d_ar, d2y, dh) instead of unreliable body_angle
    
    A frame is a fall candidate if ANY of these conditions are met:
    - PATH 1: (hw_ratio >= 1.2 AND dy_peak >= 5.0) - Lying posture with impact
    - PATH 2: height_drop >= height_drop_thres - Sudden height collapse
    - PATH 3: dy >= dy_fall_thres - Fast downward motion
    - PATH 4: (d_ar > 0.15 AND dh < -5) - Rapid posture change with height drop
    - PATH 5: d2y > 3.0 - Sudden acceleration (free fall detection)
    
    Args:
        features: Frame features dict from compute_frame_features
        angle_thres: DEPRECATED - kept for compatibility
        ar_thres: Aspect ratio threshold (W/H) (1.15)
        height_drop: Normalized height drop value
        height_drop_thres: Height drop threshold (0.18)
        dy_fall_thres: dy threshold for fast fall detection (10.0)
        impact_dy_thres: Minimum dy_peak for posture path (5.0)
        high_angle_thres: DEPRECATED - kept for compatibility
        
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
    # PATH 1: BBOX-BASED LYING POSTURE (NEW - no keypoints needed)
    # hw_ratio >= 1.2 means lying down (width > 1.2 * height)
    # Combined with impact velocity
    # DISABLED when border is clipped (AR unreliable)
    # =========================================================
    posture_condition = False
    if not is_touching_border:
        hw_ratio = features.get("hw_ratio", 0.0)
        dy_peak = features.get("dy_peak", 0.0)
        # Lying posture with impact
        if hw_ratio >= 1.2 and dy_peak >= impact_dy_thres:
            posture_condition = True
    
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
    # PATH 4: RAPID POSTURE TRANSITION (NEW)
    # Detects sudden aspect ratio increase + height decrease
    # Catches transitions from standing → lying
    # =========================================================
    posture_transition = False
    d_ar = features.get("d_ar", 0.0)
    dh = features.get("dh", 0.0)
    # Sudden AR increase (becoming wider) + height drop
    if d_ar > 0.15 and dh < -5:
        posture_transition = True
    
    # =========================================================
    # PATH 5: ACCELERATION DETECTION (NEW)
    # Detects sudden vertical acceleration (free fall)
    # Very reliable indicator of fall event
    # =========================================================
    acceleration_condition = False
    d2y = features.get("d2y", 0.0)
    if d2y > 3.0:  # Positive acceleration = moving down faster
        acceleration_condition = True
    
    # =========================================================
    # LEGACY FALLBACK: Keep old angle-based logic if available
    # Only used when keypoints are valid
    # =========================================================
    legacy_angle_condition = False
    if features.get("feature_valid", False) and features.get("body_angle_deg") is not None:
        body_angle = features["body_angle_deg"]
        ar = features.get("bbox_aspect_ratio", 0.0)
        dy_peak = features.get("dy_peak", 0.0)
        # Original angle-based detection
        if body_angle >= angle_thres and ar >= ar_thres and dy_peak >= impact_dy_thres:
            legacy_angle_condition = True
        # High angle variant
        if body_angle >= high_angle_thres and dy_peak >= 3.0:
            legacy_angle_condition = True
    
    # Any of the paths triggers candidate status
    return (posture_condition or height_drop_condition or dy_condition or 
            posture_transition or acceleration_condition or legacy_angle_condition)

def check_lying_posture(features, confirm_angle_thres, confirm_ar_thres):
    """
    Check if current frame shows lying-like posture.
    
    IMPROVED: Prioritizes bbox-based features over unreliable keypoints
    
    Args:
        features: Frame features dict from compute_frame_features
        confirm_angle_thres: DEPRECATED - kept for compatibility
        confirm_ar_thres: Aspect ratio threshold for confirmation (1.25)
        
    Returns:
        is_lying: True if posture appears to be lying down
    """
    if features["bbox"] is None:
        return False
    
    # =========================================================
    # PRIMARY: HW_RATIO CHECK (bbox-based, always available)
    # hw_ratio > 1.05 indicates lying posture (width > height)
    # Relaxed threshold for better recall
    # =========================================================
    hw_ratio = features.get("hw_ratio", 0.0)
    hw_condition = hw_ratio >= 1.05
    
    # =========================================================
    # SECONDARY: ASPECT RATIO CHECK (legacy compatibility)
    # Relaxed by 0.20 for better recall
    # =========================================================
    ar = features.get("bbox_aspect_ratio", 0.0)
    ar_condition = ar >= (confirm_ar_thres - 0.20)
    
    # =========================================================
    # TERTIARY: HEIGHT RATIO CHECK (temporal)
    # Low height compared to standing baseline
    # =========================================================
    height_ratio_condition = False
    # This requires FeatureBuffer baseline - check if available
    # For now, we skip this as it's computed elsewhere
    
    # =========================================================
    # LEGACY FALLBACK: Keep angle-based logic if keypoints valid
    # Only used as confirmation signal
    # =========================================================
    angle_condition = False
    if features.get("feature_valid", False) and features.get("body_angle_deg") is not None:
        body_angle = features["body_angle_deg"]
        # Use relaxed threshold
        if body_angle >= (confirm_angle_thres - 10.0):
            angle_condition = True
    
    # OR logic: any condition triggers lying posture
    # Prioritize hw_ratio (most reliable)
    return hw_condition or ar_condition or angle_condition
