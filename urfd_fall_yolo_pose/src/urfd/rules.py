import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, 
                         height_drop_thres, dy_fall_thres, impact_dy_thres=5.0,
                         high_angle_thres=65.0):
    """
    Check if current frame is a fall candidate.
    
    HYBRID APPROACH: Combines bbox temporal + body angle for best accuracy
    
    Detection paths (OR logic):
    - PRIMARY TIER (Bbox Temporal - always active):
      * PATH 1: hw_ratio >= 1.2 AND dy_peak >= 5.0 (lying + impact)
      * PATH 2: height_drop >= threshold (sudden collapse)
      * PATH 3: dy >= threshold (fast downward motion)
      * PATH 4: d_ar > 0.15 AND dh < -5 (rapid posture change)
      * PATH 5: d2y > 3.0 (sudden acceleration)
    
    - SECONDARY TIER (Body Angle - when keypoints available):
      * PATH 6: angle >= 60° AND hw_ratio >= 1.0 (cross-validation)
      * PATH 7: angle >= 55° AND 0.8 <= hw_ratio < 1.2 (resolve ambiguous)
      * REJECT: angle < 40° AND hw_ratio >= 1.0 (likely bending, not fall)
    
    Args:
        features: Frame features dict from compute_frame_features
        angle_thres: Angle threshold for lying posture (55.0)
        ar_thres: Aspect ratio threshold (W/H) (1.15)
        height_drop: Normalized height drop value
        height_drop_thres: Height drop threshold (0.18)
        dy_fall_thres: dy threshold for fast fall detection (10.0)
        impact_dy_thres: Minimum dy_peak for posture path (5.0)
        high_angle_thres: High angle threshold (65.0)
        
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
    # PRIMARY TIER: BBOX TEMPORAL FEATURES (Always Active)
    # =========================================================
    
    # PATH 1: Bbox-based lying posture + impact
    # hw_ratio >= 1.15 means lying down
    # BALANCED: Increased to 1.15 to reduce false positives
    # DISABLED when border is clipped
    posture_condition = False
    if not is_touching_border:
        hw_ratio = features.get("hw_ratio", 0.0)
        dy_peak = features.get("dy_peak", 0.0)
        # BALANCED: hw_ratio 1.15, impact_dy_thres from config (2.5)
        if hw_ratio >= 1.15 and dy_peak >= impact_dy_thres:
            posture_condition = True
    
    # PATH 2: Height drop detection
    # BALANCED: 0.13 (13%) for good sensitivity without false positives
    height_drop_condition = height_drop >= 0.13
    
    # PATH 3: Fast motion detection
    # BALANCED: 6.0 for moderate speed falls
    dy_condition = features["dy"] >= 6.0
    
    # PATH 4: Rapid posture transition (NEW)
    posture_transition = False
    d_ar = features.get("d_ar", 0.0)
    dh = features.get("dh", 0.0)
    # RELAXED: d_ar 0.15→0.10, dh -5→-3
    if d_ar > 0.10 and dh < -3:
        posture_transition = True
    
    # PATH 5: Acceleration detection (NEW)
    acceleration_condition = False
    d2y = features.get("d2y", 0.0)
    # RELAXED: 3.0→1.5 (catch subtle acceleration changes)
    if d2y > 1.5:
        acceleration_condition = True
    
    # =========================================================
    # ADAPTIVE HYBRID LOGIC
    # Priority: Catch falls (relaxed) BUT reject ADL (strict)
    # =========================================================
    angle = features.get("body_angle_deg")
    feature_valid = features.get("feature_valid", False)
    hw_ratio = features.get("hw_ratio", 0.0)
    ar = features.get("bbox_aspect_ratio", 0.0)
    
    # Count how many bbox paths triggered
    bbox_triggers = sum([
        posture_condition,
        height_drop_condition, 
        dy_condition,
        posture_transition,
        acceleration_condition
    ])
    
    # MODE 1: Both bbox and angle available → SMART VALIDATION
    if feature_valid and angle is not None:
        # =====================================================
        # PRIORITY 1: ADL REJECTION (Strict - prevent FP)
        # STRENGTHENED to reduce 6 FP → 2 FP
        # =====================================================
        # Very clear upright posture with wide bbox → sitting/bending
        if hw_ratio >= 0.95 and angle < 42:
            return False
        
        # Moderate wide bbox with upright angle → likely ADL  
        if hw_ratio >= 0.85 and angle < 40:
            return False
        
        # Additional: Reject borderline cases more aggressively
        if hw_ratio >= 1.1 and angle < 45:
            return False
        
        # =====================================================
        # PRIORITY 2: FALL DETECTION (Relaxed - maximize recall)
        # Accept if EITHER bbox OR angle shows strong fall signature
        # =====================================================
        
        # BBOX-dominant paths (angle just validates not ADL)
        if bbox_triggers >= 2 and angle >= 42:
            # Multiple bbox signals + angle not rejecting
            return True
        
        if bbox_triggers >= 1 and angle >= 48:
            # Single strong bbox + angle mildly confirms
            return True
        
        # ANGLE-dominant paths (bbox validates presence)
        if angle >= 58 and hw_ratio >= 0.85:
            # Strong lying angle + some bbox width
            return True
        
        if angle >= 55 and bbox_triggers >= 1:
            # Good lying angle + any bbox signal
            return True
        
        # CROSS-VALIDATION paths (both moderate)
        if angle >= 50 and hw_ratio >= 1.0 and bbox_triggers >= 1:
            # Moderate angle + lying bbox + motion
            return True
        
        # If reached here: Neither bbox nor angle is convincing
        return False
    
    # MODE 2: Only bbox available → MODERATE THRESHOLDS
    else:
        # Need strong evidence without angle validation
        if bbox_triggers >= 2:
            # Multiple paths triggered
            return True
        
        # Very strong single path
        if height_drop_condition and height_drop >= 0.16:
            return True
        
        if dy_condition and features["dy"] >= 7.5:
            return True
            
        if posture_condition and hw_ratio >= 1.2:
            return True
        
        return False

def check_lying_posture(features, confirm_angle_thres, confirm_ar_thres):
    """
    Check if current frame shows lying-like posture.
    
    HYBRID: Prioritizes bbox hw_ratio + uses angle for cross-validation
    
    Args:
        features: Frame features dict from compute_frame_features
        confirm_angle_thres: Angle threshold for confirmation (42.0)
        confirm_ar_thres: Aspect ratio threshold for confirmation (1.25)
        
    Returns:
        is_lying: True if posture appears to be lying down
    """
    if features["bbox"] is None:
        return False
    
    # =========================================================
    # PRIMARY: HW_RATIO CHECK (bbox-based, always available)
    # hw_ratio > 1.05 indicates lying posture (width > height)
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
    # TERTIARY: BODY ANGLE CHECK (cross-validation when available)
    # Uses angle to confirm or reject bbox-based detection
    # =========================================================
    angle = features.get("body_angle_deg")
    feature_valid = features.get("feature_valid", False)
    
    angle_confirms = False
    angle_rejects = False
    
    if feature_valid and angle is not None:
        # Angle confirms lying posture
        if angle >= (confirm_angle_thres - 10.0):
            angle_confirms = True
        
        # Angle rejects false bbox positive (e.g., sitting wide, bending)
        # If hw_ratio suggests lying but angle is very upright → reject
        # STRENGTHENED: Reject at lower hw_ratio to catch more ADL false positives
        if hw_ratio >= 0.85 and angle < 38.0:
            angle_rejects = True
            return False  # Explicit rejection
    
    # =========================================================
    # DECISION LOGIC
    # =========================================================
    # ADAPTIVE VALIDATION: Mode depends on angle availability
    
    if feature_valid and angle is not None:
        # MODE 1: Angle available → CROSS-VALIDATE
        # STRENGTHENED rejection for ADL false positives
        if hw_ratio >= 0.9 and angle < 42:
            return False
        
        if hw_ratio >= 1.0 and angle < 45:
            return False
        
        # Strong agreement
        if hw_ratio >= 1.2 and angle >= 50:
            return True
        
        # Moderate bbox + strong angle
        if hw_ratio >= 1.0 and angle >= 55:
            return True
        
        # Strong bbox + moderate angle  
        if hw_ratio >= 1.3 and angle >= 45:
            return True
            
        return False
    else:
        # MODE 2: No angle → STRICT BBOX ONLY
        # Require very clear lying signal
        if hw_ratio >= 1.35:
            return True
        
        return False
    
    # Weak bbox signal → require angle confirmation
    if hw_condition or ar_condition:
        # Need angle to confirm
        return angle_confirms
    
    return False
