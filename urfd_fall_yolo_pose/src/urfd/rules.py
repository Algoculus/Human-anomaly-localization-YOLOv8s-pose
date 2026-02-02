import numpy as np

def check_fall_candidate(features, angle_thres, ar_thres, height_drop, 
                         height_drop_thres, dy_fall_thres, impact_dy_thres=10.0,
                         high_angle_thres=65.0, high_angle_dy_peak_thres=8.0,
                         min_candidate_dy_peak=12.0):
    """
    Check if current frame is a fall candidate.
    
<<<<<<< HEAD
    ADAPTIVE CAM0-SAFE APPROACH: 
    - PRIMARY: Trigger detection WITHOUT relying on body angle (Impact + Shape drop)
    - ADAPTIVE: Use angle intelligently when bbox is unreliable or person turned away
    
    TRIGGER LOGIC (Requires BOTH groups OR special cases):
    
    GROUP 1 - IMPACT (at least 1 must be true):
      * dy_peak >= 6-8 (impact velocity)
      * dh <= -8% height (rapid height collapse)
      * d2y >= 3-4 (free fall acceleration)
    
    GROUP 2 - SHAPE DROP (at least 1 must be true):
      * height_drop >= 16-20% (significant height reduction)
      * width_increase >= 12-18% (body expansion)
      * d_ar >= 0.25 (rapid aspect ratio change)
    
    ADAPTIVE MODES:
    - BBOX-DOMINANT: When bbox is reliable (not touching border, good tracking)
    - ANGLE-ASSISTED: When bbox unreliable (border, small size) OR person turned away
    - HYBRID: When both available and reliable
    
    ANGLE USAGE:
    - DEFENSIVE: ADL rejection (small angle + no impact)
    - OFFENSIVE: Fall confirmation when bbox ambiguous (large angle + some motion)
    - BACKUP: When bbox temporarily lost but angle shows fall
    
    KEY: Works for cam0 frontal view - fall toward/away camera still shows:
      - height decrease, width increase, bottom_y moves down, dy_peak appears
=======
    ALL PATHS NOW REQUIRE MINIMUM IMPACT MOTION (dy_peak >= min_candidate_dy_peak)
    This prevents false positives from controlled lying down.
    
    A frame is a fall candidate if ANY of these conditions are met:
    - PATH 1: (body_angle >= angle_thres AND AR >= ar_thres AND dy_peak >= impact_dy_thres)
    - PATH 2: height_drop >= height_drop_thres AND dy_peak >= min_candidate_dy_peak
    - PATH 3: dy >= dy_fall_thres AND dy_peak >= min_candidate_dy_peak
    - PATH 4: body_angle >= high_angle_thres AND dy_peak >= high_angle_dy_peak_thres
>>>>>>> 15eb1a4bccf6da447e5e6450b7099f4febb912ae
    
    Args:
        features: Frame features dict from compute_frame_features
        angle_thres: Angle threshold (used adaptively)
        ar_thres: Aspect ratio threshold (W/H)
        height_drop: Normalized height drop value
<<<<<<< HEAD
        height_drop_thres: Height drop threshold
        dy_fall_thres: dy threshold
        impact_dy_thres: Minimum dy_peak for impact
        high_angle_thres: High angle threshold
=======
        height_drop_thres: Height drop threshold (0.18)
        dy_fall_thres: dy threshold for fast fall detection (10.0)
        impact_dy_thres: Minimum dy_peak for posture path (10.0)
        high_angle_thres: High angle threshold for Path 4 (65.0)
        high_angle_dy_peak_thres: Minimum dy_peak for high angle path (8.0)
        min_candidate_dy_peak: Minimum dy_peak for PATH 2 & 3 (12.0)
>>>>>>> 15eb1a4bccf6da447e5e6450b7099f4febb912ae
        
    Returns:
        is_candidate: True if fall candidate detected
    """
    if features["bbox"] is None:
        return False
    
    # =========================================================
    # EXTRACT FEATURES & ASSESS RELIABILITY
    # =========================================================
    hw_ratio = features.get("hw_ratio", 0.0)
    dy_peak = features.get("dy_peak", 0.0)
    dy = features.get("dy", 0.0)
    d2y = features.get("d2y", 0.0)
    dh = features.get("dh", 0.0)
    dw = features.get("dw", 0.0)
    d_ar = features.get("d_ar", 0.0)
    height = features.get("height", 0.0)
    width = features.get("width", 0.0)
    is_touching_border = features.get("is_touching_border", False)
    angle = features.get("body_angle_deg")
    feature_valid = features.get("feature_valid", False)
    
    # =========================================================
    # ASSESS BBOX RELIABILITY
    # Bbox may be unreliable when:
    # - Touching border (clipped)
    # - Very small (person far away or partially visible)
    # - Person turned away from camera (side/back view)
    # =========================================================
    
    bbox_reliable = True
    bbox_confidence = 1.0
    
    # Factor 1: Border touching reduces confidence
    if is_touching_border:
        bbox_reliable = False
        bbox_confidence *= 0.5
    
    # Factor 2: Small bbox (far away or occluded)
    if height > 0 and height < 80:  # pixels
        bbox_confidence *= 0.7
        if height < 50:
            bbox_reliable = False
    
    # Factor 3: Very wide bbox may indicate side/back view
    if hw_ratio > 1.5 and not is_touching_border:
        # Extremely wide - unusual, may be turned away
        bbox_confidence *= 0.8
    
    # =========================================================
<<<<<<< HEAD
    # ADAPTIVE MODE SELECTION
    # =========================================================
    
    # Determine which signals to trust more
    angle_available = feature_valid and angle is not None
    
    if bbox_reliable and not angle_available:
        # MODE 1: BBOX-ONLY (bbox good, no angle)
        detection_mode = "BBOX_ONLY"
        angle_weight = 0.0
        bbox_weight = 1.0
    elif not bbox_reliable and angle_available:
        # MODE 2: ANGLE-ASSISTED (bbox unreliable, use angle more)
        detection_mode = "ANGLE_ASSISTED"
        angle_weight = 0.6
        bbox_weight = 0.4
    elif bbox_reliable and angle_available:
        # MODE 3: HYBRID (both available and reliable)
        detection_mode = "HYBRID"
        angle_weight = 0.3
        bbox_weight = 0.7
    else:
        # MODE 4: DEGRADED (bbox unreliable, no angle)
        detection_mode = "DEGRADED"
        angle_weight = 0.0
        bbox_weight = 0.5  # Lower confidence
    
    # =========================================================
    # PRE-SCREENING: IMMEDIATE ADL REJECTION
    # Reject obvious ADL cases before impact/shape checks
    # =========================================================
    
    # ADL Pattern 1: Small angle + minimal motion + no height drop
    # → Standing, walking, normal activities
    if angle_available and angle < 35 and dy_peak < 3.0 and height_drop < 0.08:
        return False
    
    # ADL Pattern 2: Large angle BUT no impact (controlled lying)
    # → Yoga, resting, getting into bed slowly
    if angle_available and angle >= 50 and dy_peak < 2.5 and abs(dh) < 5 and d_ar < 0.15:
        return False
    
    # ADL Pattern 3: Slow gradual shape change (sitting down)
    # → Controlled posture transitions
    if hw_ratio >= 0.9 and dy_peak < 2.0 and abs(d2y) < 1.0:
        return False
    
    # =========================================================
    # GROUP 1: IMPACT DETECTION (at least 1 required)
    # Adjusted thresholds based on bbox reliability
    # =========================================================
=======
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
>>>>>>> 15eb1a4bccf6da447e5e6450b7099f4febb912ae
    
    # Impact 1: High velocity peak (6-8 depending on fps)
    # Detects sudden downward motion / impact moment
    impact_velocity_threshold = 6.0 if bbox_reliable else 5.0
    impact_velocity = dy_peak >= impact_velocity_threshold
    
    # Impact 2: Rapid height collapse (dh <= -8% of height)
    # Detects sudden body compression/collapse
    impact_height_collapse = False
    if height > 0:
        height_change_ratio = dh / height
        collapse_threshold = -0.08 if bbox_reliable else -0.10
        if height_change_ratio <= collapse_threshold:
            impact_height_collapse = True
    
    # Impact 3: Free fall acceleration (d2y >= 3-4)
    # Detects sudden acceleration changes (gravity)
    accel_threshold = 3.0 if bbox_reliable else 4.0
    impact_acceleration = abs(d2y) >= accel_threshold
    
    impact_detected = impact_velocity or impact_height_collapse or impact_acceleration
    
    # =========================================================
    # GROUP 2: SHAPE DROP (at least 1 required)
    # Adjusted thresholds based on bbox reliability
    # =========================================================
    
    # Shape 1: Significant height drop over time (16-20%)
    shape_drop_threshold = 0.16 if bbox_reliable else 0.18
    shape_height_drop = height_drop >= shape_drop_threshold
    
    # Shape 2: Width increase (12-18%)
    # Body expands horizontally when lying down
    shape_width_increase = False
    if width > 0:
        # Use d_ar as proxy (AR increases when width increases)
        if d_ar >= 0.12 or (dw > 0 and hw_ratio >= 1.0):
            shape_width_increase = True
    
    # Shape 3: Rapid aspect ratio change (>= 0.25)
    # Sudden posture transition from standing to lying
    ar_change_threshold = 0.25 if bbox_reliable else 0.30
    shape_ar_change = d_ar >= ar_change_threshold
    
    shape_drop_detected = shape_height_drop or shape_width_increase or shape_ar_change
    
    # =========================================================
    # ANGLE-ASSISTED DETECTION (when angle available)
    # Use angle to BOOST detection when bbox is ambiguous
    # =========================================================
    
    angle_suggests_fall = False
    angle_confidence_score = 0.0
    
    if angle_available:
        # Angle strongly suggests fall (lying posture)
        if angle >= 55:
            angle_suggests_fall = True
            angle_confidence_score = min((angle - 55) / 35, 1.0)  # 0-1 scale
        
        # Moderate angle may suggest fall if there's some motion
        elif angle >= 45 and (dy_peak >= 3.0 or height_drop >= 0.10):
            angle_suggests_fall = True
            angle_confidence_score = 0.5
    
    # =========================================================
    # ADAPTIVE TRIGGER DECISION
    # =========================================================
    
    if detection_mode == "BBOX_ONLY" or detection_mode == "HYBRID":
        # Standard trigger: Both impact AND shape drop
        if impact_detected and shape_drop_detected:
            return True
    
    elif detection_mode == "ANGLE_ASSISTED":
        # Relaxed trigger: Impact OR shape drop + angle confirmation
        if (impact_detected or shape_drop_detected) and angle_suggests_fall:
            return True
        
        # Or very strong angle + any motion signal
        if angle_confidence_score >= 0.7 and (impact_detected or shape_drop_detected):
            return True
    
    elif detection_mode == "DEGRADED":
        # Very strict: Need both groups + strong signals
        if impact_detected and shape_drop_detected:
            if dy_peak >= 7.0 or height_drop >= 0.20:
                return True
    
    # =========================================================
    # ENHANCED CHECKS: Very strong single signals (EMERGENCY)
    # =========================================================
    
    # Emergency 1: Catastrophic height drop (>25%)
    if height_drop >= 0.25:
        return True
    
    # Emergency 2: Extreme velocity + any shape change
    if dy_peak >= 10.0 and (height_drop >= 0.10 or d_ar >= 0.15):
        return True
    
    # Emergency 3: Strong acceleration + height drop
    if abs(d2y) >= 5.0 and height_drop >= 0.12:
        return True
    
    # Emergency 4: Angle-based fall (person turned away, bbox ambiguous)
    # This catches falls when person is facing away from camera
    if angle_available and angle >= 65 and dy_peak >= 4.0:
        # Very horizontal + some impact → likely fall
        return True
    
    # =========================================================
    # ANGLE-BASED ADL REJECTION (Final filtering)
    # Use angle DEFENSIVELY to reject false positives
    # =========================================================
    
    if angle_available:
        # Marginal case: Some impact OR shape but not both
        if (impact_detected or shape_drop_detected) and not (impact_detected and shape_drop_detected):
            # Only one group satisfied - check angle for ADL patterns
            
            # Upright posture, narrow bbox → bending, picking up objects
            if angle < 40 and hw_ratio < 1.0:
                return False
            
            # Very upright → ADL even with wider bbox (squatting, sitting)
            if angle < 35 and hw_ratio < 1.1:
                return False
            
            # Moderate angle but very slow motion → controlled movement
            if angle < 50 and dy_peak < 3.0 and abs(d2y) < 1.5:
                return False
    
    return False

def check_lying_posture(features, confirm_angle_thres, confirm_ar_thres, history=None, config=None):
    """
    Check if current frame confirms lying/fallen posture.
    
    CAM0-SAFE CONFIRMATION: Prioritize low height + motion settled
    
    BRANCH 1 - Cam0-safe (NO angle required):
      * low_height_duration >= 12-20 frames (sustained low position)
      * AND post_motion_low (|dy| < threshold for N frames)
      * Person is STAYING DOWN after impact
    
    BRANCH 2 - Angle-assisted (IF reliable):
      * angle >= 50° OR angle_change >= 20°
      * AND hw_ratio >= 1.2-1.3
      * AND had impact before (dy_peak in history)
    
    CRITICAL: Prevent "pick up object" false positives:
      * Candidate (bend down) → Stand up → NOT confirmed as fall
      * Must ensure person STAYS LOW, not transient
    
    Args:
        features: Current frame features
        confirm_angle_thres: Angle threshold for confirmation
        confirm_ar_thres: Aspect ratio threshold  
        history: List of previous frame features (for duration check)
        config: Configuration dict (for thresholds)
        
    Returns:
        is_lying: True if lying posture confirmed
    """
    if features["bbox"] is None:
        return False
    
    hw_ratio = features.get("hw_ratio", 0.0)
    dy = features.get("dy", 0.0)
    dy_peak = features.get("dy_peak", 0.0)
    height = features.get("height", 0.0)
    is_touching_border = features.get("is_touching_border", False)
    angle = features.get("body_angle_deg")
    feature_valid = features.get("feature_valid", False)
    
    # =========================================================
    # BRANCH 1: CAM0-SAFE CONFIRMATION
    # Low height + motion settled (works for any camera angle)
    # =========================================================
    
    if history is not None and len(history) >= 12:
        # Calculate low height duration
        # "Low height" means height significantly below recent maximum
        
        # Get height history
        height_history = [h.get("height", 0) for h in history[-20:] 
                         if isinstance(h, dict) and h.get("height") is not None]
        
        if len(height_history) >= 12 and height > 0:
            max_height = max(height_history)
            
            # Check if staying low (< 70% of max height)
            low_height_threshold = 0.70 * max_height
            
            # Count consecutive low height frames
            low_height_count = 0
            for i in range(len(history) - 1, max(len(history) - 20, -1), -1):
                h = history[i].get("height", 0)
                if h > 0 and h <= low_height_threshold:
                    low_height_count += 1
                else:
                    break  # Must be consecutive
            
            # Add current frame
            if height <= low_height_threshold:
                low_height_count += 1
            
            # Check motion settled (low |dy| for recent frames)
            motion_settled = True
            settle_threshold = 3.0  # pixels per frame
            settle_window = 8
            
            for i in range(max(0, len(history) - settle_window), len(history)):
                if isinstance(history[i], dict):
                    frame_dy = abs(history[i].get("dy", 0.0))
                    if frame_dy > settle_threshold:
                        motion_settled = False
                        break
            
            # Current frame motion check
            if abs(dy) > settle_threshold:
                motion_settled = False
            
            # CONFIRM if staying low + motion settled
            # Threshold: 12-20 frames depending on fps
            low_duration_threshold = config.get("low_height_confirm_frames", 15) if config else 15
            
            if low_height_count >= low_duration_threshold and motion_settled:
                # Additional check: Must have had some impact before
                # To avoid false positive on "sitting down slowly"
                dy_peaks = [h.get("dy_peak", 0) for h in history[-20:] if isinstance(h, dict)]
                max_dy_peak_in_history = max(dy_peaks) if dy_peaks else 0.0
                
                if max_dy_peak_in_history >= 4.0:  # Had impact
                    return True
    
    # =========================================================
    # BRANCH 2: ANGLE-ASSISTED CONFIRMATION  
    # Only if keypoints are reliable
    # =========================================================
    
    if feature_valid and angle is not None:
        # Check if angle indicates lying posture
        angle_lying = angle >= 50
        
        # Check if angle changed significantly (fall happened)
        angle_changed = False
        if history is not None and len(history) >= 5:
            recent_angles = [h.get("body_angle_deg") for h in history[-5:] 
                           if isinstance(h, dict) and h.get("feature_valid", False) 
                           and h.get("body_angle_deg") is not None]
            if len(recent_angles) >= 3:
                min_angle = min(recent_angles)
                if angle is not None and (angle - min_angle) >= 20:
                    angle_changed = True
        
        # Check aspect ratio (lying posture)
        ar_lying = hw_ratio >= 1.2
        
        # Check if had impact in history
        had_impact = False
        if history is not None and len(history) >= 10:
            dy_peaks = [h.get("dy_peak", 0) for h in history[-10:] if isinstance(h, dict)]
            max_dy_peak = max(dy_peaks) if dy_peaks else 0.0
            if max_dy_peak >= 5.0:
                had_impact = True
        
        # Confirm if angle + AR + impact
        if (angle_lying or angle_changed) and ar_lying and had_impact:
            return True
        
        # Very strong angle alone (> 60°) with wide bbox
        if not is_touching_border and angle >= 60 and hw_ratio >= 1.3:
            return True
    
    # =========================================================
    # STRICT BBOX-ONLY CONFIRMATION
    # Only for very clear lying posture when angle unavailable
    # =========================================================
    
    if not feature_valid or angle is None:
        # Require VERY clear lying posture
        # High threshold to avoid false positives
        
        if not is_touching_border and hw_ratio >= 1.40:
            # Very wide bbox - almost certainly lying
            # But still check motion settled to avoid transient
            if abs(dy) < 4.0 and dy_peak < 5.0:  # Low current motion
                return True
        
        # Moderate lying + recent impact + motion settled
        if hw_ratio >= 1.25 and dy_peak >= 5.0 and abs(dy) < 3.0:
            return True
    
    # =========================================================
    # ADL REJECTION (Final safety check)
    # =========================================================
    
    # Reject if angle shows upright despite wide bbox
    # (sitting, squatting, bending)
    if feature_valid and angle is not None:
        if angle < 40 and hw_ratio < 1.2:
            return False
        
        if angle < 35:  # Very upright
            return False
    
    return False
