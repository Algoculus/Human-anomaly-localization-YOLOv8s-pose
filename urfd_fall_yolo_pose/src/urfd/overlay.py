"""
Fall Detection Overlay Visualization

Creates annotated videos with:
- Bounding boxes colored by state
- Skeleton keypoints
- Confidence meters
- Debug info (top features, FSM state)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# COCO skeleton connections
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Lower body
    [6, 12], [7, 13], [6, 7],                           # Torso
    [6, 8], [7, 9], [8, 10], [9, 11],                   # Arms
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # Head
]

# State colors (BGR)
COLOR_NORMAL = (0, 255, 0)        # Green
COLOR_HYPOTHESIS = (0, 200, 255)   # Orange
COLOR_VERIFYING = (0, 165, 255)    # Dark orange
COLOR_FALL = (0, 0, 255)           # Red
COLOR_UNCERTAIN = (0, 255, 255)    # Yellow

# Confidence bar colors
COLOR_BAR_BG = (50, 50, 50)
COLOR_BAR_LOW = (0, 255, 0)
COLOR_BAR_MED = (0, 255, 255)
COLOR_BAR_HIGH = (0, 0, 255)


def get_state_color(state: str) -> Tuple[int, int, int]:
    """Get color for FSM state."""
    state_upper = state.upper()
    if "FALL" in state_upper or "CONFIRMED" in state_upper:
        return COLOR_FALL
    elif "VERIFYING" in state_upper:
        return COLOR_VERIFYING
    elif "HYPOTHESIS" in state_upper or "CANDIDATE" in state_upper:
        return COLOR_HYPOTHESIS
    else:
        return COLOR_NORMAL


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Get color based on confidence level."""
    if confidence < 0.35:
        return COLOR_BAR_LOW
    elif confidence < 0.6:
        return COLOR_BAR_MED
    else:
        return COLOR_BAR_HIGH


def draw_skeleton(img, keypoints, color, conf_thres=0.5):
    """Draw skeleton on image."""
    if keypoints is None:
        return
    
    # Draw keypoint circles
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf >= conf_thres:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
    
    # Draw skeleton lines
    for connection in SKELETON:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            kp1 = keypoints[idx1]
            kp2 = keypoints[idx2]
            if kp1[2] >= conf_thres and kp2[2] >= conf_thres:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(img, pt1, pt2, color, 2)


def draw_confidence_bar(img, x: int, y: int, width: int, height: int, 
                        confidence: float, label: str = ""):
    """Draw confidence bar with label."""
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_BAR_BG, -1)
    
    # Confidence fill
    fill_width = int(width * confidence)
    color = get_confidence_color(confidence)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 1)
    
    # Text
    text = f"{label} {confidence:.0%}"
    cv2.putText(img, text, (x + 2, y + height - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_debug_panel(img, x: int, y: int, info: Dict, width: int = 200):
    """Draw debug information panel."""
    # Semi-transparent background
    panel_height = 80
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Draw info
    line_y = y + 15
    line_height = 15
    
    # State
    state = info.get("state", "NORMAL")
    color = get_state_color(state)
    cv2.putText(img, f"State: {state}", (x + 5, line_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    line_y += line_height
    
    # Confidence
    conf = info.get("confidence", 0.0)
    cv2.putText(img, f"Conf: {conf:.2f}", (x + 5, line_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    line_y += line_height
    
    # Top features
    top_features = info.get("top_features", [])
    for name, value, score in top_features[:2]:
        short_name = name[:12]
        cv2.putText(img, f"{short_name}: {value:.2f}", (x + 5, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        line_y += line_height - 2


def draw_feature_bars(img, x: int, y: int, features: Dict, width: int = 100):
    """Draw mini feature indicator bars (simplified)."""
    bar_height = 8
    spacing = 3
    
    # HW Ratio (w/h)
    hw = features.get("hw_ratio", 0.0)
    draw_mini_bar(img, x, y, width, bar_height, hw, "HW", max_val=3.0)
    y += bar_height + spacing
    
    # Height Ratio (h/hmax)
    hr = features.get("height_ratio", 1.0)
    draw_mini_bar(img, x, y, width, bar_height, 1.0 - hr, "HR", max_val=1.0)
    y += bar_height + spacing
    
    # Floor Distance
    fd = features.get("floor_distance", 0.0)
    draw_mini_bar(img, x, y, width, bar_height, fd, "FD", max_val=1.0)


def draw_mini_bar(img, x: int, y: int, width: int, height: int, 
                  value: float, label: str, max_val: float = 1.0):
    """Draw a mini indicator bar."""
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Fill
    fill = np.clip(value / max_val, 0.0, 1.0)
    fill_width = int(width * fill)
    color = get_confidence_color(fill)
    if fill_width > 0:
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    
    # Label
    cv2.putText(img, label, (x - 25, y + height - 1), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)


def create_overlay_video(frames: List, all_tracks_data: List, 
                         output_path: str, fps: int, 
                         debug_mode: bool = True,
                         primary_indices: List = None):
    """
    Create overlay video with detection results.
    
    Args:
        frames: List of BGR images
        all_tracks_data: List of dicts per frame {track_id: {bbox, keypoints, state, score, ...}}
        output_path: Output video path
        fps: Output FPS
        debug_mode: Show debug overlays
        primary_indices: Optional list of primary person indices
    """
    if len(frames) == 0:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for idx, frame in enumerate(frames):
        overlay = frame.copy()
        
        # Get track data for this frame
        frame_tracks = all_tracks_data[idx] if idx < len(all_tracks_data) else {}
        
        # Handle different data formats
        if isinstance(frame_tracks, dict):
            tracks_to_draw = frame_tracks.items()
        elif isinstance(frame_tracks, list):
            # Legacy format: list of detections
            tracks_to_draw = [(i, d) for i, d in enumerate(frame_tracks) if d is not None]
        else:
            tracks_to_draw = []
        
        for tid, track_info in tracks_to_draw:
            if track_info is None:
                continue
                
            state = track_info.get("state", "NORMAL")
            score = track_info.get("score", 0.0)
            color = get_state_color(state)
            
            # Draw bounding box
            bbox = track_info.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = track_info.get("label", state)
                label_text = f"ID:{tid} {label} {score:.2f}"
                cv2.putText(overlay, label_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Confidence bar below bbox
                bar_y = min(y2 + 5, h - 15)
                bar_width = min(x2 - x1, 100)
                draw_confidence_bar(overlay, x1, bar_y, bar_width, 12, score, "")
                
                # Debug panel
                if debug_mode:
                    panel_x = x2 + 5
                    if panel_x + 150 > w:
                        panel_x = max(0, x1 - 155)
                    
                    debug_info = {
                        "state": state,
                        "confidence": score,
                        "top_features": track_info.get("top_features", [])
                    }
                    
                    # Feature bars (simplified features only)
                    if "features" in track_info:
                        feat_data = track_info["features"].to_dict() if hasattr(track_info["features"], 'to_dict') else track_info["features"]
                        draw_feature_bars(overlay, x1, bar_y + 18, feat_data, 
                                         width=min(80, x2 - x1))
            
            # Draw skeleton
            keypoints = track_info.get("keypoints")
            if keypoints is not None:
                draw_skeleton(overlay, keypoints, color)
        
        # Frame counter
        cv2.putText(overlay, f"Frame: {idx}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.write(overlay)
    
    writer.release()


def create_debug_overlay_frame(frame, detections: List, decision, features: Dict = None):
    """
    Create single frame with comprehensive debug overlay.
    
    Args:
        frame: BGR image
        detections: List of detection dicts
        decision: FallDecision object
        features: Optional features dict
        
    Returns:
        Annotated frame
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw detections
    for det in detections:
        if det is None:
            continue
            
        bbox = det.get("bbox")
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        color = get_state_color(decision.fsm_state if decision else "NORMAL")
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        keypoints = det.get("keypoints")
        if keypoints is not None:
            draw_skeleton(overlay, keypoints, color)
    
    # Decision panel (top-left)
    if decision:
        panel_width = 250
        panel_height = 100
        
        # Background
        cv2.rectangle(overlay, (5, 5), (5 + panel_width, 5 + panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(overlay, (5, 5), (5 + panel_width, 5 + panel_height), 
                     (100, 100, 100), 1)
        
        # Label with color
        label_color = COLOR_FALL if decision.label == "FALL" else \
                      COLOR_UNCERTAIN if decision.label == "UNCERTAIN" else COLOR_NORMAL
        cv2.putText(overlay, f"LABEL: {decision.label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        
        # State
        state_color = get_state_color(decision.fsm_state)
        cv2.putText(overlay, f"State: {decision.fsm_state}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        
        # Confidence bar
        draw_confidence_bar(overlay, 10, 55, panel_width - 10, 15, 
                           decision.confidence, "Conf")
        
        # Top features
        y_pos = 80
        for name, value, score in decision.top_features[:2]:
            text = f"{name[:15]}: {value:.2f} ({score:.2f})"
            cv2.putText(overlay, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_pos += 12
    
    # Features panel (bottom-left)
    if features:
        panel_y = h - 80
        
        # HW Ratio
        hw = features.get("hw_ratio", 0)
        draw_mini_bar(overlay, 35, panel_y, 80, 10, hw, "HW", 3.0)
        
        # Height Ratio
        hr = features.get("height_ratio", 1.0)
        draw_mini_bar(overlay, 35, panel_y + 15, 80, 10, 1.0 - hr, "HR", 1.0)
        
        # Floor Distance
        fd = features.get("floor_distance", 0)
        draw_mini_bar(overlay, 35, panel_y + 30, 80, 10, fd, "FD", 1.0)
    
    return overlay
