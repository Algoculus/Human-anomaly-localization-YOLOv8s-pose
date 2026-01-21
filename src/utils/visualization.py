"""
Visualization Utilities

Draws overlays for fall detection:
- Skeleton and Bounding Box
- State and Score indicators
- Alert history bars
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from loguru import logger
from pathlib import Path

from infer_pose import PoseResult, draw_pose
from rules.state_machine import State
from config import Config, get_config

class Visualizer:
    """
    Handles drawing of fall detection UI on frames.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.viz_config = self.config.visualization
        
        # Colors (BGR) - Neon/Cyberpunk Style
        self.colors = {
            State.NORMAL: (0, 255, 0),         # Neon Green
            State.FALLING: (0, 165, 255),      # Neon Orange
            State.LYING: (0, 0, 255),          # Neon Red
            State.LYING_NO_FALL: (255, 255, 0), # Cyan
            State.OCCLUDED: (128, 128, 128),   # Gray
        }
        
    def draw_frame(self, 
                   image: np.ndarray, 
                   frame_number: int,
                   state: State,
                   fall_score: float,
                   behavior_label: str = "Unknown",
                   scores_dict: Optional[dict] = None,
                   poses: List[PoseResult] = None,
                   primary_track_id: Optional[int] = None) -> np.ndarray:
        """
        Draw comprehensive overlay on frame.
        """
        # 1. Draw Poses
        color = self.colors.get(state, (255, 255, 255))
        
        if poses:
            for pose in poses:
                # Use track ID specific color if available, else state color
                draw_pose(image, pose, color_bbox=color, color_skeleton=(255, 255, 0)) # Cyan Skeleton
                
        # 2. Draw HUD (Heads Up Display)
        h, w = image.shape[:2]
        
        # Sidebar for stats
        sidebar_w = 250
        sidebar_h = 300 # Increased to fit more metrics
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (sidebar_w, sidebar_h), (10, 10, 10), -1) # Dark background
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1) # Top bar
        
        alpha = 0.7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # --- Top Bar: State & Main Score ---
        # State Indicator
        state_color = self.colors.get(state, (200, 200, 200))
        cv2.putText(image, f"STATE: {state.value}", (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)
        
        # Risk Score Bar (Top Right)
        score_w = 200
        score_x = w - score_w - 20
        # Bg
        cv2.rectangle(image, (score_x, 15), (score_x + score_w, 45), (50, 50, 50), -1)
        # Fill
        fill_w = int(score_w * min(fall_score, 1.0))
        # Gradient Color (Green -> Yellow -> Red)
        r = int(255 * min(fall_score * 2, 1.0))
        g = int(255 * min(2.0 * (1 - fall_score), 1.0))
        score_color = (0, g, r) 
        
        cv2.rectangle(image, (score_x, 15), (score_x + fill_w, 45), score_color, -1)
        cv2.putText(image, f"RISK: {fall_score:.2f}", (score_x + 10, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Sidebar: Detailed Metrics ---
        # Draw frame number
        cv2.putText(image, f"Frame: {frame_number}", (15, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw Behavior Label (Prominent)
        label_color = (0, 255, 255) if "ALERT" in behavior_label else (255, 255, 255)
        cv2.putText(image, behavior_label, (15, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)

        if scores_dict:
            y_off = 150  # Moved down to account for behavior label
            for k, v in scores_dict.items():
                if isinstance(v, (float, int)):
                    # Color code high values for alert indicators
                    val_color = (200, 200, 200)  # Default gray
                    
                    # Check for alert-worthy high values
                    if v > 0.5:
                        if 'Drop' in k or 'Impact' in k or 'Prone' in k or 'Fall' in k:
                            val_color = (0, 165, 255)  # Orange for alert values
                    
                    # Format based on value range
                    if abs(v) > 10:  # Likely degrees or pixels
                        text = f"{k:<12}: {v:.1f}"
                    else:
                        text = f"{k:<12}: {v:.3f}"
                        
                    cv2.putText(image, text, (15, y_off), 
                               cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, val_color, 1)
                    y_off += 22
                    
        return image

if __name__ == "__main__":
    # Test Visualizer
    viz = Visualizer()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock data
    scores = {'Drop Vel': 0.08, 'Prone': 0.95, 'Impact': 0.0, 'Orient': 85.0}
    img = viz.draw_frame(img, 124, State.LYING, 0.92, scores_dict=scores)
    
    out_path = Path("d:/Human-anomaly-localization-YOLOv8s-pose/output/test_viz.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Test visualization saved to {out_path}")
