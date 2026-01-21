"""
Low-light preprocessing for improving YOLO detection in dark scenes.
Uses gamma correction + CLAHE on luminance channel.
"""

import cv2
import numpy as np

def preprocess_lowlight(frame, gamma=1.3, clahe_clip=2.0, clahe_grid=8):
    """Apply gamma correction and CLAHE to improve visibility in dark scenes.
    
    Args:
        frame: Input BGR frame
        gamma: Gamma correction value (>1 brightens, <1 darkens)
        clahe_clip: CLAHE clip limit
        clahe_grid: CLAHE grid size
    
    Returns:
        Preprocessed BGR frame
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    
    # Gamma correction on Y channel
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    y_gamma = cv2.LUT(y_channel, table)
    
    # CLAHE on gamma-corrected Y
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    y_clahe = clahe.apply(y_gamma)
    
    # Reconstruct image
    ycrcb[:, :, 0] = y_clahe
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return result
