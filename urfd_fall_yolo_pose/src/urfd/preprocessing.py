"""
Low-light preprocessing for improving YOLO detection in dark scenes.
Uses gamma correction + CLAHE on luminance channel.
"""

import cv2
import numpy as np

def preprocess_lowlight(frame, gamma=1.3, clahe_clip=2.0, clahe_grid=8):
    """
    Apply gamma correction and CLAHE to improve visibility in dark scenes.
    
    Algorithm:
    1. Convert BGR to YCrCb (separate luminance from chrominance)
    2. Apply gamma correction to Y channel (brightens dark areas)
    3. Apply CLAHE to Y channel (enhances local contrast)
    4. Convert back to BGR
    
    Args:
        frame: Input BGR frame (numpy array)
        gamma: Gamma correction value (>1 brightens, <1 darkens)
        clahe_clip: CLAHE clip limit (higher = more contrast)
        clahe_grid: CLAHE grid size (smaller = more local)
    
    Returns:
        Preprocessed BGR frame
    """
    # Convert to YCrCb (Y = luminance, CrCb = chrominance)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    
    # Gamma correction: output = (input/255)^(1/gamma) * 255
    # gamma > 1 makes dark areas brighter while preserving highlights
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    y_gamma = cv2.LUT(y_channel, table)  # Apply lookup table
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Enhances local contrast without over-amplifying noise
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    y_clahe = clahe.apply(y_gamma)
    
    # Reconstruct the image with enhanced Y channel
    ycrcb[:, :, 0] = y_clahe
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return result