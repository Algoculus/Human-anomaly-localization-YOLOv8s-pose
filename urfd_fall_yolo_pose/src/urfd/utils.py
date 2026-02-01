import yaml
import random
import numpy as np
import torch

def compute_iou(bbox1, bbox2):
    """
    Compute IoU (Intersection over Union) between two bboxes.
    
    IoU = Area of Intersection / Area of Union
    Range: [0, 1] where 1 = perfect overlap
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        iou: Intersection over Union value
    """
    # Compute intersection coordinates
    x1_int = max(bbox1[0], bbox2[0])
    y1_int = max(bbox1[1], bbox2[1])
    x2_int = min(bbox1[2], bbox2[2])
    y2_int = min(bbox1[3], bbox2[3])
    
    # Compute intersection area (clamp to 0 if no overlap)
    inter_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    
    # Compute individual bbox areas
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Union = A + B - Intersection
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def compute_center_distance(bbox1, bbox2):
    """
    Compute normalized center distance between two bboxes.
    
    Distance is normalized by the max dimension of bbox1
    to make it scale-invariant.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        distance: Normalized Euclidean distance
    """
    # Compute centers
    c1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
    c2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
    
    # Compute normalization factor (max dimension of bbox1)
    h1 = bbox1[3] - bbox1[1]
    w1 = bbox1[2] - bbox1[0]
    normalize_factor = max(h1, w1)
    
    if normalize_factor <= 0:
        return 1e6  # Return large value for degenerate boxes
    
    return np.linalg.norm(c1 - c2) / normalize_factor

def validate_config(config):
    """
    Validate configuration for required keys and reasonable values.
    
    Checks:
    - All required keys are present
    - Threshold values are within valid ranges
    
    Args:
        config: Dict with configuration parameters
    
    Raises:
        ValueError: If config is invalid
    """
    required_keys = [
        "yolo_model", "imgsz", "conf_thres", "iou_thres", "keypoint_conf_thres",
        "track_iou_thres", "track_center_dist_thres", "track_max_missing",
        "cand_enter_frames", "confirm_frames", "confirm_window", "height_window",
        "max_missing_streak", "recovery_window", "score_decay",
        "output_fps", "output_dir", "seed"
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    # Validate threshold ranges
    if not (0 < config["conf_thres"] <= 1.0):
        raise ValueError(f"conf_thres must be in (0, 1], got {config['conf_thres']}")
    
    if not (0 < config["iou_thres"] <= 1.0):
        raise ValueError(f"iou_thres must be in (0, 1], got {config['iou_thres']}")
    
    if config["cand_enter_frames"] < 1:
        raise ValueError(f"cand_enter_frames must be >= 1, got {config['cand_enter_frames']}")
    
    if config["confirm_frames"] < 1:
        raise ValueError(f"confirm_frames must be >= 1, got {config['confirm_frames']}")
    
    print("Config validation passed.")

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        config: Dict with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate after loading
    validate_config(config)
    
    # Log key parameters for debugging
    print(f"Loaded config from {config_path}")
    print(f"  min_confirm_duration_frames: {config.get('min_confirm_duration_frames', 1)}")
    
    return config

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for: random, numpy, torch (CPU + CUDA)
    Also enables deterministic mode for CUDA.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_depth_path_from_rgb(rgb_path: str) -> str:
    """
    Construct Depth map path from RGB image path for URFD dataset.
    
    Structure:
    RGB: .../fall-XX-cam0-rgb/fall-XX-cam0-rgb-YYY.png
    Depth: .../fall-XX-cam0-d/fall-XX-cam0-d-YYY.png
    
    Logic:
    1. Replace 'rgb' with 'd' in folder name and filename.
    2. Check existence.
    """
    import os
    
    if "rgb" not in rgb_path:
        return None
        
    # Replace 'rgb' with 'd' strictly for the last part of path
    # Assuming standard URFD naming
    depth_path = rgb_path.replace("rgb", "d")
    
    if os.path.exists(depth_path):
        return depth_path
    return None

def load_depth_map(depth_path: str):
    """
    Load depth map (16-bit PNG) and convert to meters.
    URFD Depth is typically 1000 scales (mm).
    """
    import cv2
    import os
    if not depth_path:
        return None
        
    # Load check
    if not os.path.exists(depth_path):
        return None
        
    # Load as -1 to keep original depth info (16-bit or 8-bit)
    depth_img = cv2.imread(depth_path, -1)
    
    if depth_img is None:
        return None
        
    # Convert to meters (URFD depth is usually in mm)
    # Check max value to confirm scale. If > 255, it's mm.
    if depth_img.max() > 255:
        depth_m = depth_img.astype(np.float32) / 1000.0
    else:
        # Fallback or unknown scale? 
        # For now assume it's normalized 0-255 map? unlikely for Kinect raw.
        # But URFD paper says Kinect. It should be 16-bit.
        depth_m = depth_img.astype(np.float32) / 255.0 * 5.0 # Max range 5m guess?
        
    return depth_m
