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
        "angle_thres", "ar_thres", "confirm_angle_thres", "confirm_ar_thres",
        "height_drop_thres", "height_drop_thres_strong",
        "dy_fall_thres", "dy_window", "dy_peak_thres",
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
    
    if config["dy_window"] < 1:
        raise ValueError(f"dy_window must be >= 1, got {config['dy_window']}")
    
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
    print(f"  dy_peak_thres: {config['dy_peak_thres']}")
    print(f"  confirm_angle_thres: {config['confirm_angle_thres']}")
    print(f"  confirm_ar_thres: {config['confirm_ar_thres']}")
    print(f"  min_confirm_duration_frames: {config.get('min_confirm_duration_frames', 3)}")
    
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
