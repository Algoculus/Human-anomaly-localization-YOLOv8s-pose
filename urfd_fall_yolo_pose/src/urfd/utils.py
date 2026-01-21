import yaml
import random
import numpy as np
import torch

def validate_config(config):
    """Validate configuration for required keys and reasonable values.
    
    FIX-5: Add config validation to catch errors early
    
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
    
    # Validate ranges
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
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        config: Dict with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # FIX-5: Validate config after loading
    validate_config(config)
    
    # Log key parameters
    print(f"Loaded config from {config_path}")
    print(f"  dy_peak_thres: {config['dy_peak_thres']}")
    print(f"  confirm_angle_thres: {config['confirm_angle_thres']}")
    print(f"  confirm_ar_thres: {config['confirm_ar_thres']}")
    print(f"  min_confirm_duration_frames: {config.get('min_confirm_duration_frames', 3)}")
    
    return config

def set_seed(seed):
    """Set random seeds for reproducibility.
    
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
