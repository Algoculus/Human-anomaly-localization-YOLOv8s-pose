"""
Fall Detection Configuration Module

Contains all configurations and parameters for the fall detection system.
These values are initial estimates and can be tuned on validation data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import yaml


@dataclass
class PathConfig:
    """Paths configuration for the URFall dataset."""
    
    # Base paths
    data_root: Path = Path("d:/Human-anomaly-localization-YOLOv8s-pose/data")
    output_root: Path = Path("d:/Human-anomaly-localization-YOLOv8s-pose/output")
    
    # Data subdirectories
    @property
    def raw_download(self) -> Path:
        return self.data_root / "00_raw_download"
    
    @property
    def extracted_frames(self) -> Path:
        return self.data_root / "01_extracted_frames"
    
    @property
    def annotations(self) -> Path:
        return self.data_root / "02_annotations"
    
    @property
    def processed_data(self) -> Path:
        return self.data_root / "03_processed"

    # Output subdirectories
    @property
    def output_videos(self) -> Path:
        return self.output_root / "videos"
    
    @property
    def output_videos_fall(self) -> Path:
        return self.output_videos / "fall"
        
    @property
    def output_videos_adl(self) -> Path:
        return self.output_videos / "adl"
    
    @property
    def output_results(self) -> Path:
        return self.output_root / "results"
    
    @property
    def output_results_fall(self) -> Path:
        return self.output_results / "fall"
    
    @property
    def output_results_adl(self) -> Path:
        return self.output_results / "adl"
    
    @property
    def output_plots(self) -> Path:
        return self.output_root / "plots"
    
    @property
    def output_metrics(self) -> Path:
        return self.output_root / "metrics"
    
    @property
    def output_logs(self) -> Path:
        return self.output_root / "logs"
    
    def create_output_dirs(self):
        """Create all output directories if they don't exist."""
        for path in [self.output_videos, self.output_plots, 
                     self.output_metrics, self.output_logs]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """YOLOv8 pose model configuration."""
    
    # Model settings
    model_name: str = "yolov8s-pose.pt"
    device: str = "cuda"  # "cuda" or "cpu"
    conf_threshold: float = 0.25  # Detection confidence threshold
    iou_threshold: float = 0.45  # NMS IoU threshold
    
    # Keypoint settings
    keypoint_conf_threshold: float = 0.5  # Min confidence for valid keypoint
    min_keypoints: int = 8  # Min keypoints for reliable pose (out of 17)
    
    # Inference settings
    imgsz: int = 640  # Input image size
    verbose: bool = False


@dataclass
class PoseFeatureConfig:
    """Configuration for pose feature extraction."""
    
    # YOLOv8-pose COCO keypoint indices (17 keypoints)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    left_shoulder_idx: int = 5
    right_shoulder_idx: int = 6
    left_hip_idx: int = 11
    right_hip_idx: int = 12
    left_knee_idx: int = 13
    right_knee_idx: int = 14
    left_ankle_idx: int = 15
    right_ankle_idx: int = 16
    
    # Feature history length (for velocity calculation)
    history_length: int = 30  # ~1 second at 30fps
    
    # Smoothing parameters
    ema_alpha: float = 0.3  # Exponential moving average alpha
    median_window: int = 5  # Median filter window size


@dataclass
class FallScoreConfig:
    """Configuration for fall score calculation."""
    
    # Weights for score components (must sum to 1.0)
    # Adjusted for Safety Priority (Higher Recall)
    weight_sudden_drop: float = 0.23  # Slightly reduced
    weight_prone: float = 0.42  # Increased for sensitivity
    weight_impact: float = 0.15  # Important for true falls
    weight_sustained_lying: float = 0.20
    
    # Sudden drop detection
    drop_threshold: float = 0.03  # Lowered for higher sensitivity
    drop_window_min_sec: float = 0.3  # Min window for sudden drop (seconds)
    drop_window_max_sec: float = 1.0  # Max window for sudden drop (seconds)
    drop_sigmoid_k: float = 10.0  # Sigmoid steepness
    
    # Prone/lying detection (Recall-optimized)
    orientation_threshold_deg: float = 45.0  # Lowered: detect prone earlier/easier
    aspect_ratio_threshold: float = 0.95  # Higher threshold: very sensitive
    prone_sigmoid_k: float = 8.0  # Softer sigmoid
    
    # Impact detection (accelerometer)
    impact_sv_threshold: float = 1.5  # SV_total spike threshold (g)
    impact_sigmoid_k: float = 5.0  # Sigmoid steepness
    
    # Sustained lying
    stillness_threshold_px: float = 5.0  # Max avg movement (pixels)
    t_hold_sec: float = 3.0  # Time to confirm LYING (seconds)
    
    # Recovery
    t_recover_sec: float = 5.0  # Time to downgrade alert (seconds)


@dataclass
class StateConfig:
    """Configuration for state machine."""
    
    # State thresholds (Safety Priority: Higher Recall)
    falling_score_threshold: float = 0.40  # Aggressive recall
    lying_score_threshold: float = 0.45  # Aggressive recall
    
    # Alert thresholds with Hysteresis
    fall_alert_threshold: float = 0.50  # Lowered alert threshold
    hysteresis_exit_threshold: float = 0.35  # Harder to exit state (hysteresis)
    
    # Transition timing
    transition_window_sec: float = 2.5  # Max time between FALLING and LYING
    t_hold_sec: float = 2.0  # Stillness requirement
    occlusion_timeout_sec: float = 0.5  # Time before OCCLUDED state
    
    # Inactivity Timer (for catching slow falls)
    inactivity_alert_sec: float = 10.0  # Escalate LYING_NO_FALL after this time (increased)
    
    # Velocity thresholds for lie vs fall
    slow_liedown_velocity: float = 0.02  # Below this = slow lie-down (not fall)
    fast_fall_velocity: float = 0.04  # Above this = likely fall
    
    # Cooldown
    alert_cooldown_sec: float = 5.0  # Minimum time between alerts
    
    # State labels
    state_normal: str = "NORMAL"
    state_falling: str = "FALLING"
    state_lying: str = "LYING"
    state_occluded: str = "OCCLUDED"
    state_lying_no_fall: str = "LYING_NO_FALL"  # Resting, no alert


@dataclass
class DatasetConfig:
    """Configuration for URFall dataset."""
    
    # Dataset info
    num_fall_sequences: int = 30
    num_adl_sequences: int = 40
    fps: float = 30.0  # Approximate frame rate
    
    # Depth rescaling constants
    depth_scale_fall_cam0: float = 6000.0
    depth_scale_fall_cam1: float = 3640.0
    depth_scale_adl_cam0: float = 7000.0
    depth_max_value: int = 65535
    
    # Ground truth labels
    label_not_lying: int = -1
    label_lying: int = 1
    label_transition: int = 0  # Exclude from evaluation
    
    # Train/val/test split (by sequence count)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Multi-camera sync tolerance (ms)
    sync_tolerance_ms: float = 50.0
    
    # Column names in annotation CSV (from dataset documentation)
    # Each row = one depth frame from cam0
    annotation_columns: List[str] = field(default_factory=lambda: [
        'sequence',           # e.g., 'fall-01', 'adl-01' (camera name omitted)
        'frame_number',       # Frame number in sequence
        'label',              # -1: not lying, 1: lying on ground, 0: transitioning (exclude from classification)
        'HeightWidthRatio',   # Bounding box height to width ratio
        'MajorMinorRatio',    # Major to minor axis ratio from segmented person BLOB
        'BoundingBoxOccupancy', # Ratio of bbox occupied by person's pixels
        'MaxStdXZ',           # Std dev of pixels from centroid for X and Z axes (3D point cloud)
        'HHmaxRatio',         # Current height / standing height ratio
        'H',                  # Actual height in mm
        'D',                  # Distance of person center to floor in mm
        'P40'                 # Ratio of points in 40cm floor cuboid / total points
    ])


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    # Colors (BGR format for OpenCV)
    color_normal: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_falling: Tuple[int, int, int] = (0, 165, 255)  # Orange
    color_lying: Tuple[int, int, int] = (0, 0, 255)  # Red
    color_occluded: Tuple[int, int, int] = (128, 128, 128)  # Gray
    color_skeleton: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    
    # Drawing settings
    skeleton_thickness: int = 2
    bbox_thickness: int = 2
    text_font_scale: float = 0.7
    text_thickness: int = 2
    
    # Video output
    output_fps: float = 30.0
    codec: str = "mp4v"


@dataclass
class TrackingConfig:
    """Configuration for person tracking."""
    
    # IoU-based tracking
    iou_threshold: float = 0.3  # Min IoU for matching
    max_age: int = 30  # Max frames to keep track without detection
    min_hits: int = 3  # Min detections to confirm track


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pose_features: PoseFeatureConfig = field(default_factory=PoseFeatureConfig)
    fall_score: FallScoreConfig = field(default_factory=FallScoreConfig)
    state: StateConfig = field(default_factory=StateConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        with open(path, 'w') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert paths back to Path objects
        if 'paths' in data:
            for key in ['data_root', 'output_root']:
                if key in data['paths']:
                    data['paths'][key] = Path(data['paths'][key])
        
        # Convert tuples
        if 'visualization' in data:
            for key in ['color_normal', 'color_falling', 'color_lying', 
                        'color_occluded', 'color_skeleton']:
                if key in data['visualization']:
                    data['visualization'][key] = tuple(data['visualization'][key])
        
        return cls(
            paths=PathConfig(**data.get('paths', {})),
            model=ModelConfig(**data.get('model', {})),
            pose_features=PoseFeatureConfig(**data.get('pose_features', {})),
            fall_score=FallScoreConfig(**data.get('fall_score', {})),
            state=StateConfig(**data.get('state', {})),
            dataset=DatasetConfig(**data.get('dataset', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            tracking=TrackingConfig(**data.get('tracking', {})),
        )


# Global default configuration instance
DEFAULT_CONFIG = Config()


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get configuration, optionally loading from a file.
    
    Args:
        config_path: Optional path to config YAML file
        
    Returns:
        Config object
    """
    if config_path is not None and config_path.exists():
        return Config.load(config_path)
    return DEFAULT_CONFIG
