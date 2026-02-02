# Configuration management with Pydantic Settings
from typing import List
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application settings
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # App
    app_name: str = "Fall Detection API"
    app_env: str = "dev"  # dev, prod
    version: str = "1.0.0"
    
    # CORS
    allowed_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # WebSocket
    ws_max_fps: int = 15
    max_frame_size: int = 1024 * 1024  # 1MB
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    
    # Artifacts
    artifact_dir: Path = Path("./artifacts")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    # Core AI
    # Use realtime.yaml for WebSocket streaming (15 FPS) - thresholds scaled for lower FPS
    # Use default.yaml for video file processing (25 FPS)
    core_config_path: Path = Path("../../urfd_fall_yolo_pose/configs/realtime.yaml")
    core_config_path_offline: Path = Path("../../urfd_fall_yolo_pose/configs/default.yaml")
    yolo_model_path: Path = Path("../../urfd_fall_yolo_pose/yolov8s-pose.pt")
    core_scripts_dir: Path = Path("../../urfd_fall_yolo_pose/scripts")
    
    # Logging
    log_level: str = "INFO"
    
    # Rate limiting
    frame_rate_limit: int = 30  # max fps per camera
    burst_allowance: int = 5

@lru_cache()
def get_settings() -> Settings:
    # Cached settings instance
    return Settings()
