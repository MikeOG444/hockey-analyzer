"""
Configuration settings for Hockey Analyzer
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
DEBUG_DIR = PROJECT_ROOT / "debug_output"

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, DEBUG_DIR]:
    dir_path.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """YOLO model configuration"""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    max_detections: int = 300
    device: str = "cpu"  # or "cuda" if available

@dataclass
class VideoConfig:
    """Video processing configuration"""
    input_fps: Optional[int] = None  # Use video's native FPS
    output_fps: int = 30
    frame_skip: int = 1  # Process every N frames
    max_frames: Optional[int] = None  # Limit processing
    resize_factor: float = 1.0  # Resize frames for processing

@dataclass 
class IceDetectionConfig:
    """Ice surface detection configuration"""
    color_threshold: float = 0.7
    texture_threshold: float = 0.6
    edge_threshold: float = 0.5
    min_ice_area: int = 50000  # Minimum ice area in pixels
    debug_output: bool = False

@dataclass
class CalibrationConfig:
    """Rink calibration configuration"""
    blue_line_width_inches: float = 12.0
    glass_height_inches: float = 48.0
    column_width_inches: float = 22.0
    rink_length_feet: float = 200.0
    rink_width_feet: float = 85.0

@dataclass
class AnalysisConfig:
    """Game analysis configuration"""
    min_play_duration: float = 2.0  # seconds
    zone_entry_threshold: float = 10.0  # pixels
    shot_detection_enabled: bool = True
    team_identification_enabled: bool = True

@dataclass
class DebugConfig:
    """Debug and visualization configuration"""
    save_debug_images: bool = True
    save_debug_videos: bool = False
    debug_every_n_frames: int = 30
    max_debug_files: int = 100

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.video = VideoConfig()
        self.ice_detection = IceDetectionConfig()
        self.calibration = CalibrationConfig()
        self.analysis = AnalysisConfig()
        self.debug = DebugConfig()
        
        # Paths
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.models_dir = MODELS_DIR
        self.debug_dir = DEBUG_DIR
        
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """Get full path to model file"""
        model_name = model_name or self.model.model_name
        return self.models_dir / model_name
    
    def get_output_path(self, filename: str) -> Path:
        """Get full path for output file"""
        return self.output_dir / filename
    
    def get_debug_path(self, filename: str) -> Path:
        """Get full path for debug file"""
        return self.debug_dir / filename

# Global configuration instance
config = Config()
