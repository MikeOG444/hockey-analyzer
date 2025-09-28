"""
Hockey Analyzer - Computer Vision Hockey Game Analysis System

A comprehensive toolkit for analyzing hockey games using YOLO object detection,
ice surface detection, player tracking, and game analytics.
"""

__version__ = "0.1.0"
__author__ = "Hockey Analyzer Team"

# Main classes are available for import but not automatically loaded
# This avoids forcing heavy dependencies at package import time

__all__ = [
    "PlayerDetector",
    "IceDetector", 
    "PuckTracker",
    "GameAnalyzer",
    "RinkCalibrator"
]

# Lazy imports - classes are imported only when requested
def __getattr__(name):
    if name == "PlayerDetector":
        from .detection.player_detector import PlayerDetector
        return PlayerDetector
    elif name == "IceDetector":
        from .detection.ice_detector import IceDetector
        return IceDetector
    elif name == "PuckTracker":
        from .detection.puck_tracker import PuckTracker
        return PuckTracker
    elif name == "GameAnalyzer":
        from .analysis.game_analyzer import HockeyGameAnalyzer
        return HockeyGameAnalyzer
    elif name == "RinkCalibrator":
        from .calibration.rink_calibrator import HockeyRinkCalibrator
        return HockeyRinkCalibrator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
