"""
Hockey Analyzer - Computer Vision Hockey Game Analysis System

A comprehensive toolkit for analyzing hockey games using YOLO object detection,
ice surface detection, player tracking, and game analytics.
"""

__version__ = "0.1.0"
__author__ = "Hockey Analyzer Team"

# Import main classes for easy access
from .detection.player_detector import PlayerDetector
from .detection.ice_detector import IceDetector  
from .detection.puck_tracker import PuckTracker
from .analysis.game_analyzer import GameAnalyzer
from .calibration.rink_calibrator import RinkCalibrator

__all__ = [
    "PlayerDetector",
    "IceDetector", 
    "PuckTracker",
    "GameAnalyzer",
    "RinkCalibrator"
]
