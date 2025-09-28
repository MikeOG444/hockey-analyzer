"""
Unit tests for PlayerDetector module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from hockey_analyzer.detection.player_detector import PlayerDetector

class TestPlayerDetector:
    """Test cases for PlayerDetector class"""
    
    def test_init_default_model(self):
        """Test PlayerDetector initialization with default model"""
        detector = PlayerDetector()
        assert detector.model_name == "yolov8n.pt"
        assert detector.confidence_threshold == 0.3
    
    def test_init_custom_model(self):
        """Test PlayerDetector initialization with custom model"""
        detector = PlayerDetector(
            model_name="yolov8l.pt",
            confidence_threshold=0.5
        )
        assert detector.model_name == "yolov8l.pt"
        assert detector.confidence_threshold == 0.5
    
    @patch('hockey_analyzer.detection.player_detector.YOLO')
    def test_detect_players_basic(self, mock_yolo, sample_frame):
        """Test basic player detection functionality"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock detection results
        mock_result = Mock()
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([
            [100, 150, 150, 250],  # x1, y1, x2, y2
            [200, 180, 245, 275]
        ])
        mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.85, 0.72])
        mock_result.boxes.cls.cpu().numpy.return_value = np.array([0, 0])  # person class
        mock_model.return_value = [mock_result]
        
        detector = PlayerDetector()
        detections = detector.detect_players(sample_frame)
        
        assert len(detections) == 2
        assert detections[0]['confidence'] == 0.85
        assert detections[1]['confidence'] == 0.72
    
    @patch('hockey_analyzer.detection.player_detector.YOLO')
    def test_detect_players_empty_frame(self, mock_yolo):
        """Test player detection on empty frame"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock empty detection results
        mock_result = Mock()
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([])
        mock_result.boxes.conf.cpu().numpy.return_value = np.array([])
        mock_result.boxes.cls.cpu().numpy.return_value = np.array([])
        mock_model.return_value = [mock_result]
        
        detector = PlayerDetector()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_players(empty_frame)
        
        assert len(detections) == 0
    
    @patch('hockey_analyzer.detection.player_detector.YOLO')
    def test_confidence_filtering(self, mock_yolo, sample_frame):
        """Test that low confidence detections are filtered out"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock detection results with mixed confidence
        mock_result = Mock()
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([
            [100, 150, 150, 250],  # High confidence
            [200, 180, 245, 275],  # Low confidence
            [300, 200, 340, 290]   # High confidence
        ])
        mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.85, 0.15, 0.72])
        mock_result.boxes.cls.cpu().numpy.return_value = np.array([0, 0, 0])
        mock_model.return_value = [mock_result]
        
        detector = PlayerDetector(confidence_threshold=0.3)
        detections = detector.detect_players(sample_frame)
        
        # Should only return 2 detections (confidence > 0.3)
        assert len(detections) == 2
        assert all(d['confidence'] >= 0.3 for d in detections)
    
    def test_invalid_model_name(self):
        """Test handling of invalid model name"""
        with pytest.raises((FileNotFoundError, Exception)):
            detector = PlayerDetector(model_name="nonexistent_model.pt")
            # This might not raise immediately until model is loaded
