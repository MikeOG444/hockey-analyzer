"""
Unit tests for IceDetector module
"""

import pytest
import numpy as np
import cv2

from hockey_analyzer.detection.ice_detector import IceDetector

class TestIceDetector:
    """Test cases for IceDetector class"""
    
    def test_init(self):
        """Test IceDetector initialization"""
        detector = IceDetector()
        assert detector is not None
    
    def test_detect_ice_surface_basic(self, sample_frame):
        """Test basic ice surface detection"""
        detector = IceDetector()
        ice_mask = detector.detect_ice_surface(sample_frame)
        
        assert ice_mask is not None
        assert ice_mask.shape == sample_frame.shape[:2]  # Should be 2D mask
        assert ice_mask.dtype == np.uint8
    
    def test_detect_ice_surface_dimensions(self):
        """Test ice detection with different frame dimensions"""
        detector = IceDetector()
        
        # Test with different sizes
        for height, width in [(240, 320), (720, 1280), (1080, 1920)]:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [100, 100, 100]  # Gray background
            
            ice_mask = detector.detect_ice_surface(frame)
            assert ice_mask.shape == (height, width)
    
    def test_detect_ice_with_white_surface(self):
        """Test ice detection on frame with prominent white surface"""
        # Create frame with white ice-like surface
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:380, 50:590] = [200, 200, 200]  # White ice surface
        frame[:100, :] = [50, 100, 150]  # Darker stands
        frame[380:, :] = [50, 100, 150]  # Darker stands
        
        detector = IceDetector()
        ice_mask = detector.detect_ice_surface(frame)
        
        # Should detect the white central area as ice
        ice_pixels = np.sum(ice_mask > 0)
        total_pixels = ice_mask.shape[0] * ice_mask.shape[1]
        ice_ratio = ice_pixels / total_pixels
        
        # Should detect a reasonable amount of ice (at least 10%)
        assert ice_ratio > 0.1
    
    def test_detect_ice_no_ice_surface(self):
        """Test ice detection on frame with no clear ice surface"""
        # Create frame with varied colors, no clear ice
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        detector = IceDetector()
        ice_mask = detector.detect_ice_surface(frame)
        
        # Should still return a valid mask, even if minimal ice detected
        assert ice_mask is not None
        assert ice_mask.shape == (480, 640)
    
    def test_ice_mask_values(self, sample_frame):
        """Test that ice mask contains only valid values"""
        detector = IceDetector()
        ice_mask = detector.detect_ice_surface(sample_frame)
        
        # Mask should be binary (0 or 255) or grayscale (0-255)
        assert np.all((ice_mask >= 0) & (ice_mask <= 255))
        
        # Should have at least some variation (not all zeros or all 255s)
        unique_values = np.unique(ice_mask)
        assert len(unique_values) > 1
    
    def test_multiple_detection_methods(self, sample_frame):
        """Test that multiple detection methods work"""
        detector = IceDetector()
        
        # The detector should have multiple methods available
        # (this tests the internal structure of the class)
        assert hasattr(detector, 'detect_ice_surface')
        
        # Run detection multiple times to ensure consistency
        mask1 = detector.detect_ice_surface(sample_frame)
        mask2 = detector.detect_ice_surface(sample_frame)
        
        # Results should be identical for same input
        assert np.array_equal(mask1, mask2)
    
    def test_edge_case_small_frame(self):
        """Test ice detection on very small frame"""
        # Very small frame
        small_frame = np.ones((10, 10, 3), dtype=np.uint8) * 128
        
        detector = IceDetector()
        ice_mask = detector.detect_ice_surface(small_frame)
        
        assert ice_mask.shape == (10, 10)
        assert ice_mask.dtype == np.uint8
    
    def test_edge_case_single_color_frame(self):
        """Test ice detection on single color frame"""
        # Solid white frame (should be detected as ice)
        white_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        detector = IceDetector()
        ice_mask = detector.detect_ice_surface(white_frame)
        
        # Should detect most of the frame as ice
        ice_ratio = np.sum(ice_mask > 0) / (ice_mask.shape[0] * ice_mask.shape[1])
        assert ice_ratio > 0.5  # At least 50% should be detected as ice
