"""
Pytest configuration and shared fixtures for Hockey Analyzer tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hockey_analyzer.config.settings import config

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session") 
def sample_video_path(test_data_dir):
    """Fixture providing path to sample test video"""
    video_path = test_data_dir / "sample_game.mp4"
    # In a real implementation, you'd have a small test video here
    # For now, we'll create a placeholder
    video_path.parent.mkdir(exist_ok=True)
    if not video_path.exists():
        video_path.touch()
    return str(video_path)

@pytest.fixture
def temp_output_dir():
    """Fixture providing temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    # Create a copy of the config for testing
    test_conf = config
    test_conf.debug.save_debug_images = False  # Disable debug output in tests
    test_conf.video.max_frames = 10  # Limit frames for faster tests
    return test_conf

@pytest.fixture
def sample_frame():
    """Fixture providing a sample video frame for testing"""
    import numpy as np
    # Create a simple test frame (blue background, white rectangle for ice)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [100, 50, 0]  # Brown/ice color background
    frame[100:380, 50:590] = [200, 200, 200]  # White ice surface
    return frame

@pytest.fixture
def sample_detections():
    """Fixture providing sample YOLO detection results"""
    return [
        {
            'bbox': [100, 150, 50, 100],  # x, y, w, h
            'confidence': 0.85,
            'class': 'person',
            'class_id': 0
        },
        {
            'bbox': [200, 180, 45, 95],
            'confidence': 0.72,
            'class': 'person', 
            'class_id': 0
        },
        {
            'bbox': [350, 200, 40, 90],
            'confidence': 0.91,
            'class': 'person',
            'class_id': 0
        }
    ]
