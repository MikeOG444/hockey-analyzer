#!/usr/bin/env python3
"""
Quick test to verify the new package structure works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all main modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test package import
        import hockey_analyzer
        print("✅ Main package imported")
        
        # Test detection modules
        from hockey_analyzer.detection.player_detector import PlayerDetector
        from hockey_analyzer.detection.ice_detector import IceDetector
        from hockey_analyzer.detection.puck_tracker import PuckTracker
        print("✅ Detection modules imported")
        
        # Test analysis modules
        from hockey_analyzer.analysis.team_identifier import TeamIdentifier
        from hockey_analyzer.analysis.play_analyzer import PlayAnalyzer
        print("✅ Analysis modules imported")
        
        # Test configuration
        from hockey_analyzer.config.settings import config
        print("✅ Configuration imported")
        
        print("\\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("\\nTesting basic functionality...")
        
        # Test configuration access
        from hockey_analyzer.config.settings import config
        print(f"✅ Default model: {config.model.model_name}")
        print(f"✅ Confidence threshold: {config.model.confidence_threshold}")
        
        # Test detector initialization (without loading models)
        from hockey_analyzer.detection.player_detector import PlayerDetector
        from hockey_analyzer.detection.ice_detector import IceDetector
        
        # We won't actually load YOLO models in this test
        print("✅ Detector classes available")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_cli_available():
    """Test that CLI script exists"""
    cli_path = Path(__file__).parent / "cli.py"
    if cli_path.exists():
        print("✅ CLI script available")
        return True
    else:
        print("❌ CLI script not found")
        return False

def main():
    """Run all tests"""
    print("🏒 Hockey Analyzer - Package Structure Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_cli_available
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Package structure is working correctly.")
        print("\\nNext steps:")
        print("1. Install package: pip install -e .")
        print("2. Run CLI: python cli.py --help")
        print("3. Test analysis: python cli.py analyze data/test_videos/your_video.mp4")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
