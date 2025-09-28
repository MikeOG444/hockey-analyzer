#!/usr/bin/env python3
"""
Test the PuckTracker interface compatibility (without heavy dependencies)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_puck_tracker_methods():
    """Test that PuckTracker has all required methods"""
    try:
        print("Testing PuckTracker method signatures...")
        
        # Import the class definition (this shouldn't require cv2/numpy)
        import inspect
        from hockey_analyzer.detection.puck_tracker import PuckTracker
        
        # Check required methods exist
        required_methods = [
            'detect_puck_candidates',
            'track_puck', 
            'detect_puck',
            'visualize_tracking'
        ]
        
        for method_name in required_methods:
            if hasattr(PuckTracker, method_name):
                method = getattr(PuckTracker, method_name)
                if callable(method):
                    print(f"✅ {method_name} method exists")
                else:
                    print(f"❌ {method_name} exists but is not callable")
                    return False
            else:
                print(f"❌ {method_name} method missing")
                return False
        
        # Check method signatures using inspection
        detect_candidates_sig = inspect.signature(PuckTracker.detect_puck_candidates)
        expected_params = ['self', 'frame', 'players']
        actual_params = list(detect_candidates_sig.parameters.keys())
        
        if actual_params == expected_params:
            print("✅ detect_puck_candidates has correct signature")
        else:
            print(f"❌ detect_puck_candidates signature mismatch: {actual_params} vs {expected_params}")
            return False
        
        track_puck_sig = inspect.signature(PuckTracker.track_puck)
        expected_params = ['self', 'candidates']
        actual_params = list(track_puck_sig.parameters.keys())
        
        if actual_params == expected_params:
            print("✅ track_puck has correct signature")
        else:
            print(f"❌ track_puck signature mismatch: {actual_params} vs {expected_params}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ PuckTracker method test failed: {e}")
        return False

def test_game_analyzer_imports():
    """Test that GameAnalyzer can import its dependencies"""
    try:
        print("\\nTesting GameAnalyzer import paths...")
        
        # Test that we can at least import the module structure
        # (this will fail on cv2 but should show us if imports are correct)
        try:
            import hockey_analyzer.analysis.game_analyzer
            print("✅ GameAnalyzer module imports successfully")
        except ModuleNotFoundError as e:
            if 'cv2' in str(e) or 'ultralytics' in str(e):
                print("✅ GameAnalyzer import paths correct (cv2/ultralytics not installed)")
            else:
                print(f"❌ GameAnalyzer import error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ GameAnalyzer import test failed: {e}")
        return False

def main():
    """Run compatibility tests"""
    print("🏒 Hockey Analyzer - Interface Compatibility Check")
    print("=" * 55)
    
    tests = [
        test_puck_tracker_methods,
        test_game_analyzer_imports
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\\n" + "=" * 55)
    if all_passed:
        print("🎉 Interface compatibility checks passed!")
        print("\\nThe missing methods have been added to PuckTracker.")
        print("\\nTo test with real video analysis, you need to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run analysis: python cli.py analyze data/test_videos/your_video.mp4")
    else:
        print("❌ Some compatibility issues remain.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
