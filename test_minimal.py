#!/usr/bin/env python3
"""
Minimal test to verify package structure without heavy dependencies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_structure():
    """Test basic package structure"""
    try:
        print("Testing basic package structure...")
        
        # Test that directories exist
        package_dir = Path(__file__).parent / "hockey_analyzer"
        assert package_dir.exists(), "hockey_analyzer package directory missing"
        
        # Test key subdirectories
        subdirs = ["detection", "analysis", "calibration", "ui", "config", "utils"]
        for subdir in subdirs:
            subdir_path = package_dir / subdir
            assert subdir_path.exists(), f"{subdir} subdirectory missing"
            assert (subdir_path / "__init__.py").exists(), f"{subdir}/__init__.py missing"
        
        print("✅ Package structure looks good")
        return True
        
    except AssertionError as e:
        print(f"❌ Structure error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_import():
    """Test configuration can be imported without heavy dependencies"""
    try:
        print("Testing configuration import...")
        
        from hockey_analyzer.config.settings import config
        
        # Test basic config access
        assert hasattr(config, 'model')
        assert hasattr(config, 'video')
        assert hasattr(config, 'debug')
        
        print(f"✅ Config loaded - Model: {config.model.model_name}")
        print(f"✅ Confidence threshold: {config.model.confidence_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config import error: {e}")
        return False

def test_file_organization():
    """Test that files were moved correctly"""
    try:
        print("Testing file organization...")
        
        # Check that old src/ structure is preserved
        old_src = Path(__file__).parent / "src"
        if old_src.exists():
            print("✅ Legacy src/ directory preserved")
        
        # Check new structure has key files
        new_structure = {
            "hockey_analyzer/detection/player_detector.py": "Player detector",
            "hockey_analyzer/detection/ice_detector.py": "Ice detector", 
            "hockey_analyzer/analysis/game_analyzer.py": "Game analyzer",
            "hockey_analyzer/config/settings.py": "Configuration",
            "tests/detection/test_player_detector.py": "Player detector tests",
            "cli.py": "CLI script",
            "setup.py": "Package setup"
        }
        
        for file_path, description in new_structure.items():
            full_path = Path(__file__).parent / file_path
            if full_path.exists():
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} missing: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ File organization error: {e}")
        return False

def show_summary():
    """Show summary of the new structure"""
    print("\\n📁 NEW PACKAGE STRUCTURE:")
    print("hockey_analyzer/")
    print("├── detection/     # Player, ice, puck detection")
    print("├── analysis/      # Game analysis and team ID")
    print("├── calibration/   # Rink calibration")
    print("├── ui/           # Dashboard interface")
    print("├── config/       # Configuration management")
    print("└── utils/        # Utility functions")
    print()
    print("tests/")
    print("├── detection/    # Detection module tests")
    print("├── analysis/     # Analysis module tests")
    print("├── integration/  # End-to-end tests")
    print("└── utils/        # Utility tests")
    print()
    print("📋 READY TO USE:")
    print("• python cli.py --help")
    print("• pip install -e .")
    print("• python -m hockey_analyzer --help")

def main():
    """Run basic tests"""
    print("🏒 Hockey Analyzer - Structure Verification")
    print("=" * 50)
    
    tests = [
        test_basic_structure,
        test_config_import,
        test_file_organization
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 Package restructuring completed successfully!")
        show_summary()
    else:
        print("❌ Some issues found with the restructuring.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
