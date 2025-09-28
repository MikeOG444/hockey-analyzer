#!/usr/bin/env python3
"""
Hockey Analyzer CLI - Command Line Interface for Hockey Game Analysis

Usage:
    python -m hockey_analyzer analyze video.mp4
    python -m hockey_analyzer calibrate video.mp4
    python -m hockey_analyzer dashboard
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports to avoid loading heavy dependencies at startup
# Imports are done in functions when needed

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hockey Analyzer - Computer Vision Hockey Game Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze data/game.mp4
  %(prog)s analyze data/game.mp4 --model yolov8l.pt --output analysis_results
  %(prog)s calibrate data/game.mp4 --interactive
  %(prog)s dashboard --port 8501
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze hockey game video')
    analyze_parser.add_argument('video', help='Path to video file')
    analyze_parser.add_argument('--model', default='yolov8n.pt', 
                               help='YOLO model to use (default: yolov8n.pt)')
    analyze_parser.add_argument('--output', default='output',
                               help='Output directory (default: output)')
    analyze_parser.add_argument('--confidence', type=float, default=0.3,
                               help='Detection confidence threshold (default: 0.3)')
    analyze_parser.add_argument('--debug', action='store_true',
                               help='Enable debug output')
    
    # Calibration command  
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate rink measurements')
    calibrate_parser.add_argument('video', help='Path to video file')
    calibrate_parser.add_argument('--interactive', action='store_true',
                                 help='Interactive calibration mode')
    calibrate_parser.add_argument('--output', default='calibration.json',
                                 help='Calibration output file')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch coaching dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501,
                                 help='Dashboard port (default: 8501)')
    dashboard_parser.add_argument('--host', default='localhost',
                                 help='Dashboard host (default: localhost)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'analyze':
            return analyze_video(args)
        elif args.command == 'calibrate':
            return calibrate_rink(args)
        elif args.command == 'dashboard':
            return launch_dashboard(args)
        elif args.command == 'version':
            return show_version()
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

def analyze_video(args):
    """Analyze hockey game video"""
    # Import heavy dependencies only when needed
    from .analysis.game_analyzer import HockeyGameAnalyzer
    from .config.settings import config
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    print(f"Analyzing video: {video_path}")
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output}")
    
    # Update configuration
    config.model.model_name = args.model
    config.model.confidence_threshold = args.confidence
    config.debug.save_debug_images = args.debug
    
    # Run analysis
    analyzer = HockeyGameAnalyzer(output_dir=args.output)
    result = analyzer.analyze_game(str(video_path))
    
    print(f"Analysis complete! Results saved to: {args.output}")
    return 0

def calibrate_rink(args):
    """Calibrate rink measurements"""
    # Import only when needed
    from .calibration.rink_calibrator import HockeyRinkCalibrator
    import json
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    print(f"Calibrating rink from video: {video_path}")
    
    calibrator = HockeyRinkCalibrator()
    
    if args.interactive:
        print("Starting interactive calibration...")
        # Interactive calibration would be implemented here
        print("Interactive calibration not yet implemented")
        return 1
    else:
        # Automatic calibration
        calibration_data = calibrator.auto_calibrate(str(video_path))
        output_path = Path(args.output)
        
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to: {output_path}")
    
    return 0

def launch_dashboard(args):
    """Launch coaching dashboard"""
    import subprocess
    import sys
    
    print(f"Launching dashboard on {args.host}:{args.port}")
    
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    subprocess.run(cmd)
    return 0

def show_version():
    """Show version information"""
    from hockey_analyzer import __version__
    print(f"Hockey Analyzer v{__version__}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
