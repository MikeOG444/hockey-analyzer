# Hockey Analyzer 🏒

A comprehensive hockey video analysis system that uses computer vision and YOLO object detection to analyze hockey gameplay from LiveBarn footage.

## 🎯 Features

- **Player Detection**: YOLO-based detection and tracking of hockey players with confidence scoring
- **Ice Surface Identification**: Multi-method ice surface detection using color, texture, geometry, and markings
- **Rink Calibration**: Convert between pixel coordinates and real-world measurements
- **Game Analytics**: Analyze player positions, movements, and generate game statistics
- **Zone Detection**: Identify offensive, defensive, and neutral zones using blue line detection
- **Debug Visualization**: Comprehensive debug tools with automatic image generation
- **Coaching Dashboard**: Streamlit-based web interface for viewing analysis results

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Sufficient disk space for YOLO models (~170MB) and test videos

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MikeOG444/hockey-analyzer.git
   cd hockey-analyzer
   ```

2. **Install the package**
   ```bash
   # Install in development mode
   pip install -e .
   
   # Or install from requirements.txt
   pip install -r requirements.txt
   ```

3. **Download YOLO models**
   ```bash
   # Models will auto-download on first use, or download manually:
   # Place in the models/ directory:
   # - yolov8n.pt (6MB, fastest)
   # - yolov8s.pt (22MB, balanced) 
   # - yolov8m.pt (52MB, accurate)
   # - yolov8l.pt (88MB, most accurate)
   ```

### First Run

```bash
# Analyze a hockey game video
python cli.py analyze data/test_videos/your_game.mp4

# Or use the module directly
python -m hockey_analyzer analyze data/test_videos/your_game.mp4

# Launch the coaching dashboard
python cli.py dashboard

# Calibrate rink measurements
python cli.py calibrate data/test_videos/your_game.mp4
```

## 📁 Project Structure

```
hockey-analyzer/
├── hockey_analyzer/                 # Main package
│   ├── __init__.py                 # Package initialization
│   ├── __main__.py                 # CLI entry point
│   ├── config/                     # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py             # Global settings and configuration
│   ├── detection/                  # Object detection modules
│   │   ├── __init__.py
│   │   ├── player_detector.py      # YOLO player detection
│   │   ├── ice_detector.py         # Ice surface detection
│   │   └── puck_tracker.py         # Puck tracking
│   ├── analysis/                   # Game analysis modules
│   │   ├── __init__.py
│   │   ├── team_identifier.py      # Team identification
│   │   ├── play_analyzer.py        # Play analysis
│   │   └── game_analyzer.py        # Main game analysis
│   ├── calibration/                # Calibration modules
│   │   ├── __init__.py
│   │   └── rink_calibrator.py      # Rink calibration
│   ├── ui/                         # User interface
│   │   ├── __init__.py
│   │   └── dashboard.py            # Streamlit dashboard
│   └── utils/                      # Utility modules
│       └── __init__.py
├── tests/                          # Organized test suite
│   ├── detection/                  # Detection module tests
│   ├── analysis/                   # Analysis module tests
│   ├── integration/                # End-to-end tests
│   ├── utils/                      # Utility tests
│   └── fixtures/                   # Test data and fixtures
├── scripts/                        # Debug and utility scripts
├── data/                           # Input data
│   └── test_videos/               # Test video files
├── models/                         # YOLO model files
├── output/                         # Analysis results
├── docs/                           # Documentation
├── cli.py                          # Command-line interface
├── setup.py                       # Package setup
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎮 Usage

### Command Line Interface

The main interface is through the CLI:

```bash
# Basic analysis
python cli.py analyze path/to/video.mp4

# Advanced analysis with options
python cli.py analyze path/to/video.mp4 \
    --model yolov8l.pt \
    --confidence 0.5 \
    --output my_analysis \
    --debug

# Calibration
python cli.py calibrate path/to/video.mp4 --interactive

# Dashboard
python cli.py dashboard --port 8501
```

### Python API

```python
from hockey_analyzer import GameAnalyzer, PlayerDetector, IceDetector

# Initialize components
detector = PlayerDetector(model_name="yolov8n.pt")
ice_detector = IceDetector()
analyzer = GameAnalyzer()

# Analyze a video
result = analyzer.analyze_game("path/to/video.mp4")
print(f"Detected {len(result.play_events)} play events")
```

## 🏒 YOLO Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6MB | Fastest | Good | Real-time analysis |
| `yolov8s.pt` | 22MB | Fast | Better | Balanced performance |
| `yolov8m.pt` | 52MB | Medium | High | Quality analysis |
| `yolov8l.pt` | 88MB | Slow | Highest | Maximum accuracy |

Models are automatically downloaded to `models/` directory on first use.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/detection/      # Detection tests
python -m pytest tests/analysis/       # Analysis tests
python -m pytest tests/integration/    # Integration tests

# Run legacy tests (during migration)
python tests/detection/test_video_analysis.py
python tests/integration/test_full_system.py
```

## 🔧 Configuration

Configuration is managed through `hockey_analyzer/config/settings.py`:

```python
from hockey_analyzer.config.settings import config

# Update model settings
config.model.model_name = "yolov8l.pt"
config.model.confidence_threshold = 0.5

# Enable debug output
config.debug.save_debug_images = True

# Update analysis settings
config.analysis.min_play_duration = 3.0
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
# Make sure package is installed
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**"YOLO model not found"**
```bash
# Check models directory
ls models/
# Download manually if needed from ultralytics
```

**"No players detected"**
```bash
# Try different YOLO models
python cli.py analyze video.mp4 --model yolov8l.pt

# Lower confidence threshold
python cli.py analyze video.mp4 --confidence 0.2

# Enable debug output to see what's happening
python cli.py analyze video.mp4 --debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is for educational and research purposes. YOLO models are subject to their respective licenses.

## 🆘 Support

For issues and questions:
1. Check the documentation in `docs/` directory
2. Run with `--debug` flag to see detailed processing information
3. Check existing GitHub issues
4. Create a new issue with detailed information

---

**Happy analyzing! 🏒📊**
