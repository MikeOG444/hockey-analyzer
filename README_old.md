# Hockey Analyzer 🏒

A comprehensive hockey video analysis system that uses computer vision and YOLO object detection to analyze hockey gameplay from LiveBarn footage.

## 🎯 Features

- **Player Detection**: YOLO-based detection and tracking of hockey players with confidence scoring
- **Ice Surface Identification**: Multi-method ice surface detection using color, texture, geometry, and markings
- **Rink Calibration**: Convert between pixel coordinates and real-world measurements
- **Game Analytics**: Analyze player positions, movements, and generate game statistics
- **Zone Detection**: Identify offensive, defensive, and neutral zones using blue line detection
- **Debug Visualization**: Comprehensive debug tools with automatic image generation

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Git
- Sufficient disk space for YOLO models (~170MB) and test videos

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MikeOG444/hockey-analyzer.git
   cd hockey-analyzer
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO models**
   ```bash
   # Download the required YOLO models (they will auto-download on first use)
   # Or manually download from: https://github.com/ultralytics/assets/releases/
   # Place the following files in the project root:
   # - yolov8n.pt (6MB, fastest)
   # - yolov8s.pt (22MB, balanced) 
   # - yolov8m.pt (52MB, accurate)
   # - yolov8l.pt (88MB, most accurate)
   ```

5. **Verify installation**
   ```bash
   python test_video_analysis.py
   ```

### First Run

The system will automatically download YOLO models on first use. Test with a basic detection:

```bash
# Test basic player detection (requires test video in data/test_videos/)
python test_video_analysis.py

# Test ice surface detection
python test_ice_markings.py

# Run full system test
python test_full_system.py
```

## 📁 Project Structure

```
hockey-analyzer/
├── src/                          # Source code modules
│   ├── detection/               # Object detection algorithms
│   │   ├── basic_detector.py    # Core YOLO player detection
│   │   ├── ice_detection.py     # Ice surface detection
│   │   ├── puck_tracker.py      # Puck tracking
│   │   └── realistic_puck_tracker.py
│   ├── analysis/                # Game analysis modules
│   │   ├── team_identifier.py   # Team identification
│   │   └── play_analyzer.py     # Play analysis
│   └── ui/                      # UI components (future)
├── data/                        # Test data and videos
│   └── test_videos/            # Place your MP4 files here
├── test_*.py                    # Individual feature tests
├── debug_*.py                   # Debug utilities
├── game_analyzer.py            # Main application entry point
├── coaching_dashboard.py       # Streamlit dashboard
├── calibration.py              # Rink calibration utility
├── requirements.txt            # Python dependencies
└── yolov8*.pt                  # YOLO model files (auto-downloaded)
```

## 🧪 Testing & Development

### Core Tests
```bash
# Basic functionality
python test_video_analysis.py              # Video processing pipeline
python test_model_comparison.py            # Compare YOLO models
python test_ice_markings.py               # Ice detection algorithms

# Advanced features
python test_puck_tracking.py              # Puck tracking
python test_team_identification.py        # Player/team identification
python test_realistic_analysis.py         # End-to-end analysis
```

### Ice Detection Tests
```bash
python test_brightness_boundary_detection.py    # Boundary detection
python test_adjacent_ice_expansion.py           # Ice mask refinement
python test_edge_based_ice_mask.py              # Edge-based detection
```

### Debug Tools
```bash
python debug_detection.py                 # Debug object detection issues
python debug_medium_model.py             # Test different YOLO models
python debug_referee_detection_fixed.py  # Referee detection debugging
```

### Debug Output

Tests automatically generate debug images in `debug_*_images/` directories:
- `debug_images/` - General debug outputs
- `debug_test_*_images/` - Test-specific visualizations
- `full_system_test_output/` - Complete system analysis results

## 🎮 Main Applications

### Game Analyzer
```bash
python game_analyzer.py
```
Core analysis engine for processing hockey videos and generating statistics.

### Coaching Dashboard
```bash
python coaching_dashboard.py
```
Streamlit-based web interface for viewing analysis results (future feature).

### Calibration Tool
```bash
python calibration.py
```
Manual calibration utility for setting up pixel-to-measurement conversion.

## 🏒 YOLO Models

The system supports multiple YOLOv8 models with different performance characteristics:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6MB | Fastest | Good | Real-time analysis |
| `yolov8s.pt` | 22MB | Fast | Better | Balanced performance |
| `yolov8m.pt` | 52MB | Medium | High | Quality analysis |
| `yolov8l.pt` | 88MB | Slow | Highest | Maximum accuracy |

Models are automatically downloaded on first use.

## 🔧 Configuration

### Video Requirements
- **Format**: MP4 (other formats may work)
- **Content**: Hockey games with clear ice surface view
- **Quality**: Higher resolution recommended for better detection
- **Placement**: Store test videos in `data/test_videos/`

### Ice Detection Settings
The system uses multiple detection methods:
- **Color-based**: HSV and LAB color space analysis
- **Texture-based**: Smooth surface detection
- **Geometry-based**: Large flat area identification
- **Marking-based**: Red center line and blue line detection

### Calibration
Manual calibration requires reference frames with known measurements:
- Blue line width: 12 inches
- Glass height: 48 inches  
- Column width: 22 inches

## 🐛 Troubleshooting

### Common Issues

**"YOLO model not found"**
```bash
# Models auto-download, but you can manually download:
# They'll be placed in the project root automatically
```

**"Video not found"**
```bash
# Ensure video is in correct location:
ls -la data/test_videos/
# Videos should be .mp4 format
```

**"No players detected"**
```bash
# Try different YOLO models:
python test_model_comparison.py
# Check video quality and player visibility
```

**Debug images not generating**
```bash
# Ensure write permissions in project directory
# Debug directories are auto-created
ls -la debug_*_images/
```

### Performance Issues

- **Slow processing**: Use smaller YOLO model (`yolov8n.pt`)
- **Low accuracy**: Use larger YOLO model (`yolov8l.pt`)
- **Memory issues**: Process shorter video segments
- **GPU not used**: Install CUDA-compatible PyTorch version

## 🤝 Contributing

### Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**
   ```bash
   # Run relevant tests
   python test_*.py
   
   # Check debug outputs
   ls -la debug_*_images/
   ```

3. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: description of changes"
   git push origin feature/your-feature-name
   ```

### Code Style
- Follow PEP 8 Python style guidelines
- Use descriptive variable names (`player_count` not `pc`)
- Add docstrings to public methods
- Include user-friendly print statements for status updates

### Testing
- Add test scripts for new features following `test_*.py` pattern
- Ensure debug visualizations are generated for visual verification
- Test with multiple YOLO models when relevant

## 📊 Data Flow

```
Video Input → Player Detection (YOLO) → Ice Surface Detection → Calibration → Analysis → Statistics/Visualization
     ↓              ↓                      ↓                    ↓            ↓
Debug Images   Bounding Boxes        Ice Masks          Measurements    Game Stats
```

## 🔗 Dependencies

Key libraries:
- **OpenCV**: Video processing and computer vision
- **Ultralytics**: YOLOv8 object detection
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization and debug images
- **Pandas**: Data analysis
- **Streamlit**: Web dashboard (future)

See `requirements.txt` for complete dependency list.

## 📝 License

This project is for educational and research purposes. YOLO models are subject to their respective licenses.

## 🆘 Support

For issues and questions:
1. Check debug images in `debug_*_images/` directories
2. Run individual test scripts to isolate problems
3. Review console output for error messages
4. Open GitHub issue with debug images and error logs

---

**Happy analyzing! 🏒📊**
