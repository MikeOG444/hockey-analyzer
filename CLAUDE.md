# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hockey video analysis system built in Python that uses computer vision and YOLO object detection to analyze hockey gameplay from LiveBarn footage. The system focuses on player detection, ice surface identification, and rink calibration for tracking purposes.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Individual test scripts for different features
python test_video_analysis.py          # Basic video processing test
python test_model_comparison.py        # Compare YOLO model performance
python test_ice_markings.py           # Ice surface detection tests
python test_brightness_boundary_detection.py  # Ice boundary detection
python test_adjacent_ice_expansion.py  # Ice mask refinement tests
python debug_detection.py             # Debug object detection issues

# Specific rink tests
python test_brightness_boundary_detection_rink1.py  # Rink-specific calibration
```

### Key Models
The project uses YOLOv8 models stored in the root directory:
- `yolov8n.pt` - Nano model (fastest, least accurate)
- `yolov8s.pt` - Small model
- `yolov8m.pt` - Medium model  
- `yolov8l.pt` - Large model (slowest, most accurate)

## Architecture

### Core Modules (`src/`)

#### Detection (`src/detection/`)
- `basic_detector.py` - Basic YOLO-based player detection
- `ice_detection.py` - Comprehensive ice surface detection using multiple methods (color, texture, geometry, markings)
- `improved_detector.py` - Enhanced detection with ice surface filtering
- Contains YOLOv8 model file (`yolov8n.pt`)

#### Calibration (`calibration.py`)
- `HockeyRinkCalibrator` class for converting between pixel coordinates and real-world measurements
- Handles blue line detection, zone identification (offensive/defensive/neutral)
- Supports manual calibration using annotated reference frames
- Key rink dimensions: blue line width (12"), glass height (48"), column width (22")

#### Analysis (`src/analysis/`)
Currently empty - placeholder for future analytics features

#### UI (`src/ui/`)
Currently empty - placeholder for user interface components

### Test Data Structure
- `data/` - Contains test videos and images
- Multiple debug image directories for visual debugging of different detection algorithms
- Debug images are generated automatically during test runs

### Key Classes and Functions

#### BasicHockeyDetector (`src/detection/basic_detector.py`)
- `detect_players(frame)` - YOLO-based player detection
- `analyze_frame(frame)` - Complete frame analysis returning player count and positions
- Uses COCO class 0 (person) with confidence threshold of 0.5

#### IceSurfaceDetector (`src/detection/ice_detection.py`)
- `detect_ice_surface(frame)` - Multi-method ice surface detection
- `detect_white_surfaces()` - Color-based ice detection using HSV and LAB color spaces
- `detect_smooth_surfaces()` - Texture analysis for smooth ice areas
- `detect_large_flat_areas()` - Geometric constraints for ice identification
- `detect_by_ice_markings()` - Uses red/blue line detection to infer ice boundaries
- `is_on_ice(position)` - Check if coordinate is on detected ice surface

#### HockeyRinkCalibrator (`calibration.py`)
- `add_calibration_point()` - Add reference measurements for pixel-to-inch conversion
- `detect_blue_lines()` - Automated blue line detection
- `detect_zones()` - Identify offensive, defensive, and neutral zones
- `track_puck_zone()` - Determine current zone of puck position
- `pixel_to_inches()` - Convert between pixel distances and real measurements

## Development Notes

### Ice Detection Strategy
The system uses a multi-layered approach for ice surface detection:
1. **Color detection** - Identifies white/light colored areas using HSV and LAB color spaces
2. **Texture analysis** - Ice surfaces are typically smoother than surrounding areas
3. **Geometric constraints** - Ice is usually the largest contiguous flat area
4. **Ice markings reference** - Uses red center line and blue lines as reference points

### Calibration Approach
Manual calibration requires annotated reference frames with colored rectangles representing known dimensions. The system supports:
- Horizontal measurements (rink width spans)
- Vertical measurements (glass height)  
- Depth measurements (structural column widths)

### Testing Strategy
Each major component has dedicated test scripts that generate debug images for visual verification. Test files follow the pattern `test_*.py` and automatically create corresponding `debug_*_images/` directories.

## Data Dependencies

### Required Video Format
- Expects MP4 videos in `data/test_videos/` directory
- Currently configured for LiveBarn hockey footage
- Videos should show clear view of ice surface with visible markings

### YOLO Models
Pre-trained YOLOv8 models are stored in root directory and automatically downloaded by ultralytics package if missing.