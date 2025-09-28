import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json

@dataclass
class CalibrationPoint:
    """Stores a calibration reference point with pixel and real-world coordinates"""
    name: str
    pixel_coords: Tuple[int, int, int, int]  # x1, y1, x2, y2
    real_world_size: float  # in inches
    orientation: str  # 'horizontal', 'vertical', 'depth'

@dataclass
class RinkDimensions:
    """Stores rink measurements and calibration data"""
    blue_line_width: float = 12.0  # inches
    glass_height: float = 48.0     # inches  
    column_width: float = 22.0     # inches
    horizontal_span: float = 276.5 # inches
    
class HockeyRinkCalibrator:
    """Calibrates hockey rink dimensions from annotated frames"""
    
    def __init__(self):
        self.rink_dims = RinkDimensions()
        self.calibration_points: List[CalibrationPoint] = []
        self.pixels_per_inch_horizontal: Optional[float] = None
        self.pixels_per_inch_vertical: Optional[float] = None
        
    def add_calibration_point(self, name: str, pixel_coords: Tuple[int, int, int, int], 
                            real_world_size: float, orientation: str):
        """Add a calibration reference point"""
        point = CalibrationPoint(name, pixel_coords, real_world_size, orientation)
        self.calibration_points.append(point)
        
    def calculate_calibration_from_annotations(self, annotated_frame_path: str):
        """
        Calculate pixel-to-inch ratios from user's annotated frame
        User needs to provide pixel coordinates for their colored rectangles
        """
        # These would come from measuring the colored rectangles in your image
        # You'll need to provide these pixel measurements
        
        calibration_data = {
            'pink_rectangle': {
                'pixel_coords': (100, 150, 500, 200),  # Replace with actual measurements
                'real_world_size': 276.5,
                'orientation': 'horizontal'
            },
            'cyan_glass': {
                'pixel_coords': (800, 300, 850, 450),  # Replace with actual measurements  
                'real_world_size': 48.0,
                'orientation': 'vertical'
            },
            'green_column': {
                'pixel_coords': (50, 250, 100, 350),   # Replace with actual measurements
                'real_world_size': 22.0,
                'orientation': 'depth'
            }
        }
        
        # Calculate calibration ratios
        for name, data in calibration_data.items():
            self.add_calibration_point(name, data['pixel_coords'], 
                                     data['real_world_size'], data['orientation'])
            
        self._calculate_calibration_ratios()
        
    def _calculate_calibration_ratios(self):
        """Calculate pixels per inch for different orientations"""
        
        for point in self.calibration_points:
            x1, y1, x2, y2 = point.pixel_coords
            
            if point.orientation == 'horizontal':
                pixel_distance = abs(x2 - x1)
                self.pixels_per_inch_horizontal = pixel_distance / point.real_world_size
                
            elif point.orientation == 'vertical':
                pixel_distance = abs(y2 - y1) 
                self.pixels_per_inch_vertical = pixel_distance / point.real_world_size
                
        print(f"Horizontal calibration: {self.pixels_per_inch_horizontal:.2f} pixels/inch")
        print(f"Vertical calibration: {self.pixels_per_inch_vertical:.2f} pixels/inch")
        
    def pixel_to_inches(self, pixel_distance: float, orientation: str) -> float:
        """Convert pixel distance to inches"""
        if orientation == 'horizontal' and self.pixels_per_inch_horizontal:
            return pixel_distance / self.pixels_per_inch_horizontal
        elif orientation == 'vertical' and self.pixels_per_inch_vertical:
            return pixel_distance / self.pixels_per_inch_vertical
        else:
            raise ValueError("Calibration not available for this orientation")
            
    def detect_blue_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect blue lines in the frame using color and width filtering"""
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue line color range (adjust these values based on your rink)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_lines = []
        expected_width_pixels = self.rink_dims.blue_line_width * self.pixels_per_inch_horizontal
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on expected blue line characteristics
            if (w > h * 3 and  # Horizontal line (wider than tall)
                abs(h - expected_width_pixels) < expected_width_pixels * 0.3):  # Width tolerance
                blue_lines.append((x, y, x + w, y + h))
                
        return blue_lines
        
    def detect_zones(self, frame: np.ndarray) -> dict:
        """Detect offensive, defensive, and neutral zones"""
        
        blue_lines = self.detect_blue_lines(frame)
        
        if len(blue_lines) < 2:
            return {"error": "Could not detect both blue lines"}
            
        # Sort blue lines by x-coordinate (left to right)
        blue_lines.sort(key=lambda line: line[0])
        
        left_blue = blue_lines[0]
        right_blue = blue_lines[1]
        
        frame_height, frame_width = frame.shape[:2]
        
        zones = {
            "left_zone": {
                "bounds": (0, left_blue[2]),  # From left edge to left blue line
                "type": "defensive_or_offensive"  # Depends on team orientation
            },
            "neutral_zone": {
                "bounds": (left_blue[2], right_blue[0]),  # Between blue lines
                "type": "neutral"
            },
            "right_zone": {
                "bounds": (right_blue[0], frame_width),  # From right blue line to right edge
                "type": "defensive_or_offensive"  # Depends on team orientation
            }
        }
        
        return zones
        
    def track_puck_zone(self, frame: np.ndarray, puck_position: Tuple[int, int]) -> str:
        """Determine which zone the puck is currently in"""
        
        zones = self.detect_zones(frame)
        
        if "error" in zones:
            return "unknown"
            
        puck_x, puck_y = puck_position
        
        for zone_name, zone_data in zones.items():
            if zone_data["type"] != "neutral":
                continue
                
            left_bound, right_bound = zone_data["bounds"]
            if left_bound <= puck_x <= right_bound:
                return zone_data["type"]
                
        # Check other zones
        for zone_name, zone_data in zones.items():
            if zone_data["type"] == "neutral":
                continue
                
            left_bound, right_bound = zone_data["bounds"]
            if left_bound <= puck_x <= right_bound:
                return f"{zone_name}"
                
        return "unknown"
        
    def visualize_calibration(self, frame: np.ndarray, output_path: str = None):
        """Visualize the calibration points and detected zones"""
        
        vis_frame = frame.copy()
        
        # Draw calibration points
        for point in self.calibration_points:
            x1, y1, x2, y2 = point.pixel_coords
            color = (0, 255, 0) if point.orientation == 'horizontal' else (255, 0, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"{point.name}: {point.real_world_size}\"", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Detect and draw blue lines
        blue_lines = self.detect_blue_lines(frame)
        for line in blue_lines:
            x1, y1, x2, y2 = line
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis_frame, "Blue Line", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Detect and draw zones
        zones = self.detect_zones(frame)
        if "error" not in zones:
            for zone_name, zone_data in zones.items():
                left_bound, right_bound = zone_data["bounds"]
                cv2.line(vis_frame, (left_bound, 0), (left_bound, frame.shape[0]), 
                        (0, 255, 255), 2)
                cv2.line(vis_frame, (right_bound, 0), (right_bound, frame.shape[0]), 
                        (0, 255, 255), 2)
                
                # Label zone
                mid_x = (left_bound + right_bound) // 2
                cv2.putText(vis_frame, zone_data["type"], (mid_x, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_frame)
            
        return vis_frame

def main():
    """Example usage of the hockey tracking system"""
    
    # Initialize calibrator
    calibrator = HockeyRinkCalibrator()
    
    # Load your annotated frame
    frame_path = "your_annotated_frame.jpg"  # Replace with your image path
    
    # You need to measure the pixel coordinates of your colored rectangles
    # These are example coordinates - replace with your actual measurements
    calibrator.add_calibration_point(
        "horizontal_span", 
        (100, 150, 650, 200),  # Replace with actual pink rectangle coordinates
        276.5, 
        "horizontal"
    )
    
    calibrator.add_calibration_point(
        "glass_height",
        (800, 300, 850, 420),  # Replace with actual cyan rectangle coordinates  
        48.0,
        "vertical"
    )
    
    # Calculate calibration ratios
    calibrator._calculate_calibration_ratios()
    
    # Load and process a frame
    # frame = cv2.imread(frame_path)
    # zones = calibrator.detect_zones(frame)
    # print("Detected zones:", zones)
    
    # Visualize results
    # vis_frame = calibrator.visualize_calibration(frame, "calibration_output.jpg")
    
    print("Calibration system ready!")
    print("Next steps:")
    print("1. Measure pixel coordinates of your colored rectangles")
    print("2. Update the calibration_point coordinates in the code")
    print("3. Test with your LiveBarn footage")

if __name__ == "__main__":
    main()