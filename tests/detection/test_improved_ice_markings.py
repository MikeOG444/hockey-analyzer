import cv2
import numpy as np

def detect_actual_ice_markings():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Detecting actual ice markings (filtering out arena decorations)...")
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create ice surface mask first - focus on the white ice area
    # White/light areas (ice surface)
    ice_lower = np.array([0, 0, 180])
    ice_upper = np.array([180, 30, 255])
    ice_mask = cv2.inRange(hsv, ice_lower, ice_upper)
    
    # Clean up ice mask
    kernel = np.ones((5, 5), np.uint8)
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel)
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the largest white area (should be ice surface)
    contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ice_only_mask = np.zeros(ice_mask.shape, dtype=np.uint8)
        cv2.fillPoly(ice_only_mask, [largest_contour], 255)
    else:
        ice_only_mask = ice_mask
    
    # Now detect blue lines ONLY within ice area
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Combine blue detection with ice area
    blue_on_ice = cv2.bitwise_and(blue_mask, ice_only_mask)
    
    # Detect red center line ONLY within ice area
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Combine red detection with ice area
    red_on_ice = cv2.bitwise_and(red_mask, ice_only_mask)
    
    # Find lines with more restrictive parameters
    red_lines = cv2.HoughLinesP(red_on_ice, 1, np.pi/180, threshold=30, minLineLength=80, maxLineGap=15)
    blue_lines = cv2.HoughLinesP(blue_on_ice, 1, np.pi/180, threshold=30, minLineLength=80, maxLineGap=15)
    
    # Filter lines by length and orientation
    def filter_hockey_lines(lines):
        if lines is None:
            return None
        
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Calculate angle (hockey lines are mostly horizontal)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # Filter: reasonable length and roughly horizontal
            if length > 60 and (angle < 20 or angle > 160):
                filtered_lines.append(line)
        
        return np.array(filtered_lines) if filtered_lines else None
    
    red_lines_filtered = filter_hockey_lines(red_lines)
    blue_lines_filtered = filter_hockey_lines(blue_lines)
    
    # Create visualization - FIXED VERSION
    debug_frame = frame.copy()
    
    # Draw filtered lines
    if red_lines_filtered is not None:
        for line in red_lines_filtered:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        print(f"Red lines on ice: {len(red_lines_filtered)}")
    else:
        print("No red lines detected on ice")
    
    if blue_lines_filtered is not None:
        for line in blue_lines_filtered:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        print(f"Blue lines on ice: {len(blue_lines_filtered)}")
    else:
        print("No blue lines detected on ice")
    
    # Show ice area with green tint (fixed)
    ice_colored = np.zeros_like(frame)
    ice_colored[:, :, 1] = ice_only_mask  # Green channel
    debug_frame = cv2.addWeighted(debug_frame, 0.8, ice_colored, 0.2, 0)
    
    # Save results
    cv2.imwrite('debug_ice_markings_filtered.jpg', debug_frame)
    cv2.imwrite('debug_ice_mask.jpg', ice_only_mask)
    cv2.imwrite('debug_blue_on_ice.jpg', blue_on_ice)
    cv2.imwrite('debug_red_on_ice.jpg', red_on_ice)
    
    print("Debug images saved:")
    print("- debug_ice_markings_filtered.jpg (final result)")
    print("- debug_ice_mask.jpg (detected ice surface)")
    print("- debug_blue_on_ice.jpg (blue detection on ice only)")
    print("- debug_red_on_ice.jpg (red detection on ice only)")

if __name__ == "__main__":
    detect_actual_ice_markings()