import cv2
import numpy as np

def detect_blue_line_focused():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Focused blue line detection using hex #807c94...")
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Convert hex #807c94 to HSV and create wide range around it
    # #807c94 = RGB(128, 124, 148) 
    target_bgr = np.uint8([[[148, 124, 128]]])  # BGR format for OpenCV
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    print(f"Target HSV for #807c94: {target_hsv}")
    
    # Create very wide range around the target color
    hue_range = 40  # Wide hue tolerance
    sat_range = 80  # Wide saturation tolerance  
    val_range = 80  # Wide value tolerance
    
    lower_bound = np.array([
        max(0, target_hsv[0] - hue_range),
        max(0, target_hsv[1] - sat_range), 
        max(0, target_hsv[2] - val_range)
    ])
    
    upper_bound = np.array([
        min(179, target_hsv[0] + hue_range),
        min(255, target_hsv[1] + sat_range),
        min(255, target_hsv[2] + val_range)
    ])
    
    print(f"Detection range: {lower_bound} to {upper_bound}")
    
    # Detect the color
    color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find lines - adjusted for more vertical orientation
    lines = cv2.HoughLinesP(color_mask, 1, np.pi/180, threshold=10, minLineLength=30, maxLineGap=15)
    
    # Filter for blue line characteristics
    def filter_blue_lines(lines):
        if lines is None:
            return None
            
        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate properties
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # Blue lines are more vertical (perpendicular to boards)
            # Accept angles roughly 60-120 degrees (more vertical than horizontal)
            if length > 20 and (60 <= angle <= 120):
                filtered.append(line)
        
        return np.array(filtered) if filtered else None
    
    filtered_lines = filter_blue_lines(lines)
    
    # Create visualization
    debug_frame = frame.copy()
    
    # Show color detection areas
    color_overlay = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    debug_frame = cv2.addWeighted(debug_frame, 0.7, color_overlay, 0.3, 0)
    
    # Draw detected lines
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow lines
        print(f"Blue lines detected: {len(filtered_lines)}")
    else:
        print("No blue lines detected")
    
    # Save debug images
    cv2.imwrite('debug_blue_focused.jpg', debug_frame)
    cv2.imwrite('debug_color_detection.jpg', color_mask)
    
    print(f"Color pixels detected: {np.sum(color_mask > 0)}")
    print("Debug images saved:")
    print("- debug_blue_focused.jpg (lines overlaid)")
    print("- debug_color_detection.jpg (raw color detection)")

if __name__ == "__main__":
    detect_blue_line_focused()