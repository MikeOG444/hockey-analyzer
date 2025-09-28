import cv2
import numpy as np
from ultralytics import YOLO

def detect_ice_markings():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Detecting ice markings in first frame...")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect red center line
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Detect blue lines
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Find lines
    red_lines = cv2.HoughLinesP(red_mask, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    blue_lines = cv2.HoughLinesP(blue_mask, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    
    # Visualize results
    debug_frame = frame.copy()
    
    if red_lines is not None:
        for line in red_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        print(f"Red lines detected: {len(red_lines)}")
    else:
        print("No red lines detected")
    
    if blue_lines is not None:
        for line in blue_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        print(f"Blue lines detected: {len(blue_lines)}")
    else:
        print("No blue lines detected")
    
    cv2.imwrite('debug_ice_markings.jpg', debug_frame)
    print("Ice markings debug saved as 'debug_ice_markings.jpg'")

if __name__ == "__main__":
    detect_ice_markings()