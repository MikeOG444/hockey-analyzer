import cv2
import numpy as np
from src.detection.improved_detector import RealisticHockeyDetector

def debug_first_frame():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    detector = RealisticHockeyDetector()
    cap = cv2.VideoCapture(video_path)
    
    # Get first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame")
        return
    
    print("🔍 Debugging first frame detection")
    print("Expected: 11 people on ice (5 blue + 5 black + 1 goalie)")
    print("=" * 60)
    
    # Test with very low confidence
    all_people = detector.detect_all_people(frame, confidence_threshold=0.15)
    
    print(f"Total detections with confidence ≥ 0.15: {len(all_people)}")
    
    # Sort by confidence
    all_people.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Show all detections
    for i, person in enumerate(all_people):
        pos = person['position']
        conf = person['confidence']
        bbox = person['bbox']
        print(f"Person {i+1:2d}: pos{pos}, conf={conf:.2f}, bbox={bbox}")
    
    # Create visualization
    debug_frame = frame.copy()
    
    # Draw all detections
    for i, person in enumerate(all_people):
        x, y, w, h = person['bbox']
        conf = person['confidence']
        
        # Color code by confidence
        if conf >= 0.5:
            color = (0, 255, 0)  # Green - high confidence
        elif conf >= 0.3:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
        
        # Draw bounding box
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw confidence and number
        label = f"{i+1}: {conf:.2f}"
        cv2.putText(debug_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save debug image
    cv2.imwrite('debug_first_frame.jpg', debug_frame)
    print(f"\n📸 Debug image saved as 'debug_first_frame.jpg'")
    print("Green boxes = high confidence (≥0.5)")
    print("Yellow boxes = medium confidence (0.3-0.5)")  
    print("Red boxes = low confidence (<0.3)")
    
    # Analyze the gap
    detected_count = len(all_people)
    expected_count = 11
    
    if detected_count < expected_count:
        print(f"\n⚠️  Missing {expected_count - detected_count} people")
        print("Possible causes:")
        print("- People too close together (overlapping)")
        print("- Very low confidence detections")
        print("- People partially occluded")
        print("- Non-standard poses (goalie)")
    
    return all_people, debug_frame

if __name__ == "__main__":
    debug_first_frame()