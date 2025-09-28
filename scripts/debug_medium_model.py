import cv2
from ultralytics import YOLO
import numpy as np

def debug_medium_model():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    # Load YOLOv8 Medium model
    print("Loading YOLOv8 Medium model...")
    model = YOLO('yolov8m.pt')
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame")
        return
    
    print("Debugging YOLOv8 Medium detection on first frame")
    print("Expected: 11 people on ice + 2 on bench")
    print("=" * 60)
    
    # Run detection
    results = model(frame)
    
    all_people = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Person class
                x, y, w, h = box.xywh[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                if confidence >= 0.15:
                    person = {
                        'position': (int(x), int(y)),
                        'bbox': (int(x-w/2), int(y-h/2), int(w), int(h)),
                        'confidence': confidence
                    }
                    all_people.append(person)
    
    # Sort by confidence
    all_people.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Total people detected: {len(all_people)}")
    print("\nDetailed detections:")
    
    # Categorize by likely location (ice vs bench)
    frame_height = frame.shape[0]
    ice_people = []
    bench_people = []
    
    for i, person in enumerate(all_people):
        pos = person['position']
        conf = person['confidence']
        bbox = person['bbox']
        
        # Simple categorization by Y position
        if pos[1] > frame_height * 0.85:  # Bottom 15% - likely bench
            category = "BENCH"
            bench_people.append(person)
        else:
            category = "ICE"
            ice_people.append(person)
        
        print(f"Person {i+1:2d}: pos{pos}, conf={conf:.2f}, bbox={bbox}, location={category}")
    
    print(f"\nCategorization:")
    print(f"On ice: {len(ice_people)} people")
    print(f"On bench: {len(bench_people)} people")
    
    # Create enhanced visualization
    debug_frame = frame.copy()
    
    for i, person in enumerate(all_people):
        x, y, w, h = person['bbox']
        conf = person['confidence']
        pos = person['position']
        
        # Color code by location and confidence
        if pos[1] > frame_height * 0.85:  # Bench area
            color = (255, 0, 255)  # Magenta for bench
            location = "BENCH"
        else:  # Ice area
            if conf >= 0.5:
                color = (0, 255, 0)  # Green - high confidence ice
            elif conf >= 0.3:
                color = (0, 255, 255)  # Yellow - medium confidence ice
            else:
                color = (0, 0, 255)  # Red - low confidence ice
            location = "ICE"
        
        # Draw bounding box
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 3)
        
        # Draw labels
        label = f"{i+1}: {conf:.2f}"
        cv2.putText(debug_frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(debug_frame, location, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    cv2.putText(debug_frame, "Green=High Conf Ice, Yellow=Med Conf Ice", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_frame, "Red=Low Conf Ice, Magenta=Bench", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save debug image
    cv2.imwrite('debug_medium_model.jpg', debug_frame)
    print(f"\nDebug image saved as 'debug_medium_model.jpg'")
    
    # Validation check
    if len(ice_people) == 11 and len(bench_people) == 2:
        print("Perfect detection! All 11 ice players + 2 bench players detected correctly.")
    elif len(ice_people) == 11:
        print(f"Great ice detection! All 11 ice players detected. {len(bench_people)} bench detections.")
    else:
        print(f"Ice detection: {len(ice_people)}/11 players detected.")
    
    return all_people

if __name__ == "__main__":
    debug_medium_model()