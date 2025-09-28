import cv2
from src.detection.improved_detector import RealisticHockeyDetector

def test_lower_confidence():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    detector = RealisticHockeyDetector()
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    
    if ret:
        print("🔍 Testing different confidence thresholds:")
        print("=" * 50)
        
        for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]:
            all_people = detector.detect_all_people(frame, confidence_threshold=threshold)
            categorized = detector.categorize_people(all_people, frame)
            on_ice_count = len(categorized['likely_on_ice'])
            
            confidences = [p['confidence'] for p in categorized['likely_on_ice']]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            status = "✅" if 8 <= on_ice_count <= 17 else "⚠️"
            
            print(f"Confidence ≥ {threshold}: {on_ice_count} people on ice {status} (avg conf: {avg_conf:.2f})")
            
            # Show individual detections for best threshold
            if threshold == 0.25:
                print("   Detailed detections:")
                for i, person in enumerate(categorized['likely_on_ice'][:8]):
                    pos = person['position']
                    conf = person['confidence']
                    print(f"   Person {i+1}: pos{pos}, conf={conf:.2f}")
    
    cap.release()

if __name__ == "__main__":
    test_lower_confidence()