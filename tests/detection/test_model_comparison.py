import cv2
from ultralytics import YOLO
import time

def test_different_yolo_models():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    # Models to test (from smallest to largest)
    models_to_test = {
        'yolov8n.pt': 'YOLOv8 Nano (current)',
        'yolov8s.pt': 'YOLOv8 Small', 
        'yolov8m.pt': 'YOLOv8 Medium',
        'yolov8l.pt': 'YOLOv8 Large'
    }
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame")
        return
    
    print("Testing different YOLO models on first frame")
    print("Expected: 11 people on ice")
    print("=" * 60)
    
    results = {}
    
    for model_file, model_name in models_to_test.items():
        print(f"\nTesting {model_name}...")
        
        try:
            # Load model (will download if not present)
            model = YOLO(model_file)
            
            # Time the inference
            start_time = time.time()
            model_results = model(frame)
            inference_time = time.time() - start_time
            
            # Count people with different confidence thresholds
            people_counts = {}
            all_detections = []
            
            for result in model_results:
                for box in result.boxes:
                    if box.cls == 0:  # Person class
                        confidence = float(box.conf[0].cpu().numpy())
                        all_detections.append(confidence)
            
            # Count at different thresholds
            for threshold in [0.15, 0.25, 0.35, 0.5]:
                count = len([c for c in all_detections if c >= threshold])
                people_counts[threshold] = count
            
            results[model_name] = {
                'inference_time': inference_time,
                'counts': people_counts,
                'all_confidences': sorted(all_detections, reverse=True)
            }
            
            # Display results
            print(f"  Inference time: {inference_time:.3f}s")
            print("  People detected by confidence threshold:")
            for threshold, count in people_counts.items():
                status = "✅" if count >= 9 else "⚠️" if count >= 7 else "❌"
                print(f"    ≥{threshold}: {count} people {status}")
            
            # Show top confidence scores
            if all_detections:
                top_5 = sorted(all_detections, reverse=True)[:5]
                print(f"  Top 5 confidence scores: {[f'{c:.2f}' for c in top_5]}")
            
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
            continue
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    for model_name, data in results.items():
        best_count = max(data['counts'].values())
        best_threshold = min([t for t, c in data['counts'].items() if c == best_count])
        
        print(f"{model_name}:")
        print(f"  Best detection: {best_count}/11 people at confidence ≥{best_threshold}")
        print(f"  Speed: {data['inference_time']:.3f}s")
        print(f"  Efficiency: {best_count/data['inference_time']:.1f} detections/second")
        print()
    
    return results

if __name__ == "__main__":
    test_different_yolo_models()