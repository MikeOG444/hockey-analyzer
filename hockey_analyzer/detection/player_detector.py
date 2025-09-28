import cv2
from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
    def detect_players(self, frame):
        """Detect players in a frame"""
        results = self.model(frame)
        
        players = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Person class in COCO dataset
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.5:  # Filter low confidence detections
                        players.append({
                            'position': (int(x), int(y)),
                            'bbox': (int(x-w/2), int(y-h/2), int(w), int(h)),
                            'confidence': float(confidence)
                        })
        
        return players
    
    def analyze_frame(self, frame):
        """Analyze a single frame"""
        players = self.detect_players(frame)
        
        return {
            'player_count': len(players),
            'players': players,
            'timestamp': None  # We'll add this later
        }

# Test function
if __name__ == "__main__":
    detector = PlayerDetector()
    print("Detector ready for testing!")