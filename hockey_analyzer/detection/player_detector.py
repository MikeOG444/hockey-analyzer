import cv2
from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
    def detect_players(self, frame, ice_mask=None):
        """
        Detect players in a frame, optionally filtering by ice surface
        
        Args:
            frame: Input video frame
            ice_mask: Optional binary mask of ice surface (255=ice, 0=not ice)
                     If provided, only detections within ice area are returned
        
        Returns:
            List of player detections with position, bbox, confidence
        """
        results = self.model(frame)
        
        players = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Person class in COCO dataset
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.5:  # Filter low confidence detections
                        # Convert to integer coordinates
                        center_x, center_y = int(x), int(y)
                        bbox = (int(x-w/2), int(y-h/2), int(w), int(h))
                        
                        # If ice mask provided, check if detection is on ice
                        if ice_mask is not None:
                            # Check if player center is within ice mask
                            if (0 <= center_y < ice_mask.shape[0] and 
                                0 <= center_x < ice_mask.shape[1]):
                                if ice_mask[center_y, center_x] == 0:
                                    continue  # Skip detections not on ice
                            else:
                                continue  # Skip detections outside frame bounds
                        
                        players.append({
                            'position': (center_x, center_y),
                            'bbox': bbox,
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