import cv2
from ultralytics import YOLO
import numpy as np
import cv2

class PlayerDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
    def detect_players(self, frame, ice_mask=None):
        """
        Detect players in a frame, using polygon masking for ice-only detection
        
        Args:
            frame: Input video frame
            ice_mask: Optional binary mask of ice surface (255=ice, 0=not ice)
                     If provided, frame is masked to show only ice area before detection
        
        Returns:
            List of player detections with position, bbox, confidence
        """
        
        # OPTION 2: Polygon Masking - Only detect within ice area
        detection_frame = frame
        if ice_mask is not None:
            # Create masked frame - black out everything except ice
            detection_frame = cv2.bitwise_and(frame, frame, mask=ice_mask)
            # This makes YOLO only "see" the ice area - much more efficient!
        
        # Run YOLO detection on masked frame (or original if no mask)
        results = self.model(detection_frame)
        
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
                        
                        # Additional edge player validation using OR logic
                        if ice_mask is not None:
                            if not self._is_player_on_ice_or_logic(bbox, ice_mask):
                                continue  # Skip if neither center nor feet are on ice
                        
                        players.append({
                            'position': (center_x, center_y),
                            'bbox': bbox,
                            'confidence': float(confidence)
                        })
        
        return players
    
    def _is_player_on_ice_or_logic(self, bbox, ice_mask):
        """
        Smart edge detection using OR logic:
        - Bottom half players: Check center (feet hidden by boards)
        - Top half players: Check feet area (might be on bench)
        - Returns True if EITHER center OR feet are on ice
        """
        x, y, w, h = bbox
        
        # Center point of player (good for bottom edge players)
        center_x, center_y = x + w//2, y + h//2
        
        # Feet area (bottom of bounding box - good for top edge players)  
        feet_x, feet_y = x + w//2, y + h
        
        # Check center point
        center_on_ice = False
        if (0 <= center_y < ice_mask.shape[0] and 0 <= center_x < ice_mask.shape[1]):
            center_on_ice = ice_mask[center_y, center_x] > 0
        
        # Check feet area
        feet_on_ice = False
        if (0 <= feet_y < ice_mask.shape[0] and 0 <= feet_x < ice_mask.shape[1]):
            feet_on_ice = ice_mask[feet_y, feet_x] > 0
            
        # OR logic: Player is on ice if EITHER center OR feet are on ice
        return center_on_ice or feet_on_ice
    
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