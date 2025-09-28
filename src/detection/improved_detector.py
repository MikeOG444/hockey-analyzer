# src/detection/improved_detector.py
import cv2
from ultralytics import YOLO
import numpy as np

class RealisticHockeyDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO('yolov8m.pt')
        print("✅ YOLO model loaded")
        
    def detect_all_people(self, frame, confidence_threshold=0.3):
        """Detect all people in frame"""
        results = self.model(frame)
        
        all_people = []
        
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Person class
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > confidence_threshold:
                        person = {
                            'position': (int(x), int(y)),
                            'bbox': (int(x-w/2), int(y-h/2), int(w), int(h)),
                            'confidence': float(confidence),
                            'area': int(w * h)
                        }
                        all_people.append(person)
        
        return all_people
    
    def classify_ice_areas(self, frame):
        """Identify different areas of the rink"""
        height, width = frame.shape[:2]
        
        areas = {
            'ice_surface': {
                'y_range': (height * 0.2, height * 0.8),  # Middle area of frame
                'description': 'Main playing surface'
            },
            'bench_area': {
                'y_range': (height * 0.8, height),  # Bottom of frame
                'description': 'Player benches'
            },
            'crowd_upper': {
                'y_range': (0, height * 0.2),  # Top of frame  
                'description': 'Upper crowd/stands'
            },
            'boards_area': {
                'y_range': (height * 0.7, height * 0.9),  # Along boards
                'description': 'Along the boards'
            }
        }
        
        return areas
    
    def categorize_people(self, people, frame):
        """Categorize detected people by likely location"""
        height, width = frame.shape[:2]
        
        categorized = {
            'likely_on_ice': [],
            'likely_bench': [],
            'likely_crowd': [],
            'uncertain': []
        }
        
        for person in people:
            x, y = person['position']
            
            if y < height * 0.2:  # Top area - crowd
                categorized['likely_crowd'].append(person)
            elif y > height * 0.85:  # Bottom area - benches
                categorized['likely_bench'].append(person)  
            elif height * 0.2 <= y <= height * 0.8:  # Middle area - ice
                categorized['likely_on_ice'].append(person)
            else:
                categorized['uncertain'].append(person)
        
        return categorized
    
    def analyze_frame_comprehensive(self, frame):
        """Comprehensive analysis of all people in frame"""
        all_people = self.detect_all_people(frame, confidence_threshold=0.3)
        categorized = self.categorize_people(all_people, frame)
        
        analysis = {
            'total_people_detected': len(all_people),
            'on_ice_count': len(categorized['likely_on_ice']),
            'bench_count': len(categorized['likely_bench']),
            'crowd_count': len(categorized['likely_crowd']),
            'uncertain_count': len(categorized['uncertain']),
            'categorized_people': categorized,
            'all_people': all_people
        }
        
        # Validate ice count
        ice_count = analysis['on_ice_count']
        if 8 <= ice_count <= 20:
            analysis['ice_count_reasonable'] = True
        else:
            analysis['ice_count_reasonable'] = False
            
        return analysis

    def detect_players(self, frame, confidence_threshold=0.3):
        """Original method for backward compatibility"""
        players, all_detections = self.detect_players_with_details(frame, confidence_threshold)
        return players
    
    def detect_players_with_details(self, frame, confidence_threshold=0.3):
        """Detect players with lower confidence threshold for hockey"""
        results = self.model(frame)
        
        all_detections = []
        players = []
        
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Person class
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    detection = {
                        'position': (int(x), int(y)),
                        'bbox': (int(x-w/2), int(y-h/2), int(w), int(h)),
                        'confidence': float(confidence),
                        'center': (int(x), int(y))
                    }
                    
                    all_detections.append(detection)
                    
                    # Lower confidence threshold for hockey
                    if confidence > confidence_threshold:
                        players.append(detection)
        
        return players, all_detections
    
    def filter_ice_players(self, players, frame_height, frame_width):
        """Filter to keep only players likely on the ice"""
        ice_players = []
        
        for player in players:
            x, y = player['position']
            
            # Basic filtering - keep players in lower 2/3 of frame (ice area)
            if y > frame_height * 0.33:  # Below top third (avoid crowd/benches)
                ice_players.append(player)
        
        return ice_players
    
    def analyze_frame_detailed(self, frame):
        """Detailed analysis with multiple confidence levels"""
        height, width = frame.shape[:2]
        
        # Try different confidence thresholds
        results = {}
        
        for threshold in [0.3, 0.4, 0.5, 0.6]:
            players, all_detections = self.detect_players_with_details(frame, threshold)
            ice_players = self.filter_ice_players(players, height, width)
            
            results[f'conf_{int(threshold*10)}'] = {
                'threshold': threshold,
                'total_detections': len(all_detections),
                'players_above_threshold': len(players),
                'ice_players': len(ice_players),
                'players': ice_players
            }
        
        return results

# Test function
if __name__ == "__main__":
    detector = RealisticHockeyDetector()
    print("Realistic hockey detector ready!")