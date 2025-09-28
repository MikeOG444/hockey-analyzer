import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class PuckCandidate:
    """Represents a potential puck location with confidence metrics"""
    position: Tuple[int, int]
    confidence: float
    size: float
    timestamp: float
    detection_method: str
    velocity: Optional[Tuple[float, float]] = None

class PuckTracker:
    """
    Multi-method puck tracking system for hockey analysis
    Uses motion detection, color filtering, and contextual analysis
    """
    
    def __init__(self, ice_detector=None):
        self.ice_detector = ice_detector
        self.puck_history = deque(maxlen=30)  # Last 30 detections (1 second at 30fps)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=120
        )
        
        # Puck characteristics (based on official hockey puck dimensions)
        self.puck_diameter_inches = 3.0  # Official puck diameter
        self.min_puck_area = 5  # Minimum pixel area for puck
        self.max_puck_area = 200  # Maximum pixel area for puck
        
        # Tracking parameters
        self.max_velocity = 150  # Max pixels per frame (very fast puck)
        self.prediction_frames = 5  # Frames to predict ahead when puck is lost
        
        # Motion detection parameters
        self.motion_threshold = 25
        self.min_contour_area = 3
        
    def detect_puck_candidates(self, frame: np.ndarray, players: List[Dict]) -> List[PuckCandidate]:
        """
        Detect potential puck locations using multiple methods
        """
        candidates = []
        
        # Method 1: Motion-based detection
        motion_candidates = self._detect_puck_by_motion(frame)
        candidates.extend(motion_candidates)
        
        # Method 2: Dark circular object detection
        circle_candidates = self._detect_puck_by_shape(frame)
        candidates.extend(circle_candidates)
        
        # Method 3: Contextual detection (near players' sticks)
        context_candidates = self._detect_puck_by_context(frame, players)
        candidates.extend(context_candidates)
        
        # Filter candidates by ice surface
        if self.ice_detector and self.ice_detector.ice_mask is not None:
            candidates = self._filter_by_ice_surface(candidates)
        
        # Remove duplicates and rank by confidence
        candidates = self._consolidate_candidates(candidates)
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _detect_puck_by_motion(self, frame: np.ndarray) -> List[PuckCandidate]:
        """Detect puck using background subtraction and motion analysis"""
        candidates = []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (puck should be small)
            if self.min_contour_area <= area <= self.max_puck_area:
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate confidence based on shape and size
                    confidence = self._calculate_motion_confidence(contour, area)
                    
                    if confidence > 0.3:
                        candidates.append(PuckCandidate(
                            position=(cx, cy),
                            confidence=confidence,
                            size=area,
                            timestamp=cv2.getTickCount(),
                            detection_method="motion"
                        ))
        
        return candidates
    
    def _detect_puck_by_shape(self, frame: np.ndarray) -> List[PuckCandidate]:
        """Detect puck using HoughCircles and dark object detection"""
        candidates = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: HoughCircles for perfect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=2, maxRadius=15
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Check if it's dark enough to be a puck
                roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                          max(0, x-r):min(gray.shape[1], x+r)]
                
                if roi.size > 0:
                    avg_intensity = np.mean(roi)
                    surrounding_intensity = np.mean(gray[max(0, y-r*2):min(gray.shape[0], y+r*2),
                                                        max(0, x-r*2):min(gray.shape[1], x+r*2)])
                    
                    # Puck should be darker than surroundings
                    contrast = surrounding_intensity - avg_intensity
                    if contrast > 20:  # Significant contrast
                        confidence = min(contrast / 100.0, 1.0)
                        area = math.pi * r * r
                        
                        candidates.append(PuckCandidate(
                            position=(x, y),
                            confidence=confidence,
                            size=area,
                            timestamp=cv2.getTickCount(),
                            detection_method="circle"
                        ))
        
        # Method 2: Dark blob detection
        # Threshold for dark objects
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of dark objects
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_puck_area <= area <= self.max_puck_area:
                # Check if shape is roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.4:  # Reasonably circular
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            confidence = circularity * 0.7  # Shape-based confidence
                            
                            candidates.append(PuckCandidate(
                                position=(cx, cy),
                                confidence=confidence,
                                size=area,
                                timestamp=cv2.getTickCount(),
                                detection_method="dark_blob"
                            ))
        
        return candidates
    
    def _detect_puck_by_context(self, frame: np.ndarray, players: List[Dict]) -> List[PuckCandidate]:
        """Detect puck using contextual information (near players)"""
        candidates = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for player in players:
            px, py = player['position']
            search_radius = 100  # Search within 100 pixels of player
            
            # Define search area around player
            x1 = max(0, px - search_radius)
            y1 = max(0, py - search_radius)
            x2 = min(gray.shape[1], px + search_radius)
            y2 = min(gray.shape[0], py + search_radius)
            
            roi = gray[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Look for small dark objects near player
                _, binary = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.min_puck_area <= area <= 50:  # Smaller area for contextual detection
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"]) + x1
                            cy = int(M["m01"] / M["m00"]) + y1
                            
                            # Calculate distance from player
                            distance = math.sqrt((cx - px)**2 + (cy - py)**2)
                            proximity_score = max(0, 1 - distance / search_radius)
                            
                            confidence = proximity_score * 0.5  # Context-based confidence
                            
                            candidates.append(PuckCandidate(
                                position=(cx, cy),
                                confidence=confidence,
                                size=area,
                                timestamp=cv2.getTickCount(),
                                detection_method="context"
                            ))
        
        return candidates
    
    def _calculate_motion_confidence(self, contour, area: float) -> float:
        """Calculate confidence score for motion-based detection"""
        # Factor 1: Size appropriateness
        ideal_area = 20  # Ideal puck area in pixels
        size_score = max(0, 1 - abs(area - ideal_area) / ideal_area)
        
        # Factor 2: Shape compactness
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * math.pi * area / (perimeter * perimeter)
        else:
            compactness = 0
        
        # Factor 3: Aspect ratio (should be close to 1 for circular objects)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        
        # Combine factors
        confidence = (size_score * 0.4 + compactness * 0.4 + aspect_ratio * 0.2)
        return confidence
    
    def _filter_by_ice_surface(self, candidates: List[PuckCandidate]) -> List[PuckCandidate]:
        """Filter candidates to only those on the ice surface"""
        filtered = []
        for candidate in candidates:
            if self.ice_detector.is_on_ice(candidate.position):
                filtered.append(candidate)
        return filtered
    
    def _consolidate_candidates(self, candidates: List[PuckCandidate]) -> List[PuckCandidate]:
        """Remove duplicate candidates that are too close to each other"""
        if not candidates:
            return []
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        consolidated = []
        min_distance = 20  # Minimum distance between candidates
        
        for candidate in candidates:
            too_close = False
            for existing in consolidated:
                distance = math.sqrt(
                    (candidate.position[0] - existing.position[0])**2 +
                    (candidate.position[1] - existing.position[1])**2
                )
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                consolidated.append(candidate)
        
        return consolidated
    
    def track_puck(self, candidates: List[PuckCandidate]) -> Optional[PuckCandidate]:
        """
        Track puck across frames using temporal consistency
        """
        if not candidates:
            return self._predict_puck_position()
        
        best_candidate = None
        
        if self.puck_history:
            # Use tracking history to select best candidate
            last_position = self.puck_history[-1].position
            
            # Find candidate closest to predicted position
            min_distance = float('inf')
            for candidate in candidates:
                distance = math.sqrt(
                    (candidate.position[0] - last_position[0])**2 +
                    (candidate.position[1] - last_position[1])**2
                )
                
                # Consider both distance and confidence
                score = candidate.confidence - (distance / 100.0)
                
                if distance < self.max_velocity and score > (best_candidate.confidence if best_candidate else 0):
                    best_candidate = candidate
                    
            # If no good temporal match, take highest confidence
            if best_candidate is None and candidates:
                best_candidate = candidates[0]
        else:
            # No history, take highest confidence candidate
            best_candidate = candidates[0] if candidates else None
        
        # Update puck history
        if best_candidate:
            # Calculate velocity if we have history
            if self.puck_history:
                last_pos = self.puck_history[-1].position
                dt = 1.0 / 30.0  # Assume 30fps
                vx = (best_candidate.position[0] - last_pos[0]) / dt
                vy = (best_candidate.position[1] - last_pos[1]) / dt
                best_candidate.velocity = (vx, vy)
            
            self.puck_history.append(best_candidate)
        
        return best_candidate
    
    def _predict_puck_position(self) -> Optional[PuckCandidate]:
        """Predict puck position when no candidates are found"""
        if len(self.puck_history) < 2:
            return None
        
        # Use last known velocity to predict position
        last_puck = self.puck_history[-1]
        if last_puck.velocity:
            vx, vy = last_puck.velocity
            dt = 1.0 / 30.0  # Assume 30fps
            
            predicted_x = int(last_puck.position[0] + vx * dt)
            predicted_y = int(last_puck.position[1] + vy * dt)
            
            return PuckCandidate(
                position=(predicted_x, predicted_y),
                confidence=0.3,  # Low confidence for prediction
                size=last_puck.size,
                timestamp=cv2.getTickCount(),
                detection_method="prediction",
                velocity=last_puck.velocity
            )
        
        return None
    
    def get_puck_possession(self, puck_position: Tuple[int, int], players: List[Dict]) -> Optional[Dict]:
        """
        Determine which player has possession of the puck
        """
        if not players:
            return None
        
        min_distance = float('inf')
        closest_player = None
        possession_threshold = 50  # pixels
        
        px, py = puck_position
        
        for player in players:
            player_x, player_y = player['position']
            distance = math.sqrt((px - player_x)**2 + (py - player_y)**2)
            
            if distance < min_distance and distance < possession_threshold:
                min_distance = distance
                closest_player = player
        
        if closest_player:
            closest_player['possession_distance'] = min_distance
            return closest_player
        
        return None
    
    def visualize_tracking(self, frame: np.ndarray, candidates: List[PuckCandidate], 
                          tracked_puck: Optional[PuckCandidate]) -> np.ndarray:
        """Draw puck tracking visualization on frame"""
        vis_frame = frame.copy()
        
        # Draw all candidates
        for i, candidate in enumerate(candidates):
            x, y = candidate.position
            color = (0, 255, 255)  # Yellow for candidates
            radius = max(3, int(math.sqrt(candidate.size)))
            
            cv2.circle(vis_frame, (x, y), radius, color, 1)
            cv2.putText(vis_frame, f"{candidate.confidence:.2f}", 
                       (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw tracked puck
        if tracked_puck:
            x, y = tracked_puck.position
            color = (0, 0, 255)  # Red for tracked puck
            radius = max(5, int(math.sqrt(tracked_puck.size)))
            
            cv2.circle(vis_frame, (x, y), radius, color, 2)
            cv2.putText(vis_frame, f"PUCK: {tracked_puck.confidence:.2f}", 
                       (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
            # Draw velocity vector if available
            if tracked_puck.velocity:
                vx, vy = tracked_puck.velocity
                scale = 0.1  # Scale down velocity for visualization
                end_x = int(x + vx * scale)
                end_y = int(y + vy * scale)
                cv2.arrowedLine(vis_frame, (x, y), (end_x, end_y), color, 2)
        
        # Draw puck history trail
        if len(self.puck_history) > 1:
            # Convert deque to list for slicing
            history_list = list(self.puck_history)
            trail_points = [puck.position for puck in history_list[-10:]]  # Last 10 positions
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)  # Fade trail
                color = (int(255 * alpha), 0, 0)
                cv2.line(vis_frame, trail_points[i-1], trail_points[i], color, 2)
        
        return vis_frame