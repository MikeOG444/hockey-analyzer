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
    Realistic hockey puck tracker that accounts for:
    - Only 1 puck exists on ice
    - Puck is often hidden/obscured
    - Puck is very small and fast
    - Detection should be conservative to avoid false positives
    """

    def __init__(self, ice_detector=None):
        self.ice_detector = ice_detector
        self.puck_history = deque(maxlen=30)  # Track last 30 detections

        # Realistic puck constraints
        self.min_puck_area = 8    # Very small minimum
        self.max_puck_area = 50   # Much smaller maximum (puck is tiny)
        self.confidence_threshold = 0.6  # Higher threshold to reduce false positives

        # Detection rate expectations (realistic for hockey)
        self.expected_detection_rate = 0.3  # Only detect puck ~30% of the time
        self.frames_since_detection = 0
        self.max_frames_without_detection = 90  # 3 seconds at 30fps

        # Temporal tracking
        self.max_velocity = 200  # Max pixels per frame for very fast puck
        self.lost_puck_frames = 0

    def detect_puck_candidates(self, frame: np.ndarray, players: List[Dict]) -> List[PuckCandidate]:
        """
        Detect potential puck locations - returns list of candidates for compatibility
        
        This method maintains compatibility with the GameAnalyzer interface
        """
        puck = self.detect_puck(frame, players)
        return [puck] if puck is not None else []

    def track_puck(self, candidates: List[PuckCandidate]) -> Optional[PuckCandidate]:
        """
        Track puck across frames using temporal consistency
        
        Maintains compatibility with GameAnalyzer interface
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
                    min_distance = distance
                    
            # If no good temporal match, take highest confidence
            if best_candidate is None and candidates:
                best_candidate = max(candidates, key=lambda c: c.confidence)
        else:
            # No history, take highest confidence candidate
            best_candidate = max(candidates, key=lambda c: c.confidence) if candidates else None
        
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

    def detect_puck(self, frame: np.ndarray, players: List[Dict]) -> Optional[PuckCandidate]:
        """
        Main puck detection method - returns single best puck or None

        Hockey reality:
        - Puck is often hidden behind players, sticks, or boards
        - Very small and fast-moving
        - Better to miss detection than have false positives
        """
        self.frames_since_detection += 1

        # Step 1: Find potential puck candidates (very conservative)
        candidates = self._find_conservative_candidates(frame)

        # Step 2: Filter by ice surface
        if self.ice_detector and self.ice_detector.ice_mask is not None:
            candidates = self._filter_by_ice_surface(candidates)

        # Step 3: Apply hockey constraints
        candidates = self._apply_hockey_constraints(candidates)

        # Step 4: Use temporal tracking to select best candidate
        best_puck = self._select_best_puck(candidates)

        # Step 5: Update tracking state
        if best_puck:
            self._update_tracking_history(best_puck)
            self.frames_since_detection = 0
            self.lost_puck_frames = 0
        else:
            self.lost_puck_frames += 1
            # Try prediction if we recently had a puck
            if self.lost_puck_frames < 15:  # Within 0.5 seconds
                best_puck = self._predict_puck_position()

        return best_puck

    def _find_conservative_candidates(self, frame: np.ndarray) -> List[PuckCandidate]:
        """Find puck candidates using very conservative detection"""
        candidates = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Dark circular objects (most reliable for pucks)
        # Use very strict thresholding for dark objects
        _, dark_mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)  # Very dark threshold

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of dark objects
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Very strict size constraints
            if self.min_puck_area <= area <= self.max_puck_area:
                # Check shape - puck should be roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)

                    # High circularity requirement
                    if circularity > 0.6:
                        # Get centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # Additional validation: check local contrast
                            confidence = self._calculate_puck_confidence(gray, (cx, cy), area, circularity)

                            if confidence > self.confidence_threshold:
                                candidates.append(PuckCandidate(
                                    position=(cx, cy),
                                    confidence=confidence,
                                    size=area,
                                    timestamp=cv2.getTickCount(),
                                    detection_method="conservative_dark"
                                ))

        # Method 2: HoughCircles (very conservative parameters)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,  # Increased minDist
            param1=80, param2=35,  # Stricter parameters
            minRadius=3, maxRadius=8  # Small radius for distant puck
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Validate this is actually a dark object
                roi = gray[max(0, y-r*2):min(gray.shape[0], y+r*2),
                          max(0, x-r*2):min(gray.shape[1], x+r*2)]

                if roi.size > 0:
                    center_intensity = gray[y, x] if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1] else 255
                    avg_surrounding = np.mean(roi)

                    # Puck must be significantly darker than surroundings
                    if center_intensity < 60 and (avg_surrounding - center_intensity) > 30:
                        area = math.pi * r * r
                        confidence = min(((avg_surrounding - center_intensity) / 100.0), 1.0)

                        if confidence > self.confidence_threshold:
                            candidates.append(PuckCandidate(
                                position=(x, y),
                                confidence=confidence,
                                size=area,
                                timestamp=cv2.getTickCount(),
                                detection_method="circle"
                            ))

        return candidates

    def _calculate_puck_confidence(self, gray: np.ndarray, position: Tuple[int, int],
                                 area: float, circularity: float) -> float:
        """Calculate confidence that this is actually a puck"""
        x, y = position

        if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
            return 0.0

        # Factor 1: Local contrast (puck should be darker than surroundings)
        radius = int(math.sqrt(area / math.pi)) + 2
        x1, y1 = max(0, x-radius*2), max(0, y-radius*2)
        x2, y2 = min(gray.shape[1], x+radius*2), min(gray.shape[0], y+radius*2)

        center_region = gray[max(0, y-radius//2):min(gray.shape[0], y+radius//2+1),
                           max(0, x-radius//2):min(gray.shape[1], x+radius//2+1)]
        surrounding_region = gray[y1:y2, x1:x2]

        if center_region.size == 0 or surrounding_region.size == 0:
            return 0.0

        center_intensity = np.mean(center_region)
        surrounding_intensity = np.mean(surrounding_region)
        contrast = surrounding_intensity - center_intensity

        contrast_score = min(contrast / 80.0, 1.0) if contrast > 0 else 0.0

        # Factor 2: Size appropriateness (prefer smaller objects)
        ideal_area = 15  # Ideal puck size
        size_score = max(0, 1 - abs(area - ideal_area) / ideal_area)

        # Factor 3: Shape circularity
        shape_score = circularity

        # Factor 4: Darkness requirement (puck should be very dark)
        darkness_score = max(0, 1 - center_intensity / 60.0)

        # Combined score with weights
        final_confidence = (
            contrast_score * 0.3 +
            size_score * 0.2 +
            shape_score * 0.3 +
            darkness_score * 0.2
        )

        return final_confidence

    def _filter_by_ice_surface(self, candidates: List[PuckCandidate]) -> List[PuckCandidate]:
        """Filter candidates to only those on the ice surface"""
        filtered = []
        for candidate in candidates:
            if self.ice_detector.is_on_ice(candidate.position):
                filtered.append(candidate)
        return filtered

    def _apply_hockey_constraints(self, candidates: List[PuckCandidate]) -> List[PuckCandidate]:
        """Apply hockey-specific constraints"""
        if not candidates:
            return []

        # Constraint 1: Only keep highest confidence candidates
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        # Constraint 2: Remove candidates that are too close together
        # (there's only 1 puck, so nearby detections are duplicates)
        filtered = []
        min_distance = 25  # Minimum distance between detections

        for candidate in candidates:
            too_close = False
            for existing in filtered:
                distance = math.sqrt(
                    (candidate.position[0] - existing.position[0])**2 +
                    (candidate.position[1] - existing.position[1])**2
                )
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                filtered.append(candidate)

        # Constraint 3: Limit to maximum 3 candidates (conservative)
        return filtered[:3]

    def _select_best_puck(self, candidates: List[PuckCandidate]) -> Optional[PuckCandidate]:
        """Select the single best puck candidate using temporal tracking"""
        if not candidates:
            return None

        # If we have no history, take the highest confidence candidate
        if not self.puck_history:
            return candidates[0] if candidates[0].confidence > self.confidence_threshold else None

        # Use temporal consistency to select best candidate
        last_position = self.puck_history[-1].position
        best_candidate = None
        best_score = 0

        for candidate in candidates:
            # Calculate distance from last known position
            distance = math.sqrt(
                (candidate.position[0] - last_position[0])**2 +
                (candidate.position[1] - last_position[1])**2
            )

            # Temporal consistency score (closer to last position is better)
            if distance < self.max_velocity:
                temporal_score = 1.0 - (distance / self.max_velocity)
            else:
                temporal_score = 0.0  # Too far, likely false positive

            # Combined score
            total_score = candidate.confidence * 0.6 + temporal_score * 0.4

            if total_score > best_score and total_score > 0.5:  # Minimum combined threshold
                best_score = total_score
                best_candidate = candidate

        # If no good temporal match, only accept very high confidence candidates
        if best_candidate is None and candidates:
            if candidates[0].confidence > 0.8:  # Very high confidence required
                best_candidate = candidates[0]

        return best_candidate

    def _update_tracking_history(self, puck: PuckCandidate):
        """Update puck tracking history with velocity calculation"""
        if self.puck_history:
            # Calculate velocity from last position
            last_pos = self.puck_history[-1].position
            dt = 1.0 / 30.0  # Assume 30fps
            vx = (puck.position[0] - last_pos[0]) / dt
            vy = (puck.position[1] - last_pos[1]) / dt
            puck.velocity = (vx, vy)

        self.puck_history.append(puck)

    def _predict_puck_position(self) -> Optional[PuckCandidate]:
        """Predict puck position when not detected (puck often hidden)"""
        if len(self.puck_history) < 2:
            return None

        last_puck = self.puck_history[-1]
        if not last_puck.velocity:
            return None

        # Predict position based on last known velocity
        vx, vy = last_puck.velocity
        dt = self.lost_puck_frames / 30.0  # Time since last detection

        predicted_x = int(last_puck.position[0] + vx * dt)
        predicted_y = int(last_puck.position[1] + vy * dt)

        # Only predict if within reasonable bounds
        if 0 <= predicted_x < 1920 and 0 <= predicted_y < 1080:  # Assume HD video
            return PuckCandidate(
                position=(predicted_x, predicted_y),
                confidence=max(0.2, 0.5 - dt * 0.1),  # Decreasing confidence over time
                size=last_puck.size,
                timestamp=cv2.getTickCount(),
                detection_method="prediction",
                velocity=last_puck.velocity
            )

        return None

    def get_detection_stats(self) -> Dict:
        """Get realistic detection statistics"""
        total_frames = self.frames_since_detection + len([p for p in self.puck_history if p.detection_method != "prediction"])
        detections = len([p for p in self.puck_history if p.detection_method != "prediction"])

        return {
            'total_frames_processed': total_frames,
            'puck_detections': detections,
            'detection_rate': detections / max(total_frames, 1),
            'frames_since_last_detection': self.frames_since_detection,
            'is_puck_currently_tracked': len(self.puck_history) > 0 and self.frames_since_detection < 10
        }

    def visualize_tracking(self, frame: np.ndarray, puck: Optional[PuckCandidate]) -> np.ndarray:
        """Draw realistic puck tracking visualization"""
        vis_frame = frame.copy()

        # Draw tracked puck (only if confident)
        if puck and puck.confidence > 0.4:
            x, y = puck.position

            # Color based on detection method
            if puck.detection_method == "prediction":
                color = (0, 255, 255)  # Yellow for prediction
                thickness = 1
            else:
                color = (0, 0, 255)  # Red for actual detection
                thickness = 2

            # Draw puck
            radius = max(4, int(math.sqrt(puck.size)))
            cv2.circle(vis_frame, (x, y), radius, color, thickness)

            # Add confidence and method
            text = f"PUCK: {puck.confidence:.2f} ({puck.detection_method})"
            cv2.putText(vis_frame, text, (x-30, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw velocity vector if available
            if puck.velocity and puck.detection_method != "prediction":
                vx, vy = puck.velocity
                scale = 0.05  # Scale for visualization
                end_x = int(x + vx * scale)
                end_y = int(y + vy * scale)
                cv2.arrowedLine(vis_frame, (x, y), (end_x, end_y), color, 2)

        # Draw puck trail (last few positions)
        if len(self.puck_history) > 1:
            trail_points = [(p.position[0], p.position[1]) for p in list(self.puck_history)[-8:]]
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                color = (int(100 * alpha), 0, int(255 * alpha))
                cv2.line(vis_frame, trail_points[i-1], trail_points[i], color, 1)

        # Add detection statistics
        stats = self.get_detection_stats()
        info_text = [
            f"Puck Detection Rate: {stats['detection_rate']*100:.1f}%",
            f"Frames since detection: {stats['frames_since_last_detection']}",
            f"Status: {'TRACKING' if stats['is_puck_currently_tracked'] else 'SEARCHING'}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, vis_frame.shape[0] - 60 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_frame