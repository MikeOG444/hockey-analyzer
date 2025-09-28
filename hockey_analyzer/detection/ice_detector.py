# src/detection/ice_detection.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

class IceDetector:
    def __init__(self):
        self.ice_mask = None
        self.ice_characteristics = None
        
    def detect_ice_surface(self, frame):
        """Detect ice surface using color, texture, and geometric features"""
        
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Method 1: White/light color detection
            ice_mask_color = self.detect_white_surfaces(hsv, lab)
            
            # Method 2: Texture analysis - ice is smoother than boards/crowd
            ice_mask_texture = self.detect_smooth_surfaces(gray)
            
            # Method 3: Geometric constraints - ice is typically the largest flat area
            ice_mask_geometric = self.detect_large_flat_areas(gray)
            
            # Method 4: Use ice markings as reference points (with error handling)
            try:
                ice_mask_markings = self.detect_by_ice_markings(frame)
            except Exception as e:
                print(f"Warning: Ice marking detection failed: {e}")
                ice_mask_markings = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Combine all methods
            combined_mask = self.combine_ice_masks([
                ice_mask_color,
                ice_mask_texture, 
                ice_mask_geometric,
                ice_mask_markings
            ])
            
            # Clean up the mask
            final_mask = self.clean_ice_mask(combined_mask)
            
            self.ice_mask = final_mask
            return final_mask
            
        except Exception as e:
            print(f"Warning: Ice surface detection failed: {e}")
            # Return a basic mask covering most of the frame as fallback
            fallback_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            self.ice_mask = fallback_mask
            return fallback_mask
    
    def detect_white_surfaces(self, hsv, lab):
        """Detect white/light colored areas"""
        
        # HSV thresholds for white/light surfaces
        # White has low saturation and high value
        lower_white = np.array([0, 0, 180])      # Low saturation, high brightness
        upper_white = np.array([180, 30, 255])   # Any hue, low sat, high brightness
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Also check in LAB space - L channel for lightness
        l_channel = lab[:, :, 0]
        light_mask = cv2.threshold(l_channel, 160, 255, cv2.THRESH_BINARY)[1]
        
        # Combine both approaches
        combined_white = cv2.bitwise_or(white_mask, light_mask)
        
        return combined_white
    
    def detect_smooth_surfaces(self, gray):
        """Ice surface should be smoother than boards/crowd"""
        
        # Calculate texture using standard deviation in local neighborhoods
        kernel = np.ones((15, 15), np.float32) / 225
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        texture = np.sqrt(sqr_mean - mean**2)
        
        # Ice should have low texture variance
        smooth_threshold = np.percentile(texture, 30)  # Bottom 30% for smoothness
        smooth_mask = (texture < smooth_threshold).astype(np.uint8) * 255
        
        return smooth_mask
    
    def detect_large_flat_areas(self, gray):
        """Ice is typically the largest contiguous flat area"""
        
        # Edge detection to find boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for largest area (likely ice surface)
        if contours:
            # Find largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask from largest contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            return mask
        
        return np.zeros(gray.shape, dtype=np.uint8)
    
    def detect_by_ice_markings(self, frame):
        """Use ice markings to infer ice surface boundaries"""
        
        # Detect red center line
        red_lines = self.detect_colored_lines(frame, 'red')
        
        # Detect blue lines  
        blue_lines = self.detect_colored_lines(frame, 'blue')
        
        # If we find ice markings, expand around them
        marking_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if red_lines is not None or blue_lines is not None:
            # Create mask around detected lines
            if red_lines is not None and len(red_lines) > 0:
                for line in red_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(marking_mask, (x1, y1), (x2, y2), 255, thickness=50)
            if blue_lines is not None and len(blue_lines) > 0:
                for line in blue_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(marking_mask, (x1, y1), (x2, y2), 255, thickness=50)
                
            # Expand the marking areas to cover ice surface
            kernel = np.ones((50, 50), np.uint8)
            expanded_mask = cv2.dilate(marking_mask, kernel, iterations=3)
            
            return expanded_mask
            
        return marking_mask
    
    def detect_colored_lines(self, frame, color):
        """Detect red or blue lines on ice"""
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if color == 'red':
            # Red color ranges (handle wrap-around)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50]) 
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
        elif color == 'blue':
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
        # Find lines using HoughLines
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        # Return lines if found, otherwise None
        return lines if lines is not None and len(lines) > 0 else None
    
    def combine_ice_masks(self, masks):
        """Combine multiple ice detection methods"""
        
        # Start with all zeros
        combined = np.zeros(masks[0].shape, dtype=np.uint8)
        
        # Weight different methods
        weights = [0.3, 0.2, 0.3, 0.2]  # color, texture, geometric, markings
        
        for mask, weight in zip(masks, weights):
            if mask is not None:
                # Normalize mask to 0-1
                normalized = mask.astype(np.float32) / 255.0
                combined = combined.astype(np.float32) + (normalized * weight * 255)
        
        # Threshold the combined result
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        final_mask = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)[1]
        
        return final_mask
    
    def clean_ice_mask(self, mask):
        """Clean up the ice mask using morphological operations"""
        
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        kernel_large = np.ones((15, 15), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
        
        return cleaned
    
    def is_on_ice(self, position):
        """Check if a position is on the ice surface"""
        if self.ice_mask is None:
            return True  # Default to True if no ice mask available
            
        x, y = position
        if 0 <= x < self.ice_mask.shape[1] and 0 <= y < self.ice_mask.shape[0]:
            return self.ice_mask[y, x] > 0
        
        return False

# Test the ice detection
if __name__ == "__main__":
    detector = IceDetector()
    print("Ice surface detector ready!")