# src/detection/ice_detection.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

class IceDetector:
    def __init__(self):
        self.ice_mask = None
        self.ice_characteristics = None
        
    def detect_ice_surface(self, frame):
        """Simplified ice detection using only proven methods (color + texture)"""
        
        try:
            # Convert to needed color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            print("🧊 Running simplified ice detection (color + texture only)...")
            
            # Method 1: Color detection (primary method)
            ice_mask_color = self.detect_white_surfaces(hsv, lab)
            color_percentage = (np.sum(ice_mask_color > 0) / ice_mask_color.size) * 100
            print(f"   Color detection: {color_percentage:.1f}% of frame")
            
            # Method 2: Texture detection (refinement method)
            ice_mask_texture = self.detect_smooth_surfaces(gray)
            texture_percentage = (np.sum(ice_mask_texture > 0) / ice_mask_texture.size) * 100
            print(f"   Texture detection: {texture_percentage:.1f}% of frame")
            
            # NEW APPROACH: Intersection + Geometric Filtering + Expansion
            # Step 1: Find high-confidence ice (where color AND texture agree)
            definite_ice = cv2.bitwise_and(ice_mask_color, ice_mask_texture)
            definite_percentage = (np.sum(definite_ice > 0) / definite_ice.size) * 100
            print(f"   High-confidence ice (raw): {definite_percentage:.1f}% of frame")
            
            # Step 2: Filter seeds by geometric constraints (eliminate stands/boards)
            filtered_seeds = self.filter_seeds_by_ice_geometry(definite_ice, frame.shape)
            filtered_percentage = (np.sum(filtered_seeds > 0) / filtered_seeds.size) * 100
            print(f"   After geometric filtering: {filtered_percentage:.1f}% of frame")
            
            # Step 3: Expand from filtered seeds through white areas
            combined_mask = self.expand_ice_from_seed_areas(filtered_seeds, ice_mask_color)
            combined_percentage = (np.sum(combined_mask > 0) / combined_mask.size) * 100
            print(f"   After expansion: {combined_percentage:.1f}% of frame")
            
            # Clean up the final mask
            final_mask = self.clean_ice_mask(combined_mask)
            final_percentage = (np.sum(final_mask > 0) / final_mask.size) * 100
            print(f"   Final cleaned mask: {final_percentage:.1f}% of frame")
            
            self.ice_mask = final_mask
            return final_mask
            
        except Exception as e:
            print(f"❌ Ice surface detection failed: {e}")
            # Return a conservative fallback - assume center 40% of frame is ice
            fallback_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            h, w = frame.shape[:2]
            y1, y2 = int(h * 0.2), int(h * 0.8)
            x1, x2 = int(w * 0.1), int(w * 0.9)
            fallback_mask[y1:y2, x1:x2] = 255
            print("   Using conservative fallback mask (center 60% of frame)")
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

    def expand_ice_from_seed_areas(self, seed_mask, color_mask, max_expansion=80):
        """
        Flood fill expansion from high-confidence seed areas through white regions
        Respects boundaries and doesn't bridge across non-ice gaps
        
        Args:
            seed_mask: High-confidence ice areas (intersection of color + texture)
            color_mask: All detected white/light areas 
            max_expansion: Not used in flood fill (kept for interface compatibility)
        """
        if np.sum(seed_mask) == 0:
            print("   ⚠️  No high-confidence ice areas found, using color mask as fallback")
            return color_mask
        
        print("   🌊 Using flood fill expansion from seed areas...")
        
        # Use flood fill to expand from seeds through white areas only
        expanded_mask = self.flood_fill_from_seeds(seed_mask, color_mask)
        
        return expanded_mask

    def filter_seeds_by_ice_geometry(self, intersection_mask, frame_shape):
        """
        Filter intersection seeds to only include those in geometrically plausible ice locations
        Eliminates false positives in stands, boards, and other non-ice areas
        
        Args:
            intersection_mask: Raw intersection of color + texture detection
            frame_shape: (height, width, channels) of original frame
        """
        h, w = frame_shape[:2]
        
        # Create plausible ice region mask
        ice_region_mask = self.create_plausible_ice_region(w, h)
        
        # Filter intersection seeds to only include those in plausible locations
        filtered_seeds = cv2.bitwise_and(intersection_mask, ice_region_mask)
        
        # Statistics
        original_seeds = np.sum(intersection_mask > 0)
        filtered_seeds_count = np.sum(filtered_seeds > 0)
        eliminated_count = original_seeds - filtered_seeds_count
        
        print(f"   Geometric filter: kept {filtered_seeds_count}, eliminated {eliminated_count} false positives")
        
        return filtered_seeds
    
    def create_plausible_ice_region(self, width, height):
        """
        Create mask for where ice is geometrically expected to be in a hockey arena
        Based on typical camera angles and rink positioning
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Ice surface is typically:
        # - Centered horizontally and slightly lower vertically (camera angle)
        # - Takes up about 60-75% of frame width  
        # - Takes up about 40-60% of frame height
        # - Roughly elliptical shape due to perspective
        
        center_x = width // 2
        center_y = int(height * 0.55)  # Slightly below center due to camera angle
        
        # Ice dimensions (conservative to avoid cutting off real ice)
        ice_width = int(width * 0.75)   # 75% of frame width
        ice_height = int(height * 0.50)  # 50% of frame height
        
        # Create elliptical region (more realistic than rectangle)
        cv2.ellipse(mask, 
                   (center_x, center_y),           # Center point
                   (ice_width // 2, ice_height // 2),  # Radii
                   0,                              # Rotation angle
                   0, 360,                         # Start/end angles (full ellipse)
                   255,                            # Fill color
                   -1)                             # Filled
        
        # Optional: Add some padding around the ellipse for safety
        # Dilate slightly to ensure we don't cut off real ice edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def flood_fill_from_seeds(self, seed_mask, color_mask):
        """
        Flood fill from seed points through continuous white areas
        Stops at boundaries and doesn't bridge gaps
        """
        import cv2
        import numpy as np
        
        # Find all seed points (non-zero pixels in seed mask)
        seed_points = np.where(seed_mask > 0)
        num_seeds = len(seed_points[0])
        print(f"   Found {num_seeds} seed pixels for flood fill")
        
        if num_seeds == 0:
            return seed_mask
        
        # Create working masks
        h, w = seed_mask.shape
        result_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create a mask that defines floodable areas (white areas from color detection)
        # We need to invert this for OpenCV floodFill (0 = can fill, non-zero = barrier)
        flood_barrier = np.where(color_mask > 0, 0, 255).astype(np.uint8)
        
        # Track statistics
        regions_filled = 0
        total_pixels_filled = 0
        
        # Sample seed points (don't flood fill from every single pixel)
        # Take every Nth seed point to avoid redundant fills
        step = max(1, num_seeds // 50)  # Max 50 flood fill operations
        sampled_indices = range(0, num_seeds, step)
        
        print(f"   Sampling {len(sampled_indices)} seed points for flood fill")
        
        for i in sampled_indices:
            y, x = seed_points[0][i], seed_points[1][i]
            
            # Skip if this point is already filled
            if result_mask[y, x] > 0:
                continue
            
            # Create a temporary mask for this flood fill operation
            temp_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # Must be 2 pixels larger
            
            # Perform flood fill
            # Start from seed point, fill with value 255, through areas where barrier is 0
            pixels_filled = cv2.floodFill(
                result_mask,           # Image to fill
                temp_mask,            # Mask (must be 2 pixels larger)
                (x, y),               # Seed point
                255,                  # Fill value
                loDiff=(0,),          # Lower difference threshold
                upDiff=(0,),          # Upper difference threshold  
                flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )[0]
            
            # Apply barrier constraint: only keep fills that are in white areas
            before_constraint = np.sum(result_mask > 0)
            result_mask = cv2.bitwise_and(result_mask, color_mask)
            after_constraint = np.sum(result_mask > 0)
            
            if pixels_filled > 0:
                regions_filled += 1
                total_pixels_filled += after_constraint - (before_constraint - pixels_filled)
        
        print(f"   Flood fill completed: {regions_filled} regions, {total_pixels_filled} pixels")
        
        return result_mask
    
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