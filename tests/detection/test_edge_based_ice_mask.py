import cv2
import numpy as np

def create_ice_mask_from_edges():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Creating ice mask using conservative edge detection...")
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Conservative edge detection (like your successful image)
    edges = cv2.Canny(blurred, 50, 150)
    
    # The ice surface is the large dark area surrounded by edges
    # Invert edges so ice areas become white
    inverted_edges = cv2.bitwise_not(edges)
    
    # Clean up with morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(inverted_edges, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours and get the largest one (should be ice surface)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create clean ice mask
        ice_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.fillPoly(ice_mask, [largest_contour], 255)
        
        # Optional: erode slightly to avoid board edges
        erode_kernel = np.ones((10, 10), np.uint8)
        ice_mask = cv2.erode(ice_mask, erode_kernel, iterations=1)
        
    else:
        print("No contours found")
        ice_mask = np.zeros(edges.shape, dtype=np.uint8)
    
    return edges, inverted_edges, cleaned, ice_mask, frame

def test_ice_mask_quality(ice_mask, frame):
    """Test the ice mask by applying it to player detection"""
    from src.detection.improved_detector import RealisticHockeyDetector
    
    # Detect players
    detector = RealisticHockeyDetector()
    all_people = detector.detect_all_people(frame, confidence_threshold=0.25)
    
    # Classify players using ice mask
    on_ice_players = []
    off_ice_players = []
    
    for person in all_people:
        x, y = person['position']
        
        # Check if player position is within ice mask
        if (0 <= x < ice_mask.shape[1] and 0 <= y < ice_mask.shape[0] and 
            ice_mask[y, x] > 0):
            on_ice_players.append(person)
        else:
            off_ice_players.append(person)
    
    return on_ice_players, off_ice_players

def main():
    edges, inverted_edges, cleaned, ice_mask, frame = create_ice_mask_from_edges()
    
    # Test the mask quality
    on_ice, off_ice = test_ice_mask_quality(ice_mask, frame)
    
    print(f"Ice mask created successfully")
    print(f"Players on ice: {len(on_ice)}")
    print(f"Players off ice: {len(off_ice)}")
    print(f"Ice area pixels: {np.sum(ice_mask > 0)}")
    print(f"Ice coverage: {np.sum(ice_mask > 0) / (ice_mask.shape[0] * ice_mask.shape[1]) * 100:.1f}%")
    
    # Create visualization
    debug_frame = frame.copy()
    
    # Show ice mask as green overlay
    ice_colored = np.zeros_like(frame)
    ice_colored[:, :, 1] = ice_mask  # Green channel
    debug_frame = cv2.addWeighted(debug_frame, 0.7, ice_colored, 0.3, 0)
    
    # Draw players with different colors
    for player in on_ice:
        x, y, w, h = player['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green for on ice
        
    for player in off_ice:
        x, y, w, h = player['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red for off ice
    
    # Save debug images
    cv2.imwrite('debug_original_edges.jpg', edges)
    cv2.imwrite('debug_inverted_edges.jpg', inverted_edges)
    cv2.imwrite('debug_cleaned_edges.jpg', cleaned)
    cv2.imwrite('debug_ice_mask_final.jpg', ice_mask)
    cv2.imwrite('debug_players_on_ice.jpg', debug_frame)
    
    print("\nDebug images saved:")
    print("- debug_original_edges.jpg (conservative edge detection)")
    print("- debug_inverted_edges.jpg (inverted edges)")
    print("- debug_cleaned_edges.jpg (after morphological cleanup)")
    print("- debug_ice_mask_final.jpg (final ice mask)")
    print("- debug_players_on_ice.jpg (players classified by ice mask)")
    
    return ice_mask

if __name__ == "__main__":
    main()