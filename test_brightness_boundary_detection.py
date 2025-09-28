import cv2
import numpy as np
import os

def create_brightness_boundary_mask():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    # Create debug folder
    debug_folder = "debug_test_brightness_boundary_detection_images"
    os.makedirs(debug_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Creating ice mask using brightness boundary detection...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Get seed ice area (main visible ice)
    edges = cv2.Canny(blurred, 50, 150)
    potential_ice = np.ones(edges.shape, dtype=np.uint8) * 255
    thick_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    ice_candidate = cv2.bitwise_and(potential_ice, cv2.bitwise_not(thick_edges))
    
    contours, _ = cv2.findContours(ice_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        seed_ice_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.fillPoly(seed_ice_mask, [largest_contour], 255)
    else:
        seed_ice_mask = np.zeros(edges.shape, dtype=np.uint8)
    
    # Get ice brightness characteristics
    ice_brightness = cv2.mean(gray, mask=seed_ice_mask)[0]
    
    # Create expanded brightness mask (more permissive to catch glass areas)
    brightness_diff = np.abs(gray.astype(np.float32) - ice_brightness)
    ice_like_areas = (brightness_diff < 35).astype(np.uint8) * 255  # More permissive
    
    # Identify the dark camera-side area
    # The camera side should be much darker than ice
    dark_threshold = ice_brightness - 50  # Much darker than ice
    dark_areas = (gray < dark_threshold).astype(np.uint8) * 255
    
    # Find the boundary between ice-like brightness and dark areas
    # Dilate the dark areas slightly to create a boundary zone
    dark_boundary = cv2.dilate(dark_areas, np.ones((10, 10), np.uint8), iterations=1)
    
    # The ice mask should include ice-like areas but exclude the dark boundary
    ice_mask_candidate = cv2.bitwise_and(ice_like_areas, cv2.bitwise_not(dark_boundary))
    
    # Clean up the result
    ice_mask_candidate = cv2.morphologyEx(ice_mask_candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ice_mask_candidate = cv2.morphologyEx(ice_mask_candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # Get largest connected components (keep multiple ice areas if they exist)
    contours, _ = cv2.findContours(ice_mask_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_ice_mask = np.zeros(edges.shape, dtype=np.uint8)
    
    if contours:
        # Keep areas larger than a minimum size
        large_contours = [c for c in contours if cv2.contourArea(c) > 500]
        if large_contours:
            cv2.fillPoly(final_ice_mask, large_contours, 255)
    
    # Test the mask
    from src.detection.improved_detector import RealisticHockeyDetector
    detector = RealisticHockeyDetector()
    all_people = detector.detect_all_people(frame, confidence_threshold=0.25)
    
    height, width = final_ice_mask.shape
    on_ice_players = []
    off_ice_players = []
    
    for person in all_people:
        x, y = person['position']
        if (0 <= x < width and 0 <= y < height and final_ice_mask[y, x] > 0):
            on_ice_players.append(person)
        else:
            off_ice_players.append(person)
    
    # Create visualization
    debug_frame = frame.copy()
    
    # Show ice mask as green overlay
    ice_colored = np.zeros_like(frame)
    ice_colored[:, :, 1] = final_ice_mask
    debug_frame = cv2.addWeighted(debug_frame, 0.7, ice_colored, 0.3, 0)
    
    # Draw players
    for player in on_ice_players:
        x, y, w, h = player['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    for player in off_ice_players:
        x, y, w, h = player['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    # Save debug images
    cv2.imwrite(f'{debug_folder}/01_seed_ice_mask.jpg', seed_ice_mask)
    cv2.imwrite(f'{debug_folder}/02_ice_like_areas.jpg', ice_like_areas)
    cv2.imwrite(f'{debug_folder}/03_dark_areas.jpg', dark_areas)
    cv2.imwrite(f'{debug_folder}/04_dark_boundary.jpg', dark_boundary)
    cv2.imwrite(f'{debug_folder}/05_final_ice_mask.jpg', final_ice_mask)
    cv2.imwrite(f'{debug_folder}/06_players_classified.jpg', debug_frame)
    
    ice_coverage = np.sum(final_ice_mask > 0) / (height * width) * 100
    
    print(f"Brightness boundary ice mask results:")
    print(f"Players on ice: {len(on_ice_players)}")
    print(f"Players off ice: {len(off_ice_players)}")
    print(f"Ice coverage: {ice_coverage:.1f}%")
    print(f"Ice brightness: {ice_brightness:.1f}")
    print(f"Dark threshold: {dark_threshold:.1f}")
    print(f"Debug images saved to: {debug_folder}/")

if __name__ == "__main__":
    create_brightness_boundary_mask()