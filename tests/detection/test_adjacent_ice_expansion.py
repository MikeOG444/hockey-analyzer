import cv2
import numpy as np
import os

def create_expanded_ice_mask():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    # Create debug folder
    debug_folder = "debug_test_adjacent_ice_expansion_images"
    os.makedirs(debug_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Creating ice mask with adjacent area expansion...")
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Start with our previous working approach
    edges = cv2.Canny(blurred, 50, 150)
    potential_ice = np.ones(edges.shape, dtype=np.uint8) * 255
    thick_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    ice_candidate = cv2.bitwise_and(potential_ice, cv2.bitwise_not(thick_edges))
    
    # Get initial ice mask
    contours, _ = cv2.findContours(ice_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        initial_ice_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.fillPoly(initial_ice_mask, [largest_contour], 255)
    else:
        initial_ice_mask = np.zeros(edges.shape, dtype=np.uint8)
    
    # Now expand to include similar adjacent areas
    height, width = edges.shape
    
    # Create a similarity mask based on brightness/texture
    # Areas similar to the detected ice
    ice_region_brightness = cv2.mean(gray, mask=initial_ice_mask)[0]
    
    # Find areas with similar brightness
    brightness_diff = np.abs(gray.astype(np.float32) - ice_region_brightness)
    similar_brightness = (brightness_diff < 30).astype(np.uint8) * 255
    
    # Combine initial ice with similar adjacent areas
    # Use morphological dilation to "grow" the ice area into similar regions
    expanded_ice = initial_ice_mask.copy()
    
    # Iteratively expand into similar areas
    for i in range(3):  # Try expanding 3 times
        # Dilate current ice mask
        grown = cv2.dilate(expanded_ice, np.ones((10, 10), np.uint8), iterations=1)
        
        # Only keep expanded areas that are similar in brightness
        expansion_candidate = cv2.bitwise_and(grown, similar_brightness)
        
        # Don't expand across major edges (keep strong boundaries)
        strong_edges = cv2.Canny(blurred, 80, 200)  # Stronger edges than before
        expansion_candidate = cv2.bitwise_and(expansion_candidate, cv2.bitwise_not(strong_edges))
        
        # Update expanded ice
        expanded_ice = cv2.bitwise_or(expanded_ice, expansion_candidate)
    
    # Clean up the final mask
    final_ice_mask = cv2.morphologyEx(expanded_ice, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    final_ice_mask = cv2.erode(final_ice_mask, np.ones((3, 3), np.uint8), iterations=1)
    
    # Test the mask
    from src.detection.improved_detector import RealisticHockeyDetector
    detector = RealisticHockeyDetector()
    all_people = detector.detect_all_people(frame, confidence_threshold=0.25)
    
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
    cv2.imwrite(f'{debug_folder}/01_initial_ice_mask.jpg', initial_ice_mask)
    cv2.imwrite(f'{debug_folder}/02_similar_brightness.jpg', similar_brightness)
    cv2.imwrite(f'{debug_folder}/03_expanded_ice.jpg', expanded_ice)
    cv2.imwrite(f'{debug_folder}/04_final_ice_mask.jpg', final_ice_mask)
    cv2.imwrite(f'{debug_folder}/05_players_classified.jpg', debug_frame)
    
    ice_coverage = np.sum(final_ice_mask > 0) / (height * width) * 100
    
    print(f"Expanded ice mask results:")
    print(f"Players on ice: {len(on_ice_players)}")
    print(f"Players off ice: {len(off_ice_players)}")
    print(f"Ice coverage: {ice_coverage:.1f}%")
    print(f"Average ice brightness: {ice_region_brightness:.1f}")
    print(f"Debug images saved to: {debug_folder}/")

if __name__ == "__main__":
    create_expanded_ice_mask()