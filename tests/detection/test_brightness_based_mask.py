import cv2
import numpy as np
import os

def create_brightness_based_mask():
    video_path = "data/test_videos/trimpano.mp4"
    
    # Create debug folder
    debug_folder = "debug_test_brightness_based_mask_images"
    os.makedirs(debug_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Creating ice mask based primarily on brightness similarity...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Get initial ice estimation using our previous method
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
    
    # Calculate ice brightness characteristics from seed area
    ice_region_brightness = cv2.mean(gray, mask=seed_ice_mask)[0]
    
    # Create similarity mask (this was working well)
    brightness_diff = np.abs(gray.astype(np.float32) - ice_region_brightness)
    similar_brightness = (brightness_diff < 25).astype(np.uint8) * 255  # Slightly tighter threshold
    
    # Clean up the similarity mask minimally
    # Remove small isolated areas
    similar_cleaned = cv2.morphologyEx(similar_brightness, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Fill small holes
    similar_cleaned = cv2.morphologyEx(similar_cleaned, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # Only exclude areas that are clearly separated by major structural elements
    # Use only very strong edges to create exclusions
    strong_edges = cv2.Canny(blurred, 100, 250)  # Much stronger threshold
    strong_thick = cv2.dilate(strong_edges, np.ones((5, 5), np.uint8), iterations=1)
    
    # Apply structural exclusions only
    final_ice_mask = cv2.bitwise_and(similar_cleaned, cv2.bitwise_not(strong_thick))
    
    # Get the largest connected component
    contours, _ = cv2.findContours(final_ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Keep only reasonably large areas (filter out small noise)
        large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        if large_contours:
            # Create mask from all large ice areas
            final_ice_mask = np.zeros(edges.shape, dtype=np.uint8)
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
    cv2.imwrite(f'{debug_folder}/02_similar_brightness.jpg', similar_brightness)
    cv2.imwrite(f'{debug_folder}/03_strong_edges.jpg', strong_edges)
    cv2.imwrite(f'{debug_folder}/04_final_ice_mask.jpg', final_ice_mask)
    cv2.imwrite(f'{debug_folder}/05_players_classified.jpg', debug_frame)
    
    ice_coverage = np.sum(final_ice_mask > 0) / (height * width) * 100
    
    print(f"Brightness-based ice mask results:")
    print(f"Players on ice: {len(on_ice_players)}")
    print(f"Players off ice: {len(off_ice_players)}")
    print(f"Ice coverage: {ice_coverage:.1f}%")
    print(f"Ice brightness target: {ice_region_brightness:.1f}")
    print(f"Debug images saved to: {debug_folder}/")

if __name__ == "__main__":
    create_brightness_based_mask()