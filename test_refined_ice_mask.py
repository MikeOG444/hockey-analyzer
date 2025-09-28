import cv2
import numpy as np
import os

def create_refined_ice_mask():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    # Create debug folder
    debug_folder = "debug_test_refined_ice_mask_images"
    os.makedirs(debug_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Creating refined ice mask with careful edge preservation...")
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Conservative edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Instead of inverting and filling, let's find the ice area differently
    # Use the fact that ice should be a large, roughly rectangular area in the center
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask by excluding areas with too many edges (non-ice areas)
    height, width = edges.shape
    
    # Start with full image
    potential_ice = np.ones((height, width), dtype=np.uint8) * 255
    
    # Remove areas with dense edges (boards, crowd, etc.)
    # Dilate edges slightly to create exclusion zones
    thick_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Subtract edge areas from potential ice
    ice_candidate = cv2.bitwise_and(potential_ice, cv2.bitwise_not(thick_edges))
    
    # Find the largest connected component (should be ice surface)
    contours, _ = cv2.findContours(ice_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create final ice mask
        ice_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(ice_mask, [largest_contour], 255)
        
        # Smooth the mask slightly
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Erode slightly to stay away from board edges
        ice_mask = cv2.erode(ice_mask, np.ones((8, 8), np.uint8), iterations=1)
        
    else:
        ice_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Test the mask
    from src.detection.improved_detector import RealisticHockeyDetector
    detector = RealisticHockeyDetector()
    all_people = detector.detect_all_people(frame, confidence_threshold=0.25)
    
    on_ice_players = []
    off_ice_players = []
    
    for person in all_people:
        x, y = person['position']
        if (0 <= x < width and 0 <= y < height and ice_mask[y, x] > 0):
            on_ice_players.append(person)
        else:
            off_ice_players.append(person)
    
    # Create visualization
    debug_frame = frame.copy()
    
    # Show ice mask as green overlay
    ice_colored = np.zeros_like(frame)
    ice_colored[:, :, 1] = ice_mask
    debug_frame = cv2.addWeighted(debug_frame, 0.7, ice_colored, 0.3, 0)
    
    # Draw players
    for player in on_ice_players:
        x, y, w, h = player['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    for player in off_ice_players:
        x, y, w, h = player['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    # Save to organized debug folder
    cv2.imwrite(f'{debug_folder}/01_original_edges.jpg', edges)
    cv2.imwrite(f'{debug_folder}/02_thick_edges.jpg', thick_edges)
    cv2.imwrite(f'{debug_folder}/03_ice_candidate.jpg', ice_candidate)
    cv2.imwrite(f'{debug_folder}/04_final_ice_mask.jpg', ice_mask)
    cv2.imwrite(f'{debug_folder}/05_players_classified.jpg', debug_frame)
    
    ice_coverage = np.sum(ice_mask > 0) / (height * width) * 100
    
    print(f"Refined ice mask results:")
    print(f"Players on ice: {len(on_ice_players)}")
    print(f"Players off ice: {len(off_ice_players)}")
    print(f"Ice coverage: {ice_coverage:.1f}%")
    print(f"Debug images saved to: {debug_folder}/")

if __name__ == "__main__":
    create_refined_ice_mask()