import cv2
import numpy as np

def detect_lines_by_edges():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    print("Detecting lines using edge detection only...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try different edge detection sensitivities
    edge_methods = [
        ("Conservative", cv2.Canny(blurred, 50, 150)),
        ("Moderate", cv2.Canny(blurred, 30, 100)), 
        ("Sensitive", cv2.Canny(blurred, 20, 60)),
        ("Very Sensitive", cv2.Canny(blurred, 10, 40))
    ]
    
    results = {}
    
    for method_name, edges in edge_methods:
        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=40, maxLineGap=20)
        
        # Filter for roughly vertical lines (blue line orientation)
        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                # More vertical than horizontal (45-135 degrees)
                if length > 30 and (45 <= angle <= 135):
                    vertical_lines.append(line)
        
        results[method_name] = {
            'edges': edges,
            'all_lines': len(lines) if lines is not None else 0,
            'vertical_lines': vertical_lines
        }
        
        print(f"{method_name}: {len(vertical_lines)} vertical lines from {results[method_name]['all_lines']} total")
    
    # Create visualization showing best method
    best_method = max(results.items(), key=lambda x: len(x[1]['vertical_lines']))
    method_name, method_data = best_method
    
    debug_frame = frame.copy()
    
    # Draw the vertical lines from best method
    for line in method_data['vertical_lines']:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    
    # Save debug images
    cv2.imwrite('debug_edge_lines.jpg', debug_frame)
    cv2.imwrite('debug_best_edges.jpg', method_data['edges'])
    
    # Save all edge detection results for comparison
    for i, (name, data) in enumerate(results.items()):
        cv2.imwrite(f'debug_edges_{i}_{name.lower().replace(" ", "_")}.jpg', data['edges'])
    
    print(f"\nBest method: {method_name} with {len(method_data['vertical_lines'])} vertical lines")
    print("Debug images saved:")
    print("- debug_edge_lines.jpg (detected vertical lines)")
    print("- debug_best_edges.jpg (edge detection used)")
    print("- debug_edges_*.jpg (all edge detection methods)")

if __name__ == "__main__":
    detect_lines_by_edges()