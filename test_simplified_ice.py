#!/usr/bin/env python3
"""
Simplified Ice Detection Debug Script

This script will:
1. Test the new simplified ice detection (color + texture only)
2. Compare it with the old 4-method approach
3. Show you the difference in results
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_simplified_ice_detection(video_path, frame_number=0):
    """Test the new simplified ice detection approach"""
    
    try:
        import cv2
        import numpy as np
        from hockey_analyzer.detection.ice_detector import IceDetector
        
        print(f"🏒 Simplified Ice Detection Test")
        print(f"Video: {video_path}")
        print(f"Frame: {frame_number}")
        print("=" * 60)
        
        # Load video and frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Error: Could not read frame {frame_number}")
            return False
        
        print(f"✅ Loaded frame {frame_number} ({frame.shape[1]}x{frame.shape[0]})")
        
        # Test simplified ice detection
        ice_detector = IceDetector()
        
        print("\\n🔍 Testing NEW simplified ice detection...")
        simplified_mask = ice_detector.detect_ice_surface(frame)
        
        # Also test individual components for comparison
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("\n📊 Breaking down the intersection + expansion approach:")
        color_mask = ice_detector.detect_white_surfaces(hsv, lab)
        texture_mask = ice_detector.detect_smooth_surfaces(gray)
        
        # Show the intermediate steps
        intersection_mask = cv2.bitwise_and(color_mask, texture_mask)
        intersection_percentage = (np.sum(intersection_mask > 0) / intersection_mask.size) * 100
        print(f"   Intersection (seeds): {intersection_percentage:.1f}% of frame")
        
        # Test geometric filtering
        if intersection_percentage > 0:
            # Test geometric filtering step
            filtered_seeds = ice_detector.filter_seeds_by_ice_geometry(intersection_mask, frame.shape)
            filtered_percentage = (np.sum(filtered_seeds > 0) / filtered_seeds.size) * 100
            print(f"   After geometric filtering: {filtered_percentage:.1f}% of frame")
            
            # Test flood fill from filtered seeds
            flood_filled = ice_detector.flood_fill_from_seeds(filtered_seeds, color_mask)
            flood_percentage = (np.sum(flood_filled > 0) / flood_filled.size) * 100
            print(f"   After flood fill: {flood_percentage:.1f}% of frame")
        
        # Create comparison visualization
        debug_image = create_simplified_debug_visualization(
            frame, color_mask, texture_mask, simplified_mask
        )
        
        # Save debug image
        debug_path = f"debug_simplified_ice_frame_{frame_number}.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"\\n✅ Debug visualization saved: {debug_path}")
        
        # Analyze results
        analyze_simplified_results(simplified_mask, frame.shape[:2])
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_simplified_debug_visualization(frame, color_mask, texture_mask, final_mask):
    """Create debug visualization showing intersection + geometric filtering + flood fill"""
    
    import cv2
    import numpy as np
    from hockey_analyzer.detection.ice_detector import IceDetector
    
    # Resize for display
    display_height = 200
    aspect_ratio = frame.shape[1] / frame.shape[0]
    display_width = int(display_height * aspect_ratio)
    
    # Resize all images
    frame_small = cv2.resize(frame, (display_width, display_height))
    color_small = cv2.resize(color_mask, (display_width, display_height))
    texture_small = cv2.resize(texture_mask, (display_width, display_height))
    final_small = cv2.resize(final_mask, (display_width, display_height))
    
    # Calculate intermediate steps for new approach
    ice_detector = IceDetector()
    
    # Step 1: Largest connected component
    largest_white = ice_detector.find_largest_connected_component(color_small)
    
    # Step 2: Lines within largest area
    lines_within = cv2.bitwise_and(texture_small, largest_white)
    
    # Step 3: Boundary mask
    boundary_mask = ice_detector.create_ice_boundary_mask(largest_white)
    
    # Step 4: Final result with boundary constraint
    boundary_constrained = cv2.bitwise_and(
        cv2.bitwise_or(largest_white, lines_within), 
        boundary_mask
    )
    
    # Create colored visualizations for new approach
    color_vis = cv2.applyColorMap(color_small, cv2.COLORMAP_HOT)
    texture_vis = cv2.applyColorMap(texture_small, cv2.COLORMAP_COOL)
    largest_vis = cv2.applyColorMap(largest_white, cv2.COLORMAP_WINTER)
    lines_vis = cv2.applyColorMap(lines_within, cv2.COLORMAP_SPRING)
    boundary_vis = cv2.applyColorMap(boundary_mask, cv2.COLORMAP_RAINBOW)
    final_vis = cv2.applyColorMap(final_small, cv2.COLORMAP_VIRIDIS)
    
    # Create overlay showing final mask on original
    overlay = frame_small.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 1] = final_small  # Green channel for ice
    overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
    
    # Create 3x2 grid to show the boundary approach
    top_row = np.hstack([frame_small, color_vis, texture_vis])
    middle_row = np.hstack([largest_vis, lines_vis, boundary_vis]) 
    bottom_row = np.hstack([final_vis, overlay, np.zeros_like(overlay)])
    
    debug_grid = np.vstack([top_row, middle_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    color_white = (255, 255, 255)
    thickness = 1
    
    labels = [
        (5, 15, "Original"),
        (display_width + 5, 15, "Color"),
        (display_width * 2 + 5, 15, "Texture"),
        (display_width * 3 + 5, 15, "Intersection"),
        (5, display_height + 15, "Geo Region"),
        (display_width + 5, display_height + 15, "Filtered Seeds"),
        (display_width * 2 + 5, display_height + 15, "Final Result"),
        (display_width * 3 + 5, display_height + 15, "Ice Overlay")
    ]
    
    for x, y, text in labels:
        cv2.putText(debug_grid, text, (x, y), font, font_scale, color_white, thickness)
    
    return debug_grid

def analyze_simplified_results(ice_mask, frame_shape):
    """Analyze the simplified ice detection results"""
    
    total_pixels = frame_shape[0] * frame_shape[1]
    ice_pixels = np.sum(ice_mask > 0)
    ice_percentage = (ice_pixels / total_pixels) * 100
    
    print(f"\\n📈 SIMPLIFIED ICE DETECTION RESULTS:")
    print(f"   Ice coverage: {ice_percentage:.1f}% of frame")
    print(f"   Ice pixels: {ice_pixels:,} / {total_pixels:,}")
    
    # Analysis
    print(f"\\n💡 Analysis:")
    if 15 <= ice_percentage <= 50:
        print(f"   ✅ Ice coverage looks realistic for hockey video")
    elif ice_percentage < 15:
        print(f"   ⚠️  Low ice coverage - may be missing ice surface")
        print(f"      → Consider adjusting color thresholds")
    elif ice_percentage > 50:
        print(f"   ⚠️  High ice coverage - may be detecting too much")
        print(f"      → May include stands or background areas")
    
    print(f"\\n🎯 Next Steps:")
    print(f"   1. Review the debug image to see if ice detection is accurate")
    print(f"   2. Check that green overlay covers only the playing surface")
    print(f"   3. If it looks good, we can implement player filtering")
    print(f"   4. If adjustments needed, we can tune color/texture thresholds")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test simplified ice detection")
    parser.add_argument('video', help='Path to hockey video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame to analyze')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    success = test_simplified_ice_detection(str(video_path), args.frame)
    
    if success:
        print(f"\\n🎉 Simplified ice detection test completed!")
    else:
        print(f"\\n❌ Test failed")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
