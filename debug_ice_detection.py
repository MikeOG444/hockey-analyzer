#!/usr/bin/env python3
"""
Ice Detection Debug Script

This script will:
1. Load a frame from your hockey video
2. Run all ice detection methods
3. Show you what each method detects
4. Display the final combined ice mask
5. Help you understand if ice detection is working correctly
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_ice_detection(video_path, frame_number=0, save_debug=True):
    """Debug ice detection on a specific frame"""
    
    try:
        import cv2
        import numpy as np
        from hockey_analyzer.detection.ice_detector import IceDetector
        
        print(f"🏒 Ice Detection Debug")
        print(f"Video: {video_path}")
        print(f"Frame: {frame_number}")
        print("=" * 50)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return False
        
        # Go to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Error: Could not read frame {frame_number}")
            return False
        
        print(f"✅ Loaded frame {frame_number}")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Initialize ice detector
        ice_detector = IceDetector()
        
        # Convert frame for different methods
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("\\n🔍 Running individual detection methods...")
        
        # Method 1: Color-based detection
        print("1. Color-based detection (white/light areas)...")
        ice_mask_color = ice_detector.detect_white_surfaces(hsv, lab)
        color_percentage = (np.sum(ice_mask_color > 0) / ice_mask_color.size) * 100
        print(f"   Detected {color_percentage:.1f}% of frame as white/light")
        
        # Method 2: Texture analysis
        print("2. Texture analysis (smooth surfaces)...")
        ice_mask_texture = ice_detector.detect_smooth_surfaces(gray)
        texture_percentage = (np.sum(ice_mask_texture > 0) / ice_mask_texture.size) * 100
        print(f"   Detected {texture_percentage:.1f}% of frame as smooth")
        
        # Method 3: Geometric constraints
        print("3. Geometric analysis (largest flat area)...")
        ice_mask_geometric = ice_detector.detect_large_flat_areas(gray)
        geometric_percentage = (np.sum(ice_mask_geometric > 0) / ice_mask_geometric.size) * 100
        print(f"   Detected {geometric_percentage:.1f}% of frame as largest area")
        
        # Method 4: Ice markings
        print("4. Ice markings detection (red/blue lines)...")
        try:
            ice_mask_markings = ice_detector.detect_by_ice_markings(frame)
            markings_percentage = (np.sum(ice_mask_markings > 0) / ice_mask_markings.size) * 100
            print(f"   Detected {markings_percentage:.1f}% of frame around markings")
        except Exception as e:
            print(f"   ⚠️  Ice markings detection failed: {e}")
            ice_mask_markings = np.zeros(frame.shape[:2], dtype=np.uint8)
            markings_percentage = 0
        
        # Combine all methods
        print("\\n🔗 Combining all methods...")
        combined_mask = ice_detector.combine_ice_masks([
            ice_mask_color,
            ice_mask_texture,
            ice_mask_geometric,
            ice_mask_markings
        ])
        
        # Final cleaning
        final_mask = ice_detector.clean_ice_mask(combined_mask)
        final_percentage = (np.sum(final_mask > 0) / final_mask.size) * 100
        print(f"   Final ice mask covers {final_percentage:.1f}% of frame")
        
        # Create visualization
        if save_debug:
            print("\\n🎨 Creating debug visualization...")
            debug_image = create_debug_visualization(
                frame, 
                ice_mask_color, 
                ice_mask_texture, 
                ice_mask_geometric, 
                ice_mask_markings, 
                final_mask
            )
            
            # Save debug image
            debug_path = f"debug_ice_detection_frame_{frame_number}.jpg"
            cv2.imwrite(debug_path, debug_image)
            print(f"✅ Debug image saved: {debug_path}")
        
        # Analysis and recommendations
        print("\\n📊 Analysis:")
        analyze_ice_detection_results(
            color_percentage, 
            texture_percentage, 
            geometric_percentage, 
            markings_percentage, 
            final_percentage
        )
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install opencv-python numpy scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Error during ice detection debug: {e}")
        return False

def create_debug_visualization(frame, color_mask, texture_mask, geometric_mask, markings_mask, final_mask):
    """Create a comprehensive debug visualization"""
    
    import cv2
    import numpy as np
    
    # Resize frame for display (if too large)
    display_height = 400
    aspect_ratio = frame.shape[1] / frame.shape[0]
    display_width = int(display_height * aspect_ratio)
    
    # Resize all images
    frame_small = cv2.resize(frame, (display_width, display_height))
    color_small = cv2.resize(color_mask, (display_width, display_height))
    texture_small = cv2.resize(texture_mask, (display_width, display_height))
    geometric_small = cv2.resize(geometric_mask, (display_width, display_height))
    markings_small = cv2.resize(markings_mask, (display_width, display_height))
    final_small = cv2.resize(final_mask, (display_width, display_height))
    
    # Convert masks to color for display
    color_vis = cv2.applyColorMap(color_small, cv2.COLORMAP_JET)
    texture_vis = cv2.applyColorMap(texture_small, cv2.COLORMAP_JET)
    geometric_vis = cv2.applyColorMap(geometric_small, cv2.COLORMAP_JET)
    markings_vis = cv2.applyColorMap(markings_small, cv2.COLORMAP_JET)
    final_vis = cv2.applyColorMap(final_small, cv2.COLORMAP_JET)
    
    # Create overlay of final mask on original
    overlay = frame_small.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 1] = final_small  # Green channel for ice
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # Create grid layout (3x2)
    row1 = np.hstack([frame_small, color_vis, texture_vis])
    row2 = np.hstack([geometric_vis, markings_vis, final_vis])
    row3 = np.hstack([overlay, np.zeros_like(overlay), np.zeros_like(overlay)])
    
    debug_grid = np.vstack([row1, row2, row3])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    # Label each section
    labels = [
        (10, 30, "Original Frame"),
        (display_width + 10, 30, "Color Detection"),
        (display_width * 2 + 10, 30, "Texture Detection"),
        (10, display_height + 30, "Geometric Detection"),
        (display_width + 10, display_height + 30, "Markings Detection"),
        (display_width * 2 + 10, display_height + 30, "Final Ice Mask"),
        (10, display_height * 2 + 30, "Ice Overlay")
    ]
    
    for x, y, text in labels:
        cv2.putText(debug_grid, text, (x, y), font, font_scale, color, thickness)
    
    return debug_grid

def analyze_ice_detection_results(color_pct, texture_pct, geometric_pct, markings_pct, final_pct):
    """Analyze detection results and provide recommendations"""
    
    print(f"   Color detection: {color_pct:.1f}% of frame")
    print(f"   Texture detection: {texture_pct:.1f}% of frame")
    print(f"   Geometric detection: {geometric_pct:.1f}% of frame")
    print(f"   Markings detection: {markings_pct:.1f}% of frame")
    print(f"   FINAL RESULT: {final_pct:.1f}% of frame")
    
    print("\\n💡 Recommendations:")
    
    # Expected ice surface should be roughly 20-40% of frame in typical hockey video
    if final_pct < 10:
        print("   ⚠️  Very small ice detection - may be missing ice surface")
        print("      → Try adjusting color thresholds")
        print("      → Check if lighting is unusual")
        
    elif final_pct > 60:
        print("   ⚠️  Very large ice detection - may be detecting stands/background")
        print("      → Tighten color detection thresholds")
        print("      → Improve geometric constraints")
        
    else:
        print("   ✅ Ice detection percentage looks reasonable")
    
    # Check individual methods
    if markings_pct > 0:
        print("   ✅ Ice markings detected - good for reference")
    else:
        print("   ⚠️  No ice markings detected - relying on color/texture")
    
    if geometric_pct > 40:
        print("   ⚠️  Geometric detection found very large area - check for accuracy")
    
    if color_pct < 5:
        print("   ⚠️  Very little white/light area detected - may need color tuning")

def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Debug ice detection on hockey video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_ice_detection.py data/test_videos/game.mp4
  python debug_ice_detection.py game.mp4 --frame 100
  python debug_ice_detection.py game.mp4 --frame 0 --no-save
        """
    )
    
    parser.add_argument('video', help='Path to hockey video file')
    parser.add_argument('--frame', type=int, default=0, 
                       help='Frame number to analyze (default: 0)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save debug image')
    
    args = parser.parse_args()
    
    # Check if video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return 1
    
    # Run debug
    success = debug_ice_detection(
        str(video_path),
        frame_number=args.frame,
        save_debug=not args.no_save
    )
    
    if success:
        print("\\n🎉 Ice detection debug completed!")
        print("\\nNext steps:")
        print("1. Check the debug image to see if ice detection is accurate")
        print("2. If ice detection looks good, we can implement player filtering")
        print("3. If ice detection needs improvement, we'll tune the parameters")
        return 0
    else:
        print("\\n❌ Ice detection debug failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
