#!/usr/bin/env python3
"""
Test the Polygon Masking approach for ice-only detection

This test will:
1. Show the original frame
2. Show the masked frame (polygon masking)
3. Compare detections on full frame vs masked frame
4. Validate the OR logic for edge players
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_polygon_masking(video_path, frame_number=0):
    """Test the polygon masking approach"""
    
    try:
        import cv2
        import numpy as np
        from hockey_analyzer.detection.player_detector import PlayerDetector
        from hockey_analyzer.detection.ice_detector import IceDetector
        
        print(f"🚀 Testing POLYGON MASKING Approach")
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
        
        # Initialize detectors
        player_detector = PlayerDetector()
        ice_detector = IceDetector()
        
        print("\\n🏒 Step 1: Detecting ice surface...")
        ice_mask = ice_detector.detect_ice_surface(frame)
        
        if ice_mask is not None:
            ice_pixels = np.sum(ice_mask > 0)
            ice_percentage = (ice_pixels / ice_mask.size) * 100
            print(f"✅ Ice detection: {ice_percentage:.1f}% of frame")
        else:
            print("❌ Ice detection failed!")
            return False
        
        print("\\n🎭 Step 2: Creating masked frame...")
        # Create the polygon-masked frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=ice_mask)
        
        print("\\n👥 Step 3: Comparing detection approaches...")
        
        # OLD: Full frame detection (should detect crowd/bench)
        print("   🔴 OLD: Full frame detection...")
        players_full_frame = player_detector.detect_players(frame, ice_mask=None)
        
        # NEW: Polygon masked detection (should only detect ice players)
        print("   🟢 NEW: Polygon masked detection...")
        players_masked = player_detector.detect_players(frame, ice_mask=ice_mask)
        
        # Show the difference
        print(f"\\n📊 RESULTS COMPARISON:")
        print(f"   🔴 Full frame detections: {len(players_full_frame)} (includes crowd/bench)")
        print(f"   🟢 Polygon masked detections: {len(players_masked)} (ice-only with OR logic)")
        print(f"   📉 Efficiency gain: {len(players_full_frame) - len(players_masked)} false detections eliminated")
        
        if len(players_full_frame) > 0:
            efficiency = ((len(players_full_frame) - len(players_masked)) / len(players_full_frame)) * 100
            print(f"   📊 Detection efficiency: {efficiency:.1f}% noise eliminated")
        
        # Test the OR logic specifically
        print(f"\\n🧠 Step 4: Testing OR Logic for edge players...")
        test_or_logic_validation(players_masked, ice_mask)
        
        # Create visualization
        debug_image = create_polygon_masking_visualization(
            frame, masked_frame, ice_mask, players_full_frame, players_masked
        )
        
        # Save debug image
        debug_path = f"debug_polygon_masking_frame_{frame_number}.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"\\n✅ Polygon masking visualization saved: {debug_path}")
        
        # Analysis
        analyze_polygon_masking_results(players_full_frame, players_masked, ice_percentage)
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_or_logic_validation(players, ice_mask):
    """Test the OR logic for edge players"""
    
    print(f"   Found {len(players)} players on ice:")
    for i, player in enumerate(players):
        x, y, w, h = player['bbox']
        center_x, center_y = x + w//2, y + h//2
        feet_x, feet_y = x + w//2, y + h
        
        # Check center
        center_on_ice = False
        if (0 <= center_y < ice_mask.shape[0] and 0 <= center_x < ice_mask.shape[1]):
            center_on_ice = ice_mask[center_y, center_x] > 0
        
        # Check feet  
        feet_on_ice = False
        if (0 <= feet_y < ice_mask.shape[0] and 0 <= feet_x < ice_mask.shape[1]):
            feet_on_ice = ice_mask[feet_y, feet_x] > 0
        
        logic_result = "center" if center_on_ice and not feet_on_ice else "feet" if feet_on_ice and not center_on_ice else "both" if center_on_ice and feet_on_ice else "neither"
        
        print(f"     Player {i+1}: Center({center_x},{center_y})={center_on_ice}, Feet({feet_x},{feet_y})={feet_on_ice} → {logic_result}")

def create_polygon_masking_visualization(frame, masked_frame, ice_mask, players_full, players_masked):
    """Create visualization showing polygon masking approach"""
    
    import cv2
    import numpy as np
    
    # Create 2x2 grid layout
    height, width = frame.shape[:2]
    display_height = height // 2
    display_width = width // 2
    
    # Resize all images for display
    frame_small = cv2.resize(frame, (display_width, display_height))
    masked_small = cv2.resize(masked_frame, (display_width, display_height))
    ice_mask_small = cv2.resize(ice_mask, (display_width, display_height))
    
    # Create ice mask visualization (colored)
    ice_vis = cv2.applyColorMap(ice_mask_small, cv2.COLORMAP_WINTER)
    
    # Add detection overlays
    frame_with_detections = frame_small.copy()
    masked_with_detections = masked_small.copy()
    
    # Draw full frame detections in RED
    for player in players_full:
        x, y, w, h = player['bbox']
        # Scale coordinates to display size
        x, y, w, h = x//2, y//2, w//2, h//2
        cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame_with_detections, (x + w//2, y + h//2), 3, (0, 0, 255), -1)
    
    # Draw masked detections in GREEN  
    for player in players_masked:
        x, y, w, h = player['bbox']
        # Scale coordinates to display size
        x, y, w, h = x//2, y//2, w//2, h//2
        cv2.rectangle(masked_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(masked_with_detections, (x + w//2, y + h//2), 3, (0, 255, 0), -1)
    
    # Create 2x2 grid
    top_row = np.hstack([frame_with_detections, ice_vis])
    bottom_row = np.hstack([masked_small, masked_with_detections])
    grid = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    cv2.putText(grid, f"Original + Detections ({len(players_full)})", (10, 25), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(grid, "Ice Polygon Mask", (display_width + 10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(grid, "Masked Frame (Ice Only)", (10, display_height + 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(grid, f"Masked + Detections ({len(players_masked)})", (display_width + 10, display_height + 25), font, font_scale, (0, 255, 0), thickness)
    
    return grid

def analyze_polygon_masking_results(players_full, players_masked, ice_percentage):
    """Analyze the effectiveness of polygon masking"""
    
    print(f"\\n💡 POLYGON MASKING ANALYSIS:")
    
    # Check efficiency gains
    noise_eliminated = len(players_full) - len(players_masked)
    if noise_eliminated > 0:
        print(f"   ✅ Polygon masking eliminated {noise_eliminated} false detections!")
        efficiency = (noise_eliminated / len(players_full)) * 100 if len(players_full) > 0 else 0
        print(f"   📊 Detection efficiency: {efficiency:.1f}% noise reduction")
    elif noise_eliminated == 0:
        print(f"   ⚠️  No difference between approaches - frame may have no crowd/bench visible")
    else:
        print(f"   ❌ Unexpected result - masked approach found more detections")
    
    # Evaluate ice coverage
    print(f"   🏒 Ice coverage: {ice_percentage:.1f}% of frame")
    if 10 <= ice_percentage <= 50:
        print(f"   ✅ Ice coverage looks appropriate for polygon masking")
    else:
        print(f"   ⚠️  Unusual ice coverage - verify ice detection quality")
    
    # Expected results
    print(f"\\n🎯 EXPECTED BENEFITS:")
    print(f"   • 🚀 Faster processing (YOLO processes {ice_percentage:.1f}% instead of 100%)")
    print(f"   • 🎯 No crowd/bench noise in detections")
    print(f"   • 🧠 OR logic handles edge players properly")
    print(f"   • ⚡ Better real-time performance potential")
    
    if len(players_masked) <= 12:
        print(f"   ✅ Player count ({len(players_masked)}) looks realistic for hockey")
    else:
        print(f"   ⚠️  High player count - verify ice detection accuracy")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test polygon masking approach")
    parser.add_argument('video', help='Path to hockey video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame to analyze')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    success = test_polygon_masking(str(video_path), args.frame)
    
    if success:
        print(f"\\n🎉 Polygon masking test completed!")
        print(f"📝 Check the debug image to see the masking in action")
    else:
        print(f"\\n❌ Test failed")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
