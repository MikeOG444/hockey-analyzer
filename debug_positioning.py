#!/usr/bin/env python3
"""
Debug Detection vs Ice Mask Positioning

This script will show exactly where player detections are vs where the ice mask is
to understand why all detections are being filtered out.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_detection_positioning(video_path, frame_number=0):
    """Debug exactly where detections are vs ice mask"""
    
    try:
        import cv2
        import numpy as np
        from hockey_analyzer.detection.player_detector import PlayerDetector
        from hockey_analyzer.detection.ice_detector import IceDetector
        
        print(f"🔍 Debug Detection vs Ice Mask Positioning")
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
        
        # Get ice mask
        print("\\n🏒 Getting ice mask...")
        ice_mask = ice_detector.detect_ice_surface(frame)
        
        # Get all player detections (without filtering)
        print("\\n👥 Getting all player detections...")
        all_players = player_detector.detect_players(frame, ice_mask=None)
        
        print(f"\\n📍 DETAILED POSITION ANALYSIS:")
        print(f"   Total detections: {len(all_players)}")
        
        ice_players = 0
        non_ice_players = 0
        
        for i, player in enumerate(all_players):
            x, y = player['position']
            confidence = player['confidence']
            
            # Check if this position is on ice
            if (0 <= y < ice_mask.shape[0] and 0 <= x < ice_mask.shape[1]):
                mask_value = ice_mask[y, x]
                on_ice = mask_value > 0
                
                if on_ice:
                    ice_players += 1
                    status = "✅ ON ICE"
                else:
                    non_ice_players += 1
                    status = "❌ NOT ON ICE"
                
                print(f"   Player {i+1}: ({x:4d}, {y:4d}) conf={confidence:.2f} mask={mask_value:3d} {status}")
            else:
                non_ice_players += 1
                print(f"   Player {i+1}: ({x:4d}, {y:4d}) conf={confidence:.2f} ❌ OUT OF BOUNDS")
        
        print(f"\\n📊 SUMMARY:")
        print(f"   🟢 Players on ice: {ice_players}")
        print(f"   🔴 Players not on ice: {non_ice_players}")
        print(f"   📈 Ice detection coverage: {(np.sum(ice_mask > 0) / ice_mask.size) * 100:.1f}%")
        
        # Create detailed debug visualization
        debug_image = create_detailed_debug_visualization(frame, ice_mask, all_players)
        
        # Save debug image
        debug_path = f"debug_positioning_frame_{frame_number}.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"\\n✅ Detailed debug visualization saved: {debug_path}")
        
        # Analysis and recommendations
        analyze_positioning_results(ice_players, non_ice_players, ice_mask, frame.shape[:2])
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_detailed_debug_visualization(frame, ice_mask, players):
    """Create detailed visualization showing exact positioning"""
    
    import cv2
    import numpy as np
    
    # Create overlay frame
    overlay = frame.copy()
    
    # Create ice mask overlay (green tint)
    ice_overlay = np.zeros_like(overlay)
    ice_overlay[:, :, 1] = ice_mask  # Green channel
    overlay = cv2.addWeighted(overlay, 0.6, ice_overlay, 0.4, 0)
    
    # Draw ice mask boundaries in bright green
    contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    
    # Draw each player detection with detailed info
    for i, player in enumerate(players):
        x, y = player['position']
        bbox = player['bbox']
        confidence = player['confidence']
        
        # Check if on ice
        if (0 <= y < ice_mask.shape[0] and 0 <= x < ice_mask.shape[1]):
            mask_value = ice_mask[y, x]
            on_ice = mask_value > 0
        else:
            on_ice = False
            mask_value = 0
        
        # Color based on ice status
        if on_ice:
            color = (0, 255, 0)  # Green for on ice
            label = f"P{i+1} ON ICE"
        else:
            color = (0, 0, 255)  # Red for not on ice
            label = f"P{i+1} OFF ICE"
        
        # Draw bounding box
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
        
        # Draw center point (larger)
        cv2.circle(overlay, (x, y), 8, color, -1)
        cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
        
        # Draw detailed label
        label_text = f"{label} ({x},{y})"
        cv2.putText(overlay, label_text, (bbox[0], bbox[1] - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw confidence
        conf_text = f"conf={confidence:.2f}"
        cv2.putText(overlay, conf_text, (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw player number in center
        cv2.putText(overlay, str(i+1), (x-10, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 50
    cv2.putText(overlay, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, "Green areas = Ice surface", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(overlay, "Green boxes = Players on ice", (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(overlay, "Red boxes = Players off ice", (10, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return overlay

def analyze_positioning_results(ice_players, non_ice_players, ice_mask, frame_shape):
    import numpy as np
    """Analyze the positioning results and provide recommendations"""
    
    total_players = ice_players + non_ice_players
    ice_percentage = (np.sum(ice_mask > 0) / ice_mask.size) * 100
    
    print(f"\\n💡 DETAILED ANALYSIS:")
    
    if ice_players == 0 and total_players > 0:
        print(f"   🔴 ISSUE: All {total_players} detections are OFF ice")
        print(f"   🎯 This suggests:")
        if ice_percentage < 10:
            print(f"      • Ice detection is too restrictive ({ice_percentage:.1f}%)")
            print(f"      • Try adjusting ice detection thresholds")
        elif ice_percentage > 40:
            print(f"      • Ice detection may be OK ({ice_percentage:.1f}%)")
            print(f"      • Players may actually be on bench/sidelines in this frame")
            print(f"      • Try a different frame with active gameplay")
        else:
            print(f"      • Ice coverage looks reasonable ({ice_percentage:.1f}%)")
            print(f"      • Check if players are positioned correctly vs ice area")
    
    elif ice_players > 0:
        print(f"   ✅ SUCCESS: {ice_players} players detected on ice")
        filter_rate = (non_ice_players / total_players) * 100
        print(f"   📊 Filter effectiveness: {filter_rate:.1f}% filtered out")
    
    else:
        print(f"   ⚠️  No players detected at all - check detection confidence thresholds")
    
    print(f"\\n🎯 RECOMMENDATIONS:")
    if ice_players == 0:
        print(f"   1. Check the debug image to see ice mask vs player positions")
        print(f"   2. Try frame with active gameplay (players skating)")
        print(f"   3. Consider adjusting ice detection sensitivity if needed")
        print(f"   4. Verify this is actually a hockey game frame")
    else:
        print(f"   ✅ Pipeline is working correctly!")
        print(f"   • Continue with game analysis")
        print(f"   • Monitor filter effectiveness in logs")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Debug detection positioning vs ice mask")
    parser.add_argument('video', help='Path to hockey video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame to analyze')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    success = debug_detection_positioning(str(video_path), args.frame)
    
    if success:
        print(f"\\n🎉 Positioning debug completed!")
    else:
        print(f"\\n❌ Debug failed")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
