#!/usr/bin/env python3
"""
Test the fixed detection pipeline that filters players by ice surface

This test will:
1. Run player detection WITHOUT ice filtering (old way)
2. Run player detection WITH ice filtering (new way) 
3. Show the difference in number of detections
4. Create a visualization showing filtered vs unfiltered results
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_fixed_detection_pipeline(video_path, frame_number=0):
    """Test the fixed detection pipeline with ice filtering"""
    
    try:
        import cv2
        import numpy as np
        from hockey_analyzer.detection.player_detector import PlayerDetector
        from hockey_analyzer.detection.ice_detector import IceDetector
        
        print(f"🚀 Testing FIXED Detection Pipeline")
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
        
        print("\\n🔍 Step 1: Detecting ice surface...")
        ice_mask = ice_detector.detect_ice_surface(frame)
        
        if ice_mask is not None:
            ice_pixels = np.sum(ice_mask > 0)
            ice_percentage = (ice_pixels / ice_mask.size) * 100
            print(f"✅ Ice detection: {ice_percentage:.1f}% of frame")
        else:
            print("❌ Ice detection failed!")
            return False
        
        print("\\n👥 Step 2: Player detection comparison...")
        
        # OLD WAY: Detect players without ice filtering
        print("   🔴 OLD: Detecting players on ENTIRE frame...")
        players_unfiltered = player_detector.detect_players(frame, ice_mask=None)
        
        # NEW WAY: Detect players with ice filtering  
        print("   🟢 NEW: Detecting players ONLY on ice surface...")
        players_filtered = player_detector.detect_players(frame, ice_mask=ice_mask)
        
        # Show the difference
        print(f"\\n📊 RESULTS COMPARISON:")
        print(f"   🔴 Unfiltered detections: {len(players_unfiltered)} (includes crowd/bench)")
        print(f"   🟢 Ice-filtered detections: {len(players_filtered)} (ice players only)")
        print(f"   📉 Reduction: {len(players_unfiltered) - len(players_filtered)} detections filtered out")
        
        reduction_percentage = 0
        if len(players_unfiltered) > 0:
            reduction_percentage = ((len(players_unfiltered) - len(players_filtered)) / len(players_unfiltered)) * 100
            print(f"   📊 Filter effectiveness: {reduction_percentage:.1f}% of detections removed")
        
        # Create visualization
        debug_image = create_comparison_visualization(
            frame, ice_mask, players_unfiltered, players_filtered
        )
        
        # Save debug image
        debug_path = f"debug_fixed_pipeline_frame_{frame_number}.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"\\n✅ Comparison visualization saved: {debug_path}")
        
        # Analysis
        analyze_pipeline_fix(players_unfiltered, players_filtered, ice_percentage)
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_comparison_visualization(frame, ice_mask, players_unfiltered, players_filtered):
    """Create side-by-side comparison of filtered vs unfiltered detections"""
    
    import cv2
    import numpy as np
    
    # Create side-by-side layout
    height, width = frame.shape[:2]
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Left side: Unfiltered detections (with ice overlay)
    left_frame = frame.copy()
    ice_overlay = np.zeros_like(left_frame)
    ice_overlay[:, :, 1] = ice_mask  # Green channel for ice
    left_frame = cv2.addWeighted(left_frame, 0.7, ice_overlay, 0.3, 0)
    
    # Draw all unfiltered detections in RED
    for player in players_unfiltered:
        x, y = player['position']
        bbox = player['bbox']
        confidence = player['confidence']
        
        # Draw bounding box
        cv2.rectangle(left_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        # Draw center point
        cv2.circle(left_frame, (x, y), 5, (0, 0, 255), -1)
        # Draw confidence
        cv2.putText(left_frame, f"{confidence:.2f}", (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Right side: Filtered detections
    right_frame = frame.copy()
    right_frame = cv2.addWeighted(right_frame, 0.7, ice_overlay, 0.3, 0)
    
    # Draw only ice-filtered detections in GREEN
    for player in players_filtered:
        x, y = player['position']
        bbox = player['bbox']
        confidence = player['confidence']
        
        # Draw bounding box
        cv2.rectangle(right_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        # Draw center point
        cv2.circle(right_frame, (x, y), 5, (0, 255, 0), -1)
        # Draw confidence
        cv2.putText(right_frame, f"{confidence:.2f}", (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Combine into comparison image
    comparison[:, :width] = left_frame
    comparison[:, width:] = right_frame
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Left label
    cv2.putText(comparison, f"BEFORE: {len(players_unfiltered)} detections", 
               (10, 30), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(comparison, "Includes crowd/bench", 
               (10, 60), font, 0.5, (0, 0, 255), 1)
    
    # Right label  
    cv2.putText(comparison, f"AFTER: {len(players_filtered)} detections", 
               (width + 10, 30), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(comparison, "Ice players only", 
               (width + 10, 60), font, 0.5, (0, 255, 0), 1)
    
    return comparison

def analyze_pipeline_fix(players_unfiltered, players_filtered, ice_percentage):
    """Analyze the effectiveness of the pipeline fix"""
    
    print(f"\\n💡 ANALYSIS:")
    
    # Check if the fix is working
    if len(players_filtered) < len(players_unfiltered):
        print(f"   ✅ Ice filtering is working! Removed {len(players_unfiltered) - len(players_filtered)} false detections")
    elif len(players_filtered) == len(players_unfiltered):
        print(f"   ⚠️  No filtering occurred - check if ice mask is accurate")
    else:
        print(f"   ❌ Something went wrong - filtered count is higher than unfiltered")
    
    # Evaluate ice detection quality
    if 15 <= ice_percentage <= 50:
        print(f"   ✅ Ice detection looks good ({ice_percentage:.1f}%)")
    elif ice_percentage < 15:
        print(f"   ⚠️  Ice detection may be too restrictive ({ice_percentage:.1f}%)")
    elif ice_percentage > 50:
        print(f"   ⚠️  Ice detection may be too broad ({ice_percentage:.1f}%)")
    
    # Expected results
    print(f"\\n🎯 Expected Results:")
    print(f"   • Should detect 2-12 players on ice (typical hockey)")
    print(f"   • Should filter out spectators, bench players, officials in stands")
    print(f"   • Ice coverage should be 15-50% of frame")
    
    if len(players_filtered) <= 12:
        print(f"   ✅ Player count looks realistic for hockey")
    else:
        print(f"   ⚠️  High player count - may need stricter filtering")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test fixed detection pipeline")
    parser.add_argument('video', help='Path to hockey video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame to analyze')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    success = test_fixed_detection_pipeline(str(video_path), args.frame)
    
    if success:
        print(f"\\n🎉 Pipeline fix test completed!")
        print(f"📝 Check the debug image to see the filtering in action")
    else:
        print(f"\\n❌ Test failed")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
