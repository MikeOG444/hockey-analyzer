import cv2
import os
import numpy as np
from src.detection.basic_detector import BasicHockeyDetector
from src.detection.ice_detection import IceSurfaceDetector
from src.detection.puck_tracker import PuckTracker
import time

def test_puck_tracking():
    """Test the puck tracking system on hockey video"""
    
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
    
    print("🏒 Initializing Hockey Analysis System...")
    
    # Initialize all components
    player_detector = BasicHockeyDetector()
    ice_detector = IceSurfaceDetector()
    puck_tracker = PuckTracker(ice_detector=ice_detector)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {fps} fps, {total_frames} frames, {width}x{height}")
    
    # Create output video writer for debugging
    debug_output_path = "debug_puck_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(debug_output_path, fourcc, fps, (width, height))
    
    # Statistics
    frame_count = 0
    puck_detections = 0
    possession_changes = 0
    last_possession_player = None
    processing_times = []
    
    # Process subset of frames for testing (first 300 frames = 10 seconds)
    test_frames = min(300, total_frames)
    
    print(f"🎯 Processing first {test_frames} frames for puck tracking test...")
    
    try:
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Step 1: Detect ice surface (only every 30 frames to save processing)
            if frame_count % 30 == 0:
                ice_mask = ice_detector.detect_ice_surface(frame)
                print(f"Frame {frame_count}: Ice surface updated")
            
            # Step 2: Detect players
            players = player_detector.detect_players(frame)
            
            # Step 3: Detect puck candidates
            puck_candidates = puck_tracker.detect_puck_candidates(frame, players)
            
            # Step 4: Track puck across frames
            tracked_puck = puck_tracker.track_puck(puck_candidates)
            
            # Step 5: Determine possession
            current_possession = None
            if tracked_puck:
                puck_detections += 1
                current_possession = puck_tracker.get_puck_possession(tracked_puck.position, players)
                
                # Track possession changes
                if current_possession and current_possession != last_possession_player:
                    if last_possession_player is not None:
                        possession_changes += 1
                        print(f"Frame {frame_count}: Possession change detected!")
                    last_possession_player = current_possession
            
            # Step 6: Create visualization
            vis_frame = frame.copy()
            
            # Draw players
            for player in players:
                x, y = player['position']
                bbox = player['bbox']
                confidence = player['confidence']
                
                # Color code for possession
                color = (0, 255, 0)  # Green for normal players
                if current_possession and player == current_possession:
                    color = (0, 0, 255)  # Red for player with possession
                
                # Draw bounding box and center
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), 
                             (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                cv2.circle(vis_frame, (x, y), 5, color, -1)
                cv2.putText(vis_frame, f"{confidence:.2f}", (x-10, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw puck tracking
            vis_frame = puck_tracker.visualize_tracking(vis_frame, puck_candidates, tracked_puck)
            
            # Add info overlay
            info_text = [
                f"Frame: {frame_count}/{test_frames}",
                f"Players: {len(players)}",
                f"Puck candidates: {len(puck_candidates)}",
                f"Puck detected: {'YES' if tracked_puck else 'NO'}",
                f"Detection rate: {(puck_detections/max(frame_count, 1)*100):.1f}%",
                f"Possession changes: {possession_changes}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(vis_frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write debug frame
            out.write(vis_frame)
            
            # Track processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            frame_count += 1
            
            # Progress update every 50 frames
            if frame_count % 50 == 0:
                avg_time = np.mean(processing_times[-50:])
                fps_actual = 1.0 / avg_time
                print(f"Processed {frame_count} frames - Avg: {avg_time:.3f}s/frame ({fps_actual:.1f} fps)")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*50)
        print("🏒 PUCK TRACKING TEST RESULTS")
        print("="*50)
        print(f"Frames processed: {frame_count}")
        print(f"Puck detections: {puck_detections}")
        print(f"Detection rate: {(puck_detections/max(frame_count, 1)*100):.1f}%")
        print(f"Possession changes: {possession_changes}")
        
        if processing_times:
            avg_processing_time = np.mean(processing_times)
            max_processing_time = np.max(processing_times)
            min_processing_time = np.min(processing_times)
            
            print(f"\nPerformance:")
            print(f"  Average processing time: {avg_processing_time:.3f}s/frame")
            print(f"  Real-time factor: {(1/30)/avg_processing_time:.2f}x")
            print(f"  Range: {min_processing_time:.3f}s - {max_processing_time:.3f}s")
        
        print(f"\n📹 Debug video saved: {debug_output_path}")
        print("🔍 Review the debug video to assess puck tracking accuracy")
        
        # Recommendations
        print("\n💡 Next steps:")
        if puck_detections / max(frame_count, 1) < 0.3:
            print("  - Puck detection rate is low. Consider:")
            print("    • Adjusting detection thresholds")
            print("    • Improving ice surface detection")
            print("    • Training a custom puck detection model")
        else:
            print("  - Good puck detection rate! Consider:")
            print("    • Fine-tuning tracking parameters")
            print("    • Adding team identification")
            print("    • Implementing play analysis")

def test_puck_detection_methods():
    """Test individual puck detection methods"""
    
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
    
    # Initialize components
    ice_detector = IceSurfaceDetector()
    puck_tracker = PuckTracker(ice_detector=ice_detector)
    
    # Open video and get a few frames
    cap = cv2.VideoCapture(video_path)
    
    # Test on frames 100-120 (varied content)
    test_frame_numbers = [100, 110, 120, 150, 200]
    
    print("🔬 Testing individual puck detection methods...")
    
    for frame_num in test_frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"\n--- Frame {frame_num} ---")
        
        # Test each method individually
        ice_mask = ice_detector.detect_ice_surface(frame)
        
        # Motion-based detection
        motion_candidates = puck_tracker._detect_puck_by_motion(frame)
        print(f"Motion detection: {len(motion_candidates)} candidates")
        
        # Shape-based detection
        shape_candidates = puck_tracker._detect_puck_by_shape(frame)
        print(f"Shape detection: {len(shape_candidates)} candidates")
        
        # Context-based detection (need players)
        from src.detection.basic_detector import BasicHockeyDetector
        player_detector = BasicHockeyDetector()
        players = player_detector.detect_players(frame)
        context_candidates = puck_tracker._detect_puck_by_context(frame, players)
        print(f"Context detection: {len(context_candidates)} candidates (with {len(players)} players)")
        
        # Combined detection
        all_candidates = puck_tracker.detect_puck_candidates(frame, players)
        print(f"Combined detection: {len(all_candidates)} final candidates")
        
        # Show top candidate details
        if all_candidates:
            top_candidate = all_candidates[0]
            print(f"  Best candidate: pos({top_candidate.position[0]}, {top_candidate.position[1]}), "
                  f"conf={top_candidate.confidence:.3f}, method={top_candidate.detection_method}")
    
    cap.release()
    print("\n✅ Method comparison complete")

if __name__ == "__main__":
    # Run main puck tracking test
    print("Choose test mode:")
    print("1. Full puck tracking test (with debug video)")
    print("2. Detection methods comparison")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_puck_detection_methods()
    else:
        test_puck_tracking()