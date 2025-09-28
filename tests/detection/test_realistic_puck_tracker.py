import cv2
import os
import numpy as np
from src.detection.basic_detector import BasicHockeyDetector
from src.detection.ice_detection import IceSurfaceDetector
from src.detection.realistic_puck_tracker import RealisticPuckTracker
import time

def test_realistic_puck_tracker():
    """Test the realistic puck tracking system"""

    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return

    print("🏒 REALISTIC PUCK TRACKER TEST")
    print("=" * 60)
    print("Expected behavior:")
    print("- Only 1 puck detected at a time (never hundreds)")
    print("- Detection rate 20-40% (puck often hidden)")
    print("- Conservative detection to avoid false positives")
    print("")

    # Initialize components
    player_detector = BasicHockeyDetector()
    ice_detector = IceSurfaceDetector()
    puck_tracker = RealisticPuckTracker(ice_detector=ice_detector)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Video: {fps} fps, {total_frames} frames, {width}x{height}")

    # Process test frames
    test_frames = min(300, total_frames)  # 10 seconds
    frame_count = 0
    puck_detections = 0
    prediction_frames = 0
    processing_times = []

    print(f"🎯 Testing {test_frames} frames...\n")

    try:
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # Update ice surface occasionally
            if frame_count % 60 == 0:  # Every 2 seconds
                ice_detector.detect_ice_surface(frame)

            # Detect players
            players = player_detector.detect_players(frame)

            # Detect puck using realistic tracker
            tracked_puck = puck_tracker.detect_puck(frame, players)

            # Count detections
            if tracked_puck:
                if tracked_puck.detection_method == "prediction":
                    prediction_frames += 1
                else:
                    puck_detections += 1

            # Create debug visualization every 30 frames
            if frame_count % 30 == 0:
                vis_frame = frame.copy()

                # Draw players
                for player in players:
                    x, y = player['position']
                    bbox = player['bbox']
                    cv2.rectangle(vis_frame, (bbox[0], bbox[1]),
                                 (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)

                # Draw puck tracking
                vis_frame = puck_tracker.visualize_tracking(vis_frame, tracked_puck)

                # Get current stats
                stats = puck_tracker.get_detection_stats()

                # Add frame info
                info_text = [
                    f"Frame: {frame_count}/{test_frames}",
                    f"Players: {len(players)}",
                    f"Puck: {'YES' if tracked_puck else 'NO'}",
                    f"Method: {tracked_puck.detection_method if tracked_puck else 'NONE'}",
                    f"Confidence: {tracked_puck.confidence:.2f}" if tracked_puck else "Confidence: N/A",
                    f"Detection Rate: {stats['detection_rate']*100:.1f}%"
                ]

                for i, text in enumerate(info_text):
                    cv2.putText(vis_frame, text, (10, 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Save debug frame
                debug_filename = f"debug_realistic_puck_frame_{frame_count:04d}.jpg"
                cv2.imwrite(debug_filename, vis_frame)

                if frame_count <= 60:  # First 2 seconds
                    print(f"Frame {frame_count:3d}: Puck={'YES' if tracked_puck else 'NO':3s} "
                          f"Method={tracked_puck.detection_method if tracked_puck else 'NONE':12s} "
                          f"Conf={tracked_puck.confidence:.2f}" if tracked_puck else "Conf=N/A")

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            frame_count += 1

            # Progress update
            if frame_count % 50 == 0:
                current_stats = puck_tracker.get_detection_stats()
                print(f"Progress: {frame_count}/{test_frames} frames "
                      f"(Detection rate: {current_stats['detection_rate']*100:.1f}%)")

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Final analysis
        total_puck_frames = puck_detections + prediction_frames
        detection_rate = puck_detections / max(frame_count, 1)
        total_rate = total_puck_frames / max(frame_count, 1)

        print("\n" + "="*60)
        print("🏒 REALISTIC PUCK TRACKER RESULTS")
        print("="*60)
        print(f"Frames processed: {frame_count}")
        print(f"Actual puck detections: {puck_detections}")
        print(f"Prediction frames: {prediction_frames}")
        print(f"Total puck tracking: {total_puck_frames}")
        print("")
        print(f"Detection rate: {detection_rate*100:.1f}% (actual detections only)")
        print(f"Total tracking rate: {total_rate*100:.1f}% (including predictions)")
        print("")

        # Performance metrics
        if processing_times:
            avg_time = np.mean(processing_times)
            print(f"Average processing time: {avg_time:.3f}s/frame")
            print(f"Real-time performance: {(1/30)/avg_time:.1f}x")

        print("\n🎯 HOCKEY REALISM ASSESSMENT:")

        # Check realistic constraints
        if detection_rate <= 0.5:
            print(f"  ✅ Realistic detection rate ({detection_rate*100:.1f}% ≤ 50%)")
        else:
            print(f"  ⚠️  High detection rate ({detection_rate*100:.1f}% > 50%) - may have false positives")

        if total_rate <= 0.6:
            print(f"  ✅ Realistic total tracking ({total_rate*100:.1f}% ≤ 60%)")
        else:
            print(f"  ⚠️  High tracking rate ({total_rate*100:.1f}% > 60%)")

        # Get final stats from tracker
        final_stats = puck_tracker.get_detection_stats()
        if final_stats['frames_since_last_detection'] < 30:
            print("  ✅ Recently detected puck (good tracking)")
        else:
            print("  ⚠️  Puck lost for extended period (normal in hockey)")

        print("\n💡 RECOMMENDATIONS:")
        if detection_rate < 0.2:
            print("  - Detection rate very low - consider lowering thresholds slightly")
        elif detection_rate > 0.4:
            print("  - Detection rate high - may need stricter validation")
        else:
            print("  - Detection rate looks realistic for hockey puck tracking")

        print(f"\n📸 Debug images saved: debug_realistic_puck_frame_*.jpg")
        print("🔍 Review debug images to assess puck detection accuracy")

def compare_old_vs_new():
    """Compare old multi-detection vs new single-puck detection"""

    from src.detection.puck_tracker import PuckTracker

    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return

    print("🆚 COMPARING OLD vs NEW PUCK DETECTION")
    print("=" * 60)

    # Initialize both trackers
    ice_detector = IceSurfaceDetector()
    old_tracker = PuckTracker(ice_detector=ice_detector)
    new_tracker = RealisticPuckTracker(ice_detector=ice_detector)
    player_detector = BasicHockeyDetector()

    cap = cv2.VideoCapture(video_path)

    # Test on 5 representative frames
    test_frames = [50, 100, 150, 200, 250]

    for frame_num in test_frames:
        print(f"\n--- Frame {frame_num} ---")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Update ice detection
        ice_detector.detect_ice_surface(frame)
        players = player_detector.detect_players(frame)

        # OLD METHOD: Multiple candidates
        old_candidates = old_tracker.detect_puck_candidates(frame, players)
        old_tracked = old_tracker.track_puck(old_candidates)

        print(f"OLD METHOD:")
        print(f"  Candidates found: {len(old_candidates)}")
        print(f"  Final puck: {'YES' if old_tracked else 'NO'}")

        # NEW METHOD: Single puck
        new_puck = new_tracker.detect_puck(frame, players)

        print(f"NEW METHOD:")
        print(f"  Puck detected: {'YES' if new_puck else 'NO'}")
        if new_puck:
            print(f"  Confidence: {new_puck.confidence:.3f}")
            print(f"  Method: {new_puck.detection_method}")

    cap.release()
    print(f"\n✅ Comparison complete - new method should show much fewer detections")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test realistic puck tracker")
    print("2. Compare old vs new detection")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        compare_old_vs_new()
    else:
        test_realistic_puck_tracker()