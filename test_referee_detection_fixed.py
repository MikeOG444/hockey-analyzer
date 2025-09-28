import cv2
import numpy as np
from src.analysis.team_identifier import RefereeDetector
from src.detection.improved_detector import RealisticHockeyDetector

def test_referee_detection():
    """Test referee detection with realistic hockey constraints"""

    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"

    # Initialize components
    detector = RealisticHockeyDetector()
    referee_detector = RefereeDetector()

    print("🔍 TESTING REFEREE DETECTION WITH HOCKEY RULES")
    print("=" * 60)
    print("Expected: 1 referee in this video")
    print("Hockey rules: Max 3 referees, usually 1-2")
    print("")

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    total_refs_detected = 0
    max_refs_in_frame = 0

    # Test multiple frames to see consistency
    test_frames = [0, 100, 200, 300, 400]  # Different time points

    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1

        # Detect all players
        players = detector.detect_players(frame)

        # Detect referees with new constraints
        referee_indices = referee_detector.detect_referees(frame, players)

        refs_this_frame = len(referee_indices)
        total_refs_detected += refs_this_frame
        max_refs_in_frame = max(max_refs_in_frame, refs_this_frame)

        print(f"Frame {frame_num:3d}: {len(players)} players, {refs_this_frame} referees")

        # Show referee details
        for ref_idx in referee_indices:
            if ref_idx < len(players):
                score = referee_detector._calculate_referee_score(frame, players[ref_idx])
                bbox = players[ref_idx]['bbox']
                print(f"  Referee {ref_idx}: score={score:.3f}, bbox={bbox}")

        # Create debug visualization for first frame
        if frame_num == test_frames[0]:
            debug_frame = frame.copy()

            # Draw all players
            for i, player in enumerate(players):
                x, y, w, h = player['bbox']
                color = (0, 255, 0) if i in referee_indices else (255, 0, 0)
                thickness = 3 if i in referee_indices else 1
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, thickness)

                # Label referees
                if i in referee_indices:
                    score = referee_detector._calculate_referee_score(frame, player)
                    cv2.putText(debug_frame, f"REF: {score:.2f}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(debug_frame, f"P{i}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Add info overlay
            info_text = [
                f"Frame {frame_num}: {len(players)} players detected",
                f"Referees: {refs_this_frame}/{referee_detector.max_referees} max",
                f"Threshold: {referee_detector.stripe_threshold}",
                "Green boxes = Referees, Blue boxes = Players"
            ]

            y_pos = 30
            for text in info_text:
                cv2.putText(debug_frame, text, (10, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 25

            cv2.imwrite('debug_referee_detection_fixed.jpg', debug_frame)
            print(f"  Debug image saved: debug_referee_detection_fixed.jpg")

    cap.release()

    # Summary statistics
    avg_refs = total_refs_detected / frame_count if frame_count > 0 else 0

    print(f"\n📊 REFEREE DETECTION SUMMARY:")
    print(f"  Frames tested: {frame_count}")
    print(f"  Average referees per frame: {avg_refs:.1f}")
    print(f"  Maximum referees in any frame: {max_refs_in_frame}")
    print(f"  Total referee detections: {total_refs_detected}")

    # Validate against hockey rules
    print(f"\n✅ HOCKEY RULES VALIDATION:")
    if max_refs_in_frame <= referee_detector.max_referees:
        print(f"  ✅ Never exceeded max referees ({referee_detector.max_referees})")
    else:
        print(f"  ❌ Exceeded max referees: {max_refs_in_frame} > {referee_detector.max_referees}")

    if avg_refs <= 2.0:
        print(f"  ✅ Reasonable average ({avg_refs:.1f} ≤ 2.0 expected)")
    else:
        print(f"  ❌ Too many referees on average: {avg_refs:.1f}")

    # Expected for this specific video
    if max_refs_in_frame <= 1:
        print(f"  ✅ Correct for this video (expected 1 referee)")
    else:
        print(f"  ⚠️  Still detecting too many ({max_refs_in_frame}) for this video (expected 1)")
        print("      Consider increasing stripe_threshold or improving pattern detection")

    return max_refs_in_frame, avg_refs

if __name__ == "__main__":
    test_referee_detection()