import cv2
import numpy as np
from src.analysis.team_identifier import RefereeDetector
from src.detection.improved_detector import RealisticHockeyDetector

def debug_referee_scoring():
    """Debug referee scoring to understand why detection is failing"""

    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"

    # Initialize components
    detector = RealisticHockeyDetector()
    referee_detector = RefereeDetector()

    print("🔍 DEBUGGING REFEREE SCORING BREAKDOWN")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame")
        return

    # Detect all players
    players = detector.detect_players(frame)
    print(f"Total players detected: {len(players)}")
    print("")

    # Analyze each player's referee score in detail
    for i, player in enumerate(players):
        print(f"Player {i}: bbox={player['bbox']}")

        # Get detailed scoring breakdown
        score_breakdown = analyze_player_referee_score(frame, player, referee_detector)

        for key, value in score_breakdown.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        print(f"  FINAL SCORE: {score_breakdown['final_score']:.3f}")
        print(f"  IS REFEREE: {'YES' if score_breakdown['final_score'] > referee_detector.stripe_threshold else 'NO'}")
        print("")

    return score_breakdown

def analyze_player_referee_score(frame, player, referee_detector):
    """Analyze a player's referee score with detailed breakdown"""
    bbox = player['bbox']
    x, y, w, h = bbox

    # Extract jersey region (upper body)
    jersey_h = int(h * 0.6)
    roi = frame[y:y+jersey_h, x:x+w]

    breakdown = {
        'roi_size': f"{roi.shape if roi.size > 0 else 'EMPTY'}",
        'roi_valid': roi.size > 0 and roi.shape[0] >= 20 and roi.shape[1] >= 10
    }

    if not breakdown['roi_valid']:
        breakdown['final_score'] = 0.0
        breakdown['reason'] = 'ROI too small or empty'
        return breakdown

    # Convert to grayscale for pattern analysis
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Calculate row-wise intensity for horizontal stripe detection
    row_intensities = np.mean(gray_roi, axis=1)
    breakdown['row_intensities_length'] = len(row_intensities)

    if len(row_intensities) < 15:
        breakdown['final_score'] = 0.0
        breakdown['reason'] = 'Not enough rows for analysis'
        return breakdown

    # Smooth the signal to reduce noise
    kernel_size = max(3, len(row_intensities) // 10)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(row_intensities, kernel, mode='valid')
    breakdown['smoothed_length'] = len(smoothed)

    if len(smoothed) < 10:
        breakdown['final_score'] = 0.0
        breakdown['reason'] = 'Smoothed signal too short'
        return breakdown

    # Calculate local variance
    local_variance = []
    window_size = max(3, len(smoothed) // 8)

    for i in range(len(smoothed) - window_size + 1):
        window = smoothed[i:i+window_size]
        variance = np.var(window)
        local_variance.append(variance)

    if not local_variance:
        breakdown['final_score'] = 0.0
        breakdown['reason'] = 'No variance data'
        return breakdown

    avg_variance = np.mean(local_variance)
    max_variance = np.max(local_variance)

    # Contrast score
    contrast_score = avg_variance / (max_variance + 1e-6)
    breakdown['avg_variance'] = avg_variance
    breakdown['max_variance'] = max_variance
    breakdown['contrast_score'] = contrast_score

    # Stripe transitions
    second_derivative = np.diff(smoothed, n=2)
    zero_crossings = np.where(np.diff(np.signbit(second_derivative)))[0]
    stripe_transitions = len(zero_crossings)
    transition_score = min(1.0, stripe_transitions / 6.0)
    breakdown['stripe_transitions'] = stripe_transitions
    breakdown['transition_score'] = transition_score

    # Intensity range
    min_intensity = np.min(smoothed)
    max_intensity = np.max(smoothed)
    intensity_range = max_intensity - min_intensity
    contrast_range_score = min(1.0, intensity_range / 128.0)
    breakdown['min_intensity'] = min_intensity
    breakdown['max_intensity'] = max_intensity
    breakdown['intensity_range'] = intensity_range
    breakdown['contrast_range_score'] = contrast_range_score

    # Black/white pattern
    black_threshold = 80
    white_threshold = 170
    has_black = np.any(smoothed < black_threshold)
    has_white = np.any(smoothed > white_threshold)
    bw_pattern_score = 1.0 if (has_black and has_white) else 0.0
    breakdown['has_black'] = has_black
    breakdown['has_white'] = has_white
    breakdown['bw_pattern_score'] = bw_pattern_score

    # Size penalty
    size_penalty = min(1.0, (roi.shape[0] * roi.shape[1]) / 800.0)
    breakdown['roi_area'] = roi.shape[0] * roi.shape[1]
    breakdown['size_penalty'] = size_penalty

    # Final score
    final_score = (contrast_score * transition_score * contrast_range_score * bw_pattern_score * size_penalty)
    breakdown['final_score'] = final_score

    # Save ROI for visual inspection
    roi_filename = f'debug_player_{x}_{y}_roi.jpg'
    cv2.imwrite(roi_filename, gray_roi)
    breakdown['roi_saved'] = roi_filename

    return breakdown

if __name__ == "__main__":
    debug_referee_scoring()