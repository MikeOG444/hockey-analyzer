import cv2
import os
from src.detection.basic_detector import BasicHockeyDetector

def test_hockey_video():
    # Path to your video
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
    
    print(f"📹 Loading video: {video_path}")
    
    # Initialize detector
    detector = BasicHockeyDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video info:")
    print(f"   - Duration: {duration:.1f} seconds")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    
    # Test on a few frames
    test_frames = [0, int(total_frames * 0.1), int(total_frames * 0.5), int(total_frames * 0.8)]
    
    results = []
    
    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            print(f"\n🔍 Analyzing frame {frame_idx} ({frame_idx/total_frames*100:.1f}% through video)")
            
            # Analyze frame
            analysis = detector.analyze_frame(frame)
            results.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'player_count': analysis['player_count'],
                'players': analysis['players']
            })
            
            print(f"   - Detected {analysis['player_count']} players")
            
            # Show player positions and confidence
            for i, player in enumerate(analysis['players']):
                conf = player['confidence']
                pos = player['position']
                print(f"     Player {i+1}: position {pos}, confidence {conf:.2f}")
    
    cap.release()
    
    # Summary
    print(f"\n📈 Summary:")
    player_counts = [r['player_count'] for r in results]
    avg_players = sum(player_counts) / len(player_counts)
    print(f"   - Average players detected: {avg_players:.1f}")
    print(f"   - Range: {min(player_counts)} - {max(player_counts)} players")
    
    # Basic validation
    if avg_players < 5:
        print("⚠️  Low player detection - might need to adjust detection threshold")
    elif avg_players > 15:
        print("⚠️  High player detection - might be detecting non-players")
    else:
        print("✅ Player detection looks reasonable for hockey!")
    
    return results

if __name__ == "__main__":
    print("🏒 Hockey Video Analysis Test")
    print("=" * 40)
    
    results = test_hockey_video()
    
    if results:
        print(f"\n✅ Test completed successfully!")
        print("Next steps: Review detection accuracy and adjust if needed")