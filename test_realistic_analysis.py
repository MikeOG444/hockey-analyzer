import cv2
import os
from src.detection.improved_detector import RealisticHockeyDetector

def realistic_hockey_analysis():
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    detector = RealisticHockeyDetector()
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Test several frames throughout the video
    test_frames = [
        int(total_frames * 0.1),   # 10%
        int(total_frames * 0.3),   # 30% 
        int(total_frames * 0.5),   # 50%
        int(total_frames * 0.7),   # 70%
        int(total_frames * 0.9)    # 90%
    ]
    
    print(f"🏒 Analyzing {len(test_frames)} frames from hockey video")
    print(f"Expected on ice: 10-17 people (players + refs + goalies)")
    print("=" * 60)
    
    results = []
    
    for i, frame_idx in enumerate(test_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            analysis = detector.analyze_frame_comprehensive(frame)
            results.append(analysis)
            
            print(f"\n📊 Frame {i+1} (at {frame_idx/total_frames*100:.0f}% through video):")
            print(f"   Total people detected: {analysis['total_people_detected']}")
            print(f"   👥 On ice: {analysis['on_ice_count']} {'✅' if analysis['ice_count_reasonable'] else '⚠️'}")
            print(f"   🏒 On bench: {analysis['bench_count']}")
            print(f"   👥 In crowd: {analysis['crowd_count']}")
            print(f"   ❓ Uncertain: {analysis['uncertain_count']}")
            
            # Show confidence levels of ice players
            ice_people = analysis['categorized_people']['likely_on_ice']
            if ice_people:
                confidences = [p['confidence'] for p in ice_people]
                avg_conf = sum(confidences) / len(confidences)
                print(f"   📈 Avg confidence on ice: {avg_conf:.2f}")
    
    cap.release()
    
    # Summary across all frames
    if results:
        ice_counts = [r['on_ice_count'] for r in results]
        total_counts = [r['total_people_detected'] for r in results]
        
        print(f"\n📈 SUMMARY:")
        print(f"   Average people on ice: {sum(ice_counts)/len(ice_counts):.1f}")
        print(f"   Range on ice: {min(ice_counts)} - {max(ice_counts)}")
        print(f"   Average total detected: {sum(total_counts)/len(total_counts):.1f}")
        
        reasonable_counts = sum(1 for r in results if r['ice_count_reasonable'])
        print(f"   Reasonable ice counts: {reasonable_counts}/{len(results)} frames")
        
        if sum(ice_counts)/len(ice_counts) >= 8:
            print("   ✅ Detection looks good for hockey analysis!")
        else:
            print("   ⚠️  May need to lower confidence threshold or adjust filtering")

if __name__ == "__main__":
    realistic_hockey_analysis()