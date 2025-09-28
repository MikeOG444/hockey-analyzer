import cv2
import os
import numpy as np
from src.detection.basic_detector import BasicHockeyDetector
from src.detection.ice_detection import IceSurfaceDetector
from src.analysis.team_identifier import TeamIdentifier, RefereeDetector
import time

def test_team_identification():
    """Test team identification system"""
    
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
    
    print("🏒 Initializing Team Identification System...")
    
    # Initialize components
    player_detector = BasicHockeyDetector()
    ice_detector = IceSurfaceDetector()
    team_identifier = TeamIdentifier(expected_teams=2)
    referee_detector = RefereeDetector()
    
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
    
    # Create output video for debugging
    debug_output_path = "debug_team_identification.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(debug_output_path, fourcc, fps, (width, height))
    
    # Statistics tracking
    frame_count = 0
    team_stats = []
    processing_times = []
    
    # Process subset of frames
    test_frames = min(200, total_frames)
    print(f"🎯 Processing first {test_frames} frames for team identification test...")
    
    try:
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Detect ice surface (every 30 frames)
            if frame_count % 30 == 0:
                ice_mask = ice_detector.detect_ice_surface(frame)
                print(f"Frame {frame_count}: Ice surface updated")
            
            # Detect players
            players = player_detector.detect_players(frame)
            
            # Identify teams
            player_teams = team_identifier.identify_teams(frame, players)
            
            # Detect referees
            referees = referee_detector.detect_referees(frame, players)
            
            # Get team statistics
            frame_team_stats = team_identifier.get_team_stats(player_teams)
            team_stats.append(frame_team_stats)
            
            # Create visualization
            vis_frame = team_identifier.visualize_team_identification(frame, players, player_teams)
            
            # Mark referees
            for ref_idx in referees:
                if ref_idx < len(players):
                    bbox = players[ref_idx]['bbox']
                    x, y, w, h = bbox
                    # Draw referee indicator
                    cv2.rectangle(vis_frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 255), 3)
                    cv2.putText(vis_frame, "REF", (x, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add frame info
            info_text = [
                f"Frame: {frame_count}/{test_frames}",
                f"Players: {len(players)}",
                f"Team 1: {frame_team_stats['team_counts'].get(0, 0)} players",
                f"Team 2: {frame_team_stats['team_counts'].get(1, 0)} players",
                f"Referees: {len(referees)}",
                f"Avg Confidence: {frame_team_stats['avg_confidence']:.2f}",
                f"Teams Detected: {frame_team_stats['teams_detected']}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(vis_frame, text, (10, height - 200 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write debug frame
            out.write(vis_frame)
            
            # Track processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 50 == 0:
                avg_time = np.mean(processing_times[-50:])
                print(f"Processed {frame_count} frames - Avg: {avg_time:.3f}s/frame")
                
                # Show current team info
                team_info = team_identifier.get_team_info()
                for team_id, colors in team_info.items():
                    print(f"  {colors.team_name}: RGB{colors.primary_color}, conf={colors.confidence:.2f}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final analysis
        print("\n" + "="*60)
        print("🏒 TEAM IDENTIFICATION TEST RESULTS")
        print("="*60)
        
        if team_stats:
            # Calculate overall statistics
            avg_players_per_frame = np.mean([stats['total_players'] for stats in team_stats])
            avg_confidence = np.mean([stats['avg_confidence'] for stats in team_stats])
            
            # Team distribution analysis
            team1_counts = [stats['team_counts'].get(0, 0) for stats in team_stats]
            team2_counts = [stats['team_counts'].get(1, 0) for stats in team_stats]
            
            print(f"Frames analyzed: {frame_count}")
            print(f"Average players per frame: {avg_players_per_frame:.1f}")
            print(f"Average team assignment confidence: {avg_confidence:.2f}")
            print(f"Team 1 average: {np.mean(team1_counts):.1f} players")
            print(f"Team 2 average: {np.mean(team2_counts):.1f} players")
            
            # Team color information
            final_team_info = team_identifier.get_team_info()
            print(f"\nFinal Team Colors:")
            for team_id, colors in final_team_info.items():
                r, g, b = colors.primary_color
                print(f"  {colors.team_name}: RGB({r}, {g}, {b}) - Confidence: {colors.confidence:.2f}")
            
            # Performance stats
            if processing_times:
                avg_processing_time = np.mean(processing_times)
                print(f"\nPerformance:")
                print(f"  Average processing time: {avg_processing_time:.3f}s/frame")
                print(f"  Real-time factor: {(1/30)/avg_processing_time:.2f}x")
        
        print(f"\n📹 Debug video saved: {debug_output_path}")
        
        # Recommendations
        print("\n💡 Analysis:")
        if avg_confidence < 0.5:
            print("  ⚠️  Low team identification confidence. Consider:")
            print("    • Better lighting conditions")
            print("    • More distinct team jersey colors")
            print("    • Manual team color calibration")
        elif avg_confidence > 0.7:
            print("  ✅ Good team identification performance!")
        
        balance_ratio = abs(np.mean(team1_counts) - np.mean(team2_counts))
        if balance_ratio > 2:
            print("  ⚠️  Unbalanced team detection. Possible issues:")
            print("    • One team has more distinctive colors")
            print("    • Camera angle favoring one side")
            print("    • Different jersey visibility")
        else:
            print("  ✅ Balanced team detection")

def test_color_extraction():
    """Test color extraction from individual players"""
    
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
    
    # Initialize components
    player_detector = BasicHockeyDetector()
    team_identifier = TeamIdentifier()
    
    cap = cv2.VideoCapture(video_path)
    
    # Get a frame with many players (frame 100)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read test frame")
        return
    
    print("🎨 Testing color extraction from players...")
    
    # Detect players
    players = player_detector.detect_players(frame)
    print(f"Found {len(players)} players in test frame")
    
    # Create color analysis visualization
    vis_frame = frame.copy()
    
    for i, player in enumerate(players):
        # Extract jersey colors
        colors = team_identifier._extract_jersey_colors(frame, player)
        
        if colors:
            print(f"\nPlayer {i+1}:")
            for j, color in enumerate(colors):
                print(f"  Color {j+1}: RGB{color}")
            
            # Visualize on frame
            bbox = player['bbox']
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(vis_frame, f"P{i+1}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw jersey region
            jersey_region = team_identifier._get_jersey_region(player)
            jx, jy, jw, jh = jersey_region
            cv2.rectangle(vis_frame, (jx, jy), (jx+jw, jy+jh), (0, 255, 0), 1)
            
            # Show extracted colors
            for j, color in enumerate(colors[:3]):  # Show up to 3 colors
                # Convert RGB to BGR for OpenCV
                bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                cv2.rectangle(vis_frame, (x + j*20, y + h + 10), 
                             (x + j*20 + 15, y + h + 25), bgr_color, -1)
        else:
            print(f"\nPlayer {i+1}: No colors extracted")
    
    # Save color analysis image
    cv2.imwrite("debug_color_extraction.jpg", vis_frame)
    print(f"\n📸 Color extraction visualization saved: debug_color_extraction.jpg")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Full team identification test (with debug video)")
    print("2. Color extraction analysis")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_color_extraction()
    else:
        test_team_identification()