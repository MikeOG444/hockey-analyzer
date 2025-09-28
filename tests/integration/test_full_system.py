import cv2
import os
import time
from game_analyzer import HockeyGameAnalyzer

def test_complete_system():
    """Test the complete hockey analysis system end-to-end"""
    
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        print("Please ensure you have a test video in the data/test_videos/ directory")
        return
    
    print("🏒 COMPLETE HOCKEY ANALYSIS SYSTEM TEST")
    print("=" * 60)
    
    # Create analyzer
    output_dir = "full_system_test_output"
    analyzer = HockeyGameAnalyzer(output_dir=output_dir)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Test with limited frames for demonstration
    max_frames = 300  # ~10 seconds at 30fps
    
    print(f"🎯 Testing with {max_frames} frames (~10 seconds of video)")
    print("This is a demonstration - for full game analysis, remove the max_frames limit\n")
    
    try:
        # Run complete analysis
        start_time = time.time()
        
        result = analyzer.analyze_game(
            video_path=video_path,
            max_frames=max_frames,
            create_debug_video=True
        )
        
        total_time = time.time() - start_time
        
        # Display comprehensive results
        print("\n" + "🎉 ANALYSIS COMPLETE!" + "\n" + "=" * 60)
        
        # Performance metrics
        print("⚡ PERFORMANCE METRICS:")
        print(f"  Total processing time: {total_time:.1f} seconds")
        print(f"  Frames processed: {result.game_stats.total_frames:,}")
        print(f"  Processing speed: {result.game_stats.avg_fps:.1f} fps")
        print(f"  Real-time factor: {(result.game_info['video_duration']/total_time):.1f}x")
        
        # Detection accuracy
        print(f"\n🎯 DETECTION ACCURACY:")
        print(f"  Player detection rate: {result.game_stats.player_detection_rate*100:.1f}%")
        print(f"  Puck detection rate: {result.game_stats.puck_detection_rate*100:.1f}%")  
        print(f"  Team identification confidence: {result.game_stats.team_confidence*100:.1f}%")
        
        # Game analysis results
        print(f"\n🏒 GAME ANALYSIS:")
        print(f"  Total plays detected: {result.game_stats.total_plays}")
        print(f"  Zone entries: {result.game_stats.zone_entries}")
        print(f"  Turnovers: {result.game_stats.turnovers}")
        print(f"  Shots detected: {result.game_stats.shots}")
        
        # Team information
        print(f"\n👥 TEAM INFORMATION:")
        teams = result.team_info.get('teams', {})
        for team_id, team_data in teams.items():
            print(f"  Team {team_id + 1}: {team_data.team_name}")
            color = team_data.primary_color
            confidence = team_data.confidence
            print(f"    Jersey Color: RGB{color}")
            print(f"    Confidence: {confidence*100:.1f}%")
        
        # Play breakdown
        if result.zone_analysis.get('play_distribution'):
            print(f"\n🎭 PLAY TYPE BREAKDOWN:")
            play_types = result.zone_analysis['play_distribution'].get('play_types', {})
            for play_type, count in sorted(play_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {play_type.title()}: {count}")
        
        # Recent significant events
        print(f"\n🔍 RECENT SIGNIFICANT EVENTS:")
        if result.play_events:
            # Show last few events
            for event in result.play_events[-5:]:
                timestamp = f"{event.timestamp:.1f}s"
                print(f"  {timestamp}: {event.play_type.value} - {event.description}")
        
        # Output files
        print(f"\n📁 OUTPUT FILES GENERATED:")
        output_files = [
            "game_analysis.json",
            "play_events.csv", 
            "game_summary.txt",
            "game_analysis_debug.mp4"
        ]
        
        for filename in output_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✅ {filename} ({size:,} bytes)")
            else:
                print(f"  ❌ {filename} (not found)")
        
        # Recommendations for coaches
        print(f"\n💡 COACHING INSIGHTS:")
        
        # Detection quality assessment
        if result.game_stats.player_detection_rate > 0.8:
            print("  ✅ Excellent player tracking - reliable for detailed analysis")
        elif result.game_stats.player_detection_rate > 0.6:
            print("  ⚠️  Good player tracking - suitable for general analysis")
        else:
            print("  ⚠️  Player tracking needs improvement - check camera angle/quality")
        
        # Puck tracking assessment
        if result.game_stats.puck_detection_rate > 0.4:
            print("  ✅ Good puck tracking - play analysis should be accurate")
        elif result.game_stats.puck_detection_rate > 0.2:
            print("  ⚠️  Moderate puck tracking - some plays may be missed")
        else:
            print("  ⚠️  Low puck detection - consider camera positioning or quality")
        
        # Team identification assessment
        if result.game_stats.team_confidence > 0.7:
            print("  ✅ Strong team identification - player assignments reliable")
        elif result.game_stats.team_confidence > 0.5:
            print("  ⚠️  Moderate team identification - verify player assignments")
        else:
            print("  ⚠️  Weak team identification - manual verification recommended")
        
        # Usage recommendations
        print(f"\n🎯 NEXT STEPS FOR COACHES:")
        print("  1. Review the debug video to verify detection accuracy")
        print("  2. Check play_events.csv for detailed event timeline")
        print("  3. Use game_summary.txt for quick overview")
        print("  4. Run coaching dashboard for interactive analysis:")
        print("     streamlit run coaching_dashboard.py")
        
        # Performance recommendations
        if result.game_stats.avg_fps < 5:
            print("  ⚠️  Processing is slow - consider using a more powerful computer")
        elif result.game_stats.avg_fps > 10:
            print("  ✅ Good processing speed - suitable for regular use")
        
        print(f"\n🏒 System test completed successfully!")
        print(f"📊 Check {output_dir}/ directory for all analysis results")
        
    except Exception as e:
        print(f"❌ Error during system test: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """Test individual components separately"""
    
    video_path = "data/test_videos/CP_CT_North_Auto_Full_Pan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
    
    print("🧪 INDIVIDUAL COMPONENT TESTING")
    print("=" * 50)
    
    # Test each component
    test_scripts = [
        ("Player Detection", "python test_model_comparison.py"),
        ("Ice Surface Detection", "python test_ice_markings.py"),
        ("Puck Tracking", "python test_puck_tracking.py"),
        ("Team Identification", "python test_team_identification.py")
    ]
    
    print("Run these individual tests to verify each component:")
    for component, command in test_scripts:
        print(f"  {component}: {command}")
    
    print(f"\nOr run the complete system test above.")

if __name__ == "__main__":
    print("🏒 Hockey Analysis System Test Suite")
    print("Choose test mode:")
    print("1. Complete system test (recommended)")
    print("2. Individual component testing info")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_individual_components()
    else:
        test_complete_system()