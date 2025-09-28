import cv2
import os
import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import time

# Import our components
from ..detection.player_detector import PlayerDetector
from ..detection.ice_detector import IceDetector
from ..detection.puck_tracker import PuckTracker
from .team_identifier import TeamIdentifier, RefereeDetector
from .play_analyzer import PlayAnalyzer, PlayEvent
from ..calibration.rink_calibrator import HockeyRinkCalibrator

@dataclass
class GameStats:
    """Overall game statistics"""
    total_frames: int
    processing_time: float
    avg_fps: float
    player_detection_rate: float
    puck_detection_rate: float
    team_confidence: float
    total_plays: int
    zone_entries: int
    turnovers: int
    shots: int
    
@dataclass 
class GameAnalysisResult:
    """Complete game analysis result"""
    game_info: Dict
    game_stats: GameStats
    play_events: List[PlayEvent]
    team_info: Dict
    zone_analysis: Dict
    timestamps: List[float]

class HockeyGameAnalyzer:
    """
    Main class for complete hockey game analysis
    """
    
    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize all components
        print("🏒 Initializing Hockey Game Analyzer...")
        self.player_detector = PlayerDetector()
        self.ice_detector = IceDetector()  
        self.puck_tracker = PuckTracker(ice_detector=self.ice_detector)
        self.team_identifier = TeamIdentifier(expected_teams=2)
        self.referee_detector = RefereeDetector()
        self.play_analyzer = PlayAnalyzer()
        self.calibrator = HockeyRinkCalibrator()
        
        # Analysis state
        self.frame_count = 0
        self.start_time = None
        self.all_play_events = []
        self.processing_stats = []
        
    def analyze_game(self, video_path: str, max_frames: Optional[int] = None, 
                    create_debug_video: bool = True) -> GameAnalysisResult:
        """
        Analyze complete hockey game
        """
        print(f"📹 Starting game analysis: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
        
        print(f"📊 Video info: {fps} fps, {total_frames} frames ({duration:.1f}s), {width}x{height}")
        print(f"🎯 Processing {frames_to_process} frames...")
        
        # Setup debug video if requested
        debug_writer = None
        if create_debug_video:
            debug_path = os.path.join(self.output_dir, "game_analysis_debug.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (width, height))
            print(f"📹 Debug video will be saved to: {debug_path}")
        
        # Initialize calibration (simplified - would need manual calibration for production)
        self._initialize_calibration(width, height)
        
        # Analysis statistics
        player_detections = 0
        puck_detections = 0  
        team_confidences = []
        processing_times = []
        
        # Ice detection cache - detect once and reuse
        ice_mask = None
        ice_detection_interval = 30  # Frames between ice re-detection
        
        self.start_time = time.time()
        
        try:
            while self.frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start_time = time.time()
                timestamp = self.frame_count / fps
                
                # Step 1: Ice surface detection (cache and reuse for performance)
                if self.frame_count % ice_detection_interval == 0:
                    ice_mask = self.ice_detector.detect_ice_surface(frame)
                    if self.frame_count == 0:
                        print("✅ Initial ice surface detection completed")
                        ice_pixels = np.sum(ice_mask > 0) if ice_mask is not None else 0
                        ice_percentage = (ice_pixels / ice_mask.size) * 100 if ice_mask is not None else 0
                        print(f"🏒 Ice coverage: {ice_percentage:.1f}% of frame")
                
                # Step 2: Player detection WITH polygon masking (EFFICIENT!)
                # YOLO only processes ice area - much faster and no crowd noise
                players = self.player_detector.detect_players(frame, ice_mask=ice_mask)
                if players:
                    player_detections += 1
                
                # Step 3: Team identification
                player_teams = self.team_identifier.identify_teams(frame, players)
                if player_teams:
                    avg_confidence = np.mean([pt.team_confidence for pt in player_teams])
                    team_confidences.append(avg_confidence)
                
                # Step 4: Referee detection
                referees = self.referee_detector.detect_referees(frame, players)
                
                # Step 5: Puck tracking (with ice mask filtering)
                puck_candidates = self.puck_tracker.detect_puck_candidates(frame, players, ice_mask=ice_mask)
                tracked_puck = self.puck_tracker.track_puck(puck_candidates)
                if tracked_puck:
                    puck_detections += 1
                
                puck_position = tracked_puck.position if tracked_puck else None
                
                # Step 6: Play analysis
                play_events = self.play_analyzer.analyze_frame(
                    self.frame_count, timestamp, players, player_teams, puck_position
                )
                self.all_play_events.extend(play_events)
                
                # Step 7: Create visualization for debug video
                if debug_writer:
                    vis_frame = self._create_debug_visualization(
                        frame, players, player_teams, referees, 
                        puck_candidates, tracked_puck, play_events
                    )
                    debug_writer.write(vis_frame)
                
                # Track processing time
                frame_time = time.time() - frame_start_time
                processing_times.append(frame_time)
                
                self.frame_count += 1
                
                # Progress updates
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    progress = self.frame_count / frames_to_process * 100
                    avg_fps = self.frame_count / elapsed
                    
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{frames_to_process} frames)")
                    print(f"  Processing: {avg_fps:.1f} fps, Elapsed: {elapsed:.1f}s")
                    print(f"  🏒 Ice-only players: {len(players)}, Puck: {'✅' if tracked_puck else '❌'}")
                    
                    # Show polygon masking effectiveness
                    if ice_mask is not None:
                        ice_pixels = np.sum(ice_mask > 0)
                        ice_percentage = (ice_pixels / ice_mask.size) * 100
                        print(f"  🎯 Ice ROI: {ice_percentage:.1f}% - polygon masking active")
                    
                    # Show recent play events
                    recent_plays = [pe for pe in play_events if pe.frame_number >= self.frame_count - 30]
                    if recent_plays:
                        print(f"  Recent plays: {[pe.play_type.value for pe in recent_plays]}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if debug_writer:
                debug_writer.release()
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_processing_time = time.time() - self.start_time
        avg_processing_fps = self.frame_count / total_processing_time if total_processing_time > 0 else 0
        
        game_stats = GameStats(
            total_frames=self.frame_count,
            processing_time=total_processing_time,
            avg_fps=avg_processing_fps,
            player_detection_rate=player_detections / max(self.frame_count, 1),
            puck_detection_rate=puck_detections / max(self.frame_count, 1),
            team_confidence=np.mean(team_confidences) if team_confidences else 0.0,
            total_plays=len(self.all_play_events),
            zone_entries=len([pe for pe in self.all_play_events if pe.play_type.value == 'zone_entry']),
            turnovers=len([pe for pe in self.all_play_events if pe.play_type.value == 'turnover']),
            shots=len([pe for pe in self.all_play_events if pe.play_type.value == 'shot'])
        )
        
        # Generate comprehensive analysis
        result = self._generate_analysis_result(video_path, game_stats)
        
        # Save results
        self._save_analysis_results(result)
        
        return result
    
    def _initialize_calibration(self, width: int, height: int):
        """Initialize basic calibration - would be enhanced for production"""
        # Simple zone setup based on frame width
        zone_width = width // 3
        
        zones = {
            "left_zone": {"bounds": (0, zone_width)},
            "neutral_zone": {"bounds": (zone_width, zone_width * 2)},
            "right_zone": {"bounds": (zone_width * 2, width)}
        }
        
        self.play_analyzer.update_zone_boundaries(zones)
        print(f"✅ Initialized zones: Left(0-{zone_width}), Neutral({zone_width}-{zone_width*2}), Right({zone_width*2}-{width})")
    
    def _create_debug_visualization(self, frame, players, player_teams, referees, 
                                  puck_candidates, tracked_puck, play_events):
        """Create comprehensive debug visualization"""
        vis_frame = frame.copy()
        
        # 1. Draw players with team colors
        vis_frame = self.team_identifier.visualize_team_identification(
            vis_frame, players, player_teams
        )
        
        # 2. Mark referees
        for ref_idx in referees:
            if ref_idx < len(players):
                bbox = players[ref_idx]['bbox']
                x, y, w, h = bbox
                cv2.rectangle(vis_frame, (x-3, y-3), (x+w+3, y+h+3), (255, 255, 255), 3)
                cv2.putText(vis_frame, "REF", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 3. Draw puck tracking
        vis_frame = self.puck_tracker.visualize_tracking(
            vis_frame, puck_candidates, tracked_puck
        )
        
        # 4. Draw play analysis
        recent_plays = self.play_analyzer.get_recent_plays(5.0)
        vis_frame = self.play_analyzer.visualize_plays(vis_frame, recent_plays)
        
        # 5. Add comprehensive info overlay
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Players: {len(players)} (Refs: {len(referees)})",
            f"Puck: {'TRACKED' if tracked_puck else 'LOST'}",
            f"Candidates: {len(puck_candidates)}",
            f"Total Plays: {len(self.all_play_events)}",
            f"Recent Events: {len(play_events)}"
        ]
        
        # Add team stats
        team_stats = self.team_identifier.get_team_stats(player_teams)
        for team_id, count in team_stats.get('team_counts', {}).items():
            info_lines.append(f"Team {team_id+1}: {count}")
        
        # Draw info box background
        box_height = len(info_lines) * 25 + 20
        cv2.rectangle(vis_frame, (10, 10), (300, box_height), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (300, box_height), (255, 255, 255), 2)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            cv2.putText(vis_frame, line, (20, 35 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def _generate_analysis_result(self, video_path: str, game_stats: GameStats) -> GameAnalysisResult:
        """Generate comprehensive analysis result"""
        
        # Game info
        game_info = {
            'video_path': video_path,
            'analysis_date': datetime.now().isoformat(),
            'video_duration': self.frame_count / 30.0,  # Assume 30fps
            'total_frames_analyzed': self.frame_count
        }
        
        # Team information
        team_info = {
            'teams': self.team_identifier.get_team_info(),
            'team_stats': self.team_identifier.get_team_stats([])  # Overall stats
        }
        
        # Zone analysis
        play_stats = self.play_analyzer.get_play_statistics(time_window=float('inf'))  # All plays
        zone_analysis = {
            'zone_entries_by_team': {},
            'time_in_zones': {},
            'play_distribution': play_stats
        }
        
        # Create timestamps array
        timestamps = [i / 30.0 for i in range(self.frame_count)]
        
        return GameAnalysisResult(
            game_info=game_info,
            game_stats=game_stats,
            play_events=self.all_play_events,
            team_info=team_info,
            zone_analysis=zone_analysis,
            timestamps=timestamps
        )
    
    def _save_analysis_results(self, result: GameAnalysisResult):
        """Save analysis results to files"""
        
        # Save main analysis JSON
        analysis_path = os.path.join(self.output_dir, "game_analysis.json")
        
        # Convert dataclasses to dict for JSON serialization
        json_data = {
            'game_info': result.game_info,
            'game_stats': asdict(result.game_stats),
            'team_info': result.team_info,
            'zone_analysis': result.zone_analysis,
            'play_events': [self._play_event_to_dict(pe) for pe in result.play_events],
            'timestamps': result.timestamps[:100]  # Limit size
        }
        
        with open(analysis_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Analysis saved to: {analysis_path}")
        
        # Save play events as CSV for easy analysis
        self._save_play_events_csv(result.play_events)
        
        # Generate summary report
        self._generate_summary_report(result)
    
    def _play_event_to_dict(self, play_event: PlayEvent) -> Dict:
        """Convert PlayEvent to dictionary"""
        return {
            'play_type': play_event.play_type.value,
            'timestamp': play_event.timestamp,
            'frame_number': play_event.frame_number,
            'zone': play_event.zone.value,
            'team_id': play_event.team_id,
            'description': play_event.description,
            'confidence': play_event.confidence,
            'puck_position': play_event.puck_position
        }
    
    def _save_play_events_csv(self, play_events: List[PlayEvent]):
        """Save play events as CSV"""
        import csv
        
        csv_path = os.path.join(self.output_dir, "play_events.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'play_type', 'zone', 'team_id', 'description', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for event in play_events:
                writer.writerow({
                    'timestamp': f"{event.timestamp:.2f}",
                    'play_type': event.play_type.value,
                    'zone': event.zone.value,
                    'team_id': event.team_id,
                    'description': event.description,
                    'confidence': f"{event.confidence:.2f}"
                })
        
        print(f"📊 Play events CSV saved to: {csv_path}")
    
    def _generate_summary_report(self, result: GameAnalysisResult):
        """Generate human-readable summary report"""
        report_path = os.path.join(self.output_dir, "game_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("🏒 HOCKEY GAME ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Video: {result.game_info['video_path']}\n")
            f.write(f"Analysis Date: {result.game_info['analysis_date']}\n")
            f.write(f"Game Duration: {result.game_info['video_duration']:.1f} seconds\n")
            f.write(f"Frames Analyzed: {result.game_info['total_frames_analyzed']}\n\n")
            
            # Performance stats
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Processing Speed: {result.game_stats.avg_fps:.1f} fps\n")
            f.write(f"  Player Detection Rate: {result.game_stats.player_detection_rate*100:.1f}%\n")
            f.write(f"  Puck Detection Rate: {result.game_stats.puck_detection_rate*100:.1f}%\n")
            f.write(f"  Team ID Confidence: {result.game_stats.team_confidence*100:.1f}%\n\n")
            
            # Game stats
            f.write("GAME STATISTICS:\n")
            f.write(f"  Total Plays Detected: {result.game_stats.total_plays}\n")
            f.write(f"  Zone Entries: {result.game_stats.zone_entries}\n")
            f.write(f"  Turnovers: {result.game_stats.turnovers}\n")
            f.write(f"  Shots: {result.game_stats.shots}\n\n")
            
            # Play breakdown
            if result.zone_analysis.get('play_distribution'):
                f.write("PLAY TYPE DISTRIBUTION:\n")
                play_dist = result.zone_analysis['play_distribution'].get('play_types', {})
                for play_type, count in sorted(play_dist.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {play_type.title()}: {count}\n")
                f.write("\n")
            
            # Team information
            f.write("TEAM INFORMATION:\n")
            teams = result.team_info.get('teams', {})
            for team_id, team_data in teams.items():
                if hasattr(team_data, 'team_name'):
                    # team_data is a TeamColors dataclass
                    f.write(f"  Team {team_id + 1}: {team_data.team_name}\n")
                    f.write(f"    Primary Color: RGB{team_data.primary_color}\n")
                    f.write(f"    Confidence: {team_data.confidence*100:.1f}%\n")
                else:
                    # team_data is a dictionary
                    f.write(f"  Team {team_id + 1}: {team_data.get('team_name', 'Unknown')}\n")
                    f.write(f"    Primary Color: RGB{team_data.get('primary_color', 'Unknown')}\n")
                    f.write(f"    Confidence: {team_data.get('confidence', 0)*100:.1f}%\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Analysis complete! Review debug video and play events CSV for detailed insights.\n")
        
        print(f"📋 Summary report saved to: {report_path}")

def main():
    """Main entry point for game analysis"""
    parser = argparse.ArgumentParser(description='Hockey Game Analysis Tool')
    parser.add_argument('video', help='Path to hockey video file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (for testing)')
    parser.add_argument('--no-debug-video', action='store_true', help='Skip debug video creation')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HockeyGameAnalyzer(output_dir=args.output_dir)
    
    try:
        # Run analysis
        result = analyzer.analyze_game(
            video_path=args.video,
            max_frames=args.max_frames,
            create_debug_video=not args.no_debug_video
        )
        
        print("\n" + "="*60)
        print("🏒 GAME ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Processed {result.game_stats.total_frames} frames")
        print(f"⚡ Average processing speed: {result.game_stats.avg_fps:.1f} fps")
        print(f"🎯 Detected {result.game_stats.total_plays} total plays")
        print(f"📹 Check {args.output_dir}/ for detailed results")
        
        # Show top play types
        play_dist = result.zone_analysis.get('play_distribution', {}).get('play_types', {})
        if play_dist:
            print(f"\n📈 Top play types:")
            for play_type, count in sorted(play_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {play_type}: {count}")
        
        print("\n✅ Analysis complete! Check the output directory for results.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()