import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import math
import time

class PlayType(Enum):
    """Types of hockey plays"""
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    PASS = "pass"
    SHOT = "shot"
    SAVE = "save"
    TURNOVER = "turnover"
    FACEOFF = "faceoff"
    POWER_PLAY = "power_play"
    PENALTY_KILL = "penalty_kill"
    LINE_CHANGE = "line_change"
    BREAKAWAY = "breakaway"
    TWO_ON_ONE = "two_on_one"
    THREE_ON_TWO = "three_on_two"

class Zone(Enum):
    """Hockey rink zones"""
    DEFENSIVE_ZONE = "defensive"
    NEUTRAL_ZONE = "neutral"
    OFFENSIVE_ZONE = "offensive"
    UNKNOWN = "unknown"

@dataclass
class PlayEvent:
    """Represents a detected play event"""
    play_type: PlayType
    timestamp: float
    frame_number: int
    zone: Zone
    team_id: int
    players_involved: List[int]
    puck_position: Optional[Tuple[int, int]]
    description: str
    confidence: float
    duration: float = 0.0
    outcome: Optional[str] = None

@dataclass
class ZoneState:
    """Current state of a zone"""
    zone: Zone
    team1_players: int
    team2_players: int
    puck_present: bool
    possession_team: Optional[int]

class PlayAnalyzer:
    """
    Analyze hockey plays and generate coaching insights
    """
    
    def __init__(self, calibrator=None):
        self.calibrator = calibrator
        self.play_history = deque(maxlen=1000)  # Store recent plays
        self.zone_history = deque(maxlen=60)    # 2 seconds of zone states at 30fps
        self.possession_history = deque(maxlen=30)  # 1 second of possession
        
        # Play detection thresholds
        self.zone_entry_threshold = 50  # pixels moved into zone
        self.pass_distance_min = 80     # minimum pixels for pass
        self.pass_velocity_min = 30     # minimum velocity for pass
        self.shot_zone_radius = 150     # pixels from goal for shot detection
        self.breakaway_distance = 200   # minimum lead distance for breakaway
        
        # Time tracking
        self.frame_time = 1.0 / 30.0    # Assume 30fps
        self.play_sequence_timeout = 3.0  # seconds
        
        # Zone boundaries (will be updated by calibrator)
        self.zone_boundaries = {
            Zone.DEFENSIVE_ZONE: (0, 200),      # Left zone
            Zone.NEUTRAL_ZONE: (200, 600),      # Center zone  
            Zone.OFFENSIVE_ZONE: (600, 800)     # Right zone
        }
        
    def update_zone_boundaries(self, zones_dict: Dict):
        """Update zone boundaries from calibrator"""
        if "left_zone" in zones_dict:
            left_bounds = zones_dict["left_zone"]["bounds"]
            self.zone_boundaries[Zone.DEFENSIVE_ZONE] = left_bounds
            
        if "neutral_zone" in zones_dict:
            neutral_bounds = zones_dict["neutral_zone"]["bounds"]
            self.zone_boundaries[Zone.NEUTRAL_ZONE] = neutral_bounds
            
        if "right_zone" in zones_dict:
            right_bounds = zones_dict["right_zone"]["bounds"]
            self.zone_boundaries[Zone.OFFENSIVE_ZONE] = right_bounds
    
    def analyze_frame(self, frame_number: int, timestamp: float, 
                     players: List[Dict], player_teams: List, 
                     puck_position: Optional[Tuple[int, int]]) -> List[PlayEvent]:
        """
        Analyze a single frame and detect play events
        """
        detected_plays = []
        
        # Get current zone state
        current_zone_state = self._analyze_zone_state(players, player_teams, puck_position)
        self.zone_history.append(current_zone_state)
        
        # Update possession tracking
        current_possession = self._determine_possession(players, player_teams, puck_position)
        self.possession_history.append(current_possession)
        
        # Detect various play types
        if len(self.zone_history) >= 10:  # Need some history for analysis
            
            # Zone entries/exits
            zone_plays = self._detect_zone_changes(frame_number, timestamp, puck_position)
            detected_plays.extend(zone_plays)
            
            # Possession changes and turnovers
            possession_plays = self._detect_possession_changes(frame_number, timestamp, puck_position)
            detected_plays.extend(possession_plays)
            
            # Passing plays
            pass_plays = self._detect_passes(frame_number, timestamp, players, player_teams, puck_position)
            detected_plays.extend(pass_plays)
            
            # Offensive plays (shots, breakaways, etc.)
            offensive_plays = self._detect_offensive_plays(frame_number, timestamp, players, player_teams, puck_position)
            detected_plays.extend(offensive_plays)
            
            # Special situations
            special_plays = self._detect_special_situations(frame_number, timestamp, players, player_teams)
            detected_plays.extend(special_plays)
        
        # Store detected plays
        for play in detected_plays:
            self.play_history.append(play)
        
        return detected_plays
    
    def _analyze_zone_state(self, players: List[Dict], player_teams: List, 
                           puck_position: Optional[Tuple[int, int]]) -> Dict[Zone, ZoneState]:
        """Analyze current state of each zone"""
        zone_states = {}
        
        for zone in Zone:
            if zone == Zone.UNKNOWN:
                continue
                
            team1_count = 0
            team2_count = 0
            puck_in_zone = False
            
            zone_bounds = self.zone_boundaries.get(zone, (0, 800))
            left_bound, right_bound = zone_bounds
            
            # Count players in zone
            for player, team_info in zip(players, player_teams):
                player_x = player['position'][0]
                if left_bound <= player_x <= right_bound:
                    if hasattr(team_info, 'team_id'):
                        if team_info.team_id == 0:
                            team1_count += 1
                        elif team_info.team_id == 1:
                            team2_count += 1
                    else:
                        # Fallback if team_info is different format
                        team1_count += 1
            
            # Check if puck is in zone
            if puck_position:
                puck_x = puck_position[0]
                puck_in_zone = left_bound <= puck_x <= right_bound
            
            # Determine possession team in zone
            possession_team = None
            if puck_in_zone:
                if team1_count > team2_count:
                    possession_team = 0
                elif team2_count > team1_count:
                    possession_team = 1
            
            zone_states[zone] = ZoneState(
                zone=zone,
                team1_players=team1_count,
                team2_players=team2_count,
                puck_present=puck_in_zone,
                possession_team=possession_team
            )
        
        return zone_states
    
    def _determine_possession(self, players: List[Dict], player_teams: List, 
                            puck_position: Optional[Tuple[int, int]]) -> Optional[int]:
        """Determine which team has puck possession"""
        if not puck_position or not players:
            return None
        
        min_distance = float('inf')
        closest_team = None
        
        for player, team_info in zip(players, player_teams):
            player_pos = player['position']
            distance = math.sqrt((puck_position[0] - player_pos[0])**2 + 
                               (puck_position[1] - player_pos[1])**2)
            
            if distance < min_distance and distance < 60:  # Within possession range
                min_distance = distance
                if hasattr(team_info, 'team_id'):
                    closest_team = team_info.team_id
        
        return closest_team
    
    def _detect_zone_changes(self, frame_number: int, timestamp: float, 
                           puck_position: Optional[Tuple[int, int]]) -> List[PlayEvent]:
        """Detect zone entries and exits"""
        plays = []
        
        if len(self.zone_history) < 10 or not puck_position:
            return plays
        
        # Get current and previous puck zones
        current_zone = self._get_puck_zone(puck_position)
        
        # Look back to find zone change
        prev_zones = []
        for i in range(min(10, len(self.zone_history))):
            zone_state = self.zone_history[-(i+1)]
            for zone, state in zone_state.items():
                if state.puck_present:
                    prev_zones.append(zone)
                    break
        
        if prev_zones and len(prev_zones) >= 2:
            prev_zone = prev_zones[0]
            
            if current_zone != prev_zone and current_zone != Zone.UNKNOWN:
                # Determine possession team
                possession_team = self.possession_history[-1] if self.possession_history else None
                
                if current_zone == Zone.OFFENSIVE_ZONE and prev_zone != Zone.OFFENSIVE_ZONE:
                    # Zone entry detected
                    plays.append(PlayEvent(
                        play_type=PlayType.ZONE_ENTRY,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        zone=current_zone,
                        team_id=possession_team if possession_team else 0,
                        players_involved=[],
                        puck_position=puck_position,
                        description=f"Zone entry into {current_zone.value}",
                        confidence=0.8
                    ))
                
                elif prev_zone == Zone.OFFENSIVE_ZONE and current_zone != Zone.OFFENSIVE_ZONE:
                    # Zone exit detected
                    plays.append(PlayEvent(
                        play_type=PlayType.ZONE_EXIT,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        zone=prev_zone,
                        team_id=possession_team if possession_team else 0,
                        players_involved=[],
                        puck_position=puck_position,
                        description=f"Zone exit from {prev_zone.value}",
                        confidence=0.8
                    ))
        
        return plays
    
    def _detect_possession_changes(self, frame_number: int, timestamp: float, 
                                 puck_position: Optional[Tuple[int, int]]) -> List[PlayEvent]:
        """Detect possession changes and turnovers"""
        plays = []
        
        if len(self.possession_history) < 5:
            return plays
        
        # Look for possession changes
        current_possession = self.possession_history[-1]
        prev_possession = self.possession_history[-3] if len(self.possession_history) >= 3 else None
        
        if (current_possession is not None and prev_possession is not None and 
            current_possession != prev_possession):
            
            current_zone = self._get_puck_zone(puck_position) if puck_position else Zone.UNKNOWN
            
            plays.append(PlayEvent(
                play_type=PlayType.TURNOVER,
                timestamp=timestamp,
                frame_number=frame_number,
                zone=current_zone,
                team_id=prev_possession,  # Team that lost possession
                players_involved=[],
                puck_position=puck_position,
                description=f"Turnover in {current_zone.value} zone",
                confidence=0.7
            ))
        
        return plays
    
    def _detect_passes(self, frame_number: int, timestamp: float, 
                      players: List[Dict], player_teams: List,
                      puck_position: Optional[Tuple[int, int]]) -> List[PlayEvent]:
        """Detect passing plays"""
        plays = []
        
        # This is a simplified pass detection - would need more sophisticated tracking
        # for production use (tracking puck trajectory, player intentions, etc.)
        
        if not puck_position or len(self.possession_history) < 10:
            return plays
        
        # Look for rapid puck movement between players of same team
        possession_changes = 0
        same_team_possession = 0
        
        for i in range(min(10, len(self.possession_history))):
            if i > 0:
                curr = self.possession_history[-(i+1)]
                prev = self.possession_history[-i]
                
                if curr != prev and curr is not None and prev is not None:
                    if curr == prev:  # Same team
                        same_team_possession += 1
                    possession_changes += 1
        
        # If puck moved quickly between players of same team, likely a pass
        if possession_changes >= 2 and same_team_possession > 0:
            current_possession = self.possession_history[-1]
            current_zone = self._get_puck_zone(puck_position)
            
            plays.append(PlayEvent(
                play_type=PlayType.PASS,
                timestamp=timestamp,
                frame_number=frame_number,
                zone=current_zone,
                team_id=current_possession if current_possession else 0,
                players_involved=[],
                puck_position=puck_position,
                description=f"Pass completed in {current_zone.value} zone",
                confidence=0.6
            ))
        
        return plays
    
    def _detect_offensive_plays(self, frame_number: int, timestamp: float, 
                              players: List[Dict], player_teams: List,
                              puck_position: Optional[Tuple[int, int]]) -> List[PlayEvent]:
        """Detect shots, breakaways, and other offensive plays"""
        plays = []
        
        if not puck_position:
            return plays
        
        current_zone = self._get_puck_zone(puck_position)
        
        # Detect potential shots (puck in offensive zone near goal)
        if current_zone == Zone.OFFENSIVE_ZONE:
            # Simplified shot detection - check if puck is near goal area
            # (would need goal post detection for accurate implementation)
            zone_bounds = self.zone_boundaries[Zone.OFFENSIVE_ZONE]
            goal_area_x = zone_bounds[1] - 100  # Assume goal is near zone end
            
            distance_to_goal = abs(puck_position[0] - goal_area_x)
            
            if distance_to_goal < self.shot_zone_radius:
                current_possession = self.possession_history[-1] if self.possession_history else None
                
                plays.append(PlayEvent(
                    play_type=PlayType.SHOT,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    zone=current_zone,
                    team_id=current_possession if current_possession else 0,
                    players_involved=[],
                    puck_position=puck_position,
                    description="Shot attempt detected",
                    confidence=0.5  # Low confidence without more sophisticated detection
                ))
        
        # Detect breakaways (one player significantly ahead)
        if len(self.zone_history) > 0:
            zone_states = self.zone_history[-1]
            possession_team = self.possession_history[-1] if self.possession_history else None
            
            for zone, state in zone_states.items():
                if (state.puck_present and possession_team is not None):
                    if possession_team == 0:
                        attacking_players = state.team1_players
                        defending_players = state.team2_players
                    else:
                        attacking_players = state.team2_players
                        defending_players = state.team1_players
                    
                    # Simple breakaway detection: 1 attacker vs 0-1 defenders in offensive zone
                    if (zone == Zone.OFFENSIVE_ZONE and attacking_players == 1 and 
                        defending_players <= 1):
                        
                        plays.append(PlayEvent(
                            play_type=PlayType.BREAKAWAY,
                            timestamp=timestamp,
                            frame_number=frame_number,
                            zone=zone,
                            team_id=possession_team,
                            players_involved=[],
                            puck_position=puck_position,
                            description="Breakaway opportunity",
                            confidence=0.7
                        ))
        
        return plays
    
    def _detect_special_situations(self, frame_number: int, timestamp: float, 
                                 players: List[Dict], player_teams: List) -> List[PlayEvent]:
        """Detect special situations like power plays, line changes"""
        plays = []
        
        # Detect uneven player situations (potential power play/penalty kill)
        team_counts = defaultdict(int)
        for player, team_info in zip(players, player_teams):
            if hasattr(team_info, 'team_id'):
                team_counts[team_info.team_id] += 1
        
        total_players = sum(team_counts.values())
        
        # Look for significant player imbalances
        if len(team_counts) == 2:
            team_list = list(team_counts.items())
            team1_count = team_list[0][1]
            team2_count = team_list[1][1]
            
            if abs(team1_count - team2_count) >= 2 and total_players >= 8:
                # Likely power play situation
                pp_team = team_list[0][0] if team1_count > team2_count else team_list[1][0]
                
                plays.append(PlayEvent(
                    play_type=PlayType.POWER_PLAY,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    zone=Zone.UNKNOWN,
                    team_id=pp_team,
                    players_involved=[],
                    puck_position=None,
                    description=f"Power play situation ({team1_count} vs {team2_count})",
                    confidence=0.8
                ))
        
        return plays
    
    def _get_puck_zone(self, puck_position: Tuple[int, int]) -> Zone:
        """Determine which zone the puck is in"""
        puck_x = puck_position[0]
        
        for zone, bounds in self.zone_boundaries.items():
            if bounds[0] <= puck_x <= bounds[1]:
                return zone
        
        return Zone.UNKNOWN
    
    def get_recent_plays(self, time_window: float = 30.0) -> List[PlayEvent]:
        """Get plays from recent time window"""
        if not self.play_history:
            return []
        
        current_time = self.play_history[-1].timestamp
        cutoff_time = current_time - time_window
        
        recent_plays = []
        for play in reversed(self.play_history):
            if play.timestamp >= cutoff_time:
                recent_plays.append(play)
            else:
                break
        
        return list(reversed(recent_plays))
    
    def get_play_statistics(self, time_window: float = 300.0) -> Dict:
        """Get play statistics for analysis"""
        recent_plays = self.get_recent_plays(time_window)
        
        if not recent_plays:
            return {}
        
        # Count plays by type
        play_counts = defaultdict(int)
        zone_counts = defaultdict(int)
        team_plays = defaultdict(int)
        
        for play in recent_plays:
            play_counts[play.play_type.value] += 1
            zone_counts[play.zone.value] += 1
            if play.team_id is not None:
                team_plays[play.team_id] += 1
        
        # Calculate possession statistics
        possession_changes = sum(1 for play in recent_plays if play.play_type == PlayType.TURNOVER)
        zone_entries = sum(1 for play in recent_plays if play.play_type == PlayType.ZONE_ENTRY)
        
        return {
            'total_plays': len(recent_plays),
            'play_types': dict(play_counts),
            'zones': dict(zone_counts),
            'team_plays': dict(team_plays),
            'possession_changes': possession_changes,
            'zone_entries': zone_entries,
            'time_window': time_window
        }
    
    def visualize_plays(self, frame: np.ndarray, recent_plays: List[PlayEvent]) -> np.ndarray:
        """Visualize recent plays on frame"""
        vis_frame = frame.copy()
        
        # Draw zone boundaries
        for zone, bounds in self.zone_boundaries.items():
            left_bound, right_bound = bounds
            color = (100, 100, 100)
            cv2.line(vis_frame, (left_bound, 0), (left_bound, frame.shape[0]), color, 2)
            cv2.line(vis_frame, (right_bound, 0), (right_bound, frame.shape[0]), color, 2)
            
            # Zone labels
            mid_x = (left_bound + right_bound) // 2
            cv2.putText(vis_frame, zone.value.upper(), (mid_x - 30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show recent plays
        play_colors = {
            PlayType.ZONE_ENTRY: (0, 255, 0),
            PlayType.ZONE_EXIT: (0, 255, 255),
            PlayType.PASS: (255, 255, 0),
            PlayType.SHOT: (0, 0, 255),
            PlayType.TURNOVER: (255, 0, 255),
            PlayType.BREAKAWAY: (255, 100, 0)
        }
        
        y_offset = 50
        for i, play in enumerate(recent_plays[-5:]):  # Show last 5 plays
            color = play_colors.get(play.play_type, (255, 255, 255))
            text = f"{play.play_type.value}: {play.description}"
            
            cv2.putText(vis_frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Mark play location if available
            if play.puck_position:
                cv2.circle(vis_frame, play.puck_position, 10, color, 2)
        
        return vis_frame