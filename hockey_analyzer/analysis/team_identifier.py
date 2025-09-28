import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import colorsys

@dataclass
class TeamColors:
    """Store team color information"""
    primary_color: Tuple[int, int, int]  # BGR format
    secondary_color: Optional[Tuple[int, int, int]] = None
    team_name: str = "Unknown"
    confidence: float = 0.0

@dataclass
class PlayerTeamInfo:
    """Store player team identification info"""
    player_id: int
    team_id: int
    team_confidence: float
    dominant_colors: List[Tuple[int, int, int]]
    jersey_region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

class TeamIdentifier:
    """
    Identify teams based on jersey colors and player clustering
    """
    
    def __init__(self, expected_teams: int = 2):
        self.expected_teams = expected_teams
        self.team_colors = {}  # team_id -> TeamColors
        self.player_team_history = defaultdict(list)  # player_id -> [team_assignments]
        self.color_clusters = None
        self.frame_count = 0
        
        # Color analysis parameters
        self.jersey_roi_ratio = 0.6  # Focus on top 60% of player bbox for jersey
        self.min_color_samples = 50  # Minimum pixels to analyze
        self.color_stability_frames = 10  # Frames to confirm team assignment
        
    def identify_teams(self, frame: np.ndarray, players: List[Dict]) -> List[PlayerTeamInfo]:
        """
        Identify team for each player based on jersey colors
        """
        self.frame_count += 1
        player_teams = []
        
        # Extract jersey colors for all players
        jersey_colors = []
        valid_players = []
        
        for i, player in enumerate(players):
            colors = self._extract_jersey_colors(frame, player)
            if colors is not None:
                jersey_colors.append(colors)
                valid_players.append((i, player))
        
        if len(jersey_colors) < 4:  # Need minimum players for clustering
            # Return default assignments
            return [PlayerTeamInfo(i, 0, 0.0, []) for i in range(len(players))]
        
        # Cluster players into teams based on jersey colors
        team_assignments = self._cluster_players_by_color(jersey_colors)
        
        # Update team color models
        self._update_team_colors(jersey_colors, team_assignments)
        
        # Create player team info
        for (player_idx, player), team_id, colors in zip(valid_players, team_assignments, jersey_colors):
            confidence = self._calculate_team_confidence(colors, team_id)
            
            # Update player history for stability
            self.player_team_history[player_idx].append(team_id)
            if len(self.player_team_history[player_idx]) > self.color_stability_frames:
                self.player_team_history[player_idx].pop(0)
            
            # Use majority vote from recent history for stable assignment
            if len(self.player_team_history[player_idx]) >= 3:
                stable_team = Counter(self.player_team_history[player_idx]).most_common(1)[0][0]
            else:
                stable_team = team_id
            
            player_teams.append(PlayerTeamInfo(
                player_id=player_idx,
                team_id=stable_team,
                team_confidence=confidence,
                dominant_colors=colors,
                jersey_region=self._get_jersey_region(player)
            ))
        
        # Fill in any missing players with default assignment
        while len(player_teams) < len(players):
            player_teams.append(PlayerTeamInfo(len(player_teams), 0, 0.0, []))
        
        return player_teams
    
    def _extract_jersey_colors(self, frame: np.ndarray, player: Dict) -> Optional[List[Tuple[int, int, int]]]:
        """Extract dominant colors from player's jersey region"""
        bbox = player['bbox']
        x, y, w, h = bbox
        
        # Focus on jersey area (upper portion of player)
        jersey_y = y
        jersey_h = int(h * self.jersey_roi_ratio)
        jersey_region = (x, jersey_y, w, jersey_h)
        
        # Extract jersey ROI
        roi = frame[jersey_y:jersey_y+jersey_h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # Convert to RGB for color analysis
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Remove ice-colored pixels (white/light colors)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Mask out white/ice colors
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)
        
        # Also mask very dark colors (likely shadows/equipment)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 40])
        dark_mask = cv2.inRange(roi_hsv, lower_dark, upper_dark)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, dark_mask)
        jersey_mask = cv2.bitwise_not(combined_mask)
        
        # Get jersey pixels
        jersey_pixels = roi_rgb[jersey_mask > 0]
        
        if len(jersey_pixels) < self.min_color_samples:
            return None
        
        # Cluster jersey colors to find dominant colors
        n_colors = min(3, len(jersey_pixels) // 20)  # Adaptive number of clusters
        if n_colors < 1:
            return None
        
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(jersey_pixels)
            
            # Get dominant colors sorted by cluster size
            colors = []
            for i, center in enumerate(kmeans.cluster_centers_):
                cluster_size = np.sum(kmeans.labels_ == i)
                colors.append((tuple(center.astype(int)), cluster_size))
            
            # Sort by cluster size and return top colors
            colors.sort(key=lambda x: x[1], reverse=True)
            dominant_colors = [color[0] for color in colors[:2]]  # Top 2 colors
            
            return dominant_colors
            
        except Exception:
            # Fallback to simple mean color
            mean_color = np.mean(jersey_pixels, axis=0)
            return [tuple(mean_color.astype(int))]
    
    def _cluster_players_by_color(self, jersey_colors: List[List[Tuple[int, int, int]]]) -> List[int]:
        """Cluster players into teams based on jersey colors"""
        
        # Create feature vectors for clustering
        # Use primary color (first dominant color) for each player
        color_features = []
        for colors in jersey_colors:
            if colors:
                # Convert RGB to HSV for better color clustering
                rgb_color = colors[0]
                hsv_color = colorsys.rgb_to_hsv(rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255)
                # Use H and S channels (ignore brightness V for team identification)
                color_features.append([hsv_color[0] * 360, hsv_color[1] * 100])
            else:
                color_features.append([0, 0])  # Default
        
        color_features = np.array(color_features)
        
        if len(color_features) < self.expected_teams:
            return [0] * len(jersey_colors)
        
        try:
            # Cluster into teams
            kmeans = KMeans(n_clusters=self.expected_teams, random_state=42, n_init=10)
            team_assignments = kmeans.fit_predict(color_features)
            return team_assignments.tolist()
            
        except Exception:
            # Fallback: alternate assignment
            return [i % self.expected_teams for i in range(len(jersey_colors))]
    
    def _update_team_colors(self, jersey_colors: List[List[Tuple[int, int, int]]], team_assignments: List[int]):
        """Update team color models based on current frame analysis"""
        
        team_color_samples = defaultdict(list)
        
        # Collect color samples for each team
        for colors, team_id in zip(jersey_colors, team_assignments):
            if colors:
                team_color_samples[team_id].extend(colors)
        
        # Update team color models
        for team_id, color_samples in team_color_samples.items():
            if not color_samples:
                continue
                
            # Calculate average team colors
            primary_color = np.mean(color_samples, axis=0).astype(int)
            
            # Calculate color variance for confidence
            color_distances = [self._color_distance(color, primary_color) for color in color_samples]
            color_consistency = 1.0 - (np.mean(color_distances) / 100.0)  # Normalize
            
            self.team_colors[team_id] = TeamColors(
                primary_color=tuple(primary_color),
                team_name=f"Team {team_id + 1}",
                confidence=max(0.0, min(1.0, color_consistency))
            )
    
    def _calculate_team_confidence(self, player_colors: List[Tuple[int, int, int]], team_id: int) -> float:
        """Calculate confidence of team assignment for a player"""
        
        if team_id not in self.team_colors or not player_colors:
            return 0.0
        
        team_color = self.team_colors[team_id].primary_color
        
        # Find closest player color to team color
        min_distance = float('inf')
        for color in player_colors:
            distance = self._color_distance(color, team_color)
            min_distance = min(min_distance, distance)
        
        # Convert distance to confidence (closer = higher confidence)
        confidence = max(0.0, 1.0 - min_distance / 100.0)
        
        return confidence
    
    def _color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate perceptual distance between two RGB colors"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # Simple Euclidean distance in RGB space
        # Could be improved with LAB color space for better perceptual accuracy
        return np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
    
    def _get_jersey_region(self, player: Dict) -> Tuple[int, int, int, int]:
        """Get jersey region coordinates for visualization"""
        bbox = player['bbox']
        x, y, w, h = bbox
        jersey_h = int(h * self.jersey_roi_ratio)
        return (x, y, w, jersey_h)
    
    def get_team_info(self) -> Dict[int, TeamColors]:
        """Get current team color information"""
        return self.team_colors.copy()
    
    def visualize_team_identification(self, frame: np.ndarray, players: List[Dict], 
                                   player_teams: List[PlayerTeamInfo]) -> np.ndarray:
        """Visualize team identification on frame"""
        vis_frame = frame.copy()
        
        # Define team colors for visualization
        team_viz_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        
        for player, team_info in zip(players, player_teams):
            if team_info.team_id >= len(team_viz_colors):
                continue
                
            viz_color = team_viz_colors[team_info.team_id]
            
            # Draw player bounding box with team color
            bbox = player['bbox']
            x, y, w, h = bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), viz_color, 2)
            
            # Draw jersey region
            if team_info.jersey_region:
                jx, jy, jw, jh = team_info.jersey_region
                cv2.rectangle(vis_frame, (jx, jy), (jx + jw, jy + jh), viz_color, 1)
            
            # Add team label
            team_text = f"T{team_info.team_id + 1}: {team_info.team_confidence:.2f}"
            cv2.putText(vis_frame, team_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, viz_color, 1)
            
            # Show dominant colors
            if team_info.dominant_colors:
                for i, color in enumerate(team_info.dominant_colors[:2]):
                    # Convert RGB to BGR for OpenCV
                    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                    cv2.rectangle(vis_frame, (x + i*15, y + h + 5), 
                                 (x + i*15 + 10, y + h + 15), bgr_color, -1)
        
        # Add team color legend
        legend_y = 50
        for team_id, team_colors in self.team_colors.items():
            if team_id < len(team_viz_colors):
                viz_color = team_viz_colors[team_id]
                
                # Draw team color box
                cv2.rectangle(vis_frame, (10, legend_y), (30, legend_y + 20), viz_color, -1)
                
                # Team info text
                team_text = f"{team_colors.team_name} (conf: {team_colors.confidence:.2f})"
                cv2.putText(vis_frame, team_text, (35, legend_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                legend_y += 30
        
        return vis_frame
    
    def get_team_stats(self, player_teams: List[PlayerTeamInfo]) -> Dict:
        """Get team statistics for current frame"""
        team_counts = Counter([pt.team_id for pt in player_teams])
        
        stats = {
            'team_counts': dict(team_counts),
            'total_players': len(player_teams),
            'avg_confidence': np.mean([pt.team_confidence for pt in player_teams]) if player_teams else 0.0,
            'teams_detected': len(team_counts)
        }
        
        return stats

class RefereeDetector:
    """Detect referees/officials who wear striped jerseys"""

    def __init__(self):
        self.stripe_threshold = 0.8  # Much higher threshold - referees have very distinct stripes
        self.max_referees = 3  # Hockey rule: maximum 3 refs (professional), usually 1-2

    def detect_referees(self, frame: np.ndarray, players: List[Dict]) -> List[int]:
        """
        Detect referees among players based on striped jersey pattern
        Returns list of player indices that are likely referees

        Hockey rules applied:
        - Maximum 3 referees (professional level)
        - Usually 1-2 referees in most games
        - Returns most confident detections only
        """
        if not players:
            return []

        # Calculate referee scores for all players
        referee_candidates = []

        for i, player in enumerate(players):
            stripe_score = self._calculate_referee_score(frame, player)
            if stripe_score > self.stripe_threshold:
                referee_candidates.append((i, stripe_score))

        # Sort by confidence and limit to realistic number
        referee_candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply hockey rules: max 3 referees, prefer fewer
        max_refs = min(self.max_referees, len(referee_candidates))

        # For most games, expect 1-2 refs. Only allow 3 if confidence is very high
        if max_refs > 2:
            # Only allow 3rd referee if their score is significantly high
            if len(referee_candidates) >= 3 and referee_candidates[2][1] < 0.7:
                max_refs = 2

        # Return the most confident referee detections
        referees = [idx for idx, score in referee_candidates[:max_refs]]

        return referees
    
    def _calculate_referee_score(self, frame: np.ndarray, player: Dict) -> float:
        """Calculate likelihood score that a player is a referee based on jersey pattern"""
        bbox = player['bbox']
        x, y, w, h = bbox

        # Extract jersey region (upper body)
        jersey_h = int(h * 0.6)
        roi = frame[y:y+jersey_h, x:x+w]

        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 10:
            return 0.0

        # Convert to grayscale for pattern analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Referee detection based on actual stripe characteristics:
        # 1. Alternating black and white horizontal stripes
        # 2. High contrast between adjacent rows
        # 3. Regular pattern spacing

        # Calculate row-wise intensity for horizontal stripe detection
        row_intensities = np.mean(gray_roi, axis=1)

        if len(row_intensities) < 15:
            return 0.0

        # Smooth the signal to reduce noise
        kernel_size = max(3, len(row_intensities) // 10)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(row_intensities, kernel, mode='valid')

        if len(smoothed) < 10:
            return 0.0

        # Look for alternating pattern - referees have high contrast stripes
        # Calculate local variance to find high-contrast regions
        local_variance = []
        window_size = max(3, len(smoothed) // 8)

        for i in range(len(smoothed) - window_size + 1):
            window = smoothed[i:i+window_size]
            variance = np.var(window)
            local_variance.append(variance)

        if not local_variance:
            return 0.0

        avg_variance = np.mean(local_variance)
        max_variance = np.max(local_variance)

        # Referees should have consistent high contrast (variance) across the jersey
        contrast_score = avg_variance / (max_variance + 1e-6)

        # Look for alternating peaks and valleys (stripes)
        # Use second derivative to find stripe boundaries
        second_derivative = np.diff(smoothed, n=2)
        zero_crossings = np.where(np.diff(np.signbit(second_derivative)))[0]

        # Count potential stripe transitions
        stripe_transitions = len(zero_crossings)
        transition_score = min(1.0, stripe_transitions / 6.0)  # Normalize (referees have ~3-5 stripes = 6-10 transitions)

        # Check for black/white alternating pattern
        # Referees should have areas close to black (low intensity) and white (high intensity)
        min_intensity = np.min(smoothed)
        max_intensity = np.max(smoothed)
        intensity_range = max_intensity - min_intensity

        # Strong black/white contrast indicates stripes
        contrast_range_score = min(1.0, intensity_range / 128.0)  # Normalize to [0,1]

        # Check if we have actual black and white regions (not just gray variations)
        # Adaptive thresholds based on the actual intensity distribution
        black_threshold = min_intensity + 0.2 * intensity_range  # Bottom 20% of range
        white_threshold = min_intensity + 0.8 * intensity_range  # Top 80% of range

        has_black = np.any(smoothed < black_threshold)
        has_white = np.any(smoothed > white_threshold)

        # Referee jerseys should have both black and white regions
        # Give partial credit even if only one is detected strongly
        if has_black and has_white:
            bw_pattern_score = 1.0  # Perfect stripes
        elif has_black or has_white:
            bw_pattern_score = 0.5  # Partial credit for having contrast
        else:
            bw_pattern_score = 0.0  # No contrast detected

        # Combine scores using weighted average instead of multiplication
        # This prevents one low score from zeroing out the entire result
        scores = [contrast_score, transition_score, contrast_range_score, bw_pattern_score]
        weights = [0.2, 0.3, 0.3, 0.2]  # Emphasize transitions and contrast range

        final_score = sum(s * w for s, w in zip(scores, weights))

        # Apply additional penalty for very small regions (likely noise)
        size_penalty = min(1.0, (roi.shape[0] * roi.shape[1]) / 800.0)  # Penalize tiny regions
        final_score *= size_penalty

        return min(1.0, max(0.0, final_score))

    def _is_referee(self, frame: np.ndarray, player: Dict) -> bool:
        """Legacy method - check if player is likely a referee"""
        score = self._calculate_referee_score(frame, player)
        return score > self.stripe_threshold