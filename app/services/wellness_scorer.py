"""
Wellness Scoring Service
Calculates wellness metrics from posture data
"""
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta


class WellnessScorer:
    """
    Calculates comprehensive wellness metrics from posture snapshots
    """
    
    def __init__(self):
        """Initialize wellness scorer"""
        pass
    
    def calculate_wellness_metrics(
        self, 
        posture_snapshots: List[Dict],
        period_type: str = "daily"
    ) -> Dict:
        """
        Calculate aggregated wellness metrics from posture snapshots
        
        Args:
            posture_snapshots: List of posture snapshot dictionaries
            period_type: Type of period (hourly, daily, weekly)
            
        Returns:
            Dictionary with wellness metrics
        """
        if not posture_snapshots:
            return self._get_default_metrics(period_type)
        
        # Extract posture types
        posture_types = [snap['posture_type'] for snap in posture_snapshots]
        total_snapshots = len(posture_snapshots)
        
        # Calculate posture distribution
        posture_distribution = self._calculate_posture_distribution(posture_types)
        
        # Calculate ergonomic score
        ergonomic_score = self._calculate_ergonomic_score(posture_distribution)
        
        # Calculate posture quality score
        posture_quality = self._calculate_posture_quality(posture_snapshots)
        
        # Calculate activity level score
        activity_level = self._calculate_activity_level(posture_snapshots)
        
        # Assess stress level
        stress_level = self._assess_stress_level(posture_snapshots, posture_distribution)
        
        # Calculate fatigue indicator
        fatigue_indicator = self._calculate_fatigue(posture_snapshots)
        
        # Assess engagement level
        engagement_level = self._assess_engagement(posture_snapshots, posture_distribution)
        
        # Estimate mood
        mood_estimate = self._estimate_mood(posture_snapshots, stress_level, engagement_level)
        
        # Calculate activity breakdown
        activity_breakdown = self._calculate_activity_breakdown(posture_snapshots)
        
        # Calculate pain risks
        pain_risks = self._calculate_pain_risks(posture_snapshots, posture_distribution)
        
        # Calculate movement patterns
        movement_patterns = self._calculate_movement_patterns(posture_snapshots)
        
        return {
            'period_type': period_type,
            'date': datetime.utcnow(),
            'ergonomic_score': ergonomic_score,
            'posture_quality_score': posture_quality,
            'activity_level_score': activity_level,
            'stress_level': stress_level,
            'fatigue_indicator': fatigue_indicator,
            'engagement_level': engagement_level,
            'mood_estimate': mood_estimate,
            'good_posture_percent': posture_distribution.get('good', 0),
            'forward_head_percent': posture_distribution.get('forward_head', 0),
            'slouched_percent': posture_distribution.get('slouched', 0),
            'poor_posture_percent': posture_distribution.get('slouched', 0) + posture_distribution.get('collapsed', 0),
            'sitting_time_minutes': activity_breakdown['sitting_minutes'],
            'standing_time_minutes': activity_breakdown['standing_minutes'],
            'walking_time_minutes': activity_breakdown['walking_minutes'],
            'break_count': activity_breakdown['break_count'],
            'neck_pain_risk': pain_risks['neck'],
            'back_pain_risk': pain_risks['back'],
            'shoulder_pain_risk': pain_risks['shoulder'],
            'fidgeting_frequency': movement_patterns['fidgeting_frequency'],
            'position_changes': movement_patterns['position_changes'],
            'break_compliance_score': self._calculate_break_compliance(activity_breakdown)
        }
    
    def _calculate_posture_distribution(self, posture_types: List[str]) -> Dict[str, float]:
        """Calculate percentage distribution of posture types"""
        if not posture_types:
            return {}
        
        total = len(posture_types)
        distribution = {}
        
        for posture_type in set(posture_types):
            count = posture_types.count(posture_type)
            distribution[posture_type] = round((count / total) * 100, 2)
        
        return distribution
    
    def _calculate_ergonomic_score(self, posture_distribution: Dict[str, float]) -> float:
        """
        Calculate ergonomic score (0-100)
        Based on posture distribution
        """
        # Weights for different postures
        weights = {
            'good': 1.0,
            'moderate': 0.7,
            'forward_head': 0.4,
            'slouched': 0.3,
            'leaning': 0.5,
            'twisted': 0.3,
            'collapsed': 0.1
        }
        
        weighted_sum = 0
        for posture_type, percentage in posture_distribution.items():
            weight = weights.get(posture_type, 0.5)
            weighted_sum += (percentage / 100) * weight
        
        ergonomic_score = weighted_sum * 100
        return round(min(100, max(0, ergonomic_score)), 2)
    
    def _calculate_posture_quality(self, posture_snapshots: List[Dict]) -> float:
        """
        Calculate overall posture quality (0-100)
        Based on average spine scores and neck angles
        """
        spine_scores = [snap.get('spine_score', 70) for snap in posture_snapshots]
        neck_angles = [snap.get('neck_angle', 160) for snap in posture_snapshots]
        
        avg_spine_score = np.mean(spine_scores)
        avg_neck_angle = np.mean(neck_angles)
        
        # Normalize neck angle to 0-100 scale (160-180 degrees is good)
        neck_score = ((avg_neck_angle - 140) / 40) * 100
        neck_score = max(0, min(100, neck_score))
        
        # Combine scores
        posture_quality = (avg_spine_score * 0.6 + neck_score * 0.4)
        
        return round(posture_quality, 2)
    
    def _calculate_activity_level(self, posture_snapshots: List[Dict]) -> float:
        """
        Calculate activity level score (0-100)
        Based on variety of activities and movement
        """
        activities = [snap.get('activity', 'sitting') for snap in posture_snapshots]
        
        # Count unique activities
        unique_activities = len(set(activities))
        
        # Calculate sitting percentage
        sitting_count = activities.count('sitting')
        sitting_percent = (sitting_count / len(activities)) * 100
        
        # Lower sitting time = higher activity score
        activity_score = 100 - (sitting_percent * 0.8)
        
        # Bonus for activity variety
        activity_score += unique_activities * 5
        
        return round(min(100, max(0, activity_score)), 2)
    
    def _assess_stress_level(
        self, 
        posture_snapshots: List[Dict], 
        posture_distribution: Dict[str, float]
    ) -> str:
        """
        Assess stress level based on posture patterns
        
        Returns: low, moderate, high, very_high
        """
        # Poor posture indicators of stress
        poor_posture_percent = (
            posture_distribution.get('slouched', 0) +
            posture_distribution.get('forward_head', 0) +
            posture_distribution.get('collapsed', 0)
        )
        
        # Check shoulder symmetry variation (tension indicator)
        shoulder_symmetries = [snap.get('shoulder_symmetry', 0.9) for snap in posture_snapshots]
        symmetry_std = np.std(shoulder_symmetries)
        
        # Stress assessment
        if poor_posture_percent > 70 or symmetry_std > 0.2:
            return "very_high"
        elif poor_posture_percent > 50 or symmetry_std > 0.15:
            return "high"
        elif poor_posture_percent > 30 or symmetry_std > 0.1:
            return "moderate"
        else:
            return "low"
    
    def _calculate_fatigue(self, posture_snapshots: List[Dict]) -> float:
        """
        Calculate fatigue indicator (0-1)
        Based on posture degradation over time
        """
        if len(posture_snapshots) < 3:
            return 0.3  # Default moderate fatigue
        
        # Split into first half and second half
        mid = len(posture_snapshots) // 2
        first_half = posture_snapshots[:mid]
        second_half = posture_snapshots[mid:]
        
        # Calculate average spine score for each half
        first_half_spine = np.mean([snap.get('spine_score', 70) for snap in first_half])
        second_half_spine = np.mean([snap.get('spine_score', 70) for snap in second_half])
        
        # If posture degrades, fatigue is higher
        degradation = first_half_spine - second_half_spine
        
        # Normalize to 0-1 scale
        fatigue = max(0, degradation / 30)  # 30-point drop = max fatigue
        fatigue = min(1, fatigue + 0.2)  # Base fatigue of 0.2
        
        return round(fatigue, 2)
    
    def _assess_engagement(
        self, 
        posture_snapshots: List[Dict], 
        posture_distribution: Dict[str, float]
    ) -> str:
        """
        Assess engagement level
        
        Returns: disengaged, neutral, engaged, highly_engaged
        """
        good_posture_percent = posture_distribution.get('good', 0)
        slouched_percent = posture_distribution.get('slouched', 0)
        
        # Good posture = engaged
        if good_posture_percent > 60:
            return "highly_engaged"
        elif good_posture_percent > 40:
            return "engaged"
        elif slouched_percent > 50:
            return "disengaged"
        else:
            return "neutral"
    
    def _estimate_mood(
        self, 
        posture_snapshots: List[Dict], 
        stress_level: str, 
        engagement_level: str
    ) -> str:
        """
        Estimate mood based on posture and other indicators
        
        Returns: negative, neutral, positive
        """
        # Get average neck angle
        neck_angles = [snap.get('neck_angle', 160) for snap in posture_snapshots]
        avg_neck_angle = np.mean(neck_angles)
        
        # Low neck angle (head down) = negative mood
        # High stress = negative mood
        if avg_neck_angle < 150 or stress_level in ['high', 'very_high']:
            return "negative"
        elif engagement_level in ['engaged', 'highly_engaged'] and avg_neck_angle > 160:
            return "positive"
        else:
            return "neutral"
    
    def _calculate_activity_breakdown(self, posture_snapshots: List[Dict]) -> Dict:
        """Calculate time spent in each activity"""
        activities = [snap.get('activity', 'sitting') for snap in posture_snapshots]
        
        # Assume each snapshot represents 1 second
        total_seconds = len(activities)
        
        sitting_count = activities.count('sitting')
        standing_count = activities.count('standing')
        walking_count = activities.count('walking')
        
        # Detect breaks (transitions from sitting to standing/walking)
        break_count = 0
        for i in range(1, len(activities)):
            if activities[i-1] == 'sitting' and activities[i] in ['standing', 'walking']:
                break_count += 1
        
        return {
            'sitting_minutes': round(sitting_count / 60, 1),
            'standing_minutes': round(standing_count / 60, 1),
            'walking_minutes': round(walking_count / 60, 1),
            'break_count': break_count
        }
    
    def _calculate_pain_risks(
        self, 
        posture_snapshots: List[Dict], 
        posture_distribution: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate pain risk scores (0-1) for different body parts
        """
        # Neck pain risk
        neck_angles = [snap.get('neck_angle', 160) for snap in posture_snapshots]
        avg_neck_angle = np.mean(neck_angles)
        neck_risk = max(0, (160 - avg_neck_angle) / 40)  # Higher risk with lower angle
        
        # Back pain risk
        spine_scores = [snap.get('spine_score', 70) for snap in posture_snapshots]
        avg_spine_score = np.mean(spine_scores)
        back_risk = max(0, (80 - avg_spine_score) / 80)  # Higher risk with lower spine score
        
        # Shoulder pain risk
        slouched_percent = posture_distribution.get('slouched', 0)
        shoulder_risk = min(1, slouched_percent / 60)  # Higher risk with more slouching
        
        return {
            'neck': round(min(1, neck_risk), 2),
            'back': round(min(1, back_risk), 2),
            'shoulder': round(min(1, shoulder_risk), 2)
        }
    
    def _calculate_movement_patterns(self, posture_snapshots: List[Dict]) -> Dict:
        """Calculate movement and fidgeting patterns"""
        if len(posture_snapshots) < 2:
            return {'fidgeting_frequency': 0, 'position_changes': 0}
        
        # Detect position changes (different posture types)
        position_changes = 0
        for i in range(1, len(posture_snapshots)):
            if posture_snapshots[i]['posture_type'] != posture_snapshots[i-1]['posture_type']:
                position_changes += 1
        
        # Calculate fidgeting frequency (position changes per minute)
        duration_minutes = len(posture_snapshots) / 60
        fidgeting_frequency = round(position_changes / max(1, duration_minutes), 2)
        
        return {
            'fidgeting_frequency': fidgeting_frequency,
            'position_changes': position_changes
        }
    
    def _calculate_break_compliance(self, activity_breakdown: Dict) -> float:
        """
        Calculate break compliance score (0-100)
        Recommended: 5 min break per hour
        """
        sitting_minutes = activity_breakdown['sitting_minutes']
        break_count = activity_breakdown['break_count']
        
        if sitting_minutes == 0:
            return 100
        
        # Calculate expected breaks (1 per hour of sitting)
        expected_breaks = max(1, sitting_minutes / 60)
        
        # Calculate compliance
        compliance = min(100, (break_count / expected_breaks) * 100)
        
        return round(compliance, 2)
    
    def _get_default_metrics(self, period_type: str) -> Dict:
        """Return default metrics when no data available"""
        return {
            'period_type': period_type,
            'date': datetime.utcnow(),
            'ergonomic_score': 50.0,
            'posture_quality_score': 50.0,
            'activity_level_score': 50.0,
            'stress_level': 'moderate',
            'fatigue_indicator': 0.5,
            'engagement_level': 'neutral',
            'mood_estimate': 'neutral',
            'good_posture_percent': 0,
            'forward_head_percent': 0,
            'slouched_percent': 0,
            'poor_posture_percent': 0,
            'sitting_time_minutes': 0,
            'standing_time_minutes': 0,
            'walking_time_minutes': 0,
            'break_count': 0,
            'neck_pain_risk': 0.5,
            'back_pain_risk': 0.5,
            'shoulder_pain_risk': 0.5,
            'fidgeting_frequency': 0,
            'position_changes': 0,
            'break_compliance_score': 0
        }
