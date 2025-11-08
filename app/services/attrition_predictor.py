"""
Attrition Risk Prediction Service
Predicts employee attrition risk based on wellness patterns
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle


class AttritionPredictor:
    """
    Predicts employee attrition risk using wellness and behavioral patterns
    """
    
    def __init__(self):
        """Initialize attrition predictor"""
        self.model = None
        self.scaler = None
        self.model_version = "1.0.0"
    
    def predict_attrition_risk(
        self,
        wellness_history: List[Dict],
        current_metrics: Dict,
        baseline: Optional[Dict] = None
    ) -> Dict:
        """
        Predict attrition risk for a user
        
        Args:
            wellness_history: Historical wellness metrics (last 30 days)
            current_metrics: Most recent wellness metrics
            baseline: User's baseline profile
            
        Returns:
            Dictionary with attrition risk prediction
        """
        # Calculate features
        features = self._extract_features(wellness_history, current_metrics, baseline)
        
        # Calculate risk score using rule-based approach (simplified)
        risk_score = self._calculate_rule_based_risk(features)
        
        # Classify risk category
        risk_category = self._classify_risk(risk_score)
        
        # Estimate weeks to attrition
        weeks_to_attrition = self._estimate_time_horizon(risk_score, features)
        
        # Identify top risk factors
        top_risk_factors = self._identify_risk_factors(features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features, risk_category)
        
        # Calculate wellness trends
        wellness_trend = self._calculate_wellness_trend(wellness_history)
        stress_trend = self._calculate_stress_trend(wellness_history)
        engagement_trend = self._calculate_engagement_trend(wellness_history)
        
        return {
            'overall_risk_score': risk_score,
            'risk_category': risk_category,
            'estimated_weeks_to_attrition': weeks_to_attrition,
            'top_risk_factors': top_risk_factors,
            'wellness_score_trend': wellness_trend,
            'stress_trend': stress_trend,
            'engagement_trend': engagement_trend,
            'wellness_score_30d_avg': features.get('avg_wellness_30d', 50),
            'wellness_score_7d_avg': features.get('avg_wellness_7d', 50),
            'wellness_score_change_percent': features.get('wellness_change_pct', 0),
            'recommended_interventions': recommendations,
            'model_version': self.model_version,
            'confidence': self._calculate_confidence(features)
        }
    
    def _extract_features(
        self,
        wellness_history: List[Dict],
        current_metrics: Dict,
        baseline: Optional[Dict]
    ) -> Dict:
        """Extract features for prediction"""
        if not wellness_history:
            wellness_history = [current_metrics]
        
        # Recent history (last 7 days)
        recent_history = wellness_history[-7:] if len(wellness_history) >= 7 else wellness_history
        
        # Calculate averages
        avg_wellness_30d = np.mean([m.get('ergonomic_score', 50) for m in wellness_history])
        avg_wellness_7d = np.mean([m.get('ergonomic_score', 50) for m in recent_history])
        
        # Calculate trends
        wellness_values = [m.get('ergonomic_score', 50) for m in wellness_history]
        wellness_slope = self._calculate_slope(wellness_values)
        
        # Stress indicators
        avg_stress_score = self._stress_level_to_score([m.get('stress_level', 'moderate') for m in wellness_history])
        stress_escalation = self._check_stress_escalation(wellness_history)
        
        # Engagement indicators
        avg_engagement = self._engagement_to_score([m.get('engagement_level', 'neutral') for m in wellness_history])
        
        # Fatigue accumulation
        avg_fatigue = np.mean([m.get('fatigue_indicator', 0.5) for m in wellness_history])
        
        # Break compliance
        avg_break_compliance = np.mean([m.get('break_compliance_score', 50) for m in wellness_history])
        
        # Pain risk accumulation
        avg_pain_risk = np.mean([
            (m.get('neck_pain_risk', 0) + m.get('back_pain_risk', 0) + m.get('shoulder_pain_risk', 0)) / 3
            for m in wellness_history
        ])
        
        # Calculate change percentage
        if len(wellness_history) >= 14:
            first_week = wellness_history[:7]
            last_week = wellness_history[-7:]
            first_week_avg = np.mean([m.get('ergonomic_score', 50) for m in first_week])
            last_week_avg = np.mean([m.get('ergonomic_score', 50) for m in last_week])
            wellness_change_pct = ((last_week_avg - first_week_avg) / first_week_avg) * 100
        else:
            wellness_change_pct = 0
        
        features = {
            'avg_wellness_30d': round(avg_wellness_30d, 2),
            'avg_wellness_7d': round(avg_wellness_7d, 2),
            'wellness_slope': round(wellness_slope, 2),
            'wellness_change_pct': round(wellness_change_pct, 2),
            'avg_stress_score': round(avg_stress_score, 2),
            'stress_escalation': stress_escalation,
            'avg_engagement': round(avg_engagement, 2),
            'avg_fatigue': round(avg_fatigue, 2),
            'avg_break_compliance': round(avg_break_compliance, 2),
            'avg_pain_risk': round(avg_pain_risk, 2),
            'current_wellness': current_metrics.get('ergonomic_score', 50),
            'has_baseline': baseline is not None
        }
        
        return features
    
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of time series"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _stress_level_to_score(self, stress_levels: List[str]) -> float:
        """Convert stress levels to numerical scores"""
        mapping = {
            'low': 20,
            'moderate': 50,
            'high': 75,
            'very_high': 95
        }
        
        scores = [mapping.get(level, 50) for level in stress_levels]
        return np.mean(scores)
    
    def _engagement_to_score(self, engagement_levels: List[str]) -> float:
        """Convert engagement levels to numerical scores"""
        mapping = {
            'disengaged': 20,
            'neutral': 50,
            'engaged': 75,
            'highly_engaged': 95
        }
        
        scores = [mapping.get(level, 50) for level in engagement_levels]
        return np.mean(scores)
    
    def _check_stress_escalation(self, wellness_history: List[Dict]) -> bool:
        """Check if stress is escalating"""
        if len(wellness_history) < 3:
            return False
        
        stress_levels = [m.get('stress_level', 'moderate') for m in wellness_history[-5:]]
        
        # Check if recent stress levels are high
        high_stress_count = sum(1 for level in stress_levels if level in ['high', 'very_high'])
        
        return high_stress_count >= 3
    
    def _calculate_rule_based_risk(self, features: Dict) -> float:
        """
        Calculate risk score using rule-based approach (0-100)
        """
        risk_score = 0
        
        # Low wellness score (max 30 points)
        wellness_score = features['avg_wellness_7d']
        if wellness_score < 30:
            risk_score += 30
        elif wellness_score < 50:
            risk_score += 20
        elif wellness_score < 70:
            risk_score += 10
        
        # Declining trend (max 20 points)
        if features['wellness_slope'] < -3:
            risk_score += 20
        elif features['wellness_slope'] < -1:
            risk_score += 10
        
        # High stress (max 20 points)
        if features['avg_stress_score'] > 80:
            risk_score += 20
        elif features['avg_stress_score'] > 60:
            risk_score += 10
        
        # Low engagement (max 15 points)
        if features['avg_engagement'] < 40:
            risk_score += 15
        elif features['avg_engagement'] < 60:
            risk_score += 8
        
        # High fatigue (max 10 points)
        if features['avg_fatigue'] > 0.7:
            risk_score += 10
        elif features['avg_fatigue'] > 0.5:
            risk_score += 5
        
        # Poor break compliance (max 5 points)
        if features['avg_break_compliance'] < 40:
            risk_score += 5
        
        return min(100, risk_score)
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk category"""
        if risk_score >= 75:
            return 'critical'
        elif risk_score >= 50:
            return 'high'
        elif risk_score >= 30:
            return 'moderate'
        else:
            return 'low'
    
    def _estimate_time_horizon(self, risk_score: float, features: Dict) -> Optional[float]:
        """Estimate weeks until potential attrition"""
        if risk_score < 30:
            return None  # Low risk, no immediate concern
        
        # Simple estimation based on risk score and trend
        base_weeks = 52 - (risk_score * 0.5)  # Max 52 weeks (1 year)
        
        # Adjust based on trend
        if features['wellness_slope'] < -5:
            base_weeks *= 0.5  # Rapid decline
        elif features['wellness_slope'] < -2:
            base_weeks *= 0.75
        
        return max(1, round(base_weeks, 1))
    
    def _identify_risk_factors(self, features: Dict) -> List[Dict]:
        """Identify top contributing risk factors"""
        risk_factors = []
        
        # Check each factor
        if features['avg_wellness_7d'] < 50:
            risk_factors.append({
                'factor': 'Low Wellness Score',
                'severity': 'high' if features['avg_wellness_7d'] < 30 else 'moderate',
                'value': features['avg_wellness_7d']
            })
        
        if features['wellness_slope'] < -2:
            risk_factors.append({
                'factor': 'Declining Wellness Trend',
                'severity': 'high' if features['wellness_slope'] < -5 else 'moderate',
                'value': features['wellness_slope']
            })
        
        if features['avg_stress_score'] > 60:
            risk_factors.append({
                'factor': 'Elevated Stress Levels',
                'severity': 'high' if features['avg_stress_score'] > 80 else 'moderate',
                'value': features['avg_stress_score']
            })
        
        if features['avg_engagement'] < 60:
            risk_factors.append({
                'factor': 'Low Engagement',
                'severity': 'high' if features['avg_engagement'] < 40 else 'moderate',
                'value': features['avg_engagement']
            })
        
        if features['avg_fatigue'] > 0.6:
            risk_factors.append({
                'factor': 'High Fatigue Levels',
                'severity': 'high' if features['avg_fatigue'] > 0.8 else 'moderate',
                'value': features['avg_fatigue']
            })
        
        # Sort by severity
        risk_factors.sort(key=lambda x: 1 if x['severity'] == 'high' else 2)
        
        return risk_factors[:5]  # Top 5
    
    def _generate_recommendations(self, features: Dict, risk_category: str) -> List[str]:
        """Generate intervention recommendations"""
        recommendations = []
        
        if features['avg_wellness_7d'] < 50:
            recommendations.append("Schedule ergonomic assessment and workstation adjustment")
        
        if features['avg_stress_score'] > 60:
            recommendations.append("Recommend stress management resources and counseling")
        
        if features['avg_engagement'] < 60:
            recommendations.append("Conduct one-on-one meeting to discuss career goals and satisfaction")
        
        if features['avg_break_compliance'] < 50:
            recommendations.append("Encourage regular breaks and promote wellness programs")
        
        if features['avg_fatigue'] > 0.6:
            recommendations.append("Review workload distribution and consider workload adjustment")
        
        if risk_category in ['high', 'critical']:
            recommendations.append("Immediate HR intervention recommended - schedule retention conversation")
        
        return recommendations
    
    def _calculate_wellness_trend(self, wellness_history: List[Dict]) -> str:
        """Calculate wellness trend"""
        if len(wellness_history) < 3:
            return 'stable'
        
        values = [m.get('ergonomic_score', 50) for m in wellness_history]
        slope = self._calculate_slope(values)
        
        if slope > 2:
            return 'improving'
        elif slope > -2:
            return 'stable'
        elif slope > -5:
            return 'declining'
        else:
            return 'rapidly_declining'
    
    def _calculate_stress_trend(self, wellness_history: List[Dict]) -> str:
        """Calculate stress trend"""
        if len(wellness_history) < 3:
            return 'stable'
        
        stress_levels = [m.get('stress_level', 'moderate') for m in wellness_history]
        scores = [self._stress_level_to_score([level])[0] if isinstance(self._stress_level_to_score([level]), np.ndarray) else self._stress_level_to_score([level]) for level in stress_levels]
        
        slope = self._calculate_slope(scores)
        
        if slope > 5:
            return 'increasing'
        elif slope > -5:
            return 'stable'
        else:
            return 'decreasing'
    
    def _calculate_engagement_trend(self, wellness_history: List[Dict]) -> str:
        """Calculate engagement trend"""
        if len(wellness_history) < 3:
            return 'stable'
        
        engagement_levels = [m.get('engagement_level', 'neutral') for m in wellness_history]
        scores = [self._engagement_to_score([level])[0] if isinstance(self._engagement_to_score([level]), np.ndarray) else self._engagement_to_score([level]) for level in engagement_levels]
        
        slope = self._calculate_slope(scores)
        
        if slope > 5:
            return 'improving'
        elif slope > -5:
            return 'stable'
        else:
            return 'declining'
    
    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate prediction confidence"""
        # Higher confidence with more data and clearer patterns
        base_confidence = 0.7
        
        # Increase confidence if trend is clear
        if abs(features['wellness_slope']) > 3:
            base_confidence += 0.1
        
        # Increase confidence if has baseline
        if features['has_baseline']:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
