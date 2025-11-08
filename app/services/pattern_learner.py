"""
Pattern Learning Service
Handles baseline profiling and anomaly detection
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import pickle


class PatternLearner:
    """
    Learns user behavior patterns and detects anomalies
    """
    
    def __init__(self):
        """Initialize pattern learner"""
        self.isolation_forest = None
    
    def create_baseline(
        self, 
        wellness_metrics: List[Dict],
        min_days: int = 14
    ) -> Dict:
        """
        Create baseline profile from wellness metrics
        
        Args:
            wellness_metrics: List of wellness metric dictionaries
            min_days: Minimum days of data required
            
        Returns:
            Dictionary with baseline profile
        """
        if len(wellness_metrics) < min_days:
            return None
        
        # Extract key metrics
        ergonomic_scores = [m['ergonomic_score'] for m in wellness_metrics]
        posture_quality = [m['posture_quality_score'] for m in wellness_metrics]
        stress_levels = [m['stress_level'] for m in wellness_metrics]
        
        # Calculate averages
        avg_ergonomic = np.mean(ergonomic_scores)
        avg_posture = np.mean(posture_quality)
        
        # Calculate typical posture distribution
        good_posture_pcts = [m.get('good_posture_percent', 0) for m in wellness_metrics]
        slouched_pcts = [m.get('slouched_percent', 0) for m in wellness_metrics]
        
        typical_distribution = {
            'good_posture': np.mean(good_posture_pcts),
            'slouched': np.mean(slouched_pcts)
        }
        
        # Calculate typical work patterns
        sitting_times = [m.get('sitting_time_minutes', 0) for m in wellness_metrics]
        break_counts = [m.get('break_count', 0) for m in wellness_metrics]
        
        avg_sitting = np.mean(sitting_times)
        avg_breaks = np.mean(break_counts)
        
        # Calculate anomaly thresholds (mean ± 2*std)
        ergonomic_std = np.std(ergonomic_scores)
        posture_std = np.std(posture_quality)
        
        baseline = {
            'baseline_start_date': wellness_metrics[0]['date'],
            'baseline_end_date': wellness_metrics[-1]['date'],
            'is_complete': True,
            'avg_ergonomic_score': round(avg_ergonomic, 2),
            'avg_posture_quality': round(avg_posture, 2),
            'typical_posture_distribution': typical_distribution,
            'avg_daily_sitting_time': round(avg_sitting, 1),
            'avg_break_frequency': round(avg_breaks / 8, 2),  # per hour (8 hr day)
            'anomaly_threshold_ergonomic': round(avg_ergonomic - 2 * ergonomic_std, 2),
            'anomaly_threshold_posture': round(avg_posture - 2 * posture_std, 2)
        }
        
        return baseline
    
    def detect_anomalies(
        self, 
        current_metrics: Dict,
        baseline: Dict
    ) -> Dict:
        """
        Detect anomalies in current metrics compared to baseline
        
        Args:
            current_metrics: Current wellness metrics
            baseline: Baseline profile
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not baseline or not baseline.get('is_complete'):
            return {
                'has_anomaly': False,
                'anomaly_type': None,
                'severity': 'none'
            }
        
        anomalies = []
        severity_score = 0
        
        # Check ergonomic score
        current_ergonomic = current_metrics.get('ergonomic_score', 50)
        threshold_ergonomic = baseline.get('anomaly_threshold_ergonomic', 40)
        
        if current_ergonomic < threshold_ergonomic:
            anomalies.append('ergonomic_degradation')
            severity_score += 2
        
        # Check posture quality
        current_posture = current_metrics.get('posture_quality_score', 50)
        threshold_posture = baseline.get('anomaly_threshold_posture', 40)
        
        if current_posture < threshold_posture:
            anomalies.append('posture_degradation')
            severity_score += 2
        
        # Check stress level escalation
        current_stress = current_metrics.get('stress_level', 'moderate')
        if current_stress in ['high', 'very_high']:
            anomalies.append('elevated_stress')
            severity_score += 1
        
        # Check break compliance
        current_break_compliance = current_metrics.get('break_compliance_score', 50)
        if current_break_compliance < 50:
            anomalies.append('insufficient_breaks')
            severity_score += 1
        
        # Determine severity
        if severity_score >= 4:
            severity = 'critical'
        elif severity_score >= 2:
            severity = 'moderate'
        elif severity_score >= 1:
            severity = 'low'
        else:
            severity = 'none'
        
        return {
            'has_anomaly': len(anomalies) > 0,
            'anomaly_types': anomalies,
            'severity': severity,
            'severity_score': severity_score
        }
    
    def calculate_trend(
        self,
        metrics_series: List[Dict],
        metric_name: str,
        window_days: int = 7
    ) -> str:
        """
        Calculate trend for a specific metric
        
        Args:
            metrics_series: Time series of wellness metrics
            metric_name: Name of metric to analyze
            window_days: Window for trend calculation
            
        Returns:
            Trend: 'improving', 'stable', 'declining', 'rapidly_declining'
        """
        if len(metrics_series) < 3:
            return 'stable'
        
        # Extract values
        values = [m.get(metric_name, 50) for m in metrics_series[-window_days:]]
        
        if len(values) < 3:
            return 'stable'
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple slope calculation
        slope = np.polyfit(x, y, 1)[0]
        
        # Classify trend
        if slope > 2:
            return 'improving'
        elif slope > -2:
            return 'stable'
        elif slope > -5:
            return 'declining'
        else:
            return 'rapidly_declining'
    
    def train_isolation_forest(
        self,
        wellness_metrics: List[Dict]
    ):
        """
        Train isolation forest for advanced anomaly detection
        
        Args:
            wellness_metrics: Historical wellness metrics
        """
        if len(wellness_metrics) < 10:
            return
        
        # Extract features
        features = []
        for m in wellness_metrics:
            feature_vec = [
                m.get('ergonomic_score', 50),
                m.get('posture_quality_score', 50),
                m.get('activity_level_score', 50),
                m.get('fatigue_indicator', 0.5) * 100,
                m.get('good_posture_percent', 0),
                m.get('sitting_time_minutes', 0) / 60,
                m.get('break_count', 0)
            ]
            features.append(feature_vec)
        
        X = np.array(features)
        
        # Train model
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(X)
    
    def predict_anomaly_ml(
        self,
        current_metrics: Dict
    ) -> Dict:
        """
        Predict anomaly using trained isolation forest
        
        Args:
            current_metrics: Current wellness metrics
            
        Returns:
            Anomaly prediction with score
        """
        if self.isolation_forest is None:
            return {
                'is_anomaly': False,
                'anomaly_score': 0,
                'confidence': 0
            }
        
        # Prepare feature vector
        feature_vec = [
            current_metrics.get('ergonomic_score', 50),
            current_metrics.get('posture_quality_score', 50),
            current_metrics.get('activity_level_score', 50),
            current_metrics.get('fatigue_indicator', 0.5) * 100,
            current_metrics.get('good_posture_percent', 0),
            current_metrics.get('sitting_time_minutes', 0) / 60,
            current_metrics.get('break_count', 0)
        ]
        
        X = np.array([feature_vec])
        
        # Predict
        prediction = self.isolation_forest.predict(X)[0]
        anomaly_score = self.isolation_forest.score_samples(X)[0]
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(anomaly_score),
            'confidence': abs(anomaly_score)
        }
