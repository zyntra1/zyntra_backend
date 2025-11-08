"""
Wellness Forest Calculator Service
Calculates and updates forest visualization based on wellness scores
"""
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import math

from app.models.posture import WellnessMetrics, WellnessForest
from app.models.user import User


class ForestCalculator:
    """Calculate forest metrics from wellness data"""
    
    def calculate_and_update_forest(self, db: Session, user_id: int) -> WellnessForest:
        """
        Calculate forest metrics based on wellness data and update database
        """
        # Get or create forest
        forest = db.query(WellnessForest).filter(WellnessForest.user_id == user_id).first()
        if not forest:
            forest = WellnessForest(user_id=user_id)
            db.add(forest)
        
        # Get wellness data (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        wellness_metrics = db.query(WellnessMetrics).filter(
            WellnessMetrics.user_id == user_id,
            WellnessMetrics.date >= thirty_days_ago
        ).order_by(WellnessMetrics.date).all()
        
        if not wellness_metrics:
            # No data - default minimal forest
            forest.total_trees = 10
            forest.healthy_trees = 2
            forest.growing_trees = 3
            forest.wilting_trees = 3
            forest.dead_trees = 2
            forest.forest_health_score = 20.0
            forest.biodiversity_score = 20.0
            forest.growth_rate = 0.0
            forest.sunlight_level = 30.0
            forest.water_level = 30.0
            forest.soil_quality = 30.0
            forest.air_quality = 30.0
            forest.season = "winter"
            forest.weather = "cloudy"
            forest.time_of_day = "dusk"
            db.commit()
            db.refresh(forest)
            return forest
        
        # Calculate average wellness score
        avg_wellness = sum(m.ergonomic_score for m in wellness_metrics) / len(wellness_metrics)
        avg_posture = sum(m.posture_quality_score for m in wellness_metrics) / len(wellness_metrics)
        avg_activity = sum(m.activity_level_score for m in wellness_metrics) / len(wellness_metrics)
        
        # Calculate environmental factors
        latest_metrics = wellness_metrics[-1] if wellness_metrics else None
        
        # Sunlight = Mood + Engagement
        if latest_metrics:
            mood_map = {'negative': 20, 'neutral': 50, 'positive': 80}
            engagement_map = {'disengaged': 20, 'neutral': 50, 'engaged': 75, 'highly_engaged': 95}
            mood_score = mood_map.get(latest_metrics.mood_estimate, 50)
            engagement_score = engagement_map.get(latest_metrics.engagement_level, 50)
            sunlight = (mood_score + engagement_score) / 2
        else:
            sunlight = 50.0
        
        # Water = Break compliance
        if latest_metrics and latest_metrics.break_compliance_score:
            water = latest_metrics.break_compliance_score
        else:
            water = 50.0
        
        # Soil = Ergonomic score
        soil = avg_wellness
        
        # Air quality = Inverse of stress (100 - stress)
        if latest_metrics:
            stress_map = {'low': 10, 'moderate': 40, 'high': 70, 'very_high': 90}
            stress_value = stress_map.get(latest_metrics.stress_level, 50)
            air = 100 - stress_value
        else:
            air = 50.0
        
        # Calculate total trees (based on avg wellness, max 100)
        total_trees = min(100, max(10, int(avg_wellness)))
        
        # Distribute trees based on wellness score ranges
        if avg_wellness >= 80:
            # Excellent wellness - mostly healthy trees
            healthy_trees = int(total_trees * 0.80)
            growing_trees = int(total_trees * 0.15)
            wilting_trees = int(total_trees * 0.05)
            dead_trees = total_trees - healthy_trees - growing_trees - wilting_trees
        elif avg_wellness >= 60:
            # Good wellness - mix of healthy and growing
            healthy_trees = int(total_trees * 0.50)
            growing_trees = int(total_trees * 0.35)
            wilting_trees = int(total_trees * 0.10)
            dead_trees = int(total_trees * 0.05)
        elif avg_wellness >= 40:
            # Fair wellness - more growing and wilting
            healthy_trees = int(total_trees * 0.30)
            growing_trees = int(total_trees * 0.30)
            wilting_trees = int(total_trees * 0.30)
            dead_trees = int(total_trees * 0.10)
        else:
            # Poor wellness - mostly wilting and dead
            healthy_trees = int(total_trees * 0.15)
            growing_trees = int(total_trees * 0.20)
            wilting_trees = int(total_trees * 0.40)
            dead_trees = int(total_trees * 0.25)
        
        # Forest health score (weighted average)
        forest_health = (
            healthy_trees * 1.0 + 
            growing_trees * 0.7 + 
            wilting_trees * 0.3 + 
            dead_trees * 0.0
        ) / total_trees * 100 if total_trees > 0 else 0
        
        # Biodiversity score (based on activity variety)
        biodiversity = avg_activity
        
        # Growth rate (change over last 7 days)
        if len(wellness_metrics) >= 7:
            last_week = wellness_metrics[-7:]
            previous_week = wellness_metrics[-14:-7] if len(wellness_metrics) >= 14 else wellness_metrics[:7]
            
            last_week_avg = sum(m.ergonomic_score for m in last_week) / len(last_week)
            prev_week_avg = sum(m.ergonomic_score for m in previous_week) / len(previous_week)
            
            growth_rate = last_week_avg - prev_week_avg
        else:
            growth_rate = 0.0
        
        # Determine season based on overall wellness trend
        if growth_rate > 5:
            season = "spring"  # Growing
        elif avg_wellness >= 70:
            season = "summer"  # Thriving
        elif growth_rate < -5:
            season = "autumn"  # Declining
        else:
            season = "winter"  # Struggling
        
        # Determine weather based on stress
        if latest_metrics:
            if latest_metrics.stress_level == 'low':
                weather = "sunny"
            elif latest_metrics.stress_level == 'moderate':
                weather = "clear"
            elif latest_metrics.stress_level == 'high':
                weather = "cloudy"
            else:
                weather = "rainy"
        else:
            weather = "clear"
        
        # Determine time of day based on energy/fatigue
        if latest_metrics and latest_metrics.fatigue_indicator:
            if latest_metrics.fatigue_indicator < 0.3:
                time_of_day = "day"
            elif latest_metrics.fatigue_indicator < 0.6:
                time_of_day = "dusk"
            else:
                time_of_day = "night"
        else:
            time_of_day = "day"
        
        # Unlock special features based on metrics
        has_flowers = avg_wellness >= 80
        has_birds = avg_wellness >= 70
        has_butterflies = avg_wellness >= 60
        has_stream = water >= 70
        has_bench = air >= 70  # Low stress
        has_rocks = True  # Always present
        
        # Update forest model
        forest.total_trees = total_trees
        forest.healthy_trees = healthy_trees
        forest.growing_trees = growing_trees
        forest.wilting_trees = wilting_trees
        forest.dead_trees = dead_trees
        forest.forest_health_score = round(forest_health, 2)
        forest.biodiversity_score = round(biodiversity, 2)
        forest.growth_rate = round(growth_rate, 2)
        forest.sunlight_level = round(sunlight, 2)
        forest.water_level = round(water, 2)
        forest.soil_quality = round(soil, 2)
        forest.air_quality = round(air, 2)
        forest.has_flowers = has_flowers
        forest.has_birds = has_birds
        forest.has_butterflies = has_butterflies
        forest.has_stream = has_stream
        forest.has_rocks = has_rocks
        forest.has_bench = has_bench
        forest.season = season
        forest.time_of_day = time_of_day
        forest.weather = weather
        forest.last_updated = datetime.utcnow()
        
        db.commit()
        db.refresh(forest)
        return forest
    
    def get_forest(self, db: Session, user_id: int) -> Optional[WellnessForest]:
        """Get forest for a user, creating/updating if needed"""
        return self.calculate_and_update_forest(db, user_id)
