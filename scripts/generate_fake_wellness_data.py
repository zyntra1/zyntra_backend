"""
Generate Fake Wellness Data
Populates wellness metrics, posture snapshots, and attrition risks for demo users
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from faker import Faker
import random
from datetime import datetime, timedelta
import numpy as np

from app.core.database import SessionLocal
from app.models.user import User
from app.models.posture import PostureSnapshot, WellnessMetrics, AttritionRisk, PostureBaseline
from app.services.attrition_predictor import AttritionPredictor

faker = Faker()
attrition_predictor = AttritionPredictor()


def generate_fake_data_for_users(db: Session, user_ids: list[int]):
    """
    Generate comprehensive fake wellness data for specified users
    """
    print(f"Generating fake wellness data for {len(user_ids)} users...")
    
    for user_id in user_ids:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            print(f"User {user_id} not found, skipping...")
            continue
        
        print(f"\nGenerating data for user: {user.username} (ID: {user_id})")
        
        # Generate data for last 30 days
        generate_posture_snapshots(db, user_id, days=30)
        generate_wellness_metrics(db, user_id, days=30)
        generate_baseline(db, user_id)
        generate_attrition_risk(db, user_id)
        
        print(f"✓ Completed data generation for {user.username}")
    
    db.commit()
    print("\n✅ All fake data generated successfully!")


def generate_posture_snapshots(db: Session, user_id: int, days: int = 30):
    """
    Generate fake posture snapshots (hourly samples for work hours)
    """
    print("  - Generating posture snapshots...")
    
    posture_types = ['good', 'forward_head', 'slouched', 'leaning', 'moderate']
    activities = ['sitting', 'standing', 'walking']
    
    # User personality (affects posture quality)
    base_wellness = random.randint(40, 90)  # User's baseline wellness
    
    snapshots_created = 0
    
    for day in range(days):
        date = datetime.utcnow() - timedelta(days=days-day)
        
        # Only work hours (9 AM - 6 PM), hourly samples
        for hour in range(9, 18):
            timestamp = date.replace(hour=hour, minute=random.randint(0, 59))
            
            # Posture quality degrades throughout day
            time_fatigue_factor = (hour - 9) / 9  # 0 to 1
            
            # Random variation
            variation = random.uniform(-10, 10)
            
            # Calculate metrics
            posture_quality = base_wellness - (time_fatigue_factor * 20) + variation
            posture_quality = max(30, min(100, posture_quality))
            
            # Select posture type based on quality
            if posture_quality >= 75:
                posture_type = 'good'
                confidence = random.uniform(0.85, 0.95)
            elif posture_quality >= 60:
                posture_type = random.choice(['good', 'moderate'])
                confidence = random.uniform(0.75, 0.90)
            elif posture_quality >= 45:
                posture_type = random.choice(['moderate', 'forward_head', 'slouched'])
                confidence = random.uniform(0.70, 0.85)
            else:
                posture_type = random.choice(['forward_head', 'slouched', 'leaning'])
                confidence = random.uniform(0.65, 0.80)
            
            # Calculate angles
            if posture_type == 'good':
                neck_angle = random.uniform(165, 180)
                shoulder_symmetry = random.uniform(0.85, 1.0)
                spine_score = random.uniform(80, 100)
            elif posture_type == 'forward_head':
                neck_angle = random.uniform(135, 155)
                shoulder_symmetry = random.uniform(0.70, 0.90)
                spine_score = random.uniform(60, 80)
            elif posture_type == 'slouched':
                neck_angle = random.uniform(145, 165)
                shoulder_symmetry = random.uniform(0.50, 0.75)
                spine_score = random.uniform(40, 65)
            else:  # leaning or moderate
                neck_angle = random.uniform(150, 170)
                shoulder_symmetry = random.uniform(0.65, 0.85)
                spine_score = random.uniform(55, 75)
            
            # Activity (mostly sitting during work)
            if random.random() < 0.75:
                activity = 'sitting'
            elif random.random() < 0.85:
                activity = 'standing'
            else:
                activity = 'walking'
            
            snapshot = PostureSnapshot(
                user_id=user_id,
                timestamp=timestamp,
                posture_type=posture_type,
                confidence=round(confidence, 2),
                neck_angle=round(neck_angle, 2),
                shoulder_symmetry=round(shoulder_symmetry, 2),
                spine_score=round(spine_score, 2),
                activity=activity,
                session_id=f"fake_session_{user_id}_{day}",
                video_source=f"CCTV_Camera_{random.randint(1, 5)}"
            )
            
            db.add(snapshot)
            snapshots_created += 1
    
    print(f"    Created {snapshots_created} posture snapshots")


def generate_wellness_metrics(db: Session, user_id: int, days: int = 30):
    """
    Generate fake daily wellness metrics
    """
    print("  - Generating wellness metrics...")
    
    stress_levels = ['low', 'moderate', 'high', 'very_high']
    engagement_levels = ['disengaged', 'neutral', 'engaged', 'highly_engaged']
    moods = ['negative', 'neutral', 'positive']
    
    # User personality
    base_wellness = random.randint(45, 85)
    base_stress_idx = random.randint(0, 2)  # 0=low, 1=moderate, 2=high
    base_engagement_idx = random.randint(1, 3)  # 1=neutral, 2=engaged, 3=highly_engaged
    
    metrics_created = 0
    
    for day in range(days):
        date = datetime.utcnow() - timedelta(days=days-day)
        
        # Add trend (improving or declining over time)
        trend_factor = random.choice([-0.5, 0, 0.5])  # declining, stable, improving
        day_factor = trend_factor * day
        
        # Random daily variation
        daily_variation = random.uniform(-8, 8)
        
        # Calculate scores
        ergonomic_score = base_wellness + day_factor + daily_variation
        ergonomic_score = max(20, min(100, ergonomic_score))
        
        posture_quality = ergonomic_score + random.uniform(-5, 5)
        posture_quality = max(20, min(100, posture_quality))
        
        activity_level = random.uniform(40, 80)
        
        # Stress level
        stress_variation = random.randint(-1, 1)
        stress_idx = max(0, min(3, base_stress_idx + stress_variation))
        stress_level = stress_levels[stress_idx]
        
        # Fatigue (higher towards end of month)
        fatigue_indicator = 0.3 + (day / days) * 0.3 + random.uniform(-0.1, 0.1)
        fatigue_indicator = max(0, min(1, fatigue_indicator))
        
        # Engagement
        engagement_variation = random.randint(-1, 1)
        engagement_idx = max(0, min(3, base_engagement_idx + engagement_variation))
        engagement_level = engagement_levels[engagement_idx]
        
        # Mood
        if stress_level in ['high', 'very_high']:
            mood = random.choice(['negative', 'neutral'])
        elif engagement_level in ['engaged', 'highly_engaged']:
            mood = random.choice(['neutral', 'positive'])
        else:
            mood = random.choice(moods)
        
        # Posture distribution
        good_posture_pct = max(10, min(70, ergonomic_score * 0.6 + random.uniform(-10, 10)))
        forward_head_pct = random.uniform(10, 30)
        slouched_pct = random.uniform(10, 40)
        poor_posture_pct = 100 - good_posture_pct
        
        # Activity breakdown (8-hour workday = 480 minutes)
        sitting_time = random.randint(300, 420)  # 5-7 hours
        standing_time = random.randint(30, 90)
        walking_time = 480 - sitting_time - standing_time
        break_count = random.randint(2, 8)
        
        # Pain risks
        neck_pain_risk = max(0, min(1, (100 - ergonomic_score) / 100 + random.uniform(-0.2, 0.2)))
        back_pain_risk = max(0, min(1, slouched_pct / 80 + random.uniform(-0.2, 0.2)))
        shoulder_pain_risk = max(0, min(1, forward_head_pct / 70 + random.uniform(-0.2, 0.2)))
        
        # Movement patterns
        fidgeting_frequency = random.uniform(1, 8)
        position_changes = random.randint(5, 30)
        break_compliance = (break_count / 8) * 100
        
        wellness_metric = WellnessMetrics(
            user_id=user_id,
            date=date,
            period_type='daily',
            ergonomic_score=round(ergonomic_score, 2),
            posture_quality_score=round(posture_quality, 2),
            activity_level_score=round(activity_level, 2),
            stress_level=stress_level,
            fatigue_indicator=round(fatigue_indicator, 2),
            engagement_level=engagement_level,
            mood_estimate=mood,
            good_posture_percent=round(good_posture_pct, 2),
            forward_head_percent=round(forward_head_pct, 2),
            slouched_percent=round(slouched_pct, 2),
            poor_posture_percent=round(poor_posture_pct, 2),
            sitting_time_minutes=sitting_time,
            standing_time_minutes=standing_time,
            walking_time_minutes=walking_time,
            break_count=break_count,
            neck_pain_risk=round(neck_pain_risk, 2),
            back_pain_risk=round(back_pain_risk, 2),
            shoulder_pain_risk=round(shoulder_pain_risk, 2),
            fidgeting_frequency=round(fidgeting_frequency, 2),
            position_changes=position_changes,
            break_compliance_score=round(break_compliance, 2)
        )
        
        db.add(wellness_metric)
        metrics_created += 1
    
    print(f"    Created {metrics_created} wellness metrics")


def generate_baseline(db: Session, user_id: int):
    """
    Generate fake baseline profile
    """
    print("  - Generating baseline profile...")
    
    # Get existing metrics
    metrics = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == user_id
    ).order_by(WellnessMetrics.date).limit(14).all()
    
    if not metrics:
        print("    No metrics found, skipping baseline")
        return
    
    # Calculate baseline values
    avg_ergonomic = np.mean([m.ergonomic_score for m in metrics])
    avg_posture = np.mean([m.posture_quality_score for m in metrics])
    avg_sitting = np.mean([m.sitting_time_minutes for m in metrics])
    avg_breaks = np.mean([m.break_count for m in metrics])
    
    baseline = PostureBaseline(
        user_id=user_id,
        baseline_start_date=metrics[0].date,
        baseline_end_date=metrics[-1].date,
        is_complete=True,
        avg_neck_angle=random.uniform(155, 175),
        avg_shoulder_symmetry=random.uniform(0.75, 0.95),
        avg_spine_score=random.uniform(60, 85),
        typical_posture_distribution={'good': 45.0, 'slouched': 30.0, 'forward_head': 25.0},
        typical_work_hours_start=9,
        typical_work_hours_end=18,
        avg_break_frequency=round(avg_breaks / 8, 2),
        avg_daily_sitting_time=round(avg_sitting, 1),
        anomaly_threshold_neck=150.0,
        anomaly_threshold_posture=50.0,
        anomaly_threshold_stress=0.7
    )
    
    db.add(baseline)
    print("    Created baseline profile")


def generate_attrition_risk(db: Session, user_id: int):
    """
    Generate fake attrition risk prediction
    """
    print("  - Generating attrition risk prediction...")
    
    # Get wellness history
    wellness_history = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == user_id
    ).order_by(WellnessMetrics.date).all()
    
    if not wellness_history:
        print("    No wellness history found, skipping attrition risk")
        return
    
    # Convert to dictionaries
    wellness_dicts = []
    for w in wellness_history:
        wellness_dicts.append({
            'date': w.date,
            'ergonomic_score': w.ergonomic_score,
            'posture_quality_score': w.posture_quality_score,
            'activity_level_score': w.activity_level_score,
            'stress_level': w.stress_level,
            'fatigue_indicator': w.fatigue_indicator,
            'engagement_level': w.engagement_level,
            'break_compliance_score': w.break_compliance_score,
            'neck_pain_risk': w.neck_pain_risk,
            'back_pain_risk': w.back_pain_risk,
            'shoulder_pain_risk': w.shoulder_pain_risk
        })
    
    # Get baseline
    baseline = db.query(PostureBaseline).filter(
        PostureBaseline.user_id == user_id
    ).first()
    
    baseline_dict = None
    if baseline:
        baseline_dict = {
            'is_complete': baseline.is_complete,
            'avg_ergonomic_score': baseline.avg_spine_score
        }
    
    # Predict attrition risk
    current_metrics = wellness_dicts[-1] if wellness_dicts else {}
    risk_prediction = attrition_predictor.predict_attrition_risk(
        wellness_history=wellness_dicts,
        current_metrics=current_metrics,
        baseline=baseline_dict
    )
    
    attrition_risk = AttritionRisk(
        user_id=user_id,
        prediction_date=datetime.utcnow(),
        overall_risk_score=risk_prediction['overall_risk_score'],
        risk_category=risk_prediction['risk_category'],
        estimated_weeks_to_attrition=risk_prediction.get('estimated_weeks_to_attrition'),
        top_risk_factors=risk_prediction.get('top_risk_factors', []),
        wellness_score_trend=risk_prediction.get('wellness_score_trend'),
        stress_trend=risk_prediction.get('stress_trend'),
        engagement_trend=risk_prediction.get('engagement_trend'),
        wellness_score_30d_avg=risk_prediction.get('wellness_score_30d_avg'),
        wellness_score_7d_avg=risk_prediction.get('wellness_score_7d_avg'),
        wellness_score_change_percent=risk_prediction.get('wellness_score_change_percent'),
        recommended_interventions=risk_prediction.get('recommended_interventions', []),
        model_version=risk_prediction.get('model_version'),
        confidence=risk_prediction.get('confidence')
    )
    
    db.add(attrition_risk)
    print(f"    Created attrition risk: {risk_prediction['risk_category']} ({risk_prediction['overall_risk_score']:.1f})")


if __name__ == "__main__":
    # User IDs from your database
    USER_IDS = [2, 3, 4, 5, 6]  # adhithya, afish, amal, arpit, munazir
    
    db = SessionLocal()
    try:
        generate_fake_data_for_users(db, USER_IDS)
    except Exception as e:
        print(f"\n❌ Error generating fake data: {e}")
        db.rollback()
    finally:
        db.close()
