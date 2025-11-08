"""
Posture Detection Database Models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class PostureSnapshot(Base):
    """
    Individual posture snapshot captured from video analysis
    Stores skeletal keypoints and posture classification
    """
    __tablename__ = "posture_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Timestamp of capture
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Posture classification
    posture_type = Column(String(50), nullable=False)  # good, forward_head, slouched, leaning, twisted, collapsed
    confidence = Column(Float, nullable=False)
    
    # Skeletal angles (in degrees)
    neck_angle = Column(Float, nullable=True)
    shoulder_symmetry = Column(Float, nullable=True)  # 0-1 score
    spine_score = Column(Float, nullable=True)  # 0-100
    hip_knee_ankle_alignment = Column(Float, nullable=True)
    elbow_angle = Column(Float, nullable=True)
    
    # Activity classification
    activity = Column(String(50), nullable=True)  # sitting, standing, walking
    
    # MediaPipe keypoints (33 landmarks × 3 coordinates)
    keypoints_json = Column(JSON, nullable=True)
    
    # Session identifier (for grouping snapshots from same video)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Metadata
    video_source = Column(String(255), nullable=True)  # CCTV camera ID or user upload
    frame_number = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="posture_snapshots")
    
    def __repr__(self):
        return f"<PostureSnapshot(id={self.id}, user_id={self.user_id}, type={self.posture_type})>"


class WellnessMetrics(Base):
    """
    Aggregated wellness metrics calculated from posture data
    Typically computed daily or hourly
    """
    __tablename__ = "wellness_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Time period for these metrics
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # hourly, daily, weekly
    
    # Physical wellness scores (0-100)
    ergonomic_score = Column(Float, nullable=False)
    posture_quality_score = Column(Float, nullable=False)
    activity_level_score = Column(Float, nullable=False)
    
    # Behavioral indicators
    stress_level = Column(String(20), nullable=True)  # low, moderate, high, very_high
    fatigue_indicator = Column(Float, nullable=True)  # 0-1
    engagement_level = Column(String(20), nullable=True)  # disengaged, neutral, engaged, highly_engaged
    mood_estimate = Column(String(20), nullable=True)  # negative, neutral, positive
    
    # Posture breakdown (percentage of time in each posture)
    good_posture_percent = Column(Float, nullable=True)
    forward_head_percent = Column(Float, nullable=True)
    slouched_percent = Column(Float, nullable=True)
    poor_posture_percent = Column(Float, nullable=True)
    
    # Activity breakdown
    sitting_time_minutes = Column(Integer, nullable=True)
    standing_time_minutes = Column(Integer, nullable=True)
    walking_time_minutes = Column(Integer, nullable=True)
    break_count = Column(Integer, nullable=True)
    
    # Risk indicators
    neck_pain_risk = Column(Float, nullable=True)  # 0-1
    back_pain_risk = Column(Float, nullable=True)  # 0-1
    shoulder_pain_risk = Column(Float, nullable=True)  # 0-1
    
    # Movement patterns
    fidgeting_frequency = Column(Float, nullable=True)  # movements per minute
    position_changes = Column(Integer, nullable=True)
    
    # Compliance
    break_compliance_score = Column(Float, nullable=True)  # 0-100
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="wellness_metrics")
    
    def __repr__(self):
        return f"<WellnessMetrics(id={self.id}, user_id={self.user_id}, date={self.date})>"


class AttritionRisk(Base):
    """
    Attrition risk prediction based on wellness patterns
    """
    __tablename__ = "attrition_risks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Prediction date
    prediction_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Risk scores (0-100)
    overall_risk_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)  # low, moderate, high, critical
    
    # Time horizon prediction
    estimated_weeks_to_attrition = Column(Float, nullable=True)
    
    # Contributing factors (sorted by importance)
    top_risk_factors = Column(JSON, nullable=True)  # List of top 5 factors
    
    # Wellness trend indicators
    wellness_score_trend = Column(String(20), nullable=True)  # improving, stable, declining, rapidly_declining
    stress_trend = Column(String(20), nullable=True)
    engagement_trend = Column(String(20), nullable=True)
    
    # Historical comparison
    wellness_score_30d_avg = Column(Float, nullable=True)
    wellness_score_7d_avg = Column(Float, nullable=True)
    wellness_score_change_percent = Column(Float, nullable=True)
    
    # Recommendations
    recommended_interventions = Column(JSON, nullable=True)  # List of suggested actions
    
    # Model metadata
    model_version = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="attrition_risks")
    
    def __repr__(self):
        return f"<AttritionRisk(id={self.id}, user_id={self.user_id}, risk={self.risk_category})>"


class PostureBaseline(Base):
    """
    Baseline posture profile for each user
    Learned during initial 2-4 weeks
    """
    __tablename__ = "posture_baselines"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    # Baseline period
    baseline_start_date = Column(DateTime(timezone=True), nullable=False)
    baseline_end_date = Column(DateTime(timezone=True), nullable=False)
    is_complete = Column(Boolean, default=False)
    
    # Average normal posture characteristics
    avg_neck_angle = Column(Float, nullable=True)
    avg_shoulder_symmetry = Column(Float, nullable=True)
    avg_spine_score = Column(Float, nullable=True)
    
    # Normal posture distribution
    typical_posture_distribution = Column(JSON, nullable=True)
    
    # Normal work patterns
    typical_work_hours_start = Column(Integer, nullable=True)  # hour of day
    typical_work_hours_end = Column(Integer, nullable=True)
    avg_break_frequency = Column(Float, nullable=True)  # breaks per hour
    avg_daily_sitting_time = Column(Integer, nullable=True)  # minutes
    
    # Anomaly detection thresholds
    anomaly_threshold_neck = Column(Float, nullable=True)
    anomaly_threshold_posture = Column(Float, nullable=True)
    anomaly_threshold_stress = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="posture_baseline")
    
    def __repr__(self):
        return f"<PostureBaseline(user_id={self.user_id}, complete={self.is_complete})>"
