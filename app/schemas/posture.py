"""
Pydantic schemas for posture detection and wellness monitoring
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# Posture Snapshot Schemas
class PostureSnapshotBase(BaseModel):
    posture_type: str
    confidence: float
    neck_angle: Optional[float] = None
    shoulder_symmetry: Optional[float] = None
    spine_score: Optional[float] = None
    hip_knee_ankle_alignment: Optional[float] = None
    elbow_angle: Optional[float] = None
    activity: Optional[str] = None


class PostureSnapshotCreate(PostureSnapshotBase):
    user_id: Optional[int] = None
    keypoints_json: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    video_source: Optional[str] = None
    frame_number: Optional[int] = None


class PostureSnapshotResponse(PostureSnapshotBase):
    id: int
    user_id: Optional[int]
    timestamp: datetime
    session_id: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Wellness Metrics Schemas
class WellnessMetricsBase(BaseModel):
    ergonomic_score: float = Field(..., ge=0, le=100)
    posture_quality_score: float = Field(..., ge=0, le=100)
    activity_level_score: float = Field(..., ge=0, le=100)
    stress_level: Optional[str] = None
    fatigue_indicator: Optional[float] = Field(None, ge=0, le=1)
    engagement_level: Optional[str] = None
    mood_estimate: Optional[str] = None


class WellnessMetricsCreate(WellnessMetricsBase):
    user_id: int
    date: datetime
    period_type: str
    good_posture_percent: Optional[float] = None
    forward_head_percent: Optional[float] = None
    slouched_percent: Optional[float] = None
    poor_posture_percent: Optional[float] = None
    sitting_time_minutes: Optional[int] = None
    standing_time_minutes: Optional[int] = None
    walking_time_minutes: Optional[int] = None
    break_count: Optional[int] = None
    neck_pain_risk: Optional[float] = None
    back_pain_risk: Optional[float] = None
    shoulder_pain_risk: Optional[float] = None
    fidgeting_frequency: Optional[float] = None
    position_changes: Optional[int] = None
    break_compliance_score: Optional[float] = None


class WellnessMetricsResponse(WellnessMetricsBase):
    id: int
    user_id: int
    date: datetime
    period_type: str
    good_posture_percent: Optional[float]
    forward_head_percent: Optional[float]
    slouched_percent: Optional[float]
    poor_posture_percent: Optional[float]
    sitting_time_minutes: Optional[int]
    standing_time_minutes: Optional[int]
    walking_time_minutes: Optional[int]
    break_count: Optional[int]
    neck_pain_risk: Optional[float]
    back_pain_risk: Optional[float]
    shoulder_pain_risk: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Attrition Risk Schemas
class AttritionRiskBase(BaseModel):
    overall_risk_score: float = Field(..., ge=0, le=100)
    risk_category: str
    estimated_weeks_to_attrition: Optional[float] = None
    wellness_score_trend: Optional[str] = None
    stress_trend: Optional[str] = None
    engagement_trend: Optional[str] = None


class AttritionRiskCreate(AttritionRiskBase):
    user_id: int
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    wellness_score_30d_avg: Optional[float] = None
    wellness_score_7d_avg: Optional[float] = None
    wellness_score_change_percent: Optional[float] = None
    recommended_interventions: Optional[List[str]] = None
    model_version: Optional[str] = None
    confidence: Optional[float] = None


class AttritionRiskResponse(AttritionRiskBase):
    id: int
    user_id: int
    prediction_date: datetime
    top_risk_factors: Optional[List[Dict[str, Any]]]
    recommended_interventions: Optional[List[str]]
    confidence: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Demo Analysis Schemas (for CCTV upload)
class PersonAnalysis(BaseModel):
    """Analysis for a single person detected in CCTV"""
    person_id: str  # e.g., "Person 1", "Person 2"
    total_frames_detected: int
    
    # Average posture metrics
    avg_ergonomic_score: float
    avg_posture_quality: float
    dominant_posture: str
    
    # Posture breakdown
    posture_distribution: Dict[str, float]  # percentage of time in each posture
    
    # Wellness indicators
    stress_level: str
    fatigue_indicator: float
    mood_estimate: str
    engagement_level: str
    
    # Activity breakdown
    activity_distribution: Dict[str, float]
    
    # Risk assessments
    neck_pain_risk: float
    back_pain_risk: float
    overall_wellness_score: float
    
    # Movement patterns
    position_changes: int
    fidgeting_frequency: float
    
    # Sample posture snapshots (max 5)
    sample_snapshots: List[PostureSnapshotResponse]


class CCTVAnalysisResponse(BaseModel):
    """Complete analysis of CCTV video"""
    video_filename: str
    analysis_duration_seconds: float
    total_frames_processed: int
    fps: int
    
    # Persons detected
    persons_detected: int
    person_analyses: List[PersonAnalysis]
    
    # Overall statistics
    overall_avg_wellness: float
    high_risk_count: int
    
    session_id: str
    processed_at: datetime


# User Wellness Dashboard Schemas
class WellnessDashboardResponse(BaseModel):
    """Complete wellness dashboard for a user"""
    user_id: int
    username: str
    
    # Current metrics (today)
    today_ergonomic_score: Optional[float]
    today_stress_level: Optional[str]
    today_mood: Optional[str]
    
    # Trends (7 days)
    wellness_trend_7d: List[WellnessMetricsResponse]
    avg_wellness_7d: float
    
    # Trends (30 days)
    avg_wellness_30d: float
    wellness_change_percent: float  # change from previous 30 days
    
    # Attrition risk
    current_risk: Optional[AttritionRiskResponse]
    
    # Recommendations
    recommendations: List[str]
    
    # Posture breakdown (last 7 days)
    posture_quality_trend: str  # improving, stable, declining
    top_posture_issues: List[str]
    

class UserWellnessListItem(BaseModel):
    """Summary item for admin dashboard"""
    user_id: int
    username: str
    full_name: str
    
    current_wellness_score: float
    wellness_trend: str  # improving, stable, declining
    stress_level: str
    attrition_risk: str
    attrition_risk_score: float
    
    last_updated: datetime
    
    class Config:
        from_attributes = True


class AdminWellnessDashboard(BaseModel):
    """Aggregated wellness data for admin"""
    total_employees: int
    
    # Average scores
    avg_wellness_score: float
    avg_ergonomic_score: float
    avg_engagement: float
    
    # Risk distribution
    low_risk_count: int
    moderate_risk_count: int
    high_risk_count: int
    critical_risk_count: int
    
    # Employee list with wellness data
    employees: List[UserWellnessListItem]
    
    # Department insights (if available)
    wellness_distribution: Dict[str, float]
    
    last_updated: datetime


# Baseline Schemas
class PostureBaselineResponse(BaseModel):
    user_id: int
    is_complete: bool
    baseline_start_date: datetime
    baseline_end_date: Optional[datetime]
    avg_neck_angle: Optional[float]
    avg_shoulder_symmetry: Optional[float]
    avg_spine_score: Optional[float]
    
    class Config:
        from_attributes = True
