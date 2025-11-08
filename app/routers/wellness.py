"""
Wellness Monitoring API Router
Endpoints for user and admin wellness dashboards
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.user import User, Admin
from app.models.posture import WellnessMetrics, AttritionRisk, PostureBaseline
from app.schemas.posture import (
    WellnessMetricsResponse,
    AttritionRiskResponse,
    WellnessDashboardResponse,
    AdminWellnessDashboard,
    UserWellnessListItem
)
from app.utils.dependencies import get_current_user, get_current_admin

# Separate routers for user and admin
user_router = APIRouter(prefix="/wellness", tags=["Wellness - User"])
admin_router = APIRouter(prefix="/wellness", tags=["Wellness - Admin"])


@user_router.get("/dashboard", response_model=WellnessDashboardResponse)
async def get_user_wellness_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive wellness dashboard for current user
    """
    # Get today's metrics
    today = datetime.utcnow().date()
    today_metrics = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == current_user.id,
        func.date(WellnessMetrics.date) == today
    ).first()
    
    # Get last 7 days metrics
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    metrics_7d = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == current_user.id,
        WellnessMetrics.date >= seven_days_ago
    ).order_by(WellnessMetrics.date).all()
    
    # Get last 30 days metrics
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    metrics_30d = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == current_user.id,
        WellnessMetrics.date >= thirty_days_ago
    ).all()
    
    # Calculate averages
    avg_wellness_7d = sum(m.ergonomic_score for m in metrics_7d) / len(metrics_7d) if metrics_7d else 50
    avg_wellness_30d = sum(m.ergonomic_score for m in metrics_30d) / len(metrics_30d) if metrics_30d else 50
    
    # Calculate change percentage
    if len(metrics_30d) >= 14:
        first_week = metrics_30d[:7]
        last_week = metrics_30d[-7:]
        first_avg = sum(m.ergonomic_score for m in first_week) / len(first_week)
        last_avg = sum(m.ergonomic_score for m in last_week) / len(last_week)
        wellness_change = ((last_avg - first_avg) / first_avg) * 100
    else:
        wellness_change = 0
    
    # Get current attrition risk
    current_risk = db.query(AttritionRisk).filter(
        AttritionRisk.user_id == current_user.id
    ).order_by(desc(AttritionRisk.prediction_date)).first()
    
    # Generate recommendations
    recommendations = []
    if today_metrics:
        if today_metrics.ergonomic_score < 50:
            recommendations.append("Consider adjusting your workstation for better ergonomics")
        if today_metrics.stress_level in ['high', 'very_high']:
            recommendations.append("Take regular breaks to manage stress levels")
        if today_metrics.break_compliance_score < 50:
            recommendations.append("Remember to take breaks every hour")
    
    # Determine posture trend
    if len(metrics_7d) >= 3:
        recent_scores = [m.posture_quality_score for m in metrics_7d[-3:]]
        earlier_scores = [m.posture_quality_score for m in metrics_7d[:3]]
        if sum(recent_scores) > sum(earlier_scores):
            posture_trend = "improving"
        elif sum(recent_scores) < sum(earlier_scores):
            posture_trend = "declining"
        else:
            posture_trend = "stable"
    else:
        posture_trend = "stable"
    
    # Top posture issues
    top_issues = []
    if today_metrics:
        if today_metrics.forward_head_percent > 30:
            top_issues.append("Forward head posture")
        if today_metrics.slouched_percent > 30:
            top_issues.append("Slouching")
        if today_metrics.neck_pain_risk > 0.6:
            top_issues.append("Neck pain risk")
    
    return WellnessDashboardResponse(
        user_id=current_user.id,
        username=current_user.username,
        today_ergonomic_score=today_metrics.ergonomic_score if today_metrics else None,
        today_stress_level=today_metrics.stress_level if today_metrics else None,
        today_mood=today_metrics.mood_estimate if today_metrics else None,
        wellness_trend_7d=[WellnessMetricsResponse.from_orm(m) for m in metrics_7d],
        avg_wellness_7d=round(avg_wellness_7d, 2),
        avg_wellness_30d=round(avg_wellness_30d, 2),
        wellness_change_percent=round(wellness_change, 2),
        current_risk=AttritionRiskResponse.from_orm(current_risk) if current_risk else None,
        recommendations=recommendations,
        posture_quality_trend=posture_trend,
        top_posture_issues=top_issues
    )


@admin_router.get("/dashboard", response_model=AdminWellnessDashboard)
async def get_admin_wellness_dashboard(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get aggregated wellness dashboard for all employees under this admin
    """
    # Get all users under this admin
    users = db.query(User).filter(User.admin_id == current_admin.id).all()
    
    if not users:
        return AdminWellnessDashboard(
            total_employees=0,
            avg_wellness_score=0,
            avg_ergonomic_score=0,
            avg_engagement=0,
            low_risk_count=0,
            moderate_risk_count=0,
            high_risk_count=0,
            critical_risk_count=0,
            employees=[],
            wellness_distribution={},
            last_updated=datetime.utcnow()
        )
    
    # Get latest metrics for each user
    employee_data = []
    total_wellness = 0
    total_ergonomic = 0
    total_engagement = 0
    
    risk_counts = {'low': 0, 'moderate': 0, 'high': 0, 'critical': 0}
    
    for user in users:
        # Get latest wellness metrics
        latest_metrics = db.query(WellnessMetrics).filter(
            WellnessMetrics.user_id == user.id
        ).order_by(desc(WellnessMetrics.date)).first()
        
        # Get latest attrition risk
        latest_risk = db.query(AttritionRisk).filter(
            AttritionRisk.user_id == user.id
        ).order_by(desc(AttritionRisk.prediction_date)).first()
        
        if latest_metrics:
            wellness_score = latest_metrics.ergonomic_score
            total_wellness += wellness_score
            total_ergonomic += latest_metrics.ergonomic_score
            
            # Engagement score
            engagement_map = {'disengaged': 20, 'neutral': 50, 'engaged': 75, 'highly_engaged': 95}
            engagement_score = engagement_map.get(latest_metrics.engagement_level, 50)
            total_engagement += engagement_score
            
            # Determine wellness trend
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_metrics = db.query(WellnessMetrics).filter(
                WellnessMetrics.user_id == user.id,
                WellnessMetrics.date >= seven_days_ago
            ).all()
            
            if len(recent_metrics) >= 3:
                scores = [m.ergonomic_score for m in recent_metrics]
                if scores[-1] > scores[0] + 5:
                    wellness_trend = "improving"
                elif scores[-1] < scores[0] - 5:
                    wellness_trend = "declining"
                else:
                    wellness_trend = "stable"
            else:
                wellness_trend = "stable"
            
            # Risk data
            if latest_risk:
                risk_category = latest_risk.risk_category
                risk_score = latest_risk.overall_risk_score
                risk_counts[risk_category] += 1
            else:
                risk_category = "low"
                risk_score = 0
                risk_counts['low'] += 1
            
            employee_data.append(UserWellnessListItem(
                user_id=user.id,
                username=user.username,
                full_name=user.full_name or user.username,
                current_wellness_score=wellness_score,
                wellness_trend=wellness_trend,
                stress_level=latest_metrics.stress_level,
                attrition_risk=risk_category,
                attrition_risk_score=risk_score,
                last_updated=latest_metrics.date
            ))
    
    # Calculate averages
    avg_wellness = total_wellness / len(employee_data) if employee_data else 0
    avg_ergonomic = total_ergonomic / len(employee_data) if employee_data else 0
    avg_engagement = total_engagement / len(employee_data) if employee_data else 0
    
    # Wellness distribution
    wellness_distribution = {
        'excellent': sum(1 for e in employee_data if e.current_wellness_score >= 80),
        'good': sum(1 for e in employee_data if 60 <= e.current_wellness_score < 80),
        'fair': sum(1 for e in employee_data if 40 <= e.current_wellness_score < 60),
        'poor': sum(1 for e in employee_data if e.current_wellness_score < 40)
    }
    
    return AdminWellnessDashboard(
        total_employees=len(users),
        avg_wellness_score=round(avg_wellness, 2),
        avg_ergonomic_score=round(avg_ergonomic, 2),
        avg_engagement=round(avg_engagement, 2),
        low_risk_count=risk_counts['low'],
        moderate_risk_count=risk_counts['moderate'],
        high_risk_count=risk_counts['high'],
        critical_risk_count=risk_counts['critical'],
        employees=employee_data,
        wellness_distribution=wellness_distribution,
        last_updated=datetime.utcnow()
    )


@admin_router.get("/user/{user_id}", response_model=WellnessDashboardResponse)
async def get_user_wellness_by_admin(
    user_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get wellness dashboard for a specific user (admin access)
    """
    # Verify user belongs to this admin
    user = db.query(User).filter(
        User.id == user_id,
        User.admin_id == current_admin.id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or not under your management"
        )
    
    # Reuse user dashboard logic
    # (Simplified - in production, extract to shared function)
    return WellnessDashboardResponse(
        user_id=user.id,
        username=user.username,
        today_ergonomic_score=None,
        today_stress_level=None,
        today_mood=None,
        wellness_trend_7d=[],
        avg_wellness_7d=0,
        avg_wellness_30d=0,
        wellness_change_percent=0,
        current_risk=None,
        recommendations=[],
        posture_quality_trend="stable",
        top_posture_issues=[]
    )
