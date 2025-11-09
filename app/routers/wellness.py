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
from app.models.posture import WellnessMetrics, AttritionRisk, PostureBaseline, WellnessForest
from app.schemas.posture import (
    WellnessMetricsResponse,
    AttritionRiskResponse,
    WellnessDashboardResponse,
    AdminWellnessDashboard,
    UserWellnessListItem,
    WellnessForestResponse,
    AdminForestOverview,
    AIInsightsResponse
)
from app.utils.dependencies import get_current_user, get_current_admin
from app.services.forest_calculator import ForestCalculator
from app.services.ai_insights import ai_insights_generator

# Separate routers for user and admin
user_router = APIRouter(prefix="/wellness", tags=["Wellness - User"])
admin_router = APIRouter(prefix="/wellness", tags=["Wellness - Admin"])

# Initialize forest calculator
forest_calculator = ForestCalculator()


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


@user_router.get("/ai-insights", response_model=AIInsightsResponse)
async def get_ai_wellness_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated wellness insights for current user
    
    Returns 8 personalized insights based on the user's wellness data
    from the last 30 days, powered by Google Gemini AI.
    """
    # Get last 30 days metrics
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    metrics = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == current_user.id,
        WellnessMetrics.date >= thirty_days_ago
    ).order_by(WellnessMetrics.date).all()
    
    # Get latest attrition risk
    latest_risk = db.query(AttritionRisk).filter(
        AttritionRisk.user_id == current_user.id
    ).order_by(desc(AttritionRisk.prediction_date)).first()
    
    # Generate AI insights
    insights = ai_insights_generator.generate_wellness_insights(
        user_name=current_user.username,
        recent_metrics=metrics,
        latest_risk=latest_risk
    )
    
    # Calculate summary stats
    if metrics:
        avg_wellness = sum(m.ergonomic_score for m in metrics) / len(metrics)
        
        # Determine trend
        if len(metrics) >= 14:
            first_week = metrics[:7]
            last_week = metrics[-7:]
            first_avg = sum(m.ergonomic_score for m in first_week) / len(first_week)
            last_avg = sum(m.ergonomic_score for m in last_week) / len(last_week)
            
            if last_avg > first_avg + 5:
                trend = "improving"
            elif last_avg < first_avg - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
    else:
        avg_wellness = 50.0
        trend = "insufficient_data"
    
    return AIInsightsResponse(
        user_id=current_user.id,
        username=current_user.username,
        insights=insights,
        generated_at=datetime.utcnow(),
        data_period_days=len(metrics),
        avg_wellness_score=round(avg_wellness, 2),
        wellness_trend=trend
    )


@admin_router.get("/dashboard", response_model=AdminWellnessDashboard)
async def get_admin_wellness_dashboard(
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get aggregated wellness dashboard for all employees under this admin
    
    Optional filters:
    - user_id: Filter by specific employee ID
    - username: Filter by specific employee username
    """
    # Build query for users under this admin
    query = db.query(User).filter(User.admin_id == current_admin.id)
    
    # Apply filters if provided
    if user_id:
        query = query.filter(User.id == user_id)
    if username:
        query = query.filter(User.username.ilike(f"%{username}%"))
    
    users = query.all()
    
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
    
    # Get today's metrics
    today = datetime.utcnow().date()
    today_metrics = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == user.id,
        func.date(WellnessMetrics.date) == today
    ).first()
    
    # Get last 7 days metrics
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    metrics_7d = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == user.id,
        WellnessMetrics.date >= seven_days_ago
    ).order_by(WellnessMetrics.date).all()
    
    # Get last 30 days metrics
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    metrics_30d = db.query(WellnessMetrics).filter(
        WellnessMetrics.user_id == user.id,
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
        AttritionRisk.user_id == user.id
    ).order_by(desc(AttritionRisk.prediction_date)).first()
    
    # Generate recommendations
    recommendations = []
    if today_metrics:
        if today_metrics.ergonomic_score < 50:
            recommendations.append("Consider adjusting workstation for better ergonomics")
        if today_metrics.stress_level in ['high', 'very_high']:
            recommendations.append("Take regular breaks to manage stress levels")
        if today_metrics.break_compliance_score and today_metrics.break_compliance_score < 50:
            recommendations.append("Remember to take breaks every hour")
    else:
        recommendations.append("No data available for today - ensure monitoring is active")
    
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
        if today_metrics.forward_head_percent and today_metrics.forward_head_percent > 30:
            top_issues.append("Forward head posture")
        if today_metrics.slouched_percent and today_metrics.slouched_percent > 30:
            top_issues.append("Slouching")
        if today_metrics.neck_pain_risk and today_metrics.neck_pain_risk > 0.6:
            top_issues.append("Neck pain risk")
    
    return WellnessDashboardResponse(
        user_id=user.id,
        username=user.username,
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


# ============================================
# WELLNESS FOREST ENDPOINTS
# ============================================

@user_router.get("/forest", response_model=WellnessForestResponse)
async def get_my_forest(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's wellness forest visualization
    Virtual forest representation of wellness metrics
    """
    # Calculate and get forest
    forest = forest_calculator.get_forest(db, current_user.id)
    
    return WellnessForestResponse(
        user_id=forest.user_id,
        username=current_user.username,
        total_trees=forest.total_trees,
        healthy_trees=forest.healthy_trees,
        growing_trees=forest.growing_trees,
        wilting_trees=forest.wilting_trees,
        dead_trees=forest.dead_trees,
        forest_health_score=forest.forest_health_score,
        biodiversity_score=forest.biodiversity_score,
        growth_rate=forest.growth_rate,
        sunlight_level=forest.sunlight_level,
        water_level=forest.water_level,
        soil_quality=forest.soil_quality,
        air_quality=forest.air_quality,
        has_flowers=forest.has_flowers,
        has_birds=forest.has_birds,
        has_butterflies=forest.has_butterflies,
        has_stream=forest.has_stream,
        has_rocks=forest.has_rocks,
        has_bench=forest.has_bench,
        season=forest.season,
        time_of_day=forest.time_of_day,
        weather=forest.weather,
        last_updated=forest.last_updated
    )


@admin_router.get("/forests", response_model=AdminForestOverview)
async def get_all_forests(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get all employee forests overview (admin)
    """
    # Get all users under this admin
    users = db.query(User).filter(User.admin_id == current_admin.id).all()
    
    if not users:
        return AdminForestOverview(
            total_employees=0,
            avg_forest_health=0.0,
            healthiest_forest=None,
            most_improved_forest=None,
            needs_attention=[],
            forests=[],
            last_updated=datetime.utcnow()
        )
    
    # Get/calculate forests for all users
    forests_data = []
    total_health = 0.0
    healthiest_score = 0.0
    healthiest_user = None
    most_growth = -1000.0
    most_growth_user = None
    needs_attention = []
    
    for user in users:
        forest = forest_calculator.get_forest(db, user.id)
        
        forest_response = WellnessForestResponse(
            user_id=forest.user_id,
            username=user.username,
            total_trees=forest.total_trees,
            healthy_trees=forest.healthy_trees,
            growing_trees=forest.growing_trees,
            wilting_trees=forest.wilting_trees,
            dead_trees=forest.dead_trees,
            forest_health_score=forest.forest_health_score,
            biodiversity_score=forest.biodiversity_score,
            growth_rate=forest.growth_rate,
            sunlight_level=forest.sunlight_level,
            water_level=forest.water_level,
            soil_quality=forest.soil_quality,
            air_quality=forest.air_quality,
            has_flowers=forest.has_flowers,
            has_birds=forest.has_birds,
            has_butterflies=forest.has_butterflies,
            has_stream=forest.has_stream,
            has_rocks=forest.has_rocks,
            has_bench=forest.has_bench,
            season=forest.season,
            time_of_day=forest.time_of_day,
            weather=forest.weather,
            last_updated=forest.last_updated
        )
        
        forests_data.append(forest_response)
        total_health += forest.forest_health_score
        
        # Track healthiest
        if forest.forest_health_score > healthiest_score:
            healthiest_score = forest.forest_health_score
            healthiest_user = user.username
        
        # Track most improved
        if forest.growth_rate > most_growth:
            most_growth = forest.growth_rate
            most_growth_user = user.username
        
        # Track needs attention
        if forest.forest_health_score < 40:
            needs_attention.append(user.username)
    
    avg_health = total_health / len(users) if users else 0.0
    
    return AdminForestOverview(
        total_employees=len(users),
        avg_forest_health=round(avg_health, 2),
        healthiest_forest=healthiest_user,
        most_improved_forest=most_growth_user if most_growth > 0 else None,
        needs_attention=needs_attention,
        forests=forests_data,
        last_updated=datetime.utcnow()
    )


@admin_router.get("/forest/{user_id}", response_model=WellnessForestResponse)
async def get_user_forest_by_admin(
    user_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get specific user's forest (admin access)
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
    
    # Calculate and get forest
    forest = forest_calculator.get_forest(db, user.id)
    
    return WellnessForestResponse(
        user_id=forest.user_id,
        username=user.username,
        total_trees=forest.total_trees,
        healthy_trees=forest.healthy_trees,
        growing_trees=forest.growing_trees,
        wilting_trees=forest.wilting_trees,
        dead_trees=forest.dead_trees,
        forest_health_score=forest.forest_health_score,
        biodiversity_score=forest.biodiversity_score,
        growth_rate=forest.growth_rate,
        sunlight_level=forest.sunlight_level,
        water_level=forest.water_level,
        soil_quality=forest.soil_quality,
        air_quality=forest.air_quality,
        has_flowers=forest.has_flowers,
        has_birds=forest.has_birds,
        has_butterflies=forest.has_butterflies,
        has_stream=forest.has_stream,
        has_rocks=forest.has_rocks,
        has_bench=forest.has_bench,
        season=forest.season,
        time_of_day=forest.time_of_day,
        weather=forest.weather,
        last_updated=forest.last_updated
    )

