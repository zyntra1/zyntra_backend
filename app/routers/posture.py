"""
Posture Detection API Router
Admin endpoints for CCTV analysis and demo purposes
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import os
import tempfile
import uuid
from datetime import datetime

from app.core.database import get_db
from app.models.user import Admin
from app.schemas.posture import (
    CCTVAnalysisResponse,
    PersonAnalysis,
    PostureSnapshotResponse
)
from app.utils.dependencies import get_current_admin
from app.services.posture_analyzer import PostureAnalyzer
from app.services.wellness_scorer import WellnessScorer

router = APIRouter(prefix="/posture", tags=["Posture Detection - Admin"])

# Initialize services
posture_analyzer = PostureAnalyzer()
wellness_scorer = WellnessScorer()


@router.post("/analyze-cctv-demo", response_model=CCTVAnalysisResponse)
async def analyze_cctv_demo(
    file: UploadFile = File(...),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Demo endpoint: Analyze CCTV video and detect posture for all persons
    
    This is for demonstration purposes - detects persons as "Person 1", "Person 2", etc.
    Does NOT map to database users.
    
    Analyzes:
    - Posture classification (good, forward_head, slouched, etc.)
    - Wellness indicators (stress, fatigue, mood)
    - Activity patterns (sitting, standing, walking)
    - Risk assessments (neck pain, back pain, overall wellness)
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_video_path = temp_file.name
    
    try:
        start_time = datetime.utcnow()
        
        # Analyze video
        print(f"Starting posture analysis for: {file.filename}")
        analysis_result = posture_analyzer.analyze_video(
            video_path=temp_video_path,
            fps_sample=1,  # Sample every 1 second
            max_persons=5
        )
        
        # Process results for each detected person
        person_analyses = []
        
        for person_id, posture_snapshots in analysis_result['persons'].items():
            # Convert person_id to display format
            person_number = person_id.split('_')[1]
            display_id = f"Person {person_number}"
            
            # Calculate wellness metrics
            wellness_metrics = wellness_scorer.calculate_wellness_metrics(
                posture_snapshots,
                period_type="session"
            )
            
            # Calculate posture distribution
            posture_types = [snap['posture_type'] for snap in posture_snapshots]
            posture_distribution = {}
            for p_type in set(posture_types):
                count = posture_types.count(p_type)
                percentage = (count / len(posture_types)) * 100
                posture_distribution[p_type] = round(percentage, 1)
            
            # Dominant posture
            dominant_posture = max(posture_distribution, key=posture_distribution.get)
            
            # Activity distribution
            activities = [snap.get('activity', 'unknown') for snap in posture_snapshots]
            activity_distribution = {}
            for activity in set(activities):
                count = activities.count(activity)
                percentage = (count / len(activities)) * 100
                activity_distribution[activity] = round(percentage, 1)
            
            # Convert sample snapshots to response format
            sample_snapshots = []
            step = max(1, len(posture_snapshots) // 5)  # Get 5 samples
            for snap in posture_snapshots[::step][:5]:
                sample_snapshots.append(PostureSnapshotResponse(
                    id=0,  # Demo - no DB storage
                    user_id=None,
                    posture_type=snap['posture_type'],
                    confidence=snap['confidence'],
                    neck_angle=snap.get('neck_angle'),
                    shoulder_symmetry=snap.get('shoulder_symmetry'),
                    spine_score=snap.get('spine_score'),
                    hip_knee_ankle_alignment=None,
                    elbow_angle=None,
                    activity=snap.get('activity'),
                    timestamp=snap['timestamp'],
                    session_id=snap['session_id'],
                    created_at=snap['timestamp']
                ))
            
            # Calculate overall wellness score
            overall_wellness = (
                wellness_metrics['ergonomic_score'] * 0.4 +
                wellness_metrics['posture_quality_score'] * 0.3 +
                wellness_metrics['activity_level_score'] * 0.3
            )
            
            # Create person analysis
            person_analysis = PersonAnalysis(
                person_id=display_id,
                total_frames_detected=len(posture_snapshots),
                avg_ergonomic_score=wellness_metrics['ergonomic_score'],
                avg_posture_quality=wellness_metrics['posture_quality_score'],
                dominant_posture=dominant_posture,
                posture_distribution=posture_distribution,
                stress_level=wellness_metrics['stress_level'],
                fatigue_indicator=wellness_metrics['fatigue_indicator'],
                mood_estimate=wellness_metrics['mood_estimate'],
                engagement_level=wellness_metrics['engagement_level'],
                activity_distribution=activity_distribution,
                neck_pain_risk=wellness_metrics['neck_pain_risk'],
                back_pain_risk=wellness_metrics['back_pain_risk'],
                overall_wellness_score=round(overall_wellness, 2),
                position_changes=wellness_metrics['position_changes'],
                fidgeting_frequency=wellness_metrics['fidgeting_frequency'],
                sample_snapshots=sample_snapshots
            )
            
            person_analyses.append(person_analysis)
        
        # Calculate overall statistics
        if person_analyses:
            overall_avg_wellness = sum(p.overall_wellness_score for p in person_analyses) / len(person_analyses)
            high_risk_count = sum(1 for p in person_analyses if p.overall_wellness_score < 50)
        else:
            overall_avg_wellness = 0
            high_risk_count = 0
        
        end_time = datetime.utcnow()
        analysis_duration = (end_time - start_time).total_seconds()
        
        # Create response
        response = CCTVAnalysisResponse(
            video_filename=file.filename,
            analysis_duration_seconds=round(analysis_duration, 2),
            total_frames_processed=analysis_result['frames_processed'],
            fps=analysis_result['fps'],
            persons_detected=len(person_analyses),
            person_analyses=person_analyses,
            overall_avg_wellness=round(overall_avg_wellness, 2),
            high_risk_count=high_risk_count,
            session_id=analysis_result['session_id'],
            processed_at=datetime.utcnow()
        )
        
        return response
    
    except Exception as e:
        print(f"Error analyzing CCTV video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing video: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Clean up analyzer resources
        posture_analyzer.close()


@router.get("/health")
async def health_check():
    """
    Health check endpoint for posture detection service
    """
    return {
        "status": "healthy",
        "service": "posture_detection",
        "version": "1.0.0",
        "mediapipe_available": True
    }
