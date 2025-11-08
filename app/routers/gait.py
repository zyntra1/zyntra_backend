"""
Gait Recognition API Endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import os
import json
import uuid
import tempfile

from app.core.database import get_db
from app.models.user import User, Admin
from app.models.gait import GaitProfile, GaitRecognitionLog, GaitDetection
from app.schemas.gait import (
    GaitProfileResponse,
    GaitRecognitionResponse,
    GaitRecognitionLogList,
    VideoUploadResponse,
    GaitDetectionResponse
)
from app.utils.dependencies import get_current_user, get_current_admin
from app.utils.video_storage import video_storage
from app.services.gait_processor import GaitProcessor, serialize_embedding, deserialize_embedding
from app.core.config import settings

# Create separate routers for user and admin endpoints
user_router = APIRouter(prefix="/gait", tags=["Gait Recognition - User"])
admin_router = APIRouter(prefix="/gait", tags=["Gait Recognition - Admin"])

# Initialize gait processor
# Model paths configured in settings, downloaded on startup
gait_processor = GaitProcessor(model_path=settings.GAIT_MODEL_PATH)


@user_router.post("/upload-user-video", response_model=VideoUploadResponse)
async def upload_user_gait_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload gait video for a user
    Extracts gait features and stores them in the database
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    # Check if user already has a gait profile
    existing_profile = db.query(GaitProfile).filter(
        GaitProfile.user_id == current_user.id
    ).first()
    
    if existing_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has a gait profile. Delete the existing one first."
        )
    
    try:
        # Save video file to Azure
        video_path = await video_storage.save_user_video(file, current_user.id)
        
        # Create pending profile immediately
        pending_profile = GaitProfile(
            user_id=current_user.id,
            video_path=video_path,
            embedding=b'',  # Empty for now
            embedding_dimension=0,
            processing_status="processing"
        )
        db.add(pending_profile)
        db.commit()
        
        # Process video in background
        background_tasks.add_task(
            process_user_video,
            db,
            current_user.id,
            video_path
        )
        
        return VideoUploadResponse(
            message="Video uploaded successfully. Processing in background.",
            status="processing"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading video: {str(e)}"
        )


def process_user_video(db: Session, user_id: int, video_path: str):
    """
    Background task to process user's gait video
    """
    temp_video_path = None
    try:
        # Download video from Azure to temp location
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_user_{user_id}_{uuid.uuid4().hex}.mp4")
        video_storage.download_to_local(video_path, temp_video_path)
        
        # Detect person in video
        detections = gait_processor.detect_persons(temp_video_path)
        
        if len(detections) == 0:
            print(f"No person detected in video for user {user_id}")
            # Update status to failed
            profile = db.query(GaitProfile).filter(GaitProfile.user_id == user_id).first()
            if profile:
                profile.processing_status = "failed"
                db.commit()
            return
        
        # Use first detected person (assuming single person in enrollment video)
        person_track = detections[0]
        
        # Extract gait features
        embedding = gait_processor.extract_gait_features(temp_video_path, person_track)
        
        if embedding is None:
            print(f"Failed to extract gait features for user {user_id}")
            # Update status to failed
            profile = db.query(GaitProfile).filter(GaitProfile.user_id == user_id).first()
            if profile:
                profile.processing_status = "failed"
                db.commit()
            return
        
        # Update existing profile with embedding
        profile = db.query(GaitProfile).filter(GaitProfile.user_id == user_id).first()
        if profile:
            profile.embedding = serialize_embedding(embedding)
            profile.embedding_dimension = len(embedding)
            profile.processing_status = "completed"
            db.commit()
            print(f"Gait profile completed for user {user_id}")
        
    except Exception as e:
        print(f"Error processing user video: {e}")
        db.rollback()
        # Update status to failed
        try:
            profile = db.query(GaitProfile).filter(GaitProfile.user_id == user_id).first()
            if profile:
                profile.processing_status = "failed"
                db.commit()
        except:
            pass
    finally:
        # Clean up temp file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass


@user_router.get("/user-profile", response_model=GaitProfileResponse)
async def get_user_gait_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's gait profile
    """
    profile = db.query(GaitProfile).filter(
        GaitProfile.user_id == current_user.id
    ).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No gait profile found for user"
        )
    
    return profile


@user_router.get("/profile-status")
async def get_profile_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get processing status of current user's gait profile (for polling)
    """
    profile = db.query(GaitProfile).filter(
        GaitProfile.user_id == current_user.id
    ).first()
    
    if not profile:
        return {"status": "not_found", "message": "No gait profile found"}
    
    return {
        "status": profile.processing_status,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at
    }


@user_router.delete("/user-profile")
async def delete_user_gait_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete current user's gait profile
    """
    profile = db.query(GaitProfile).filter(
        GaitProfile.user_id == current_user.id
    ).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No gait profile found for user"
        )
    
    # Delete video file
    video_storage.delete_file(profile.video_path)
    
    # Delete from database
    db.delete(profile)
    db.commit()
    
    return {"message": "Gait profile deleted successfully"}


@admin_router.post("/upload-cctv-video", response_model=VideoUploadResponse)
async def upload_cctv_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Upload CCTV video for gait recognition
    Admin only endpoint
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    try:
        # Save video file
        video_path = await video_storage.save_cctv_video(file, current_admin.id)
        
        # Create recognition log
        recognition_log = GaitRecognitionLog(
            admin_id=current_admin.id,
            original_video_path=video_path,
            processing_status="pending"
        )
        
        db.add(recognition_log)
        db.commit()
        db.refresh(recognition_log)
        
        # Process video in background
        background_tasks.add_task(
            process_cctv_video,
            db,
            recognition_log.id,
            video_path
        )
        
        return VideoUploadResponse(
            message="CCTV video uploaded successfully. Processing in background.",
            log_id=recognition_log.id,
            status="processing"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading video: {str(e)}"
        )


def process_cctv_video(db: Session, log_id: int, video_path: str):
    """
    Background task to process CCTV video and perform gait recognition
    """
    temp_video_path = None
    temp_output_path = None
    try:
        # Update status
        log = db.query(GaitRecognitionLog).filter(GaitRecognitionLog.id == log_id).first()
        log.processing_status = "processing"
        db.commit()
        
        # Download video from Azure to temp location
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_cctv_{log_id}_{uuid.uuid4().hex}.mp4")
        video_storage.download_to_local(video_path, temp_video_path)
        
        # Detect persons in video
        person_tracks = gait_processor.detect_persons(temp_video_path)
        log.total_persons_detected = len(person_tracks)
        db.commit()
        
        if len(person_tracks) == 0:
            log.processing_status = "completed"
            log.completed_at = db.execute(text("SELECT NOW()")).scalar()
            db.commit()
            return
        
        # Get all gait profiles from database (only completed ones)
        gait_profiles = db.query(GaitProfile).filter(
            GaitProfile.processing_status == "completed"
        ).all()
        print(f"Found {len(gait_profiles)} enrolled users in database")
        
        database_embeddings = [
            (profile.user_id, deserialize_embedding(profile.embedding))
            for profile in gait_profiles
        ]
        
        if len(database_embeddings) == 0:
            print("⚠️  No users enrolled yet. All detections will be marked as unknown.")
        
        # Process each detected person
        recognition_results = []
        total_recognized = 0
        
        for idx, person_track in enumerate(person_tracks):
            print(f"Processing person {idx + 1}/{len(person_tracks)}...")
            
            # Extract gait features
            embedding = gait_processor.extract_gait_features(temp_video_path, person_track)
            
            if embedding is None:
                print(f"  ✗ Failed to extract gait features for person {idx}")
                # Create detection entry with no match
                detection = GaitDetection(
                    log_id=log_id,
                    person_index=idx,
                    matched_user_id=None,
                    confidence_score=None,
                    bbox_data=json.dumps(person_track['boxes']),
                    is_recognized=False
                )
                db.add(detection)
                recognition_results.append({
                    'is_recognized': False,
                    'username': None,
                    'confidence': 0
                })
                continue
            
            print(f"  ✓ Extracted gait embedding (dim: {len(embedding)})")
            
            # Match against database using gait recognition
            matched_user_id, confidence = gait_processor.match_gait_embedding(
                embedding, database_embeddings
            )
            
            is_recognized = matched_user_id is not None
            if is_recognized:
                total_recognized += 1
                user = db.query(User).filter(User.id == matched_user_id).first()
                username = user.username if user else "Unknown"
                full_name = user.full_name if user else "Unknown"
                print(f"  ✓ RECOGNIZED: {username} ({full_name}) (confidence: {confidence:.2%})")
            else:
                username = None
                full_name = None
                print(f"  ✗ Unknown person (best score: {confidence:.2%}, below threshold)")
            
            # Create detection entry
            detection = GaitDetection(
                log_id=log_id,
                person_index=idx,
                matched_user_id=matched_user_id,
                confidence_score=confidence,
                bbox_data=json.dumps(person_track['boxes']),
                is_recognized=is_recognized
            )
            db.add(detection)
            
            recognition_results.append({
                'is_recognized': is_recognized,
                'username': username,
                'full_name': full_name,
                'confidence': confidence
            })
        
        db.commit()
        
        # Generate annotated video to temp location
        temp_output_path = os.path.join(tempfile.gettempdir(), f"processed_{log_id}_{uuid.uuid4().hex}.mp4")
        
        gait_processor.create_annotated_video(
            temp_video_path,
            temp_output_path,
            person_tracks,
            recognition_results
        )
        
        # Upload processed video to Azure Blob Storage
        processed_video_path = video_storage.upload_processed_video(temp_output_path, log_id)
        
        # Update log
        log.processed_video_path = processed_video_path
        log.total_recognized = total_recognized
        log.processing_status = "completed"
        log.completed_at = db.execute(text("SELECT NOW()")).scalar()
        db.commit()
        
        print(f"\n{'='*60}")
        print(f"CCTV Recognition Summary - Log ID: {log_id}")
        print(f"{'='*60}")
        print(f"Total persons detected: {len(person_tracks)}")
        print(f"Recognized users: {total_recognized}")
        print(f"Unknown persons: {len(person_tracks) - total_recognized}")
        print(f"Recognition rate: {(total_recognized/len(person_tracks)*100) if len(person_tracks) > 0 else 0:.1f}%")
        print(f"Processed video: {processed_video_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error processing CCTV video: {e}")
        db.rollback()  # Explicitly rollback the failed transaction
        try:
            log = db.query(GaitRecognitionLog).filter(GaitRecognitionLog.id == log_id).first()
            if log:
                log.processing_status = "failed"
                db.commit()
        except Exception as inner_e:
            print(f"Error updating log status: {inner_e}")
            db.rollback()
    finally:
        # Clean up temp files
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except:
                pass


@admin_router.get("/recognition-logs", response_model=List[GaitRecognitionLogList])
async def get_recognition_logs(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get all recognition logs for current admin
    """
    logs = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.admin_id == current_admin.id
    ).order_by(GaitRecognitionLog.created_at.desc()).all()
    
    return logs


@admin_router.get("/recognition-logs/{log_id}", response_model=GaitRecognitionResponse)
async def get_recognition_log(
    log_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get detailed recognition log with detections
    """
    log = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.id == log_id,
        GaitRecognitionLog.admin_id == current_admin.id
    ).first()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recognition log not found"
        )
    
    # Build response with detection details
    detections_response = []
    for detection in log.detections:
        det_response = GaitDetectionResponse(
            id=detection.id,
            person_index=detection.person_index,
            matched_user_id=detection.matched_user_id,
            matched_username=detection.matched_user.username if detection.matched_user else None,
            matched_full_name=detection.matched_user.full_name if detection.matched_user else None,
            confidence_score=detection.confidence_score,
            is_recognized=detection.is_recognized
        )
        detections_response.append(det_response)
    
    return GaitRecognitionResponse(
        id=log.id,
        original_video_path=log.original_video_path,
        processed_video_path=log.processed_video_path,
        total_persons_detected=log.total_persons_detected,
        total_recognized=log.total_recognized,
        processing_status=log.processing_status,
        created_at=log.created_at,
        completed_at=log.completed_at,
        detections=detections_response
    )


@admin_router.get("/recognition-status/{log_id}")
async def get_recognition_status(
    log_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get processing status of a recognition log (for polling)
    """
    log = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.id == log_id,
        GaitRecognitionLog.admin_id == current_admin.id
    ).first()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recognition log not found"
        )
    
    return {
        "log_id": log.id,
        "status": log.processing_status,
        "total_persons_detected": log.total_persons_detected,
        "total_recognized": log.total_recognized,
        "created_at": log.created_at,
        "completed_at": log.completed_at
    }


@admin_router.get("/download-processed-video/{log_id}")
async def download_processed_video(
    log_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get download URL for processed video with annotations
    Returns a temporary SAS URL valid for 1 hour
    """
    log = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.id == log_id,
        GaitRecognitionLog.admin_id == current_admin.id
    ).first()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recognition log not found or you don't have permission to access it"
        )
    
    if log.processing_status == "pending" or log.processing_status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video is still being processed. Current status: {log.processing_status}"
        )
    
    if log.processing_status == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video processing failed. Please try uploading again or use the reprocess endpoint."
        )
    
    if not log.processed_video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video not available. Processing may have failed."
        )
    
    # Generate SAS URL for download
    download_url = video_storage.generate_download_url(log.processed_video_path, expiry_hours=1)
    
    if not download_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video file not found in storage. The file may have been deleted."
        )
    
    return {
        "download_url": download_url,
        "expires_in_hours": 1,
        "message": "Use this URL to download the processed video. The URL will expire in 1 hour."
    }


@admin_router.get("/stats")
async def get_gait_stats(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get gait recognition statistics
    """
    # Total users with gait profiles
    total_profiles = db.query(GaitProfile).count()
    
    # Total recognition logs
    total_logs = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.admin_id == current_admin.id
    ).count()
    
    # Total persons detected
    total_detected = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.admin_id == current_admin.id
    ).with_entities(
        db.query(GaitRecognitionLog.total_persons_detected).label('total')
    ).all()
    
    total_persons = sum([log[0] or 0 for log in total_detected])
    
    # Total recognized
    total_recognized_query = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.admin_id == current_admin.id
    ).with_entities(
        db.query(GaitRecognitionLog.total_recognized).label('total')
    ).all()
    
    total_recognized = sum([log[0] or 0 for log in total_recognized_query])
    
    return {
        "total_gait_profiles": total_profiles,
        "total_recognition_logs": total_logs,
        "total_persons_detected": total_persons,
        "total_recognized": total_recognized,
        "recognition_rate": (total_recognized / total_persons * 100) if total_persons > 0 else 0
    }


@admin_router.post("/reprocess-video/{log_id}")
async def reprocess_failed_video(
    log_id: int,
    background_tasks: BackgroundTasks,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Reprocess a failed CCTV video
    """
    log = db.query(GaitRecognitionLog).filter(
        GaitRecognitionLog.id == log_id,
        GaitRecognitionLog.admin_id == current_admin.id
    ).first()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recognition log not found"
        )
    
    if log.processing_status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video is already being processed"
        )
    
    if log.processing_status == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video has already been processed successfully"
        )
    
    # Reset status and reprocess
    log.processing_status = "pending"
    log.completed_at = None
    db.commit()
    
    # Clear old detections if any
    db.query(GaitDetection).filter(GaitDetection.log_id == log_id).delete()
    db.commit()
    
    # Reprocess in background
    background_tasks.add_task(
        process_cctv_video,
        db,
        log_id,
        log.original_video_path
    )
    
    return {
        "message": "Video queued for reprocessing",
        "log_id": log_id,
        "status": "processing"
    }
