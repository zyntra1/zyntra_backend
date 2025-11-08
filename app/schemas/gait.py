from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class GaitProfileCreate(BaseModel):
    """Schema for creating a gait profile (user video upload)"""
    pass


class GaitProfileResponse(BaseModel):
    """Response schema for gait profile"""
    id: int
    user_id: int
    video_path: str
    embedding_dimension: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class GaitDetectionResponse(BaseModel):
    """Response schema for individual detection in CCTV video"""
    id: int
    person_index: int
    matched_user_id: Optional[int] = None
    matched_username: Optional[str] = None
    matched_full_name: Optional[str] = None
    confidence_score: Optional[float] = None
    is_recognized: bool
    
    class Config:
        from_attributes = True


class GaitRecognitionResponse(BaseModel):
    """Response schema for CCTV recognition results"""
    id: int
    original_video_path: str
    processed_video_path: Optional[str] = None
    total_persons_detected: int
    total_recognized: int
    processing_status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    detections: List[GaitDetectionResponse] = []
    
    class Config:
        from_attributes = True


class GaitRecognitionLogList(BaseModel):
    """Schema for listing recognition logs"""
    id: int
    total_persons_detected: int
    total_recognized: int
    processing_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class VideoUploadResponse(BaseModel):
    """Response after video upload"""
    message: str
    profile_id: Optional[int] = None
    log_id: Optional[int] = None
    status: str
