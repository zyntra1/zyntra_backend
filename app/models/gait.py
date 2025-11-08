from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary, Float, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class GaitProfile(Base):
    """
    Stores gait embeddings for each user.
    Each user can have one gait profile extracted from their uploaded video.
    """
    __tablename__ = "gait_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    video_path = Column(String(500), nullable=False)  # Path to original video
    embedding = Column(LargeBinary, nullable=False)  # Serialized numpy array of gait embedding
    embedding_dimension = Column(Integer, nullable=False)  # Dimension of embedding vector
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", backref="gait_profile")
    
    def __repr__(self):
        return f"<GaitProfile(id={self.id}, user_id={self.user_id})>"


class GaitRecognitionLog(Base):
    """
    Stores logs of gait recognition attempts from CCTV/admin uploads.
    Contains the processed video with bounding boxes and recognition results.
    """
    __tablename__ = "gait_recognition_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("admins.id"), nullable=False)
    original_video_path = Column(String(500), nullable=False)  # Original CCTV video
    processed_video_path = Column(String(500), nullable=True)  # Video with bounding boxes
    total_persons_detected = Column(Integer, default=0)
    total_recognized = Column(Integer, default=0)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    admin = relationship("Admin", backref="gait_logs")
    detections = relationship("GaitDetection", back_populates="log", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<GaitRecognitionLog(id={self.id}, admin_id={self.admin_id}, status={self.processing_status})>"


class GaitDetection(Base):
    """
    Stores individual person detections from a recognition log.
    Links detected persons to recognized users (if matched).
    """
    __tablename__ = "gait_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(Integer, ForeignKey("gait_recognition_logs.id"), nullable=False)
    person_index = Column(Integer, nullable=False)  # Index of person in video (0, 1, 2...)
    matched_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null if no match
    confidence_score = Column(Float, nullable=True)  # Similarity score (0-1)
    bbox_data = Column(Text, nullable=True)  # JSON string of bounding boxes per frame (can be large)
    is_recognized = Column(Boolean, default=False)
    
    # Relationships
    log = relationship("GaitRecognitionLog", back_populates="detections")
    matched_user = relationship("User", backref="gait_detections")
    
    def __repr__(self):
        return f"<GaitDetection(id={self.id}, log_id={self.log_id}, matched_user_id={self.matched_user_id}, confidence={self.confidence_score})>"
