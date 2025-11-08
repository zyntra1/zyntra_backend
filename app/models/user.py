from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Admin(Base):
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_super_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship with users/employees
    employees = relationship("User", back_populates="admin", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Admin(id={self.id}, username={self.username}, email={self.email})>"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    admin_id = Column(Integer, ForeignKey("admins.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship with admin
    admin = relationship("Admin", back_populates="employees")
    
    # Posture and wellness relationships
    posture_snapshots = relationship("PostureSnapshot", back_populates="user", cascade="all, delete-orphan")
    wellness_metrics = relationship("WellnessMetrics", back_populates="user", cascade="all, delete-orphan")
    attrition_risks = relationship("AttritionRisk", back_populates="user", cascade="all, delete-orphan")
    posture_baseline = relationship("PostureBaseline", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email}, admin_id={self.admin_id})>"
