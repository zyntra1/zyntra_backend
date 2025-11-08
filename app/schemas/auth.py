from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
    user_type: Optional[str] = None  # "admin" or "user"


# Base User/Admin Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = None


class AdminBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = None


# Create Schemas (with password)
class UserCreate(UserBase):
    password: str = Field(..., min_length=6)
    admin_id: Optional[int] = None  # Optional, can be set by admin


class AdminCreate(AdminBase):
    password: str = Field(..., min_length=6)
    is_super_admin: bool = False


# Update Schemas
class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6)
    is_active: Optional[bool] = None


class AdminUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6)
    is_active: Optional[bool] = None


# Response Schemas
class UserResponse(UserBase):
    id: int
    is_active: bool
    admin_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AdminResponse(AdminBase):
    id: int
    is_active: bool
    is_super_admin: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AdminWithEmployees(AdminResponse):
    employees: list[UserResponse] = []
    
    class Config:
        from_attributes = True


# Login Schemas
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_type: str  # "admin" or "user"
    user_data: dict
