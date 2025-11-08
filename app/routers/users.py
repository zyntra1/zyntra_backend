from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models import Admin, User
from app.schemas import UserResponse, UserUpdate, UserCreate
from app.utils import get_current_admin, get_current_user, get_password_hash

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Get current logged-in user's information.
    """
    if isinstance(current_user, User):
        return current_user
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only users can access this endpoint"
        )


@router.get("/", response_model=List[UserResponse])
async def list_users_under_admin(
    skip: int = 0,
    limit: int = 100,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    List all users/employees under the current admin.
    """
    users = db.query(User).filter(
        User.admin_id == current_admin.id
    ).offset(skip).limit(limit).all()
    
    return users


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_by_admin(
    user_data: UserCreate,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Create a new user/employee under the current admin.
    """
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user under current admin
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        admin_id=current_admin.id  # Set to current admin
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get specific user by ID (must be under current admin).
    """
    user = db.query(User).filter(
        User.id == user_id,
        User.admin_id == current_admin.id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or not under your management"
        )
    
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Update a user's information (must be under current admin).
    """
    user = db.query(User).filter(
        User.id == user_id,
        User.admin_id == current_admin.id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or not under your management"
        )
    
    update_data = user_update.dict(exclude_unset=True)
    
    # Check if email is being updated and if it's already taken
    if "email" in update_data and update_data["email"] != user.email:
        existing_user = db.query(User).filter(User.email == update_data["email"]).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being updated and if it's already taken
    if "username" in update_data and update_data["username"] != user.username:
        existing_user = db.query(User).filter(User.username == update_data["username"]).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Hash password if being updated
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    # Update user
    for key, value in update_data.items():
        setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    
    return user


@router.put("/me/update", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user's own information.
    """
    if not isinstance(current_user, User):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only users can access this endpoint"
        )
    
    update_data = user_update.dict(exclude_unset=True)
    
    # Check if email is being updated and if it's already taken
    if "email" in update_data and update_data["email"] != current_user.email:
        existing_user = db.query(User).filter(User.email == update_data["email"]).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being updated and if it's already taken
    if "username" in update_data and update_data["username"] != current_user.username:
        existing_user = db.query(User).filter(User.username == update_data["username"]).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Hash password if being updated
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    # Update user
    for key, value in update_data.items():
        setattr(current_user, key, value)
    
    db.commit()
    db.refresh(current_user)
    
    return current_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Delete a user (must be under current admin).
    """
    user = db.query(User).filter(
        User.id == user_id,
        User.admin_id == current_admin.id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or not under your management"
        )
    
    db.delete(user)
    db.commit()
    
    return None
