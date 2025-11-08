from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models import Admin, User
from app.schemas import AdminResponse, AdminWithEmployees, AdminUpdate
from app.utils import get_current_admin, get_current_super_admin, get_password_hash

router = APIRouter(prefix="/api/admins", tags=["Admins"])


@router.get("/me", response_model=AdminResponse)
async def get_current_admin_info(current_admin: Admin = Depends(get_current_admin)):
    """
    Get current logged-in admin's information.
    """
    return current_admin


@router.get("/me/employees", response_model=AdminWithEmployees)
async def get_admin_employees(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get all employees/users under the current admin.
    """
    admin = db.query(Admin).filter(Admin.id == current_admin.id).first()
    return admin


@router.get("/", response_model=List[AdminResponse])
async def list_all_admins(
    skip: int = 0,
    limit: int = 100,
    current_admin: Admin = Depends(get_current_super_admin),
    db: Session = Depends(get_db)
):
    """
    List all admins (Super admin only).
    """
    admins = db.query(Admin).offset(skip).limit(limit).all()
    return admins


@router.get("/{admin_id}", response_model=AdminWithEmployees)
async def get_admin_by_id(
    admin_id: int,
    current_admin: Admin = Depends(get_current_super_admin),
    db: Session = Depends(get_db)
):
    """
    Get specific admin by ID with their employees (Super admin only).
    """
    admin = db.query(Admin).filter(Admin.id == admin_id).first()
    
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found"
        )
    
    return admin


@router.put("/me", response_model=AdminResponse)
async def update_current_admin(
    admin_update: AdminUpdate,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Update current admin's information.
    """
    update_data = admin_update.dict(exclude_unset=True)
    
    # Check if email is being updated and if it's already taken
    if "email" in update_data and update_data["email"] != current_admin.email:
        existing_admin = db.query(Admin).filter(Admin.email == update_data["email"]).first()
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being updated and if it's already taken
    if "username" in update_data and update_data["username"] != current_admin.username:
        existing_admin = db.query(Admin).filter(Admin.username == update_data["username"]).first()
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Hash password if being updated
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    # Update admin
    for key, value in update_data.items():
        setattr(current_admin, key, value)
    
    db.commit()
    db.refresh(current_admin)
    
    return current_admin


@router.delete("/{admin_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_admin(
    admin_id: int,
    current_admin: Admin = Depends(get_current_super_admin),
    db: Session = Depends(get_db)
):
    """
    Delete an admin (Super admin only).
    This will also delete all employees under this admin due to cascade.
    """
    admin = db.query(Admin).filter(Admin.id == admin_id).first()
    
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found"
        )
    
    # Prevent deleting yourself
    if admin.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    db.delete(admin)
    db.commit()
    
    return None
