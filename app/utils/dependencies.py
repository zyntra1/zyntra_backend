from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError
from app.core.database import get_db
from app.models import Admin, User
from app.schemas import TokenData
from app.utils.auth import decode_access_token

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Dependency to get the current authenticated user (either Admin or User).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Extract token string from Authorization header
    if credentials is None or not credentials.credentials:
        raise credentials_exception

    token = credentials.credentials

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    user_type: str = payload.get("user_type")
    
    if email is None or user_type is None:
        raise credentials_exception
    
    token_data = TokenData(email=email, user_type=user_type)
    
    # Check if user is an admin or regular user
    if user_type == "admin":
        user = db.query(Admin).filter(Admin.email == token_data.email).first()
    else:
        user = db.query(User).filter(User.email == token_data.email).first()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_admin(
    current_user = Depends(get_current_user)
):
    """
    Dependency to ensure the current user is an admin.
    """
    if not isinstance(current_user, Admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_super_admin(
    current_admin: Admin = Depends(get_current_admin)
):
    """
    Dependency to ensure the current user is a super admin.
    """
    if not current_admin.is_super_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required"
        )
    return current_admin


async def get_current_active_user(
    current_user = Depends(get_current_user)
):
    """
    Dependency to get current active user (Admin or User).
    """
    return current_user
