from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models import Admin, User
from app.schemas import LoginRequest, LoginResponse, AdminCreate, UserCreate, AdminResponse, UserResponse
from app.utils import verify_password, get_password_hash, create_access_token

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/admin/register", response_model=AdminResponse, status_code=status.HTTP_201_CREATED)
async def register_admin(admin_data: AdminCreate, db: Session = Depends(get_db)):
    """
    Register a new admin.
    """
    # Check if email already exists
    if db.query(Admin).filter(Admin.email == admin_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    if db.query(Admin).filter(Admin.username == admin_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new admin
    hashed_password = get_password_hash(admin_data.password)
    new_admin = Admin(
        email=admin_data.email,
        username=admin_data.username,
        hashed_password=hashed_password,
        full_name=admin_data.full_name,
        is_super_admin=admin_data.is_super_admin
    )
    
    db.add(new_admin)
    db.commit()
    db.refresh(new_admin)
    
    return new_admin


@router.post("/user/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user/employee under an admin.
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
    
    # Verify admin exists
    if user_data.admin_id:
        admin = db.query(Admin).filter(Admin.id == user_data.admin_id).first()
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="admin_id is required"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        admin_id=user_data.admin_id
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    Login for both admin and user.
    Returns JWT token and user information.
    """
    # Try to find user in Admin table
    admin = db.query(Admin).filter(Admin.email == login_data.email).first()
    
    if admin and verify_password(login_data.password, admin.hashed_password):
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        access_token = create_access_token(
            data={"sub": admin.email, "user_type": "admin"}
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user_type="admin",
            user_data={
                "id": admin.id,
                "email": admin.email,
                "username": admin.username,
                "full_name": admin.full_name,
                "is_super_admin": admin.is_super_admin
            }
        )
    
    # Try to find user in User table
    user = db.query(User).filter(User.email == login_data.email).first()
    
    if user and verify_password(login_data.password, user.hashed_password):
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        access_token = create_access_token(
            data={"sub": user.email, "user_type": "user"}
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user_type="user",
            user_data={
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "admin_id": user.admin_id
            }
        )
    
    # If neither found or password incorrect
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password"
    )


@router.post("/token", response_model=dict)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login endpoint.
    Use this with OAuth2PasswordBearer for swagger UI authentication.
    """
    # Try admin first
    admin = db.query(Admin).filter(Admin.email == form_data.username).first()
    if admin and verify_password(form_data.password, admin.hashed_password):
        access_token = create_access_token(
            data={"sub": admin.email, "user_type": "admin"}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    
    # Try user
    user = db.query(User).filter(User.email == form_data.username).first()
    if user and verify_password(form_data.password, user.hashed_password):
        access_token = create_access_token(
            data={"sub": user.email, "user_type": "user"}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
