import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import base64
from app.core.config import settings


def _prepare_password(password: str) -> bytes:
    """
    Prepare password for hashing by converting to SHA256 first.
    This allows passwords longer than 72 bytes to work with bcrypt.
    Returns a base64 encoded SHA256 hash that's always within bcrypt's 72-byte limit.
    """
    # Hash the password with SHA256 first
    sha256_hash = hashlib.sha256(password.encode('utf-8')).digest()
    # Encode to base64 to get a string representation (44 chars, well under 72 bytes)
    return base64.b64encode(sha256_hash)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    Uses SHA256 + bcrypt for compatibility with long passwords.
    """
    prepared_password = _prepare_password(plain_password)
    return bcrypt.checkpw(prepared_password, hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """
    Hash a password for storing.
    Uses SHA256 pre-hashing + bcrypt to support passwords of any length.
    """
    prepared_password = _prepare_password(password)
    # Generate salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(prepared_password, salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing the data to encode in the token
        expires_delta: Optional expiration time delta
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: The JWT token to decode
    
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None
