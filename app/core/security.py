import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests
from app.core.config import settings
from app.models.schemas import User
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

def verify_google_token(token: str) -> Optional[dict]:
    try:
        # Avoid verifying client ID in dev if we allow a placeholder, but normally we should check it
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), 
            settings.GOOGLE_CLIENT_ID if settings.GOOGLE_CLIENT_ID != "placeholder_google_client_id.apps.googleusercontent.com" else None
        )
        return idinfo
    except ValueError as e:
        logger.error(f"Token verification failed: {e}")
        return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"JWT decode error: {e}")
        return None

async def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[User]:
    if not credentials:
        return None
        
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if not payload:
        return None
        
    try:
        return User(
            id=payload.get("sub"),
            email=payload.get("email"),
            name=payload.get("name"),
            picture=payload.get("picture")
        )
    except Exception:
        return None

async def get_current_user(user: Optional[User] = Depends(get_current_user_optional)) -> User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
