from fastapi import APIRouter, HTTPException, status
from app.models.schemas import GoogleLoginRequest, AuthResponse, User
from app.core.security import verify_google_token, create_access_token
from app.api.controllers import router

auth_router = APIRouter()

@auth_router.post("/auth/google", response_model=AuthResponse)
async def google_auth(request: GoogleLoginRequest):
    idinfo = verify_google_token(request.token)
    if not idinfo:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token",
        )
        
    user_id = idinfo.get("sub")
    email = idinfo.get("email")
    name = idinfo.get("name")
    picture = idinfo.get("picture")
    
    if not user_id or not email:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incomplete user information from Google",
        )

    user = User(
        id=user_id,
        email=email,
        name=name,
        picture=picture
    )
    
    token_data = {
        "sub": user.id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture
    }
    
    access_token = create_access_token(data=token_data)
    
    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=user
    )

router.include_router(auth_router, tags=["Authentication"])
