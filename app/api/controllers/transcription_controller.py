from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import app.api.controllers.service_registry as service_registry
from app.core.config import settings
import shutil
from pathlib import Path
import uuid

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

@router.post("/", response_model=dict)
async def transcribe_media(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio or video file.
    """
    if not service_registry.transcription_service:
        raise HTTPException(status_code=503, detail="Transcription service not initialized")

    allowed_extensions = {".mp3", ".wav", ".mp4", ".m4a", ".mpeg", ".mpga", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")

    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_file_path = settings.UPLOAD_DIR / temp_filename
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = service_registry.transcription_service.transcribe(temp_file_path)
        
        temp_file_path.unlink()
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"DEBUG: Locals: {locals().keys()}")
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
