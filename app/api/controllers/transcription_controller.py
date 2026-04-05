from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.core.security import get_current_user
from app.models.schemas import User, TranscriptionResponse
import app.api.controllers.service_registry as service_registry
from app.core.config import settings

import shutil
from pathlib import Path
import uuid

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

@router.post("/", response_model=TranscriptionResponse)
async def transcribe_media(
    file: UploadFile = File(...),
    summarization_type: str = Form("extractive"),
    num_sentences: int = Form(3),
    max_length: int = Form(150),
    min_length: int = Form(40),
    current_user: User = Depends(get_current_user)
):
    """
    Transcribe an uploaded audio or video file.
    """
    if not service_registry.transcription_service:
        raise HTTPException(status_code=503, detail="Transcription service not initialized")

    summarization_type = summarization_type.lower().strip()
    if summarization_type not in {"extractive", "abstractive"}:
        raise HTTPException(status_code=400, detail="summarization_type must be 'extractive' or 'abstractive'")

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
        transcript_text = result.get("text", "")

        if not transcript_text:
            raise HTTPException(status_code=500, detail="Transcription completed but no text was generated")

        if summarization_type == "extractive":
            if service_registry.extractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Extractive summarizer not available")
            summary = service_registry.extractive_summarizer.summarize(
                transcript_text,
                num_sentences=max(1, num_sentences),
            )
        else:
            if service_registry.abstractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Abstractive summarizer not available")
            safe_min = max(10, min_length)
            safe_max = max(safe_min + 10, max_length)
            summary = service_registry.abstractive_summarizer.summarize(
                transcript_text,
                max_length=safe_max,
                min_length=safe_min,
            )
        
        temp_file_path.unlink()

        return {
            "text": transcript_text,
            "summary": summary,
            "summarization_type": summarization_type,
            "language": result.get("language", "unknown"),
            "metadata": {
                "segments_count": len(result.get("segments", [])),
                "filename": file.filename,
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
