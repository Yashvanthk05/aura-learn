from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import uuid

from app.models.schemas import AudiobookRequest, AudiobookResponse
from app.core.config import settings
from . import service_registry as svc

router = APIRouter()


@router.post("/generate-audiobook", response_model=AudiobookResponse)
async def generate_audiobook(request: AudiobookRequest):
    if svc.audiobook_generator is None:
        raise HTTPException(status_code=503, detail="Audiobook generator not available")

    try:
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.wav"
        audio_path = settings.OUTPUT_DIR / audio_filename

        svc.audiobook_generator.generate(
            text=request.text, output_path=str(audio_path), language=request.language
        )

        if not audio_path.exists():
            raise HTTPException(status_code=500, detail="Audiobook generation failed: output file missing")

        return AudiobookResponse(
            audio_url=f"/audio/{audio_filename}",
            filename=audio_filename,
            text_length=len(request.text),
            language=request.language,
            message="Audiobook generated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audiobook generation failed: {str(e)}")


@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    audio_path = settings.OUTPUT_DIR / filename

    if audio_path.suffix == "":
        wav_candidate = audio_path.with_suffix(".wav")
        if wav_candidate.exists():
            audio_path = wav_candidate

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav", filename=audio_path.name)
