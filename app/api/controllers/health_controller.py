from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.core.config import settings
from . import service_registry as svc

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded={
            "extractive_summarizer": svc.extractive_summarizer is not None,
            "abstractive_summarizer": svc.abstractive_summarizer is not None,
            "audiobook_generator": svc.audiobook_generator is not None,
            "session_manager": svc.session_manager is not None,
            "chat_service": svc.chat_service is not None,
            "vector_stores": len(svc.vector_stores)
        }
    )
