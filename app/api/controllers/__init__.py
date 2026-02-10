from fastapi import APIRouter
from .upload_controller import router as upload_router
from .document_controller import router as document_router
from .summarize_controller import router as summarize_router
from .audiobook_controller import router as audiobook_router
from .chat_controller import router as chat_router
from .search_controller import router as search_router
from .health_controller import router as health_router

router = APIRouter()
router.include_router(upload_router)
router.include_router(document_router)
router.include_router(summarize_router)
router.include_router(audiobook_router)
router.include_router(chat_router)
router.include_router(search_router)
router.include_router(health_router)
