from pathlib import Path
from typing import Optional, Dict
from fastapi import HTTPException

from app.core.config import settings
from app.services.document_service import DocumentManager
from app.services.extractive_service import ExtractiveSummarizer
from app.services.abstractive_service import AbstractiveSummarizer
from app.services.tts_service import AudiobookGenerator
from app.services.vector_store_service import HybridVectorStore
from app.services.session_service import SessionManager
from app.services.chat_service import RAGChatService
from app.services.transcription_service import TranscriptionService

document_manager: Optional[DocumentManager] = None
extractive_summarizer: Optional[ExtractiveSummarizer] = None
abstractive_summarizer: Optional[AbstractiveSummarizer] = None
audiobook_generator: Optional[AudiobookGenerator] = None
vector_stores: Dict[str, HybridVectorStore] = {}
session_manager: Optional[SessionManager] = None
chat_service: Optional[RAGChatService] = None
transcription_service: Optional[TranscriptionService] = None


def init_services():
    global document_manager, extractive_summarizer, abstractive_summarizer
    global audiobook_generator, session_manager, chat_service, transcription_service

    document_manager = DocumentManager(settings.DATA_DIR)

    extractive_model_path = Path(settings.BASE_DIR) / settings.EXTRACTIVE_MODEL_PATH
    if extractive_model_path.exists():
        extractive_summarizer = ExtractiveSummarizer(
            str(extractive_model_path),
            encoder_name=settings.SENTENCE_ENCODER
        )

    abstractive_model_path = Path(settings.BASE_DIR) / settings.ABSTRACTIVE_MODEL_PATH
    if abstractive_model_path.exists():
        abstractive_summarizer = AbstractiveSummarizer(str(abstractive_model_path))

    audiobook_generator = AudiobookGenerator(settings.TTS_MODEL)
    transcription_service = TranscriptionService()

    sessions_dir = settings.DATA_DIR / "sessions"
    session_manager = SessionManager(sessions_dir)
    chat_service = RAGChatService()


def get_or_create_vector_store(document_id: str) -> HybridVectorStore:
    if document_id not in vector_stores:
        vector_store = HybridVectorStore()
        document = document_manager.get_document(document_id)
        if document:
            vector_store.create_index(document['chunks'])
            vector_stores[document_id] = vector_store
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    return vector_stores[document_id]
