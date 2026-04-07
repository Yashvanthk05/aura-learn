import mimetypes
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import get_current_user
from app.models.schemas import User
from app.utils.pdf_processor import SUPPORTED_EXTENSIONS

from app.models.schemas import (
    CreateWorkspaceRequest,
    CreateSessionRequest, CreateSessionResponse,
    ChatQueryRequest, ChatQueryResponse,
    ConversationHistoryResponse, SessionInfoResponse,
    SessionListResponse, SessionListItem,
    SessionSourcesResponse, UploadSourceResponse, SourceFileInfo,
    Citation, ChatMessage,
)
from . import service_registry as svc

router = APIRouter()

ALLOWED_MEDIA_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".mpeg", ".mpga", ".webm"}
ALLOWED_SOURCE_EXTENSIONS = set(SUPPORTED_EXTENSIONS).union(ALLOWED_MEDIA_EXTENSIONS)
MAX_SOURCES_PER_CHAT = 10


def _sanitize_title(raw_title: Optional[str]) -> str:
    title = (raw_title or "").strip()
    if not title:
        return "New Chat"
    return title[:80]


def _is_media_extension(file_ext: str) -> bool:
    return file_ext.lower() in ALLOWED_MEDIA_EXTENSIONS


def _serialize_source(session_id: str, source: dict) -> SourceFileInfo:
    return SourceFileInfo(
        source_id=source.get("source_id", "unknown"),
        filename=source.get("filename", "unknown"),
        file_type=source.get("file_type", "unknown"),
        size_bytes=int(source.get("size_bytes", 0)),
        status=source.get("status", "ready"),
        added_at=source.get("added_at", datetime.now().isoformat()),
        source_url=f"/api/v1/chat/session/{session_id}/source/{source.get('source_id', 'unknown')}",
    )


def _safe_upload_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    root = settings.UPLOAD_DIR.resolve()
    path = Path(path_str).resolve()
    if not str(path).startswith(str(root)):
        return None
    return path


def _sanitize_metadata(metadata: Optional[dict]) -> dict:
    if not metadata:
        return {}
    clean = dict(metadata)
    sources = clean.get("sources")
    if isinstance(sources, list):
        clean["sources"] = [
            {k: v for k, v in source.items() if k != "path"}
            for source in sources
            if isinstance(source, dict)
        ]
    return clean


@router.post("/chat/workspace", response_model=CreateSessionResponse)
async def create_workspace_chat(request: CreateWorkspaceRequest, current_user: User = Depends(get_current_user)):
    title = _sanitize_title(request.title)

    document_id = svc.document_manager.create_workspace_document(current_user.id)
    session_id = svc.session_manager.create_session(
        current_user.id,
        document_id,
        {
            "title": title,
            "sources": [],
        },
    )
    session = svc.session_manager.get_session(session_id, current_user.id)

    return CreateSessionResponse(
        session_id=session_id,
        document_id=document_id,
        created_at=session["created_at"],
        message="Workspace chat created successfully",
        title=title,
        source_count=0,
    )


@router.post("/chat/session/{session_id}/sources", response_model=UploadSourceResponse)
async def add_source_to_chat(session_id: str, file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    session = svc.session_manager.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    existing_sources = svc.session_manager.get_sources(session_id, current_user.id)
    if len(existing_sources) >= MAX_SOURCES_PER_CHAT:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_SOURCES_PER_CHAT} files are allowed per chat",
        )

    filename = Path(file.filename or "source").name
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_SOURCE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Supported: {sorted(ALLOWED_SOURCE_EXTENSIONS)}",
        )

    source_id = str(uuid.uuid4())
    upload_path = settings.UPLOAD_DIR / f"{session_id}_{source_id}_{filename}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if _is_media_extension(file_ext):
            if not svc.transcription_service:
                raise HTTPException(status_code=503, detail="Transcription service not initialized")

            transcript = svc.transcription_service.transcribe(upload_path)
            transcript_text = (transcript.get("text") or "").strip()
            if not transcript_text:
                raise HTTPException(status_code=500, detail="Transcription completed but no text was generated")

            new_chunks = svc.document_manager.chunk_text_content(
                transcript_text,
                source_name=filename,
                topic="Transcript",
            )
        else:
            new_chunks = svc.document_manager.process_file_to_chunks(str(upload_path))

        if not new_chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in file")

        document_id = session["document_id"]
        total_chunks = svc.document_manager.append_chunks(document_id, current_user.id, new_chunks)
        if total_chunks is None:
            raise HTTPException(status_code=404, detail="Chat knowledge base not found")

        svc.refresh_vector_store(document_id, current_user.id)

        source_record = {
            "source_id": source_id,
            "filename": filename,
            "file_type": file_ext.lstrip(".") or "unknown",
            "size_bytes": upload_path.stat().st_size,
            "status": "ready",
            "added_at": datetime.now().isoformat(),
            "path": str(upload_path),
        }
        if not svc.session_manager.add_source(session_id, current_user.id, source_record):
            raise HTTPException(status_code=500, detail="Failed to update session sources")

        updated_sources = svc.session_manager.get_sources(session_id, current_user.id)
        return UploadSourceResponse(
            session_id=session_id,
            document_id=document_id,
            source=_serialize_source(session_id, source_record),
            total_sources=len(updated_sources),
            total_chunks=total_chunks,
            message="Source uploaded and indexed successfully",
        )
    except HTTPException:
        if upload_path.exists() and not any(
            src.get("path") == str(upload_path)
            for src in svc.session_manager.get_sources(session_id, current_user.id)
        ):
            upload_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest source file: {str(e)}")


@router.get("/chat/session/{session_id}/sources", response_model=SessionSourcesResponse)
async def list_session_sources(session_id: str, current_user: User = Depends(get_current_user)):
    session = svc.session_manager.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    raw_sources = svc.session_manager.get_sources(session_id, current_user.id)
    return SessionSourcesResponse(
        session_id=session_id,
        sources=[_serialize_source(session_id, source) for source in raw_sources],
    )


@router.get("/chat/session/{session_id}/source/{source_id}")
async def preview_session_source(session_id: str, source_id: str, current_user: User = Depends(get_current_user)):
    session = svc.session_manager.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    source = svc.session_manager.get_source(session_id, current_user.id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source file not found")

    source_path = _safe_upload_path(source.get("path", ""))
    if not source_path:
        raise HTTPException(status_code=403, detail="Invalid source path")
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file is no longer available")

    media_type = mimetypes.guess_type(str(source_path))[0] or "application/octet-stream"
    return FileResponse(
        path=source_path,
        media_type=media_type,
        filename=source.get("filename", source_path.name),
    )


@router.post("/chat/session", response_model=CreateSessionResponse)
async def create_chat_session(request: CreateSessionRequest, current_user: User = Depends(get_current_user)):
    document = svc.document_manager.get_document(request.document_id, current_user.id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    svc.get_or_create_vector_store(request.document_id, current_user.id)

    session_id = svc.session_manager.create_session(
        current_user.id, request.document_id, request.metadata
    )
    session = svc.session_manager.get_session(session_id, current_user.id)

    return CreateSessionResponse(
        session_id=session_id,
        document_id=request.document_id,
        created_at=session['created_at'],
        message="Chat session created successfully",
        title=(session.get('metadata') or {}).get('title'),
        source_count=len((session.get('metadata') or {}).get('sources', [])),
    )


@router.post("/chat/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest, current_user: User = Depends(get_current_user)):
    session = svc.session_manager.get_session(request.session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    document_id = session['document_id']

    try:
        vector_store = svc.get_or_create_vector_store(document_id, current_user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")

    try:
        if request.use_hybrid_search:
            retrieved_chunks = vector_store.search(request.query, top_k=request.top_k)
            search_method = "hybrid (FAISS + BM25 + TF-IDF)"
        else:
            retrieved_chunks = vector_store.search_faiss_only(request.query, top_k=request.top_k)
            search_method = "FAISS (semantic only)"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    conversation_history = None
    if request.include_history:
        conversation_history = svc.session_manager.get_context_for_query(
            request.session_id, current_user.id, max_history=5
        )

    try:
        response_text, citations = svc.chat_service.generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            max_context_chunks=request.max_context_chunks,
            generation_type=request.model_type or "abstractive"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

    svc.session_manager.add_message(request.session_id, current_user.id, role='user', content=request.query)
    svc.session_manager.add_message(
        request.session_id, current_user.id, role='assistant', content=response_text, citations=citations
    )

    return ChatQueryResponse(
        session_id=request.session_id,
        query=request.query,
        response=response_text,
        citations=[Citation(**c) for c in citations],
        retrieved_chunks=len(retrieved_chunks),
        search_method=search_method,
        timestamp=datetime.now().isoformat()
    )


@router.get("/chat/session/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, max_messages: Optional[int] = None, current_user: User = Depends(get_current_user)):
    session = svc.session_manager.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = svc.session_manager.get_conversation_history(
        session_id, current_user.id, max_messages=max_messages, include_citations=True
    )

    return ConversationHistoryResponse(
        session_id=session_id,
        document_id=session['document_id'],
        messages=[ChatMessage(**msg) for msg in messages],
        message_count=len(messages)
    )


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str, current_user: User = Depends(get_current_user)):
    session = svc.session_manager.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfoResponse(
        session_id=session['session_id'],
        document_id=session['document_id'],
        created_at=session['created_at'],
        updated_at=session['updated_at'],
        message_count=len(session['messages']),
        title=(session.get('metadata') or {}).get('title'),
        source_count=len((session.get('metadata') or {}).get('sources', [])),
        metadata=_sanitize_metadata(session.get('metadata'))
    )


@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    success = svc.session_manager.delete_session(session_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@router.get("/chat/sessions", response_model=SessionListResponse)
async def list_chat_sessions(document_id: Optional[str] = None, current_user: User = Depends(get_current_user)):
    sessions = svc.session_manager.list_sessions(current_user.id, document_id)
    sanitized_sessions = []
    for session in sessions:
        s = dict(session)
        s['metadata'] = _sanitize_metadata(session.get('metadata'))
        sanitized_sessions.append(SessionListItem(**s))

    return {
        "sessions": sanitized_sessions
    }
