from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime

from app.models.schemas import (
    CreateSessionRequest, CreateSessionResponse,
    ChatQueryRequest, ChatQueryResponse,
    ConversationHistoryResponse, SessionInfoResponse,
    Citation, ChatMessage,
)
from . import service_registry as svc

router = APIRouter()


@router.post("/chat/session", response_model=CreateSessionResponse)
async def create_chat_session(request: CreateSessionRequest):
    document = svc.document_manager.get_document(request.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    svc.get_or_create_vector_store(request.document_id)

    session_id = svc.session_manager.create_session(
        request.document_id, request.metadata
    )
    session = svc.session_manager.get_session(session_id)

    return CreateSessionResponse(
        session_id=session_id,
        document_id=request.document_id,
        created_at=session['created_at'],
        message="Chat session created successfully"
    )


@router.post("/chat/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    session = svc.session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    document_id = session['document_id']

    try:
        vector_store = svc.get_or_create_vector_store(document_id)
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
            request.session_id, max_history=5
        )

    try:
        response_text, citations = svc.chat_service.generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            max_context_chunks=request.max_context_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

    svc.session_manager.add_message(request.session_id, role='user', content=request.query)
    svc.session_manager.add_message(
        request.session_id, role='assistant', content=response_text, citations=citations
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
async def get_conversation_history(session_id: str, max_messages: Optional[int] = None):
    session = svc.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = svc.session_manager.get_conversation_history(
        session_id, max_messages=max_messages, include_citations=True
    )

    return ConversationHistoryResponse(
        session_id=session_id,
        document_id=session['document_id'],
        messages=[ChatMessage(**msg) for msg in messages],
        message_count=len(messages)
    )


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    session = svc.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfoResponse(
        session_id=session['session_id'],
        document_id=session['document_id'],
        created_at=session['created_at'],
        updated_at=session['updated_at'],
        message_count=len(session['messages']),
        metadata=session.get('metadata')
    )


@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    success = svc.session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@router.get("/chat/sessions")
async def list_chat_sessions(document_id: Optional[str] = None):
    sessions = svc.session_manager.list_sessions(document_id)
    return {"sessions": sessions}
