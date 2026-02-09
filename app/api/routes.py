from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import shutil
from typing import Optional

from app.models.schemas import (
    UploadPDFResponse,
    DocumentInfoResponse,
    SummarizeRequest,
    SummarizeResponse,
    AudiobookRequest,
    AudiobookResponse,
    SummarizeAndAudioRequest,
    SummarizeAndAudioResponse,
    HealthResponse,
    ChunkInfo,
    CreateSessionRequest,
    CreateSessionResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    ConversationHistoryResponse,
    SessionInfoResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Citation,
    ChatMessage )

from app.core.config import settings
from app.services.document_service import DocumentManager
from app.services.extractive_service import ExtractiveSummarizer
from app.services.abstractive_service import AbstractiveSummarizer
from app.services.tts_service import AudiobookGenerator
from app.services.vector_store_service import HybridVectorStore
from app.services.session_service import SessionManager
from app.services.chat_service import RAGChatService
from typing import Dict
from datetime import datetime

router = APIRouter()

document_manager: Optional[DocumentManager] = None
extractive_summarizer: Optional[ExtractiveSummarizer] = None
abstractive_summarizer: Optional[AbstractiveSummarizer] = None
audiobook_generator: Optional[AudiobookGenerator] = None
vector_stores: Dict[str, HybridVectorStore] = {}
session_manager: Optional[SessionManager] = None
chat_service: Optional[RAGChatService] = None

def init_services():
    global document_manager, extractive_summarizer, abstractive_summarizer, audiobook_generator, session_manager, chat_service
    
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
    
    sessions_dir = settings.DATA_DIR / "sessions"
    session_manager = SessionManager(sessions_dir)
    chat_service = RAGChatService()

@router.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    document_id = str(uuid.uuid4())
    
    upload_path = settings.UPLOAD_DIR / f"{document_id}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        result = document_manager.process_document(str(upload_path), document_id)
        
        return UploadPDFResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=result["num_chunks"],
            message=f"PDF processed successfully. {result['num_chunks']} chunks extracted."
        )
    except Exception as e:

        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.get("/document/{document_id}", response_model=DocumentInfoResponse)
async def get_document_info(document_id: str):

    document = document_manager.get_document(document_id)
    
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks_info = [
        ChunkInfo(
            chunk_id=chunk["chunk_id"],
            topic=chunk["topic"],
            page=chunk["page"],
            text_preview=chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
        )
        for chunk in document["chunks"]
    ]
    
    return DocumentInfoResponse(
        document_id=document_id,
        filename=document["filename"],
        num_chunks=document["num_chunks"],
        chunks=chunks_info
    )

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    
    chunks = document_manager.get_chunks(request.document_id, request.chunk_ids)
    
    if chunks is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found for specified IDs")
    
    combined_text = " ".join([chunk["text"] for chunk in chunks])
    
    try:
        if request.summarization_type == "extractive":
            if extractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Extractive summarizer not available")
            
            summary = extractive_summarizer.summarize(
                combined_text,
                num_sentences=request.num_sentences
            )
            
        elif request.summarization_type == "abstractive":
            if abstractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Abstractive summarizer not available")
            
            summary = abstractive_summarizer.summarize(
                combined_text,
                max_length=request.max_length,
                min_length=request.min_length
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid summarization type")
        
        return SummarizeResponse(
            document_id=request.document_id,
            summarization_type=request.summarization_type,
            summary=summary,
            num_chunks_processed=len(chunks),
            metadata={
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    
@router.post("/generate-audiobook", response_model=AudiobookResponse)
async def generate_audiobook(request: AudiobookRequest):

    if audiobook_generator is None:
        raise HTTPException(status_code=503, detail="Audiobook generator not available")
    
    try:

        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.wav"
        audio_path = settings.OUTPUT_DIR / audio_filename
        
        audiobook_generator.generate(
            text=request.text,
            output_path=str(audio_path),
            language=request.language
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audiobook generation failed: {str(e)}")

@router.post("/summarize-and-audio", response_model=SummarizeAndAudioResponse)
async def summarize_and_generate_audio(request: SummarizeAndAudioRequest):
   
    chunks = document_manager.get_chunks(request.document_id, request.chunk_ids)
    
    if chunks is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found")
    
    combined_text = " ".join([chunk["text"] for chunk in chunks])
    
    try:
        if request.summarization_type == "extractive":
            if extractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Extractive summarizer not available")
            summary = extractive_summarizer.summarize(
                combined_text,
                num_sentences=request.num_sentences
            )
        else:
            if abstractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Abstractive summarizer not available")
            summary = abstractive_summarizer.summarize(
                combined_text,
                max_length=request.max_length,
                min_length=request.min_length
            )
        
        if audiobook_generator is None:
            raise HTTPException(status_code=503, detail="Audiobook generator not available")
        
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.wav"
        audio_path = settings.OUTPUT_DIR / audio_filename
        
        audiobook_generator.generate(
            text=summary,
            output_path=str(audio_path),
            language=request.language
        )

        if not audio_path.exists():
            raise HTTPException(status_code=500, detail="Processing failed: audio output file missing")

        return SummarizeAndAudioResponse(
            document_id=request.document_id,
            summarization_type=request.summarization_type,
            summary=summary,
            audio_url=f"/audio/{audio_filename}",
            audio_filename=audio_filename,
            num_chunks_processed=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    
    audio_path = settings.OUTPUT_DIR / filename

    if audio_path.suffix == "":
        wav_candidate = audio_path.with_suffix(".wav")
        if wav_candidate.exists():
            audio_path = wav_candidate

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=audio_path.name
    )

@router.get("/health", response_model=HealthResponse)
async def health_check():

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded={
            "extractive_summarizer": extractive_summarizer is not None,
            "abstractive_summarizer": abstractive_summarizer is not None,
            "audiobook_generator": audiobook_generator is not None,
            "session_manager": session_manager is not None,
            "chat_service": chat_service is not None,
            "vector_stores": len(vector_stores)
        }
    )

@router.delete("/document/{document_id}")
async def delete_document(document_id: str):

    success = document_manager.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    for pdf_file in settings.UPLOAD_DIR.glob(f"{document_id}_*"):
        pdf_file.unlink()
    
    return {"message": "Document deleted successfully"}

@router.get("/documents")
async def list_documents():

    documents = document_manager.list_documents()
    return {"documents": documents}

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

@router.post("/chat/session", response_model=CreateSessionResponse)
async def create_chat_session(request: CreateSessionRequest):
    
    document = document_manager.get_document(request.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    get_or_create_vector_store(request.document_id)
    
    session_id = session_manager.create_session(
        request.document_id,
        request.metadata
    )
    
    session = session_manager.get_session(session_id)
    
    return CreateSessionResponse(
        session_id=session_id,
        document_id=request.document_id,
        created_at=session['created_at'],
        message="Chat session created successfully"
    )

@router.post("/chat/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    document_id = session['document_id']
    
    try:
        vector_store = get_or_create_vector_store(document_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
    
    try:
        if request.use_hybrid_search:
            retrieved_chunks = vector_store.search(
                request.query,
                top_k=request.top_k
            )
            search_method = "hybrid (FAISS + BM25 + TF-IDF)"
        else:
            retrieved_chunks = vector_store.search_faiss_only(
                request.query,
                top_k=request.top_k
            )
            search_method = "FAISS (semantic only)"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    conversation_history = None
    if request.include_history:
        conversation_history = session_manager.get_context_for_query(
            request.session_id,
            max_history=5
        )
    
    try:
        response_text, citations = chat_service.generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            max_context_chunks=request.max_context_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
    
    session_manager.add_message(
        request.session_id,
        role='user',
        content=request.query
    )
    
    session_manager.add_message(
        request.session_id,
        role='assistant',
        content=response_text,
        citations=citations
    )
    
    citation_models = [Citation(**c) for c in citations]
    
    return ChatQueryResponse(
        session_id=request.session_id,
        query=request.query,
        response=response_text,
        citations=citation_models,
        retrieved_chunks=len(retrieved_chunks),
        search_method=search_method,
        timestamp=datetime.now().isoformat()
    )

@router.get("/chat/session/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, max_messages: Optional[int] = None):
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session_manager.get_conversation_history(
        session_id,
        max_messages=max_messages,
        include_citations=True
    )
    
    message_models = [ChatMessage(**msg) for msg in messages]
    
    return ConversationHistoryResponse(
        session_id=session_id,
        document_id=session['document_id'],
        messages=message_models,
        message_count=len(messages)
    )

@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    
    session = session_manager.get_session(session_id)
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
    
    success = session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}

@router.get("/chat/sessions")
async def list_chat_sessions(document_id: Optional[str] = None):
    
    sessions = session_manager.list_sessions(document_id)
    return {"sessions": sessions}

@router.post("/search", response_model=SearchResponse)
async def search_document(request: SearchRequest):
    
    try:
        vector_store = get_or_create_vector_store(request.document_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
    
    try:
        if request.search_method == "hybrid":
            results = vector_store.search(request.query, top_k=request.top_k)
        elif request.search_method == "faiss":
            results = vector_store.search_faiss_only(request.query, top_k=request.top_k)
        elif request.search_method == "bm25":
            results = vector_store.search_bm25_only(request.query, top_k=request.top_k)
        else:
            raise HTTPException(status_code=400, detail="Invalid search method. Use 'hybrid', 'faiss', or 'bm25'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    result_models = []
    for r in results:
        result_models.append(SearchResult(
            chunk_id=r['chunk_id'],
            topic=r['topic'],
            page=r['page'],
            text=r['text'],
            score=r['score'],
            score_breakdown=r.get('score_breakdown')
        ))
    
    return SearchResponse(
        document_id=request.document_id,
        query=request.query,
        results=result_models,
        search_method=request.search_method,
        total_results=len(result_models)
    )