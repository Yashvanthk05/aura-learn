from fastapi import APIRouter, HTTPException
import uuid

from app.models.schemas import (
    SummarizeRequest, SummarizeResponse,
    SummarizeAndAudioRequest, SummarizeAndAudioResponse,
)
from app.core.config import settings
from . import service_registry as svc

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    chunks = svc.document_manager.get_chunks(request.document_id, request.chunk_ids)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found for specified IDs")

    combined_text = " ".join([chunk["text"] for chunk in chunks])

    try:
        if request.summarization_type == "extractive":
            if svc.extractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Extractive summarizer not available")
            summary = svc.extractive_summarizer.summarize(
                combined_text, num_sentences=request.num_sentences
            )
        elif request.summarization_type == "abstractive":
            if svc.abstractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Abstractive summarizer not available")
            summary = svc.abstractive_summarizer.summarize(
                combined_text, max_length=request.max_length, min_length=request.min_length
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid summarization type")

        return SummarizeResponse(
            document_id=request.document_id,
            summarization_type=request.summarization_type,
            summary=summary,
            num_chunks_processed=len(chunks),
            metadata={"chunk_ids": [chunk["chunk_id"] for chunk in chunks]}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/summarize-and-audio", response_model=SummarizeAndAudioResponse)
async def summarize_and_generate_audio(request: SummarizeAndAudioRequest):
    chunks = svc.document_manager.get_chunks(request.document_id, request.chunk_ids)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found")

    combined_text = " ".join([chunk["text"] for chunk in chunks])

    try:
        if request.summarization_type == "extractive":
            if svc.extractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Extractive summarizer not available")
            summary = svc.extractive_summarizer.summarize(
                combined_text, num_sentences=request.num_sentences
            )
        else:
            if svc.abstractive_summarizer is None:
                raise HTTPException(status_code=503, detail="Abstractive summarizer not available")
            summary = svc.abstractive_summarizer.summarize(
                combined_text, max_length=request.max_length, min_length=request.min_length
            )

        if svc.audiobook_generator is None:
            raise HTTPException(status_code=503, detail="Audiobook generator not available")

        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.wav"
        audio_path = settings.OUTPUT_DIR / audio_filename

        svc.audiobook_generator.generate(
            text=summary, output_path=str(audio_path), language=request.language
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
