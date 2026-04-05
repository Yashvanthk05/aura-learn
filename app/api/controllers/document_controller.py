from fastapi import APIRouter, HTTPException, Depends
from app.core.security import get_current_user
from app.models.schemas import User, DocumentInfoResponse, ChunkInfo


from app.core.config import settings
from . import service_registry as svc

router = APIRouter()


@router.get("/document/{document_id}", response_model=DocumentInfoResponse)
async def get_document_info(document_id: str, current_user: User = Depends(get_current_user)):
    document = svc.document_manager.get_document(document_id, current_user.id)
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


@router.delete("/document/{document_id}")
async def delete_document(document_id: str, current_user: User = Depends(get_current_user)):
    success = svc.document_manager.delete_document(document_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    for f in settings.UPLOAD_DIR.glob(f"{document_id}_*"):
        f.unlink()

    return {"message": "Document deleted successfully"}


@router.get("/documents")
async def list_documents(current_user: User = Depends(get_current_user)):
    documents = svc.document_manager.list_documents(current_user.id)
    return {"documents": documents}
