from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.core.security import get_current_user
from app.models.schemas import User, UploadPDFResponse
import uuid
import shutil
import os


from app.core.config import settings
from app.utils.pdf_processor import SUPPORTED_EXTENSIONS
from . import service_registry as svc

router = APIRouter()


@router.post("/upload", response_model=UploadPDFResponse)
async def upload_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    document_id = str(uuid.uuid4())
    upload_path = settings.UPLOAD_DIR / f"{document_id}_{file.filename}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        result = svc.document_manager.process_document(str(upload_path), current_user.id, document_id)
        return UploadPDFResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=result["num_chunks"],
            message=f"Document processed successfully. {result['num_chunks']} chunks extracted."
        )
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
