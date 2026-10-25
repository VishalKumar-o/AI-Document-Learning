
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_processing_service import (
    DocumentProcessingService,
)
from app.services.retrieval_service import RetrievalService


router = APIRouter(prefix="/documents", tags=["Documents"])

document_processing_service = DocumentProcessingService()
retrieval_service = RetrievalService()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # Generate a unique filename to avoid overwriting another upload.
    document_id = str(uuid4())
    safe_filename = Path(file.filename).name
    stored_filename = f"{document_id}.pdf"
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_path.write_bytes(content)

        chunks = document_processing_service.process_pdf(
            str(file_path)
        )

        if not chunks:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in the PDF.",
            )

        retrieval_service.add_document(
            document_id=document_id,
            chunks=chunks,
        )

        return {
            "message": "PDF uploaded, processed, and indexed successfully",
            "document_id": document_id,
            "filename": safe_filename,
            "total_pages_with_text": len(
                {chunk["page_number"] for chunk in chunks}
            ),
            "total_chunks": len(chunks),
        }

    except HTTPException:
        raise

    except Exception as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Could not process or index this PDF. Please check the file.",
        ) from error

    finally:
        await file.close()
