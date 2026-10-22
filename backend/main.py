
from pathlib import Path
from uuid import uuid4

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(
    title="AI Document Learning System",
    description="An AI-powered system for learning from documents",
    version="1.0.0"
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Document Learning System",
        "status": "Backend is running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF."
        )

    saved_path = UPLOAD_DIR / f"{uuid4().hex}.pdf"

    try:
        saved_path.write_bytes(content)

        with fitz.open(saved_path) as document:
            pages = []

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                pages.append({
                    "page": page_number,
                    "text": text
                })

        return {
            "message": "PDF uploaded and text extracted successfully.",
            "filename": file.filename,
            "total_pages": len(pages),
            "pages_with_text": sum(
                1 for page in pages if page["text"]
            ),
            "pages": pages
        }

    except Exception as error:
        saved_path.unlink(missing_ok=True)

        if isinstance(error, HTTPException):
            raise error

        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Please check the file."
        ) from error

    finally:
        await file.close()
