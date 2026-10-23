import json
import os
import socket
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import (
    delete_document as delete_document_record,
    get_all_documents,
    get_document as get_document_record,
    initialize_database,
    save_document,
)
from retrieval_service import document_search


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
OLLAMA_TIMEOUT = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Active documents and search indexes are held in memory.
# SQLite permanently stores document metadata.
documents = {}


# --------------------------------------------------
# STARTUP: RESTORE SAVED DOCUMENTS AND SEARCH INDEXES
# --------------------------------------------------

def extract_pdf_chunks(file_path):
    """Extract text chunks and page count from a saved PDF."""

    all_chunks = []

    with fitz.open(file_path) as pdf:
        page_count = len(pdf)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            all_chunks.extend(
                split_into_chunks(text, page_number)
            )

    return page_count, all_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite and rebuild indexes when FastAPI starts."""

    initialize_database()
    documents.clear()

    for record in get_all_documents():
        document_id = record["document_id"]
        file_path = record["file_path"]

        if not os.path.isfile(file_path):
            print(
                f"WARNING: PDF missing for {record['filename']}. "
                "Skipping index restoration."
            )
            continue

        try:
            page_count, all_chunks = extract_pdf_chunks(file_path)

            if not all_chunks:
                print(
                    f"WARNING: No readable text in "
                    f"{record['filename']}. Skipping."
                )
                continue

            document_search.add_document(document_id, all_chunks)

            # Refresh metadata if the saved PDF has changed.
            record["pages"] = page_count
            record["chunks"] = len(all_chunks)

            documents[document_id] = record

            print(
                f"Restored: {record['filename']} "
                f"({page_count} pages, {len(all_chunks)} chunks)"
            )

        except Exception as error:
            # One damaged PDF should not prevent the API from starting.
            print(
                f"WARNING: Could not restore "
                f"{record['filename']}: {error}"
            )

    print(f"SQLite initialized. Restored {len(documents)} document(s).")

    yield


app = FastAPI(
    title="AI Document Learning and Question Answering System",
    description=(
        "Upload multiple PDFs, manage documents, search their content, "
        "and ask questions with page-based sources."
    ),
    version="1.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about an uploaded document.",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of document passages to retrieve.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the document to search.",
    )


# --------------------------------------------------
# DOCUMENT HELPERS
# --------------------------------------------------

def public_document(document):
    """Return document metadata without its internal file path."""

    return {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "pages": document["pages"],
        "chunks": document["chunks"],
        "uploaded_at": document["uploaded_at"],
    }


def resolve_document_id(document_id):
    """Validate the selected document or choose the only available one."""

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before continuing.",
        )

    if document_id:
        if document_id not in documents:
            raise HTTPException(
                status_code=404,
                detail="Document not found. Refresh the document list.",
            )
        return document_id

    if len(documents) == 1:
        return next(iter(documents))

    raise HTTPException(
        status_code=400,
        detail=(
            "Please provide document_id to select which PDF to use. "
            "Get available IDs from GET /documents."
        ),
    )


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_into_chunks(text, page_number):
    """Split one PDF page into smaller overlapping chunks."""

    chunks = []
    text = text.strip()

    if not text:
        return chunks

    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "page": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# BUILD DOCUMENT CONTEXT
# --------------------------------------------------

def build_document_context(results):
    """Format retrieved passages with page citations."""

    context_parts = []

    for index, result in enumerate(results, start=1):
        page = result.get("page", "Unknown")
        text = result.get("text", "").strip()

        if text:
            context_parts.append(
                f"[Source {index} | Page {page}]\n{text}"
            )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# LOCAL OLLAMA LLM
# --------------------------------------------------

def generate_local_answer(question, context):
    """Generate a grounded answer using the local Ollama model."""

    prompt = f"""
You are an accurate document question-answering assistant.

Answer the user's exact question using ONLY the supplied document passages.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}

RULES:
1. Answer the exact question asked.
2. Identify the specific fact requested before answering.
3. Do not confuse events that happen one after another.
4. Prioritize passages that directly describe the requested event.
5. Explain events in chronological order when needed.
6. Use only facts supported by the passages.
7. Combine passages only when they help answer the question.
8. Cite supporting pages using (Page 4) or (Pages 3 and 4).
9. Every cited page must support the statement next to it.
10. Ignore irrelevant passages.
11. Do not invent quotations or facts.
12. Use simple English, usually one to three sentences.
13. If the passages do not contain enough evidence, respond with exactly:
I could not find enough information to answer this question
in the uploaded document.

Now answer the user's actual question using the supplied passages.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise document question-answering "
                    "assistant. Use only the provided evidence. "
                    "Answer the exact question, distinguish events "
                    "correctly, and provide accurate page citations. "
                    "Never guess or invent facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_TIMEOUT,
        ) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

        answer = (
            response_data.get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an empty answer.",
            )

        return answer

    except HTTPException:
        raise

    except urllib.error.HTTPError as error:
        error_body = error.read().decode(
            "utf-8",
            errors="replace",
        )
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {error.code}: {error_body}",
        )

    except urllib.error.URLError as error:
        reason = error.reason

        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise HTTPException(
                status_code=504,
                detail="Ollama timed out. Please try again.",
            )

        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to Ollama. Check that it is running. "
                f"Details: {reason}"
            ),
        )

    except (socket.timeout, TimeoutError):
        raise HTTPException(
            status_code=504,
            detail="The local model took too long to respond.",
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid JSON response.",
        )

    except OSError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read the Ollama response: {error}",
        )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Document Learning and Question Answering System",
        "model": OLLAMA_MODEL,
        "status": "running",
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "document_count": len(documents),
        "indexed_chunks": len(document_search.chunks),
    }


# --------------------------------------------------
# UPLOAD PDF
# --------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload, extract, index, and permanently register a PDF."""

    filename = os.path.basename(file.filename or "")

    if not filename.lower().endswith(".pdf"):
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

    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            page_count = len(pdf)
            all_chunks = []

            for page_number, page in enumerate(pdf, start=1):
                all_chunks.extend(
                    split_into_chunks(
                        page.get_text("text"),
                        page_number,
                    )
                )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the PDF: {error}",
        )

    if not all_chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found in this PDF. "
                "It may be a scanned PDF that needs OCR."
            ),
        )

    document_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    document = {
        "document_id": document_id,
        "filename": filename,
        "pages": page_count,
        "chunks": len(all_chunks),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_path": file_path,
    }

    index_added = False
    file_saved = False
    database_saved = False

    try:
        with open(file_path, "wb") as saved_file:
            saved_file.write(content)
        file_saved = True

        document_search.add_document(document_id, all_chunks)
        index_added = True

        save_document(document)
        database_saved = True

        documents[document_id] = document

    except Exception as error:
        if index_added:
            document_search.remove_document(document_id)

        if database_saved:
            delete_document_record(document_id)

        documents.pop(document_id, None)

        if file_saved and os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Could not register or index the PDF: {error}",
        )

    return {
        "message": "PDF uploaded and indexed successfully.",
        **public_document(document),
        "total_documents": len(documents),
        "total_indexed_chunks": len(document_search.chunks),
    }


# --------------------------------------------------
# LIST ALL DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def list_documents():
    """List documents registered in the active application."""

    return {
        "total_documents": len(documents),
        "documents": [
            public_document(document)
            for document in documents.values()
        ],
    }


# --------------------------------------------------
# GET ONE DOCUMENT
# --------------------------------------------------

@app.get("/documents/{document_id}")
def get_one_document(document_id: str):
    """Get metadata for one PDF."""

    document = get_document_record(document_id)

    if not document or document_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Document not found or its PDF could not be restored.",
        )

    return public_document(document)


# --------------------------------------------------
# DELETE ONE DOCUMENT
# --------------------------------------------------

@app.delete("/documents/{document_id}")
def delete_one_document(document_id: str):
    """Delete one PDF, its search index, and its database record."""

    document = documents.get(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    file_path = document["file_path"]

    try:
        # Remove the file first so a file deletion error does not
        # remove the database record or search index.
        if os.path.exists(file_path):
            os.remove(file_path)

        if not delete_document_record(document_id):
            raise HTTPException(
                status_code=500,
                detail="Could not remove the document database record.",
            )

        document_search.remove_document(document_id)
        del documents[document_id]

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not fully delete the document: {error}",
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "filename": document["filename"],
        "remaining_documents": len(documents),
    }


# --------------------------------------------------
# SHARED QUESTION-ANSWERING FUNCTION
# --------------------------------------------------

def answer_document_question(request: AskRequest):
    """Search the selected PDF and generate a grounded answer."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    document_id = resolve_document_id(request.document_id)
    document = documents[document_id]
    top_k = max(1, min(request.top_k or 5, 10))

    try:
        results = document_search.search(
            question,
            top_k=top_k,
            document_id=document_id,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {error}",
        )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant information was found in this document.",
        )

    context = build_document_context(results)

    if not context:
        raise HTTPException(
            status_code=404,
            detail="The retrieved passages contain no readable text.",
        )

    answer = generate_local_answer(question, context)

    sources = [
        {
            "page": result["page"],
            "chunk_index": result["chunk_index"],
            "score": round(result["score"], 6),
            "text": result["text"],
        }
        for result in results
    ]

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "question": question,
        "answer": answer,
        "model": OLLAMA_MODEL,
        "sources": sources,
    }


# --------------------------------------------------
# ASK QUESTIONS
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: AskRequest):
    return answer_document_question(request)


# --------------------------------------------------
# QUERY ALIAS
# --------------------------------------------------

@app.post("/query")
def query_question(request: AskRequest):
    return answer_document_question(request)


# --------------------------------------------------
# DEBUG SEARCH
# --------------------------------------------------

@app.post("/debug-search")
def debug_search(request: AskRequest):
    """Inspect retrieved passages without calling Ollama."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    document_id = resolve_document_id(request.document_id)
    document = documents[document_id]
    top_k = max(1, min(request.top_k or 5, 10))

    try:
        results = document_search.search(
            question,
            top_k=top_k,
            document_id=document_id,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {error}",
        )

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "question": question,
        "results": results,
    }
