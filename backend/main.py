
import json
import socket
import urllib.error
import urllib.request
from typing import Optional

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retrieval_service import document_search


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
OLLAMA_TIMEOUT = 300

app = FastAPI(
    title="AI Document Learning and Question Answering System",
    description=(
        "Upload a PDF, search its content, and ask questions "
        "with answers grounded in document passages."
    ),
    version="1.1.0"
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
        description="Question about the uploaded document."
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of document passages to retrieve."
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
                "text": chunk_text
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
    """
    Format retrieved passages so the model can identify
    which page supports each part of its answer.
    """

    context_parts = []

    for index, result in enumerate(results, start=1):
        page = result.get("page", "Unknown")
        text = result.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"[Source {index} | Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# LOCAL OLLAMA LLM
# --------------------------------------------------

def generate_local_answer(question, context):
    """
    Generate an answer using the local Ollama model.

    The model must answer the exact question, use document
    evidence, distinguish events, and cite supporting pages.
    """

    prompt = f"""
You are an accurate document question-answering assistant.

Your task is to answer the user's exact question using ONLY
the supplied document passages.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}

FOLLOW THESE RULES CAREFULLY:

1. Answer the exact question asked. Do not answer a different
   or broader question.

2. Find the specific fact requested before writing the answer.
   For example, if the user asks where someone LANDED after
   falling, state the exact place where they landed.

3. Do not confuse events that happen one after another.
   If a character first lands in one place and later enters
   another place, keep these events separate.

4. When the question asks about a specific event, prioritize
   the passage that directly describes that event.

5. If the passages describe several events, explain them in
   chronological order when needed to answer the question.

6. Use only facts supported by the supplied passages.
   Do not invent details, reasons, motives, or conclusions.

7. If multiple passages provide relevant evidence, combine
   them only when they help answer the actual question.

8. Cite the page that supports each factual statement.
   Use the format (Page 4) or (Pages 3 and 4).

9. Do not cite a page just because it appears in the context.
   Each cited page must support the statement next to it.

10. Ignore irrelevant passages, even if they mention the same
    character or topic.

11. Do not invent quotations. Only quote words that appear
    exactly in the supplied passages.

12. Use simple, clear English. Usually answer in one to three
    sentences. Include important details without unnecessary
    explanations.

13. If the passages do not contain enough evidence to answer,
    respond with exactly:
    I could not find enough information to answer this question
    in the uploaded document.

EXAMPLE OF PRECISE ANSWERING:

Question:
Where did Alice land after falling down the rabbit hole?

Document passage:
[Source 1 | Page 4]
"Down she came upon a heap of sticks and dry leaves,
and the fall was over."
Later in the same passage, Alice finds herself in a long,
low hall lit by lamps.

Incorrect answer:
Alice landed in a dark hall (Page 4).

Correct answer:
Alice landed on a heap of sticks and dry leaves (Page 4).

The hall was where she arrived later, not where she first landed.

Now answer the user's actual question using the supplied
document passages. Follow the same level of precision.
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
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_TIMEOUT
        ) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

        answer = (
            response_data
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an empty answer."
            )

        return answer

    except HTTPException:
        raise

    except urllib.error.HTTPError as error:
        error_body = error.read().decode(
            "utf-8",
            errors="replace"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama returned HTTP {error.code}: "
                f"{error_body}"
            )
        )

    except urllib.error.URLError as error:
        reason = error.reason

        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise HTTPException(
                status_code=504,
                detail=(
                    "Ollama timed out. Try again or use a shorter "
                    "question."
                )
            )

        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to Ollama. Check that Ollama is "
                f"running. Details: {reason}"
            )
        )

    except (socket.timeout, TimeoutError):
        raise HTTPException(
            status_code=504,
            detail=(
                "The local model took too long to respond. "
                "Try again or use a shorter question."
            )
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid JSON response."
        )

    except OSError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read the Ollama response: {error}"
        )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Document Learning and Question Answering System",
        "model": OLLAMA_MODEL,
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "indexed_chunks": len(document_search.chunks)
    }


# --------------------------------------------------
# UPLOAD PDF
# --------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text from a PDF and index its chunks."""

    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
    ):
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

    try:
        all_chunks = []

        with fitz.open(
            stream=content,
            filetype="pdf"
        ) as pdf:
            page_count = len(pdf)

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                page_chunks = split_into_chunks(
                    page_text,
                    page_number
                )

                all_chunks.extend(page_chunks)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the PDF: {str(error)}"
        )

    if not all_chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found in this PDF. "
                "It may be a scanned PDF that needs OCR."
            )
        )

    try:
        document_search.add_chunks(all_chunks)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not index the PDF: {str(error)}"
        )

    return {
        "message": "PDF uploaded and indexed successfully.",
        "filename": file.filename,
        "pages": page_count,
        "chunks_added": len(all_chunks),
        "total_indexed_chunks": len(document_search.chunks)
    }


# --------------------------------------------------
# SHARED QUESTION-ANSWERING FUNCTION
# --------------------------------------------------

def answer_document_question(request: AskRequest):
    """Retrieve relevant passages and generate a grounded answer."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    if not document_search.chunks:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking a question."
        )

    top_k = max(1, min(request.top_k or 5, 10))

    try:
        results = document_search.search(
            question,
            top_k=top_k
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(error)}"
        )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant information was found in the document."
        )

    context = build_document_context(results)

    if not context:
        raise HTTPException(
            status_code=404,
            detail="The retrieved passages contain no readable text."
        )

    answer = generate_local_answer(
        question,
        context
    )

    sources = [
        {
            "page": result["page"],
            "chunk_index": result["chunk_index"],
            "score": round(result["score"], 6),
            "text": result["text"]
        }
        for result in results
    ]

    return {
        "question": question,
        "answer": answer,
        "model": OLLAMA_MODEL,
        "sources": sources
    }


# --------------------------------------------------
# ASK QUESTIONS
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: AskRequest):
    """Ask a question about the uploaded document."""
    return answer_document_question(request)


# --------------------------------------------------
# QUERY ALIAS
# --------------------------------------------------

@app.post("/query")
def query_question(request: AskRequest):
    """Alternative endpoint for document question answering."""
    return answer_document_question(request)


# --------------------------------------------------
# DEBUG SEARCH
# --------------------------------------------------

@app.post("/debug-search")
def debug_search(request: AskRequest):
    """Return retrieval results without calling the language model."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    if not document_search.chunks:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before running document search."
        )

    top_k = max(1, min(request.top_k or 5, 10))

    try:
        results = document_search.search(
            question,
            top_k=top_k
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(error)}"
        )

    return {
        "question": question,
        "results": results
    }
