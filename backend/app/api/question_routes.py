
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.document_routes import retrieval_service
from app.services.llm_service import LLMService

router = APIRouter(prefix="/questions", tags=["Questions"])

llm_service = LLMService()


class QuestionRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/ask")
def ask_question(request: QuestionRequest):
    """Retrieve relevant passages and generate an AI answer."""

    results = retrieval_service.search(
        question=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=(
                "No indexed document or matching passages found. "
                "Upload the PDF again if the backend was restarted."
            ),
        )

    try:
        answer = llm_service.generate_answer(
            question=request.question,
            passages=results,
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected AI service error: {error}",
        ) from error

    return {
        "document_id": request.document_id,
        "question": request.question,
        "answer": answer,
        "answer_model": getattr(llm_service, "last_provider", llm_service.model),
        "total_results": len(results),
        "results": results,
    }
