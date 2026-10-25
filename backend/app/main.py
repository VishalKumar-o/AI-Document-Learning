
from fastapi import FastAPI

from app.api.health_routes import router as health_router
from app.api.document_routes import router as document_router
from app.api.question_routes import router as question_router


app = FastAPI(
    title="AI-Based Document Learning System",
    description="Document learning and question answering using NLP and RAG",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(document_router)
app.include_router(question_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to AI Document Learning System",
        "status": "Backend is running",
    }
