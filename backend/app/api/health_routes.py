
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "success",
        "message": "AI Document Learning backend is running"
    }
