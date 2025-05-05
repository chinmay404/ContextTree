"""
Health check endpoints.
"""
from fastapi import APIRouter, status

router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status of the application.
    """
    return {
        "status": "healthy",
        "message": "The service is up and running."
    }