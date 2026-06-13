"""
KrishiMitra AI — Health Check Route
GET /health — confirms the API is alive and configured correctly.
"""

from fastapi import APIRouter
from app.schemas.advisory import HealthResponse
from app.core.config import settings
from app.core.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns app status, version, and environment.
    """
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )
