"""
KrishiMitra AI -- FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logger import logger
from app.api.routes import health
from app.api.routes import advisory

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🌱 {settings.APP_NAME} v{settings.APP_VERSION} starting up...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info("API is ready to serve farmers!")
    yield
    logger.info("KrishiMitra AI shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered crop disease detection and advisory for Indian farmers.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(advisory.router)


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to KrishiMitra AI 🌾",
        "tagline": "Your AI farming expert — anytime, anywhere",
        "docs": "/docs",
        "health": "/health",
        "advisory": "POST /advisory",
    }
