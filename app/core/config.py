"""
KrishiMitra AI — Central Configuration
Loads all settings from .env file using Pydantic BaseSettings.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── App ────────────────────────────────────────────────
    APP_NAME:    str  = "KrishiMitra AI"
    APP_VERSION: str  = "1.0.0"
    APP_ENV:     str  = "development"
    DEBUG:       bool = True

    # ── API ────────────────────────────────────────────────
    API_HOST:       str = "0.0.0.0"
    API_PORT:       int = 8000
    API_RATE_LIMIT: str = "10/minute"

    # ── IBM Watson ─────────────────────────────────────────
    IBM_API_KEY:    str = "dummy_key"
    IBM_PROJECT_ID: str = "dummy_project"
    IBM_URL:        str = "https://us-south.ml.cloud.ibm.com"

    # ── Model ──────────────────────────────────────────────
    MODEL_CHECKPOINT_PATH: str   = "models/checkpoints/efficientnet_b0_best.pth"
    CONFIDENCE_THRESHOLD:  float = 0.75
    NUM_CLASSES:           int   = 38

    # ── Database ───────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./logs/krishimitra.db"

    # ── ChromaDB ───────────────────────────────────────────
    CHROMA_DB_PATH:         str = "data/knowledge_base/chroma_db"
    CHROMA_COLLECTION_NAME: str = "icar_knowledge"

    # ── Embeddings ─────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Language ───────────────────────────────────────────
    DEFAULT_LANGUAGE:    str = "en"
    SUPPORTED_LANGUAGES: str = "en,hi"

    # ── Logging ────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE:  str = "logs/krishimitra.log"

    @property
    def supported_languages_list(self) -> list:
        return self.SUPPORTED_LANGUAGES.split(",")

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
