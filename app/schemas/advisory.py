"""
KrishiMitra AI — Pydantic Schemas
Defines all request/response data models with validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum
import uuid
from datetime import datetime, timezone


def utcnow():
    """Timezone-aware UTC datetime (replaces deprecated utcnow)."""
    return datetime.now(timezone.utc)


# ── Enums ─────────────────────────────────────────────────

class Language(str, Enum):
    ENGLISH = "en"
    HINDI   = "hi"


class FeedbackRating(int, Enum):
    THUMBS_DOWN = 0
    THUMBS_UP   = 1


# ── Advisory Request ──────────────────────────────────────

class AdvisoryRequest(BaseModel):
    query:     Optional[str] = Field(default=None, max_length=500)
    language:  Language      = Field(default=Language.ENGLISH)
    crop_name: Optional[str] = Field(default=None, max_length=100)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v):
        if v:
            return v.strip()
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "My tomato leaves have brown spots",
                "language": "en",
                "crop_name": "tomato"
            }
        }
    }


# ── Advisory Response ─────────────────────────────────────

class DiseaseInfo(BaseModel):
    disease_name:  str
    confidence:    float = Field(ge=0.0, le=1.0)
    severity:      Optional[str] = None
    affected_crop: Optional[str] = None


class AdvisoryResponse(BaseModel):
    advisory_id:       str      = Field(default_factory=lambda: str(uuid.uuid4()))
    disease:           Optional[DiseaseInfo] = None
    advisory_text:     str
    recommendations:   List[str] = []
    confidence_score:  float    = Field(ge=0.0, le=1.0)
    language:          Language
    escalate_to_expert: bool    = False
    timestamp:         datetime = Field(default_factory=utcnow)
    disclaimer:        str      = Field(
        default="This advisory is AI-generated. For severe cases, consult your local KVK."
    )


# ── Feedback Schema ───────────────────────────────────────

class FeedbackRequest(BaseModel):
    advisory_id: str                    = Field(description="ID of the advisory being rated")
    rating:      FeedbackRating         = Field(description="0 = thumbs down, 1 = thumbs up")
    comment:     Optional[str]          = Field(default=None, max_length=300)


class FeedbackResponse(BaseModel):
    message:     str = "Thank you for your feedback!"
    advisory_id: str


# ── Health Check Schema ───────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    app_name:    str
    version:     str
    environment: str
    timestamp:   datetime = Field(default_factory=utcnow)
