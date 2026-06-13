"""
KrishiMitra AI — Unit Tests for Pydantic Schemas
"""

import pytest
from pydantic import ValidationError
from app.schemas.advisory import (
    AdvisoryRequest, FeedbackRequest,
    FeedbackRating, Language
)


def test_advisory_request_valid():
    """Test valid advisory request."""
    req = AdvisoryRequest(
        query="My tomato has brown spots",
        language="en",
        crop_name="tomato"
    )
    assert req.query == "My tomato has brown spots"
    assert req.language == Language.ENGLISH
    assert req.crop_name == "tomato"


def test_advisory_request_hindi():
    """Test advisory request with Hindi language."""
    req = AdvisoryRequest(query="मेरी फसल में रोग है", language="hi")
    assert req.language == Language.HINDI


def test_advisory_request_defaults():
    """Test advisory request with no fields uses defaults."""
    req = AdvisoryRequest()
    assert req.language == Language.ENGLISH
    assert req.query is None
    assert req.crop_name is None


def test_advisory_request_query_stripped():
    """Test that query whitespace is stripped."""
    req = AdvisoryRequest(query="  tomato disease  ")
    assert req.query == "tomato disease"


def test_advisory_request_query_too_long():
    """Test that query exceeding 500 chars raises error."""
    with pytest.raises(ValidationError):
        AdvisoryRequest(query="x" * 501)


def test_feedback_request_valid():
    """Test valid feedback request."""
    fb = FeedbackRequest(advisory_id="abc-123", rating=1)
    assert fb.rating == FeedbackRating.THUMBS_UP


def test_feedback_request_thumbs_down():
    """Test thumbs down feedback."""
    fb = FeedbackRequest(advisory_id="abc-123", rating=0)
    assert fb.rating == FeedbackRating.THUMBS_DOWN


def test_feedback_invalid_rating():
    """Test invalid rating raises error."""
    with pytest.raises(ValidationError):
        FeedbackRequest(advisory_id="abc-123", rating=5)
