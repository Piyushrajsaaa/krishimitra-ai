"""
KrishiMitra AI -- Unit Tests for LLM Advisory Service
Run: pytest tests/unit/test_llm.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_service import LLMService


@pytest.fixture
def llm():
    """LLM service instance (IBM will be unavailable in test env)."""
    return LLMService()


@pytest.fixture
def sample_advisory_args():
    return dict(
        disease_name="Tomato___Early_blight",
        confidence=0.88,
        crop_name="tomato",
        query="My tomato leaves have brown spots",
        context="Apply Mancozeb 2g/L. Remove infected leaves immediately.",
        language="en",
        advisory_id="test-001",
    )


# ── Initialization Tests ──────────────────────────────────

def test_llm_initializes(llm):
    """LLM service must initialize without crashing."""
    assert llm is not None


def test_ibm_not_ready_without_key(llm):
    """IBM must not be ready when API key is dummy."""
    assert llm.is_ibm_ready is False


# ── Fallback Advisory Tests (English) ────────────────────

def test_fallback_english_returns_dict(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert isinstance(result, dict)


def test_fallback_english_has_required_keys(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert "advisory_text"    in result
    assert "recommendations"  in result
    assert "source"           in result


def test_fallback_english_advisory_not_empty(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert len(result["advisory_text"]) > 10


def test_fallback_english_has_recommendations(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


def test_fallback_english_mentions_crop(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert "tomato" in result["advisory_text"].lower()


def test_fallback_english_mentions_disease(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert "blight" in result["advisory_text"].lower()


def test_fallback_source_is_set(llm, sample_advisory_args):
    result = llm.generate_advisory(**sample_advisory_args)
    assert result["source"] != ""


# ── Fallback Advisory Tests (Hindi) ──────────────────────

def test_fallback_hindi_returns_dict(llm, sample_advisory_args):
    sample_advisory_args["language"] = "hi"
    result = llm.generate_advisory(**sample_advisory_args)
    assert isinstance(result, dict)


def test_fallback_hindi_not_empty(llm, sample_advisory_args):
    sample_advisory_args["language"] = "hi"
    result = llm.generate_advisory(**sample_advisory_args)
    assert len(result["advisory_text"]) > 10


def test_fallback_hindi_has_recommendations(llm, sample_advisory_args):
    sample_advisory_args["language"] = "hi"
    result = llm.generate_advisory(**sample_advisory_args)
    assert len(result["recommendations"]) > 0


# ── Healthy Crop Tests ────────────────────────────────────

def test_healthy_crop_english(llm):
    result = llm.generate_advisory(
        disease_name="Tomato___healthy",
        confidence=0.95,
        crop_name="tomato",
        query="How is my crop?",
        context="Healthy crop management.",
        language="en",
        advisory_id="test-healthy-en",
    )
    assert "healthy" in result["advisory_text"].lower()


def test_healthy_crop_hindi(llm):
    result = llm.generate_advisory(
        disease_name="Tomato___healthy",
        confidence=0.95,
        crop_name="tomato",
        query="मेरी फसल कैसी है?",
        context="स्वस्थ फसल प्रबंधन।",
        language="hi",
        advisory_id="test-healthy-hi",
    )
    assert "स्वस्थ" in result["advisory_text"]


# ── Severity Tests ────────────────────────────────────────

def test_severity_high_confidence(llm):
    severity = llm._estimate_severity(0.90)
    assert severity == "severe"


def test_severity_medium_confidence(llm):
    severity = llm._estimate_severity(0.70)
    assert severity == "moderate"


def test_severity_low_confidence(llm):
    severity = llm._estimate_severity(0.50)
    assert severity == "mild"


# ── Context Extraction Tests ──────────────────────────────

def test_extract_recs_from_context(llm):
    context = "Apply Mancozeb 2g per litre.\nRemove infected leaves immediately.\nAvoid overhead irrigation."
    recs = llm._extract_recs_from_context(context)
    assert isinstance(recs, list)
    assert len(recs) > 0


def test_extract_recs_empty_context(llm):
    recs = llm._extract_recs_from_context("")
    assert isinstance(recs, list)
    assert len(recs) > 0  # Must return fallback
