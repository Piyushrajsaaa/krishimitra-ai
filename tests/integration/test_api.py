"""
KrishiMitra AI -- Integration Tests for API Endpoints
Run: pytest tests/integration/ -v
"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def make_test_image(color=(34, 139, 34), fmt="JPEG"):
    """Creates an in-memory test image."""
    img = Image.new("RGB", (300, 300), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ── Health Tests ──────────────────────────────────────────

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert "KrishiMitra" in response.json()["message"]


# ── Advisory Endpoint Tests ───────────────────────────────

def test_advisory_valid_image_english():
    """Valid image with English language returns 200."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"query": "My tomato has spots", "language": "en", "crop_name": "tomato"}
    )
    assert response.status_code == 200


def test_advisory_response_has_required_fields():
    """Advisory response must contain all required fields."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "en"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "advisory_id"       in data
    assert "disease"           in data
    assert "advisory_text"     in data
    assert "recommendations"   in data
    assert "confidence_score"  in data
    assert "language"          in data
    assert "escalate_to_expert" in data
    assert "disclaimer"        in data


def test_advisory_confidence_in_range():
    """Confidence score must be between 0 and 1."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "en"}
    )
    data = response.json()
    assert 0.0 <= data["confidence_score"] <= 1.0


def test_advisory_disease_info_present():
    """Disease info block must be present and valid."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "en"}
    )
    data = response.json()
    assert data["disease"] is not None
    assert "disease_name" in data["disease"]
    assert "confidence"   in data["disease"]
    assert "severity"     in data["disease"]


def test_advisory_valid_image_hindi():
    """Valid image with Hindi language returns 200."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "hi"}
    )
    assert response.status_code == 200
    assert response.json()["language"] == "hi"


def test_advisory_invalid_file_type():
    """Non-image file must return 400."""
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
    response = client.post(
        "/advisory",
        files={"image": ("doc.pdf", fake_pdf, "application/pdf")},
        data={"language": "en"}
    )
    assert response.status_code == 400


def test_advisory_missing_image():
    """Missing image must return 422."""
    response = client.post(
        "/advisory",
        data={"language": "en"}
    )
    assert response.status_code == 422


def test_advisory_png_image():
    """PNG image must be accepted."""
    buf = make_test_image(fmt="PNG")
    response = client.post(
        "/advisory",
        files={"image": ("leaf.png", buf, "image/png")},
        data={"language": "en"}
    )
    assert response.status_code == 200


def test_advisory_recommendations_is_list():
    """Recommendations must be a list."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "en"}
    )
    data = response.json()
    assert isinstance(data["recommendations"], list)


def test_advisory_invalid_language_defaults_to_en():
    """Invalid language code must default to English."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "fr"}
    )
    assert response.status_code == 200
    assert response.json()["language"] == "en"


def test_advisory_disclaimer_present():
    """Disclaimer must always be present."""
    buf = make_test_image()
    response = client.post(
        "/advisory",
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
        data={"language": "en"}
    )
    data = response.json()
    assert len(data["disclaimer"]) > 0


# ── Feedback Endpoint Tests ───────────────────────────────

def test_feedback_thumbs_up():
    """Thumbs up feedback must return 200."""
    response = client.post(
        "/feedback",
        json={"advisory_id": "test-001", "rating": 1}
    )
    assert response.status_code == 200


def test_feedback_thumbs_down():
    """Thumbs down feedback must return 200."""
    response = client.post(
        "/feedback",
        json={"advisory_id": "test-001", "rating": 0}
    )
    assert response.status_code == 200


def test_feedback_with_comment():
    """Feedback with comment must return 200."""
    response = client.post(
        "/feedback",
        json={"advisory_id": "test-001", "rating": 1, "comment": "Very helpful!"}
    )
    assert response.status_code == 200
    assert "advisory_id" in response.json()


def test_feedback_invalid_rating():
    """Invalid rating must return 422."""
    response = client.post(
        "/feedback",
        json={"advisory_id": "test-001", "rating": 5}
    )
    assert response.status_code == 422


def test_feedback_missing_advisory_id():
    """Missing advisory_id must return 422."""
    response = client.post(
        "/feedback",
        json={"rating": 1}
    )
    assert response.status_code == 422
