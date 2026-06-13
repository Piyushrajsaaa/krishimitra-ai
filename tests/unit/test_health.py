"""
KrishiMitra AI — Unit Tests for Health Endpoint
Run with: pytest tests/unit/test_health.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to KrishiMitra AI 🌾"
    assert "docs" in data
    assert "health" in data


def test_health_endpoint_status():
    """Test health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_fields():
    """Test health endpoint returns all required fields."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "KrishiMitra AI"
    assert data["version"] == "1.0.0"
    assert data["environment"] == "development"
    assert "timestamp" in data


def test_health_endpoint_content_type():
    """Test health endpoint returns JSON."""
    response = client.get("/health")
    assert response.headers["content-type"] == "application/json"


def test_invalid_endpoint_returns_404():
    """Test unknown route returns 404."""
    response = client.get("/invalid-route")
    assert response.status_code == 404
