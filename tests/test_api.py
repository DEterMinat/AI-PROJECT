"""
API Tests for Medical AI System
===============================

Unit tests for FastAPI endpoints and functionality.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.api.integrated_medical_api import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "online"

    def test_health_check_content(self, client):
        """Test health check response content."""
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "online"


class TestMedicalQA:
    """Test medical question answering endpoints."""

    def test_ask_medical_question(self, client):
        """Test basic medical question endpoint."""
        payload = {
            "question": "What are the symptoms of diabetes?",
            "user_id": "test_user"
        }
        response = client.post("/predict", json=payload)  # Use correct endpoint
        assert response.status_code in [200, 503]  # 503 if model not loaded

        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "answer" in data
            assert "processing_time" in data

    def test_ask_empty_question(self, client):
        """Test empty question handling."""
        payload = {
            "question": "",
            "user_id": "test_user"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_ask_long_question(self, client):
        """Test long question handling."""
        long_question = "What are the symptoms of " * 100
        payload = {
            "question": long_question,
            "user_id": "test_user"
        }
        response = client.post("/predict", json=payload)
        # Should handle gracefully
        assert response.status_code in [200, 503, 413]


class TestSystemStatus:
    """Test system status endpoints."""

    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/status")
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "fastapi" in data
            assert "langchain" in data
            assert "timestamp" in data["fastapi"]  # timestamp is inside fastapi object


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_method(self, client):
        """Test invalid HTTP method."""
        response = client.patch("/predict")
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__])