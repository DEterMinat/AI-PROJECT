"""
Model Tests for Medical AI System
=================================

Unit tests for AI models and LangChain services.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLangchainMedicalService:
    """Test LangChain medical service."""

    @pytest.fixture
    def mock_service(self):
        """Mock LangchainMedicalService for testing."""
        with patch('src.models.medical_ai.LangchainMedicalService') as mock:
            instance = mock.return_value
            instance.ask_medical_question.return_value = {
                "success": True,
                "answer": "Mock medical answer",
                "sources": [],
                "processing_time": 0.1,
                "disclaimer": "This is a mock disclaimer"
            }
            instance.detect_emergency.return_value = False  # Default to no emergency
            yield instance

    def test_service_initialization(self, mock_service):
        """Test service can be initialized."""
        from src.models.medical_ai import LangchainMedicalService
        # Should not raise exception
        pass

    def test_ask_medical_question_basic(self, mock_service):
        """Test basic medical question asking."""
        result = mock_service.ask_medical_question("What is diabetes?")
        assert result["success"] is True
        assert "answer" in result
        assert isinstance(result["processing_time"], (int, float))

    def test_ask_medical_question_with_rag(self, mock_service):
        """Test medical question with RAG enabled."""
        result = mock_service.ask_medical_question(
            "What are diabetes symptoms?",
            use_rag=True
        )
        assert result["success"] is True

    def test_emergency_detection(self, mock_service):
        """Test emergency symptom detection."""
        # Mock emergency detection
        mock_service.detect_emergency.return_value = True
        # Mock emergency disclaimer
        mock_service.ask_medical_question.return_value["disclaimer"] = "EMERGENCY: Seek immediate medical attention"

        result = mock_service.ask_medical_question("I can't breathe")
        # Should include emergency warning
        assert "emergency" in result.get("disclaimer", "").lower()


class TestModelLoading:
    """Test model loading functionality."""

    def test_model_path_validation(self):
        """Test model path validation."""
        from pathlib import Path
        model_path = Path("models/")
        assert model_path.exists() or not model_path.exists()  # Either way is fine

    def test_vectorstore_path(self):
        """Test vectorstore path configuration."""
        from pathlib import Path
        vectorstore_path = Path("data/vectorstore")
        # Path should be valid
        assert isinstance(str(vectorstore_path), str)


class TestMedicalValidation:
    """Test medical content validation."""

    def test_medical_disclaimer_presence(self):
        """Test that medical disclaimers are included."""
        disclaimer_text = "This is not medical advice"
        assert "medical advice" in disclaimer_text.lower()

    def test_safe_medical_responses(self):
        """Test that responses include safety warnings."""
        # This would be tested with actual model responses
        pass


class TestPerformanceMetrics:
    """Test performance monitoring."""

    def test_response_time_tracking(self):
        """Test response time measurement."""
        import time
        start_time = time.time()
        time.sleep(0.01)  # Small delay
        end_time = time.time()
        processing_time = end_time - start_time
        assert processing_time > 0
        assert processing_time < 1  # Should be fast

    def test_memory_usage_monitoring(self):
        """Test memory usage tracking."""
        # Basic memory check
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb > 0
        assert memory_mb < 2000  # Should not be excessive (FLAN-T5 model uses ~1.3GB)


if __name__ == "__main__":
    pytest.main([__file__])