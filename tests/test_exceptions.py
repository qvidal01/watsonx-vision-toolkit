"""Tests for custom exceptions module"""

import pytest
from watsonx_vision.exceptions import (
    WatsonxVisionError,
    LLMConnectionError,
    LLMResponseError,
    LLMParseError,
    LLMTimeoutError,
    DocumentAnalysisError,
    ValidationError,
    ConfigurationError,
)


class TestWatsonxVisionError:
    """Tests for base WatsonxVisionError"""

    def test_basic_error(self):
        """Test creating basic error"""
        error = WatsonxVisionError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details is None

    def test_error_with_details(self):
        """Test error with details"""
        error = WatsonxVisionError("Error occurred", details={"code": 500})
        assert "Error occurred" in str(error)
        assert "code" in str(error)
        assert error.details == {"code": 500}

    def test_error_inherits_from_exception(self):
        """Test that error inherits from Exception"""
        error = WatsonxVisionError("Test")
        assert isinstance(error, Exception)


class TestLLMConnectionError:
    """Tests for LLMConnectionError"""

    def test_connection_error(self):
        """Test creating connection error"""
        error = LLMConnectionError("Failed to connect to Watsonx")
        assert "connect" in str(error).lower()
        assert isinstance(error, WatsonxVisionError)

    def test_connection_error_with_details(self):
        """Test connection error with endpoint details"""
        error = LLMConnectionError(
            "Connection refused",
            details={"url": "https://api.example.com", "timeout": 30}
        )
        assert error.details["url"] == "https://api.example.com"


class TestLLMResponseError:
    """Tests for LLMResponseError"""

    def test_response_error(self):
        """Test creating response error"""
        error = LLMResponseError("Invalid response from LLM")
        assert isinstance(error, WatsonxVisionError)
        assert "response" in str(error).lower()

    def test_response_error_with_raw_content(self):
        """Test response error with raw content details"""
        error = LLMResponseError(
            "Unexpected format",
            details={"raw_content": "not json", "status": 200}
        )
        assert error.details["raw_content"] == "not json"


class TestLLMParseError:
    """Tests for LLMParseError"""

    def test_parse_error(self):
        """Test creating parse error"""
        error = LLMParseError("Failed to parse JSON")
        assert isinstance(error, WatsonxVisionError)

    def test_parse_error_with_content(self):
        """Test parse error with content that failed"""
        error = LLMParseError(
            "Invalid JSON",
            details={"content": "{broken json", "error": "JSONDecodeError"}
        )
        assert "broken" in str(error.details["content"])


class TestLLMTimeoutError:
    """Tests for LLMTimeoutError"""

    def test_timeout_error(self):
        """Test creating timeout error"""
        error = LLMTimeoutError("Request timed out after 30s")
        assert isinstance(error, WatsonxVisionError)
        assert "timed out" in str(error).lower()

    def test_timeout_error_with_duration(self):
        """Test timeout error with duration details"""
        error = LLMTimeoutError(
            "Request timed out",
            details={"duration_ms": 30000, "max_allowed": 30000}
        )
        assert error.details["duration_ms"] == 30000


class TestDocumentAnalysisError:
    """Tests for DocumentAnalysisError"""

    def test_analysis_error(self):
        """Test creating document analysis error"""
        error = DocumentAnalysisError("Failed to analyze passport.png")
        assert isinstance(error, WatsonxVisionError)

    def test_analysis_error_with_filename(self):
        """Test analysis error with filename details"""
        error = DocumentAnalysisError(
            "OCR failed",
            details={"filename": "blurry.png", "reason": "low resolution"}
        )
        assert error.details["filename"] == "blurry.png"


class TestValidationError:
    """Tests for ValidationError"""

    def test_validation_error(self):
        """Test creating validation error"""
        error = ValidationError("Cross-validation failed")
        assert isinstance(error, WatsonxVisionError)

    def test_validation_error_with_fields(self):
        """Test validation error with field details"""
        error = ValidationError(
            "Inconsistent data",
            details={"fields": ["name", "dob"], "severity": "high"}
        )
        assert "name" in error.details["fields"]


class TestConfigurationError:
    """Tests for ConfigurationError"""

    def test_configuration_error(self):
        """Test creating configuration error"""
        error = ConfigurationError("Missing API key")
        assert isinstance(error, WatsonxVisionError)

    def test_configuration_error_with_missing_fields(self):
        """Test configuration error with missing field details"""
        error = ConfigurationError(
            "Invalid configuration",
            details={"missing": ["api_key", "project_id"]}
        )
        assert "api_key" in error.details["missing"]


class TestExceptionHierarchy:
    """Tests for exception hierarchy"""

    def test_all_exceptions_inherit_from_base(self):
        """Test all custom exceptions inherit from WatsonxVisionError"""
        exceptions = [
            LLMConnectionError("test"),
            LLMResponseError("test"),
            LLMParseError("test"),
            LLMTimeoutError("test"),
            DocumentAnalysisError("test"),
            ValidationError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, WatsonxVisionError)
            assert isinstance(exc, Exception)

    def test_exceptions_can_be_raised_and_caught(self):
        """Test exceptions can be properly raised and caught"""
        with pytest.raises(LLMConnectionError):
            raise LLMConnectionError("Connection failed")

        with pytest.raises(WatsonxVisionError):
            raise LLMParseError("Parse failed")

    def test_exception_chaining(self):
        """Test exception chaining works"""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise LLMParseError("Wrapped error", details=str(e)) from e
        except LLMParseError as wrapped:
            assert wrapped.__cause__ is not None
            assert isinstance(wrapped.__cause__, ValueError)
