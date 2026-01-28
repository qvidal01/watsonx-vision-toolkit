"""
Custom Exceptions for Watsonx Vision Toolkit

Provides specific exception types for different error scenarios
to enable proper error handling and debugging.
"""

from typing import Optional, Any


class WatsonxVisionError(Exception):
    """Base exception for all Watsonx Vision Toolkit errors"""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class LLMConnectionError(WatsonxVisionError):
    """Raised when connection to the LLM provider fails"""
    pass


class LLMResponseError(WatsonxVisionError):
    """Raised when LLM returns an invalid or unexpected response"""
    pass


class LLMParseError(WatsonxVisionError):
    """Raised when parsing LLM response fails (e.g., invalid JSON)"""
    pass


class LLMTimeoutError(WatsonxVisionError):
    """Raised when LLM request times out"""
    pass


class DocumentAnalysisError(WatsonxVisionError):
    """Raised when document analysis fails"""
    pass


class ValidationError(WatsonxVisionError):
    """Raised when validation logic encounters an error"""
    pass


class ConfigurationError(WatsonxVisionError):
    """Raised when configuration is invalid or incomplete"""
    pass
