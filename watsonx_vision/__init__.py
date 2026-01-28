"""
Watsonx Vision Toolkit

A reusable Python package for vision-based document analysis, fraud detection,
and multi-document validation using IBM Watsonx AI or compatible LLM providers.

Extracted from IBM Watsonx Loan Preprocessing Agents project.
"""

__version__ = "0.1.0"
__author__ = "AIQSO - Quinn Vidal"
__email__ = "quinn@aiqso.io"

from .vision_llm import VisionLLM, VisionLLMConfig
from .fraud_detector import FraudDetector, FraudResult
from .cross_validator import CrossValidator, ValidationResult
from .decision_engine import DecisionEngine, Decision
from .exceptions import (
    WatsonxVisionError,
    LLMConnectionError,
    LLMResponseError,
    LLMParseError,
    LLMTimeoutError,
    DocumentAnalysisError,
    ValidationError,
    ConfigurationError,
)
from .retry import (
    RetryConfig,
    retry_with_backoff,
    retry_llm_call,
    async_retry_with_backoff,
    async_retry_llm_call,
)
from .cache import (
    CacheConfig,
    ResponseCache,
    CacheStats,
)

__all__ = [
    # Core classes
    "VisionLLM",
    "VisionLLMConfig",
    "FraudDetector",
    "FraudResult",
    "CrossValidator",
    "ValidationResult",
    "DecisionEngine",
    "Decision",
    # Exceptions
    "WatsonxVisionError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMParseError",
    "LLMTimeoutError",
    "DocumentAnalysisError",
    "ValidationError",
    "ConfigurationError",
    # Retry utilities
    "RetryConfig",
    "retry_with_backoff",
    "retry_llm_call",
    # Async retry utilities
    "async_retry_with_backoff",
    "async_retry_llm_call",
    # Cache utilities
    "CacheConfig",
    "ResponseCache",
    "CacheStats",
]
