"""Watsonx Vision Toolkit

A reusable Python package for vision-based document analysis, fraud detection,
and multi-document validation using IBM Watsonx AI or compatible LLM providers.

Extracted from IBM Watsonx Loan Preprocessing Agents project.
"""

__version__ = "0.2.1"
__author__ = "AIQSO - Quinn Vidal"
__email__ = "quinn@aiqso.io"

from .cache import (
    CacheConfig,
    CacheStats,
    ResponseCache,
)
from .cross_validator import CrossValidator, ValidationResult
from .decision_engine import Decision, DecisionEngine
from .exceptions import (
    ConfigurationError,
    DocumentAnalysisError,
    LLMConnectionError,
    LLMParseError,
    LLMResponseError,
    LLMTimeoutError,
    ValidationError,
    WatsonxVisionError,
)
from .fraud_detector import FraudDetector, FraudResult
from .retry import (
    RetryConfig,
    async_retry_llm_call,
    async_retry_with_backoff,
    retry_llm_call,
    retry_with_backoff,
)
from .vision_llm import LLMProvider, VisionLLM, VisionLLMConfig

__all__ = [
    # Core classes
    "LLMProvider",
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
