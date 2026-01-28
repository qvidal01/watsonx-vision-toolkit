# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Examples Directory** - Practical code samples
  - `basic_classification.py` - Document type classification
  - `information_extraction.py` - Extracting structured data with presets
  - `fraud_detection.py` - Single and batch fraud detection
  - `cross_validation.py` - Multi-document consistency validation
  - `retry_configuration.py` - Configuring retry behavior
  - `complete_workflow.py` - End-to-end loan processing workflow
- **CI/CD Pipeline** - GitHub Actions workflow
  - Automated testing across Python 3.9-3.13
  - Ruff linting with auto-formatting checks
  - Mypy type checking for strict type safety
  - Coverage reporting with pytest-cov
- **Retry Logic with Exponential Backoff**
  - `RetryConfig` class for configurable retry behavior
  - `retry_with_backoff` decorator for automatic retries
  - `retry_llm_call` function for non-decorator usage
  - Exponential backoff with optional jitter
  - Configurable retry exceptions, delays, and max attempts
  - Integrated into `VisionLLM` and `CrossValidator` classes
- **Custom Exception Hierarchy** - Structured error handling
  - `WatsonxVisionError` - Base exception for all toolkit errors
  - `LLMConnectionError` - Connection failures to LLM providers
  - `LLMResponseError` - Invalid or unexpected LLM responses
  - `LLMParseError` - JSON parsing failures
  - `LLMTimeoutError` - Request timeout handling
  - `DocumentAnalysisError` - Document analysis failures
  - `ValidationError` - Cross-validation logic errors
  - `ConfigurationError` - Invalid configuration detection
- **Logging Infrastructure** - Structured logging throughout
  - Debug-level logging for validation steps
  - Error-level logging for failures with context
  - Warning-level logging for type coercion
- Comprehensive test suite for `cross_validator.py` (29 tests)
- Comprehensive test suite for `vision_llm.py` (36 tests)
- Test suite for `exceptions.py` (20 tests)
- Test coverage increased from ~40% to ~90%
- Comprehensive API reference documentation (`docs/API_REFERENCE.md`)

### Changed
- `VisionLLM.analyze_image()` now raises specific exceptions instead of generic errors
- `CrossValidator._llm_validate()` includes comprehensive error handling
- `FraudDetector.validate_document()` wraps errors in `DocumentAnalysisError`

### Fixed
- Corrected severity calculation test assertions to match implementation behavior

## [0.1.0] - 2025-01-27

### Added
- **VisionLLM Module** - Unified interface for vision-based document analysis
  - Support for IBM Watsonx AI and Ollama providers
  - Document classification with customizable document types
  - Information extraction with configurable fields
  - Authenticity validation with layout and field scoring
  - Base64 image encoding utility

- **FraudDetector Module** - Document authenticity validation
  - Vision-based fraud detection with confidence scoring
  - Severity levels: NONE, LOW, MEDIUM, HIGH, CRITICAL
  - Batch document validation
  - Summary report generation
  - `SpecializedFraudDetector` for document-specific validation:
    - Tax returns
    - Bank statements
    - Passports
    - Driver's licenses

- **CrossValidator Module** - Multi-document consistency checking
  - LLM-based intelligent comparison
  - Field severity mapping (CRITICAL, HIGH, MEDIUM, LOW)
  - Support for custom field definitions
  - Batch validation for multiple document packages
  - Human-readable report generation with severity icons
  - `FinancialCrossValidator` for financial document validation

- **DecisionEngine Module** - Multi-criteria weighted decision making
  - Configurable approval/rejection thresholds
  - Custom criterion support with evaluator functions
  - Integration with fraud detection and cross-validation results
  - Decision statuses: APPROVED, REJECTED, PENDING_REVIEW, NEEDS_MORE_INFO
  - `LoanDecisionEngine` with pre-configured loan criteria:
    - Age verification
    - Income requirements
    - Debt-to-income ratio

- **Project Infrastructure**
  - Zero required dependencies (provider packages are optional)
  - Full type hints with strict mypy configuration
  - Comprehensive README with API documentation
  - MIT License

### Technical Details
- Python 3.9 - 3.13 support
- Modern build system using Hatchling (PEP 517/518)
- Optional dependencies for Watsonx (`langchain-ibm`) and Ollama (`langchain-ollama`)
- Development tools: pytest, ruff, mypy, black

[Unreleased]: https://github.com/qvidal01/watsonx-vision-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qvidal01/watsonx-vision-toolkit/releases/tag/v0.1.0
