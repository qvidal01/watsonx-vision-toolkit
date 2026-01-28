# Exceptions

Structured exception hierarchy for precise error handling.

## Exception Hierarchy

```
WatsonxVisionError (base)
├── LLMConnectionError      # Connection to LLM failed
├── LLMResponseError        # Invalid LLM response
├── LLMParseError           # JSON parsing failed
├── LLMTimeoutError         # Request timeout
├── DocumentAnalysisError   # Document analysis failed
├── ValidationError         # Validation logic error
└── ConfigurationError      # Invalid configuration
```

---

## WatsonxVisionError

Base exception for all toolkit errors.

```python
from watsonx_vision import WatsonxVisionError

try:
    result = llm.classify_document(image_data)
except WatsonxVisionError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `message` | `str` | Human-readable error message |
| `details` | `Any \| None` | Additional context |

---

## LLMConnectionError

Raised when connection to the LLM provider fails.

```python
from watsonx_vision import LLMConnectionError

try:
    result = llm.classify_document(image_data)
except LLMConnectionError as e:
    print(f"Connection failed: {e.message}")
```

**Common causes:**

- Network connectivity issues
- Provider service outage
- Invalid provider URL
- Firewall blocking requests
- DNS resolution failure

**Retryable:** Yes

---

## LLMResponseError

Raised when the LLM returns an unexpected or invalid response.

```python
from watsonx_vision import LLMResponseError

try:
    result = llm.classify_document(image_data)
except LLMResponseError as e:
    print(f"Invalid response: {e.message}")
    print(f"Raw response: {e.details}")
```

**Common causes:**

- Model returned empty response
- Response format unexpected
- Model error or hallucination
- Context length exceeded

**Retryable:** Sometimes (depends on cause)

---

## LLMParseError

Raised when JSON parsing of the LLM response fails.

```python
from watsonx_vision import LLMParseError

try:
    result = llm.extract_information(image_data)
except LLMParseError as e:
    print(f"Parse failed: {e.message}")
    if e.details:
        print(f"Raw content: {e.details.get('raw_content')}")
        print(f"Parse error: {e.details.get('error')}")
```

**Details dict:**

| Key | Type | Description |
|-----|------|-------------|
| `raw_content` | `str` | Raw LLM response |
| `error` | `str` | JSON parse error message |

**Common causes:**

- Model returned non-JSON text
- Malformed JSON in response
- Response truncated mid-JSON
- Markdown formatting in response

**Retryable:** Sometimes

---

## LLMTimeoutError

Raised when a request times out.

```python
from watsonx_vision import LLMTimeoutError

try:
    result = llm.analyze_image(image_data, prompt)
except LLMTimeoutError as e:
    print(f"Timeout: {e.message}")
```

**Common causes:**

- Large image processing
- Complex prompt requiring long generation
- Provider under heavy load
- Network latency issues

**Retryable:** Yes

---

## DocumentAnalysisError

Raised when document analysis fails.

```python
from watsonx_vision import DocumentAnalysisError

try:
    result = detector.validate_document(image_data)
except DocumentAnalysisError as e:
    print(f"Analysis failed: {e.message}")
```

**Common causes:**

- Corrupted image file
- Unsupported image format
- Image too small to analyze
- Image too large for model
- Unreadable document content

**Retryable:** No (usually data issue)

---

## ValidationError

Raised when cross-validation logic encounters an error.

```python
from watsonx_vision import ValidationError

try:
    result = validator.validate(app_data, doc_data)
except ValidationError as e:
    print(f"Validation error: {e.message}")
```

**Common causes:**

- Invalid input data format
- Missing required fields
- Incompatible data types
- Empty document data list

**Retryable:** No (data issue)

---

## ConfigurationError

Raised when configuration is invalid.

```python
from watsonx_vision import ConfigurationError

try:
    config = VisionLLMConfig(provider="invalid")
    llm = VisionLLM(config)
except ConfigurationError as e:
    print(f"Config error: {e.message}")
```

**Common causes:**

- Invalid provider name
- Missing required credentials
- Invalid parameter values
- Conflicting configuration options

**Retryable:** No (configuration issue)

---

## Usage Patterns

### Catch All Toolkit Errors

```python
from watsonx_vision import WatsonxVisionError

try:
    result = llm.classify_document(image_data)
except WatsonxVisionError as e:
    logging.error(f"Toolkit error: {e.message}")
```

### Specific Error Handling

```python
from watsonx_vision import (
    LLMConnectionError,
    LLMTimeoutError,
    LLMParseError,
    WatsonxVisionError
)

try:
    result = llm.classify_document(image_data)
except LLMConnectionError:
    # Retry or use fallback
    result = fallback_classify(image_data)
except LLMTimeoutError:
    # Increase timeout and retry
    result = classify_with_longer_timeout(image_data)
except LLMParseError as e:
    # Log raw response for debugging
    logging.error(f"Parse failed: {e.details}")
    result = None
except WatsonxVisionError as e:
    # Handle any other toolkit error
    logging.error(f"Unexpected error: {e.message}")
    raise
```

### With Retry Logic

```python
from watsonx_vision import (
    retry_llm_call, RetryConfig,
    LLMConnectionError, LLMTimeoutError
)

retry_config = RetryConfig(
    max_attempts=3,
    retry_exceptions=(LLMConnectionError, LLMTimeoutError)
)

try:
    result = retry_llm_call(
        lambda: llm.classify_document(image_data),
        retry_config
    )
except (LLMConnectionError, LLMTimeoutError) as e:
    # All retries exhausted
    logging.error(f"Failed after retries: {e.message}")
```

### Graceful Degradation

```python
from watsonx_vision import WatsonxVisionError

def safe_classify(image_data):
    """Classify with fallback to 'Unknown'."""
    try:
        result = llm.classify_document(image_data)
        return result.get("doc_type", "Unknown")
    except WatsonxVisionError as e:
        logging.warning(f"Classification failed: {e.message}")
        return "Unknown"
```

### Error Logging

```python
import logging
from watsonx_vision import WatsonxVisionError

logger = logging.getLogger(__name__)

try:
    result = llm.classify_document(image_data)
except WatsonxVisionError as e:
    logger.error(
        "Document classification failed",
        extra={
            "error_type": type(e).__name__,
            "message": e.message,
            "details": e.details
        },
        exc_info=True
    )
    raise
```
