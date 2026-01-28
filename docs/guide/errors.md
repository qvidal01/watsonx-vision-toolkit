# Error Handling

The toolkit provides a structured exception hierarchy for precise error handling.

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

## Exception Types

### WatsonxVisionError

Base exception for all toolkit errors. Catch this for general error handling.

```python
from watsonx_vision import WatsonxVisionError

try:
    result = llm.classify_document(image_data)
except WatsonxVisionError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

### LLMConnectionError

Raised when connection to the LLM provider fails.

```python
from watsonx_vision import LLMConnectionError

try:
    result = llm.classify_document(image_data)
except LLMConnectionError as e:
    print(f"Connection failed: {e.message}")
    # Retry or fallback logic
```

**Common causes:**

- Network issues
- Provider service down
- Invalid URL
- Firewall blocking

### LLMResponseError

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
- Model error/hallucination

### LLMParseError

Raised when JSON parsing of the LLM response fails.

```python
from watsonx_vision import LLMParseError

try:
    result = llm.extract_information(image_data)
except LLMParseError as e:
    print(f"Parse failed: {e.message}")
    print(f"Raw content: {e.details.get('raw_content')}")
    print(f"Parse error: {e.details.get('error')}")
```

**Common causes:**

- Model returned non-JSON response
- Malformed JSON in response
- Response truncated

### LLMTimeoutError

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
- Complex prompt
- Provider overloaded

### DocumentAnalysisError

Raised when document analysis fails.

```python
from watsonx_vision import DocumentAnalysisError

try:
    result = detector.validate_document(image_data)
except DocumentAnalysisError as e:
    print(f"Analysis failed: {e.message}")
```

**Common causes:**

- Corrupted image
- Unsupported format
- Image too small/large

### ValidationError

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

### ConfigurationError

Raised when configuration is invalid.

```python
from watsonx_vision import ConfigurationError

try:
    config = VisionLLMConfig(provider="invalid")
except ConfigurationError as e:
    print(f"Config error: {e.message}")
```

**Common causes:**

- Invalid provider name
- Missing required credentials
- Invalid parameter values

## Exception Properties

All exceptions have these properties:

| Property | Type | Description |
|----------|------|-------------|
| `message` | `str` | Human-readable error message |
| `details` | `Any \| None` | Additional context (dict, str, etc.) |

## Error Handling Patterns

### Specific to General

Handle specific exceptions first:

```python
from watsonx_vision import (
    WatsonxVisionError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMParseError
)

try:
    result = llm.classify_document(image_data)
except LLMConnectionError:
    # Handle connection issues - maybe retry
    pass
except LLMTimeoutError:
    # Handle timeout - maybe increase timeout
    pass
except LLMParseError as e:
    # Handle parse error - log raw response
    logging.error(f"Parse failed: {e.details}")
except WatsonxVisionError as e:
    # Handle any other toolkit error
    logging.error(f"Error: {e.message}")
```

### With Retry

Combine with retry logic:

```python
from watsonx_vision import (
    retry_llm_call, RetryConfig,
    LLMConnectionError, LLMTimeoutError
)

retry_config = RetryConfig(max_attempts=3)

def classify_with_retry(image_data):
    def do_classify():
        return llm.classify_document(image_data)

    return retry_llm_call(do_classify, retry_config)

try:
    result = classify_with_retry(image_data)
except (LLMConnectionError, LLMTimeoutError):
    # All retries exhausted
    print("Failed after 3 attempts")
```

### Graceful Degradation

Return defaults on failure:

```python
def safe_classify(image_data, default="Unknown"):
    try:
        result = llm.classify_document(image_data)
        return result.get("doc_type", default)
    except WatsonxVisionError as e:
        logging.warning(f"Classification failed: {e.message}")
        return default
```

### Logging Errors

Log errors with context:

```python
import logging

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
        }
    )
    raise
```

## Example: Robust Document Processing

```python
from watsonx_vision import (
    VisionLLM, VisionLLMConfig, FraudDetector,
    WatsonxVisionError, LLMConnectionError, LLMTimeoutError,
    LLMParseError, DocumentAnalysisError
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)
detector = FraudDetector(llm)

def process_document(path):
    """Process a document with comprehensive error handling."""
    try:
        image = VisionLLM.encode_image_to_base64(path)

        # Classification
        try:
            doc_type = llm.classify_document(image)
        except LLMParseError:
            logger.warning(f"Classification parse error for {path}")
            doc_type = {"doc_type": "Unknown"}

        # Fraud detection
        try:
            fraud_result = detector.validate_document(image, path)
        except DocumentAnalysisError as e:
            logger.warning(f"Fraud detection failed for {path}: {e.message}")
            fraud_result = None

        return {
            "path": path,
            "doc_type": doc_type.get("doc_type"),
            "fraud_valid": fraud_result.valid if fraud_result else None,
            "status": "processed"
        }

    except LLMConnectionError as e:
        logger.error(f"Connection error for {path}: {e.message}")
        return {"path": path, "status": "connection_error"}

    except LLMTimeoutError as e:
        logger.error(f"Timeout for {path}: {e.message}")
        return {"path": path, "status": "timeout"}

    except WatsonxVisionError as e:
        logger.error(f"Processing error for {path}: {e.message}")
        return {"path": path, "status": "error", "error": e.message}

    except Exception as e:
        logger.exception(f"Unexpected error for {path}")
        return {"path": path, "status": "unexpected_error", "error": str(e)}

# Process documents
documents = ["doc1.png", "doc2.png", "doc3.png"]
results = [process_document(doc) for doc in documents]

# Summary
successful = sum(1 for r in results if r["status"] == "processed")
print(f"Processed {successful}/{len(documents)} documents successfully")
```
