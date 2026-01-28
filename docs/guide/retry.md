# Retry Logic

The toolkit includes configurable retry logic with exponential backoff for handling transient failures.

## Overview

Retry logic automatically retries failed operations, which is essential for:

- **Network issues** - Temporary connection failures
- **Rate limiting** - API rate limit responses
- **Server errors** - 5xx errors from providers
- **Timeouts** - Slow responses under load

## Quick Start

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, RetryConfig

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0
)

# Create LLM with retry config
config = VisionLLMConfig.from_env()
config.retry_config = retry_config

llm = VisionLLM(config)
```

## RetryConfig

```python
from watsonx_vision import RetryConfig

retry_config = RetryConfig(
    max_attempts=3,           # Maximum retry attempts
    base_delay=1.0,           # Initial delay in seconds
    max_delay=60.0,           # Maximum delay cap
    exponential_base=2.0,     # Exponential multiplier
    jitter=True,              # Add random jitter
    retry_exceptions=None     # Custom exception types to retry
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_attempts` | `int` | `3` | Total attempts including first |
| `base_delay` | `float` | `1.0` | Initial delay in seconds |
| `max_delay` | `float` | `60.0` | Maximum delay cap |
| `exponential_base` | `float` | `2.0` | Delay multiplier each attempt |
| `jitter` | `bool` | `True` | Add random jitter to delays |
| `retry_exceptions` | `tuple \| None` | `None` | Exception types to retry |

## Exponential Backoff

Delays increase exponentially between attempts:

```
Attempt 1: Immediate
Attempt 2: base_delay * exponential_base^0 = 1.0s
Attempt 3: base_delay * exponential_base^1 = 2.0s
Attempt 4: base_delay * exponential_base^2 = 4.0s
...
```

With jitter, a random factor (0-100% of delay) is added to prevent thundering herd.

## Decorator Usage

Use the decorator for custom functions:

```python
from watsonx_vision import retry_with_backoff, RetryConfig

config = RetryConfig(max_attempts=5, base_delay=2.0)

@retry_with_backoff(config)
def call_external_api():
    # This will be retried on failure
    return requests.get("https://api.example.com/data")
```

## Function Usage

For non-decorator usage:

```python
from watsonx_vision import retry_llm_call, RetryConfig

config = RetryConfig(max_attempts=3)

def my_llm_call():
    return llm.analyze_image(image_data, prompt)

result = retry_llm_call(my_llm_call, config)
```

## Async Support

For async operations:

```python
from watsonx_vision import async_retry_with_backoff, async_retry_llm_call, RetryConfig

config = RetryConfig(max_attempts=3)

@async_retry_with_backoff(config)
async def async_api_call():
    return await some_async_operation()

# Or function-based
result = await async_retry_llm_call(async_operation, config)
```

## Custom Exceptions

Specify which exceptions should trigger retry:

```python
from watsonx_vision import RetryConfig, LLMConnectionError, LLMTimeoutError

retry_config = RetryConfig(
    max_attempts=3,
    retry_exceptions=(LLMConnectionError, LLMTimeoutError, ConnectionError)
)
```

By default, retries occur on:

- `LLMConnectionError`
- `LLMTimeoutError`
- `ConnectionError`
- `TimeoutError`

## Environment Variables

Configure via environment:

```bash
export VISION_RETRY_ENABLED=true
export VISION_RETRY_MAX_ATTEMPTS=5
export VISION_RETRY_BASE_DELAY=2.0
export VISION_RETRY_MAX_DELAY=120.0
```

## Logging

Retry attempts are logged for debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now retry attempts will be logged:
# DEBUG: Attempt 1 failed, retrying in 1.0s...
# DEBUG: Attempt 2 failed, retrying in 2.0s...
```

## Best Practices

### 1. Set Reasonable Limits

```python
# Production settings
retry_config = RetryConfig(
    max_attempts=3,      # Don't retry forever
    base_delay=1.0,      # Start with 1 second
    max_delay=30.0       # Cap at 30 seconds
)
```

### 2. Use Jitter

Always enable jitter in production to prevent thundering herd:

```python
retry_config = RetryConfig(jitter=True)  # Default
```

### 3. Handle Non-Retryable Errors

Some errors shouldn't be retried (auth failures, invalid input):

```python
from watsonx_vision import (
    LLMConnectionError,  # Retryable
    LLMTimeoutError,     # Retryable
    ConfigurationError,  # NOT retryable
    LLMParseError        # NOT retryable
)
```

### 4. Monitor Retry Patterns

```python
import logging

# Set up logging to track retry frequency
logging.getLogger("watsonx_vision.retry").setLevel(logging.INFO)
```

## Example: Robust Document Processing

```python
from watsonx_vision import (
    VisionLLM, VisionLLMConfig, RetryConfig,
    LLMConnectionError, LLMTimeoutError
)
import logging

logging.basicConfig(level=logging.INFO)

# Configure robust retry
retry_config = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    jitter=True
)

config = VisionLLMConfig.from_env()
config.retry_config = retry_config

llm = VisionLLM(config)

def process_document(path):
    """Process a document with automatic retry."""
    try:
        image = VisionLLM.encode_image_to_base64(path)
        result = llm.classify_document(image)
        return {"success": True, "result": result}
    except (LLMConnectionError, LLMTimeoutError) as e:
        # Retries exhausted
        return {"success": False, "error": str(e)}

# Process batch
documents = ["doc1.png", "doc2.png", "doc3.png"]
results = [process_document(doc) for doc in documents]

successful = sum(1 for r in results if r["success"])
print(f"Processed {successful}/{len(documents)} documents successfully")
```
