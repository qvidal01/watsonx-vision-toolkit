# Retry

Retry utilities with exponential backoff.

## RetryConfig

Configuration for retry behavior.

```python
from watsonx_vision import RetryConfig

retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retry_exceptions=None
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

### Default Retry Exceptions

When `retry_exceptions=None`, retries on:

- `LLMConnectionError`
- `LLMTimeoutError`
- `ConnectionError`
- `TimeoutError`

---

## retry_with_backoff

Decorator for automatic retries.

```python
from watsonx_vision import retry_with_backoff, RetryConfig

config = RetryConfig(max_attempts=5, base_delay=2.0)

@retry_with_backoff(config)
def unreliable_function():
    # This will be retried on failure
    return api_call()
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `RetryConfig` | Retry configuration |

### Behavior

1. Executes the decorated function
2. On exception (if in retry_exceptions):
   - Logs the failure
   - Calculates delay: `base_delay * exponential_base^(attempt-1)`
   - Applies max_delay cap
   - Adds jitter if enabled
   - Sleeps and retries
3. After max_attempts, raises the last exception

---

## retry_llm_call

Function-based retry for non-decorator usage.

```python
from watsonx_vision import retry_llm_call, RetryConfig

config = RetryConfig(max_attempts=3)

def my_llm_call():
    return llm.analyze_image(image_data, prompt)

result = retry_llm_call(my_llm_call, config)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `Callable` | Function to retry |
| `config` | `RetryConfig` | Retry configuration |

**Returns:** Result of the function call

**Raises:** Last exception after all retries exhausted

---

## async_retry_with_backoff

Async decorator for automatic retries.

```python
from watsonx_vision import async_retry_with_backoff, RetryConfig

config = RetryConfig(max_attempts=3)

@async_retry_with_backoff(config)
async def async_api_call():
    return await some_async_operation()
```

---

## async_retry_llm_call

Async function-based retry.

```python
from watsonx_vision import async_retry_llm_call, RetryConfig

config = RetryConfig(max_attempts=3)

async def my_async_call():
    return await llm.classify_document_async(image_data)

result = await async_retry_llm_call(my_async_call, config)
```

---

## Delay Calculation

Delays follow exponential backoff:

```
delay = min(base_delay * exponential_base^(attempt-1), max_delay)
```

With jitter (if enabled):

```
jitter_amount = random.uniform(0, delay)
final_delay = delay + jitter_amount
```

### Example Progression

With `base_delay=1.0`, `exponential_base=2.0`, `max_delay=60.0`:

| Attempt | Base Delay | With Jitter (approx) |
|---------|------------|----------------------|
| 1 | 1.0s | 1.0-2.0s |
| 2 | 2.0s | 2.0-4.0s |
| 3 | 4.0s | 4.0-8.0s |
| 4 | 8.0s | 8.0-16.0s |
| 5 | 16.0s | 16.0-32.0s |

---

## Environment Variables

Configure via environment:

| Variable | Description | Default |
|----------|-------------|---------|
| `VISION_RETRY_ENABLED` | Enable retry | `true` |
| `VISION_RETRY_MAX_ATTEMPTS` | Max attempts | `3` |
| `VISION_RETRY_BASE_DELAY` | Base delay (seconds) | `1.0` |
| `VISION_RETRY_MAX_DELAY` | Max delay (seconds) | `60.0` |

---

## Example Usage

### Basic Decorator

```python
from watsonx_vision import retry_with_backoff, RetryConfig

config = RetryConfig(max_attempts=3, base_delay=1.0)

@retry_with_backoff(config)
def fetch_data():
    return requests.get("https://api.example.com/data").json()

result = fetch_data()  # Retries up to 3 times
```

### Custom Exceptions

```python
from watsonx_vision import retry_with_backoff, RetryConfig

config = RetryConfig(
    max_attempts=5,
    retry_exceptions=(ConnectionError, TimeoutError, ValueError)
)

@retry_with_backoff(config)
def custom_operation():
    # Retries on ConnectionError, TimeoutError, or ValueError
    return do_something()
```

### With LLM Operations

```python
from watsonx_vision import (
    VisionLLM, VisionLLMConfig, RetryConfig,
    retry_llm_call
)

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

retry_config = RetryConfig(max_attempts=3, base_delay=2.0)

def classify_document(image_data):
    def do_classify():
        return llm.classify_document(image_data)

    return retry_llm_call(do_classify, retry_config)

result = classify_document(image_data)
```

### Async Usage

```python
import asyncio
from watsonx_vision import async_retry_with_backoff, RetryConfig

config = RetryConfig(max_attempts=3)

@async_retry_with_backoff(config)
async def fetch_async():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            return await response.json()

result = asyncio.run(fetch_async())
```

### Logging Retries

```python
import logging

# Enable debug logging to see retry attempts
logging.basicConfig(level=logging.DEBUG)

@retry_with_backoff(config)
def operation():
    return do_something()

# Output:
# DEBUG: Attempt 1 failed: ConnectionError, retrying in 1.2s...
# DEBUG: Attempt 2 failed: ConnectionError, retrying in 2.5s...
# DEBUG: Attempt 3 succeeded
```
