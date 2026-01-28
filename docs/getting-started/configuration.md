# Configuration

The toolkit can be configured programmatically or via environment variables.

## VisionLLMConfig

The main configuration class for the vision LLM:

```python
from watsonx_vision import VisionLLMConfig

config = VisionLLMConfig(
    provider="ollama",           # "ollama" or "watsonx"
    model_id="llava:latest",     # Model identifier
    api_key=None,                # API key (Watsonx only)
    url="http://localhost:11434", # Provider URL
    project_id=None,             # Project ID (Watsonx only)
    max_tokens=2000,             # Maximum response tokens
    temperature=0.0,             # Sampling temperature
    top_p=0.1                    # Nucleus sampling parameter
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"watsonx"` | LLM provider (`"watsonx"` or `"ollama"`) |
| `model_id` | `str` | See below | Model identifier |
| `api_key` | `str \| None` | `None` | API key for Watsonx |
| `url` | `str \| None` | `None` | Provider endpoint URL |
| `project_id` | `str \| None` | `None` | Watsonx project ID |
| `max_tokens` | `int` | `2000` | Maximum response tokens |
| `temperature` | `float` | `0.0` | Sampling temperature (0.0-1.0) |
| `top_p` | `float` | `0.1` | Nucleus sampling (0.0-1.0) |

### Default Models

- **Watsonx**: `meta-llama/llama-4-maverick-17b-128e-instruct-fp8`
- **Ollama**: `llava:latest`

## Environment Variables

Load configuration from environment variables using `from_env()`:

```python
config = VisionLLMConfig.from_env()
```

### Supported Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VISION_PROVIDER` | Provider name | `ollama` |
| `VISION_MODEL_ID` | Model identifier | `llava:13b` |
| `VISION_MAX_TOKENS` | Max response tokens | `2000` |
| `VISION_TEMPERATURE` | Sampling temperature | `0.0` |
| `VISION_TOP_P` | Nucleus sampling | `0.1` |

### Watsonx Variables

| Variable | Description |
|----------|-------------|
| `WATSONX_API_KEY` | IBM Cloud API key |
| `WATSONX_URL` | Watsonx endpoint URL |
| `WATSONX_PROJECT_ID` | Project/workspace ID |

### Ollama Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_HOST` | Alternative to OLLAMA_URL | - |

### Retry Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VISION_RETRY_ENABLED` | Enable retry logic | `true` |
| `VISION_RETRY_MAX_ATTEMPTS` | Maximum retry attempts | `3` |
| `VISION_RETRY_BASE_DELAY` | Base delay in seconds | `1.0` |
| `VISION_RETRY_MAX_DELAY` | Maximum delay in seconds | `60.0` |

### Cache Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VISION_CACHE_ENABLED` | Enable response caching | `false` |
| `VISION_CACHE_TTL` | Cache TTL in seconds | `3600` |
| `VISION_CACHE_MAX_SIZE` | Maximum cache entries | `1000` |
| `VISION_CACHE_BACKEND` | Backend type | `memory` |
| `VISION_CACHE_FILE_PATH` | File path for file backend | - |

## Using Prefixes

For multiple instances, use a prefix:

```bash
export PROD_WATSONX_API_KEY=prod-key-here
export DEV_WATSONX_API_KEY=dev-key-here
```

```python
prod_config = VisionLLMConfig.from_env(prefix="PROD_")
dev_config = VisionLLMConfig.from_env(prefix="DEV_")
```

## Explicit Overrides

Override environment variables with explicit values:

```python
config = VisionLLMConfig.from_env(
    max_tokens=4000,  # Override env variable
    temperature=0.2
)
```

## Cache Configuration

Configure response caching:

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

cache_config = CacheConfig(
    enabled=True,
    ttl=3600,           # 1 hour
    max_size=1000,      # Max entries
    backend="memory"    # or "file"
)

config = VisionLLMConfig.from_env()
llm = VisionLLM(config, cache_config=cache_config)
```

Or from environment:

```python
cache_config = CacheConfig.from_env()
```

## Retry Configuration

Configure retry behavior:

```python
from watsonx_vision import RetryConfig

retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)
```

## Complete Example

```bash
# .env file
VISION_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
VISION_MODEL_ID=llava:13b
VISION_CACHE_ENABLED=true
VISION_CACHE_TTL=7200
VISION_RETRY_MAX_ATTEMPTS=5
```

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

# Load all config from environment
config = VisionLLMConfig.from_env()
cache_config = CacheConfig.from_env()

# Create LLM with caching
llm = VisionLLM(config, cache_config=cache_config)

# Use it
image = VisionLLM.encode_image_to_base64("document.png")
result = llm.classify_document(image)
```
