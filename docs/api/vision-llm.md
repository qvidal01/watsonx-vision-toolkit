# VisionLLM

The main class for vision-based document analysis.

## VisionLLMConfig

Configuration dataclass for VisionLLM.

```python
from watsonx_vision import VisionLLMConfig, LLMProvider

config = VisionLLMConfig(
    provider=LLMProvider.WATSONX,  # or LLMProvider.OLLAMA
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    api_key="your-api-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project-id",
    max_tokens=2000,
    temperature=0.0,
    top_p=0.1
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `LLMProvider \| str` | `WATSONX` | LLM provider to use |
| `model_id` | `str` | Provider default | Model identifier |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `url` | `str \| None` | `None` | Provider URL |
| `project_id` | `str \| None` | `None` | Project/workspace ID |
| `max_tokens` | `int` | `2000` | Maximum response tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `top_p` | `float` | `0.1` | Nucleus sampling parameter |
| `retry_config` | `RetryConfig \| None` | `None` | Retry configuration |

### Class Methods

#### from_env

Load configuration from environment variables.

```python
config = VisionLLMConfig.from_env(prefix="")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | `""` | Environment variable prefix |

---

## VisionLLM

Main class for vision-based document analysis.

### Constructor

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

config = VisionLLMConfig(...)
cache_config = CacheConfig(enabled=True)  # Optional

llm = VisionLLM(config, cache_config=cache_config)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | `VisionLLMConfig` | Yes | Configuration object |
| `cache_config` | `CacheConfig \| None` | No | Cache configuration |

**Raises:**

- `ImportError`: If required provider packages are not installed
- `ConfigurationError`: If configuration is invalid

---

### analyze_image

Analyze an image with a custom prompt.

```python
result = llm.analyze_image(
    image_data="data:image/png;base64,...",
    prompt="Describe this document",
    system_prompt="You are a document analysis assistant",
    parse_json=True,
    use_cache=True
)
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_data` | `str` | Yes | - | Base64-encoded image data URI |
| `prompt` | `str` | Yes | - | User prompt for analysis |
| `system_prompt` | `str \| None` | No | `None` | Optional system context |
| `parse_json` | `bool` | No | `True` | Parse response as JSON |
| `use_cache` | `bool` | No | `True` | Use response cache |

**Returns:** `Dict` if `parse_json=True`, else `str`

**Raises:**

- `LLMConnectionError`: Connection to provider failed
- `LLMResponseError`: Invalid response from LLM
- `LLMParseError`: Failed to parse JSON response
- `LLMTimeoutError`: Request timed out

**Async variant:** `analyze_image_async()`

---

### classify_document

Classify a document image into predefined types.

```python
result = llm.classify_document(
    image_data="data:image/png;base64,...",
    document_types=["Passport", "Driver's License", "Tax Return"]
)
# Returns: {"doc_type": "Passport"}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |
| `document_types` | `List[str] \| None` | No | Custom document types |

**Default Document Types:**

- Driving License, Passport, SSN, Utility Bill, Salary Slip
- ITR (Income Tax Return), Bank Account Statement, Tax Return
- Articles of Incorporation, Personal Financial Statement, Others

**Returns:** `Dict[str, str]` with `doc_type` key

**Async variant:** `classify_document_async()`

---

### extract_information

Extract structured information from a document image.

```python
result = llm.extract_information(
    image_data="data:image/png;base64,...",
    fields=["Name", "Date of Birth", "Address"],
    date_format="YYYY-MM-DD"
)
# Returns: {"name": "John Doe", "dob": "1990-01-15", "address": "..."}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_data` | `str` | Yes | - | Base64-encoded image data URI |
| `fields` | `List[str] \| None` | No | Default PII | Fields to extract |
| `date_format` | `str` | No | `"YYYY-MM-DD"` | Output date format |

**Returns:** `Dict[str, Any]` with extracted fields

**Async variant:** `extract_information_async()`

---

### validate_authenticity

Validate document authenticity using vision analysis.

```python
result = llm.validate_authenticity(image_data="data:image/png;base64,...")
# Returns:
# {
#     "valid": True,
#     "reason": "Document appears authentic",
#     "layout_score": 95,
#     "field_score": 90,
#     "forgery_signs": []
# }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |

**Returns:** `Dict` with:

| Key | Type | Description |
|-----|------|-------------|
| `valid` | `bool` | Whether document appears authentic |
| `reason` | `str` | Explanation |
| `layout_score` | `int` | Layout consistency score (0-100) |
| `field_score` | `int` | Field formatting score (0-100) |
| `forgery_signs` | `List[str]` | Issues found |

**Async variant:** `validate_authenticity_async()`

---

### encode_image_to_base64

Static method to encode a local image file to base64 data URI.

```python
data_uri = VisionLLM.encode_image_to_base64(
    image_path="/path/to/document.png",
    mime_type="image/png"  # Optional, auto-detected
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | `str` | Yes | Path to image file |
| `mime_type` | `str \| None` | No | MIME type (auto-detected) |

**Returns:** `str` - Base64-encoded data URI

**Supported formats:** PNG, JPG, JPEG, GIF, WEBP, PDF

---

### Cache Properties and Methods

#### cache

Get the cache object (if enabled).

```python
cache = llm.cache  # ResponseCache or None
```

#### cache_enabled

Check if caching is enabled.

```python
if llm.cache_enabled:
    print("Caching is active")
```

#### cache_stats

Get cache statistics.

```python
stats = llm.cache_stats()
print(f"Hits: {stats.hits}, Misses: {stats.misses}")
```

**Returns:** `CacheStats` or `None` if cache not enabled

#### cache_clear

Clear the cache.

```python
llm.cache_clear()
```

---

## LLMProvider Enum

```python
from watsonx_vision.vision_llm import LLMProvider

class LLMProvider(Enum):
    WATSONX = "watsonx"
    OLLAMA = "ollama"
```

Can also use strings:

```python
config = VisionLLMConfig(provider="ollama")  # Works
config = VisionLLMConfig(provider=LLMProvider.OLLAMA)  # Also works
```
