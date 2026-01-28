# Response Caching

The toolkit includes a built-in caching system to reduce API calls and improve performance.

## Overview

Response caching stores LLM responses and returns cached results for identical requests. This is useful for:

- **Cost reduction** - Avoid repeated API calls for the same document
- **Faster responses** - Cached results return instantly
- **Development** - Test without hitting the API repeatedly
- **Batch processing** - Process the same document multiple times efficiently

## Quick Start

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

# Configure caching
cache_config = CacheConfig(
    enabled=True,
    ttl=3600,        # Cache for 1 hour
    max_size=1000    # Maximum 1000 entries
)

# Create LLM with caching
config = VisionLLMConfig.from_env()
llm = VisionLLM(config, cache_config=cache_config)

# First call - hits the API
result1 = llm.classify_document(image_data)

# Second call - returns from cache
result2 = llm.classify_document(image_data)
```

## CacheConfig

```python
from watsonx_vision import CacheConfig

cache_config = CacheConfig(
    enabled=True,           # Enable caching
    ttl=3600.0,             # Time-to-live in seconds
    max_size=1000,          # Maximum entries (LRU eviction)
    backend="memory",       # "memory" or "file"
    file_path=None,         # Path for file backend
    hash_images=True        # Hash image data for cache keys
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable/disable caching |
| `ttl` | `float` | `3600.0` | Cache entry lifetime in seconds |
| `max_size` | `int` | `1000` | Maximum cache entries |
| `backend` | `str` | `"memory"` | Storage backend |
| `file_path` | `str \| None` | `None` | File path for file backend |
| `hash_images` | `bool` | `True` | Hash image data in cache keys |

## Backends

### Memory Backend

In-memory LRU cache. Fast but not persistent across restarts.

```python
cache_config = CacheConfig(
    enabled=True,
    backend="memory",
    max_size=500
)
```

### File Backend

Persistent file-based cache using JSON. Survives restarts.

```python
cache_config = CacheConfig(
    enabled=True,
    backend="file",
    file_path="/tmp/vision_cache.json",
    max_size=1000
)
```

## Cache Keys

Cache keys are generated from:

- Image data (SHA256 hash if `hash_images=True`)
- Prompt text
- System prompt (if provided)
- `parse_json` parameter

Identical inputs produce the same cache key and return cached results.

## Cache Statistics

Monitor cache performance:

```python
# Get statistics
stats = llm.cache_stats()

print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Size: {stats.size}")
print(f"Max size: {stats.max_size}")
```

## Cache Management

```python
# Check if caching is enabled
if llm.cache_enabled:
    print("Caching is active")

# Clear the cache
llm.cache_clear()

# Get the cache object directly
cache = llm.cache
```

## Bypass Cache

Skip caching for specific calls:

```python
# This call bypasses the cache
result = llm.analyze_image(image_data, prompt, use_cache=False)
```

## Environment Variables

Configure caching via environment:

```bash
export VISION_CACHE_ENABLED=true
export VISION_CACHE_TTL=7200
export VISION_CACHE_MAX_SIZE=500
export VISION_CACHE_BACKEND=file
export VISION_CACHE_FILE_PATH=/tmp/cache.json
```

```python
cache_config = CacheConfig.from_env()
```

## CLI Caching

Enable caching in CLI commands:

```bash
# Enable caching
watsonx-vision classify doc.png --cache

# With custom TTL
watsonx-vision classify doc.png --cache --cache-ttl 7200
```

## Best Practices

### 1. Use for Repeated Analysis

```python
# Good - same document analyzed multiple times
for analysis_type in ["classify", "extract", "validate"]:
    result = llm.classify_document(image_data)
```

### 2. Set Appropriate TTL

- **Development**: Long TTL (hours/days)
- **Production**: Shorter TTL based on data freshness needs
- **Real-time**: Disable caching

### 3. Monitor Cache Performance

```python
stats = llm.cache_stats()
if stats.hit_rate < 0.5:
    print("Low cache hit rate - check cache configuration")
```

### 4. Clear Cache When Needed

```python
# After model updates
llm.cache_clear()

# After configuration changes
llm.cache_clear()
```

## Thread Safety

The cache is thread-safe and can be used in multi-threaded applications:

```python
from concurrent.futures import ThreadPoolExecutor

llm = VisionLLM(config, cache_config=cache_config)

def process_document(path):
    image = VisionLLM.encode_image_to_base64(path)
    return llm.classify_document(image)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_document, document_paths))
```

## Example: Batch Processing with Cache

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

# Configure with file cache for persistence
cache_config = CacheConfig(
    enabled=True,
    backend="file",
    file_path="./document_cache.json",
    ttl=86400  # 24 hours
)

config = VisionLLMConfig.from_env()
llm = VisionLLM(config, cache_config=cache_config)

# Process documents
documents = ["doc1.png", "doc2.png", "doc3.png"]
results = []

for doc_path in documents:
    image = VisionLLM.encode_image_to_base64(doc_path)
    result = llm.classify_document(image)
    results.append(result)

# Check cache performance
stats = llm.cache_stats()
print(f"Processed {len(documents)} documents")
print(f"Cache hits: {stats.hits}, misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.1%}")
```
