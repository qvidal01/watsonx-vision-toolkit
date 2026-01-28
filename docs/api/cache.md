# Cache

Response caching utilities for reducing API calls.

## CacheConfig

Configuration for response caching.

```python
from watsonx_vision import CacheConfig

cache_config = CacheConfig(
    enabled=True,
    ttl=3600.0,
    max_size=1000,
    backend="memory",
    file_path=None,
    hash_images=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable/disable caching |
| `ttl` | `float` | `3600.0` | Time-to-live in seconds |
| `max_size` | `int` | `1000` | Maximum cache entries |
| `backend` | `str` | `"memory"` | Storage backend (`"memory"` or `"file"`) |
| `file_path` | `str \| None` | `None` | File path for file backend |
| `hash_images` | `bool` | `True` | Hash image data in cache keys |

### Class Methods

#### from_env

Load configuration from environment variables.

```python
cache_config = CacheConfig.from_env(prefix="")
```

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `VISION_CACHE_ENABLED` | Enable caching (`true`/`false`) |
| `VISION_CACHE_TTL` | TTL in seconds |
| `VISION_CACHE_MAX_SIZE` | Maximum entries |
| `VISION_CACHE_BACKEND` | Backend type |
| `VISION_CACHE_FILE_PATH` | File path for file backend |

---

## ResponseCache

LRU cache with TTL support.

### Constructor

```python
from watsonx_vision import ResponseCache, CacheConfig

config = CacheConfig(enabled=True, ttl=3600)
cache = ResponseCache(config)
```

---

### get

Get a value from the cache.

```python
value = cache.get(key, default=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Cache key |
| `default` | `Any` | Default value if not found |

**Returns:** Cached value or default

---

### set

Store a value in the cache.

```python
cache.set(key, value)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Cache key |
| `value` | `Any` | Value to cache |

---

### clear

Clear all cache entries.

```python
cache.clear()
```

---

### stats

Get cache statistics.

```python
stats = cache.stats()
```

**Returns:** `CacheStats`

---

### Internal Methods

#### _generate_key

Generate a cache key from request parameters.

```python
key = cache._generate_key(
    image_data="data:image/png;base64,...",
    prompt="Classify this document",
    system_prompt=None,
    parse_json=True
)
```

Key generation:

1. If `hash_images=True`, SHA256 hash of image data
2. Otherwise, full image data
3. Combined with prompt, system_prompt, and kwargs
4. SHA256 hash of combined string

---

## CacheStats

Statistics about cache performance.

```python
@dataclass
class CacheStats:
    hits: int           # Number of cache hits
    misses: int         # Number of cache misses
    size: int           # Current number of entries
    max_size: int       # Maximum allowed entries
    hit_rate: float     # hits / (hits + misses)
```

### Example

```python
stats = cache.stats()
print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Size: {stats.size}/{stats.max_size}")
```

---

## CacheEntry

Internal dataclass for cache entries.

```python
@dataclass
class CacheEntry:
    value: Any          # Cached value
    timestamp: float    # Creation time
    ttl: float          # Time-to-live
```

---

## Example Usage

### With VisionLLM

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

cache_config = CacheConfig(
    enabled=True,
    ttl=3600,
    max_size=500
)

config = VisionLLMConfig.from_env()
llm = VisionLLM(config, cache_config=cache_config)

# First call - hits API
result1 = llm.classify_document(image_data)

# Second call - returns from cache
result2 = llm.classify_document(image_data)

# Check stats
stats = llm.cache_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")  # 50%

# Clear cache
llm.cache_clear()
```

### Standalone Cache

```python
from watsonx_vision import ResponseCache, CacheConfig

config = CacheConfig(enabled=True, ttl=3600)
cache = ResponseCache(config)

# Store value
cache.set("my-key", {"result": "data"})

# Retrieve value
value = cache.get("my-key")
print(value)  # {"result": "data"}

# Check if exists
value = cache.get("missing-key", default="not found")
print(value)  # "not found"

# Statistics
stats = cache.stats()
print(f"Hits: {stats.hits}, Misses: {stats.misses}")
```

### File Backend

```python
from watsonx_vision import CacheConfig, ResponseCache

config = CacheConfig(
    enabled=True,
    backend="file",
    file_path="/tmp/vision_cache.json",
    ttl=86400  # 24 hours
)

cache = ResponseCache(config)

# Cache persists across restarts
cache.set("key", "value")

# Later, in another process
cache2 = ResponseCache(config)
value = cache2.get("key")  # Returns "value"
```
