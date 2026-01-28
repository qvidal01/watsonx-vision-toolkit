"""
Response Caching Example

This example demonstrates how to use response caching to reduce API calls
and improve performance when processing documents.

Caching is useful for:
- Development and testing (avoid repeated API calls)
- Processing the same documents multiple times
- Reducing costs when using paid API providers
"""

import time

from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig, CacheStats
from watsonx_vision.vision_llm import LLMProvider
from watsonx_vision.cache import ResponseCache


def basic_caching():
    """Basic caching with VisionLLM."""
    print("=" * 60)
    print("Basic Caching Example")
    print("=" * 60)

    # Create LLM config
    llm_config = VisionLLMConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llava:latest",
        url="http://localhost:11434",
    )

    # Create cache config with 1-hour TTL and 100 entry limit
    cache_config = CacheConfig(
        enabled=True,
        ttl=3600.0,  # 1 hour
        max_size=100,
    )

    # Initialize VisionLLM with caching
    # llm = VisionLLM(llm_config, cache_config=cache_config)

    print(f"Cache enabled: ttl={cache_config.ttl}s, max_size={cache_config.max_size}")
    print()

    # First call would hit the LLM
    # result1 = llm.classify_document(image_data)
    # print(f"First call: {result1}")

    # Second identical call would return cached result
    # result2 = llm.classify_document(image_data)
    # print(f"Second call (cached): {result2}")

    # Check cache statistics
    # stats = llm.cache_stats()
    # print(f"Cache stats: {stats}")


def cache_statistics():
    """Demonstrate cache statistics monitoring."""
    print("=" * 60)
    print("Cache Statistics Example")
    print("=" * 60)

    # Create a standalone cache for demonstration
    cache = ResponseCache(CacheConfig(ttl=3600, max_size=100))

    # Simulate some cache operations
    cache.set("doc1", {"doc_type": "Passport"})
    cache.set("doc2", {"doc_type": "Driver License"})
    cache.set("doc3", {"doc_type": "Bank Statement"})

    # Hit some entries
    cache.get("doc1")  # Hit
    cache.get("doc1")  # Hit
    cache.get("doc2")  # Hit
    cache.get("nonexistent")  # Miss

    # Get statistics
    stats = cache.stats()
    print(f"Cache Statistics:")
    print(f"  Size: {stats.size} entries")
    print(f"  Hits: {stats.hits}")
    print(f"  Misses: {stats.misses}")
    print(f"  Hit Rate: {stats.hit_rate:.2%}")
    print(f"  Evictions: {stats.evictions}")
    print(f"  Expirations: {stats.expirations}")
    print()


def ttl_expiration():
    """Demonstrate TTL-based cache expiration."""
    print("=" * 60)
    print("TTL Expiration Example")
    print("=" * 60)

    # Create cache with 2-second TTL for demonstration
    cache = ResponseCache(CacheConfig(ttl=2.0))

    # Add an entry
    cache.set("short_lived", {"data": "This will expire soon"})
    print(f"Added entry: {cache.get('short_lived')}")

    # Entry is still valid
    time.sleep(1)
    print(f"After 1 second: {cache.get('short_lived')}")

    # Entry has expired
    time.sleep(1.5)
    result = cache.get("short_lived")
    print(f"After 2.5 seconds: {result}")  # None
    print()


def lru_eviction():
    """Demonstrate LRU eviction when cache is full."""
    print("=" * 60)
    print("LRU Eviction Example")
    print("=" * 60)

    # Create small cache that will fill up
    cache = ResponseCache(CacheConfig(max_size=3))

    # Fill the cache
    cache.set("entry1", {"value": 1})
    cache.set("entry2", {"value": 2})
    cache.set("entry3", {"value": 3})
    print(f"Cache size: {len(cache)}")

    # Access entry1 to make it recently used
    cache.get("entry1")
    print("Accessed entry1 (now recently used)")

    # Add new entry - entry2 should be evicted (least recently used)
    cache.set("entry4", {"value": 4})
    print("Added entry4")

    print(f"entry1 still exists: {cache.get('entry1') is not None}")
    print(f"entry2 was evicted: {cache.get('entry2') is None}")
    print(f"entry3 still exists: {cache.get('entry3') is not None}")
    print(f"entry4 exists: {cache.get('entry4') is not None}")

    stats = cache.stats()
    print(f"Total evictions: {stats.evictions}")
    print()


def file_based_cache():
    """Demonstrate file-based cache persistence."""
    print("=" * 60)
    print("File-Based Cache Example")
    print("=" * 60)

    import tempfile
    import os

    # Create a temporary file for the cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "vision_cache.json")

        # Create cache with file backend
        cache1 = ResponseCache(CacheConfig(
            backend="file",
            file_path=cache_path,
            ttl=3600,
        ))

        # Add some entries
        cache1.set("persistent1", {"doc_type": "Passport"})
        cache1.set("persistent2", {"doc_type": "License"})
        print(f"Added 2 entries to file cache at {cache_path}")

        # Create new cache instance - it loads from file
        cache2 = ResponseCache(CacheConfig(
            backend="file",
            file_path=cache_path,
            ttl=3600,
        ))

        # Entries persist across instances
        print(f"persistent1 from new instance: {cache2.get('persistent1')}")
        print(f"persistent2 from new instance: {cache2.get('persistent2')}")
    print()


def disable_cache_per_request():
    """Demonstrate disabling cache for specific requests."""
    print("=" * 60)
    print("Disable Cache Per Request Example")
    print("=" * 60)

    # When using VisionLLM with cache, you can disable it per-request:
    # llm = VisionLLM(config, cache_config=CacheConfig())
    #
    # Normal call uses cache
    # result1 = llm.classify_document(image_data)
    #
    # This call bypasses cache
    # result2 = llm.analyze_image(image_data, prompt, use_cache=False)
    #
    # Useful for:
    # - Forcing fresh analysis
    # - Testing
    # - When document has been updated

    cache = ResponseCache(CacheConfig())
    cache.set("key", "cached_value")

    print(f"With cache: {cache.get('key')}")
    print("use_cache=False bypasses the cache lookup")
    print()


def cache_from_environment():
    """Configure cache using environment variables."""
    print("=" * 60)
    print("Cache from Environment Variables Example")
    print("=" * 60)

    # Set environment variables (normally done outside Python)
    import os
    os.environ["VISION_CACHE_ENABLED"] = "true"
    os.environ["VISION_CACHE_TTL"] = "7200"  # 2 hours
    os.environ["VISION_CACHE_MAX_SIZE"] = "500"

    # Load config from environment
    cache_config = CacheConfig.from_env()

    print(f"Loaded from environment:")
    print(f"  Enabled: {cache_config.enabled}")
    print(f"  TTL: {cache_config.ttl} seconds")
    print(f"  Max Size: {cache_config.max_size} entries")
    print(f"  Backend: {cache_config.backend}")

    # Cleanup
    del os.environ["VISION_CACHE_ENABLED"]
    del os.environ["VISION_CACHE_TTL"]
    del os.environ["VISION_CACHE_MAX_SIZE"]
    print()


def cache_with_prefix():
    """Use prefixed environment variables for multiple cache configs."""
    print("=" * 60)
    print("Prefixed Cache Config Example")
    print("=" * 60)

    import os

    # Different configs for different environments
    os.environ["PROD_VISION_CACHE_TTL"] = "3600"  # 1 hour for production
    os.environ["DEV_VISION_CACHE_TTL"] = "60"  # 1 minute for development

    prod_config = CacheConfig.from_env(prefix="PROD_")
    dev_config = CacheConfig.from_env(prefix="DEV_")

    print(f"Production cache TTL: {prod_config.ttl}s")
    print(f"Development cache TTL: {dev_config.ttl}s")

    # Cleanup
    del os.environ["PROD_VISION_CACHE_TTL"]
    del os.environ["DEV_VISION_CACHE_TTL"]
    print()


if __name__ == "__main__":
    basic_caching()
    cache_statistics()
    ttl_expiration()
    lru_eviction()
    file_based_cache()
    disable_cache_per_request()
    cache_from_environment()
    cache_with_prefix()

    print("=" * 60)
    print("All caching examples completed!")
    print("=" * 60)
