"""
Response Cache Module

Provides caching for LLM responses to reduce API calls and improve performance.
Supports in-memory caching with TTL and size limits.
"""

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for response caching."""

    enabled: bool = True
    """Whether caching is enabled."""

    ttl: float = 3600.0
    """Time-to-live for cache entries in seconds (default: 1 hour)."""

    max_size: int = 1000
    """Maximum number of entries in cache."""

    backend: str = "memory"
    """Cache backend: 'memory' or 'file'."""

    file_path: Optional[str] = None
    """Path for file-based cache (only used if backend='file')."""

    hash_images: bool = True
    """Whether to hash image data for cache keys (reduces memory for keys)."""

    @classmethod
    def from_env(cls, prefix: str = "") -> "CacheConfig":
        """
        Create cache configuration from environment variables.

        Environment variables:
            - {PREFIX}VISION_CACHE_ENABLED: Enable/disable cache (default: true)
            - {PREFIX}VISION_CACHE_TTL: TTL in seconds (default: 3600)
            - {PREFIX}VISION_CACHE_MAX_SIZE: Max entries (default: 1000)
            - {PREFIX}VISION_CACHE_BACKEND: 'memory' or 'file' (default: memory)
            - {PREFIX}VISION_CACHE_FILE_PATH: Path for file cache

        Args:
            prefix: Optional prefix for environment variables

        Returns:
            CacheConfig instance
        """

        def get_env(*keys: str, default: Optional[str] = None) -> Optional[str]:
            for key in keys:
                if prefix:
                    value = os.environ.get(f"{prefix}{key}")
                    if value is not None:
                        return value
                value = os.environ.get(key)
                if value is not None:
                    return value
            return default

        def get_bool(*keys: str, default: bool = True) -> bool:
            value = get_env(*keys)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")

        def get_float(*keys: str, default: float = 0.0) -> float:
            value = get_env(*keys)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default

        def get_int(*keys: str, default: int = 0) -> int:
            value = get_env(*keys)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        return cls(
            enabled=get_bool("VISION_CACHE_ENABLED", default=True),
            ttl=get_float("VISION_CACHE_TTL", default=3600.0),
            max_size=get_int("VISION_CACHE_MAX_SIZE", default=1000),
            backend=get_env("VISION_CACHE_BACKEND", default="memory"),
            file_path=get_env("VISION_CACHE_FILE_PATH"),
        )


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    value: Any
    """The cached value."""

    created_at: float = field(default_factory=time.time)
    """Timestamp when entry was created."""

    hits: int = 0
    """Number of times this entry was accessed."""

    def is_expired(self, ttl: float) -> bool:
        """Check if entry has expired based on TTL."""
        return time.time() - self.created_at > ttl


class ResponseCache:
    """
    LRU cache for LLM responses with TTL support.

    Provides caching to reduce redundant LLM API calls for identical inputs.
    Useful when processing the same documents multiple times or during development.

    Example:
        >>> cache = ResponseCache(CacheConfig(ttl=3600, max_size=100))
        >>> cache.set("key1", {"result": "value"})
        >>> result = cache.get("key1")  # Returns cached value
        >>> cache.get("nonexistent")  # Returns None

        >>> # With VisionLLM
        >>> config = VisionLLMConfig(...)
        >>> cache_config = CacheConfig(ttl=1800)
        >>> llm = VisionLLM(config, cache_config=cache_config)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize response cache.

        Args:
            config: Cache configuration. If None, uses defaults.
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats()

        # Load file cache if configured
        if self.config.backend == "file" and self.config.file_path:
            self._load_from_file()

    def _generate_key(
        self,
        image_data: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a cache key from inputs.

        Args:
            image_data: Base64-encoded image data
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters to include in key

        Returns:
            SHA256 hash of the combined inputs
        """
        key_parts = []

        # Hash image data if configured (saves memory for large images)
        if self.config.hash_images:
            image_hash = hashlib.sha256(image_data.encode()).hexdigest()
            key_parts.append(f"img:{image_hash}")
        else:
            key_parts.append(f"img:{image_data[:100]}")

        key_parts.append(f"prompt:{prompt}")

        if system_prompt:
            key_parts.append(f"system:{system_prompt}")

        # Include any additional kwargs that affect output
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")

        combined = "|".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Value to return if key not found or expired

        Returns:
            Cached value or default
        """
        if not self.config.enabled:
            return default

        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return default

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired(self.config.ttl):
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expirations += 1
                logger.debug(f"Cache entry expired: {key[:16]}...")
                return default

            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1
            logger.debug(f"Cache hit: {key[:16]}...")
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.config.enabled:
            return

        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.config.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
                logger.debug(f"Cache eviction: {oldest_key[:16]}...")

            self._cache[key] = CacheEntry(value=value)
            logger.debug(f"Cache set: {key[:16]}...")

            # Persist to file if configured
            if self.config.backend == "file" and self.config.file_path:
                self._save_to_file()

    def get_or_set(
        self,
        key: str,
        factory: callable,
    ) -> Any:
        """
        Get from cache or compute and store.

        Args:
            key: Cache key
            factory: Callable to compute value if not cached

        Returns:
            Cached or computed value
        """
        result = self.get(key)
        if result is not None:
            return result

        value = factory()
        self.set(key, value)
        return value

    def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from cache.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache invalidated: {key[:16]}...")
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.clears += 1
            logger.info(f"Cache cleared: {count} entries removed")

            if self.config.backend == "file" and self.config.file_path:
                self._save_to_file()

            return count

    def stats(self) -> "CacheStats":
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats

    def _load_from_file(self) -> None:
        """Load cache from file."""
        if not self.config.file_path:
            return

        path = Path(self.config.file_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for key, entry_data in data.items():
                entry = CacheEntry(
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    hits=entry_data.get("hits", 0),
                )
                if not entry.is_expired(self.config.ttl):
                    self._cache[key] = entry

            logger.info(f"Loaded {len(self._cache)} entries from cache file")
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")

    def _save_to_file(self) -> None:
        """Save cache to file."""
        if not self.config.file_path:
            return

        try:
            path = Path(self.config.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for key, entry in self._cache.items():
                data[key] = {
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "hits": entry.hits,
                }

            with open(path, "w") as f:
                json.dump(data, f)

            logger.debug(f"Saved {len(data)} entries to cache file")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    """Number of cache hits."""

    misses: int = 0
    """Number of cache misses."""

    evictions: int = 0
    """Number of entries evicted due to size limit."""

    expirations: int = 0
    """Number of entries expired due to TTL."""

    clears: int = 0
    """Number of times cache was cleared."""

    size: int = 0
    """Current number of entries in cache."""

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "clears": self.clears,
            "size": self.size,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.2%}, size={self.size})"
        )
