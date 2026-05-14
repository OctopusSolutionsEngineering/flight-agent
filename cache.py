"""Pluggable cache backend (in-memory or Redis)."""
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from cachetools import TTLCache
import redis

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract cache interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None: ...
    
    @abstractmethod
    def delete(self, key: str) -> None: ...
    
    @abstractmethod
    def clear(self) -> None: ...
    
    @abstractmethod
    def stats(self) -> dict: ...


class InMemoryCache(CacheBackend):
    """Thread-safe in-memory TTL cache. Good for single-pod deployments."""
    
    def __init__(self, max_size: int = 1000):
        # cachetools handles TTL but uses a single TTL per cache.
        # We store (value, expiry_timestamp) tuples to support per-key TTLs.
        import time
        self._time = time
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=86400)  # max TTL
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        try:
            value, expiry = self._cache[key]
            if self._time.time() < expiry:
                self._hits += 1
                return value
            else:
                del self._cache[key]
        except KeyError:
            pass
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int) -> None:
        expiry = self._time.time() + ttl
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "backend": "in-memory",
            "size": len(self._cache),
            "max_size": self._cache.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }


class RedisCache(CacheBackend):
    """Redis-backed cache. Use for multi-pod deployments."""
    
    def __init__(self, url: str, key_prefix: str = "flight-agent:"):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = key_prefix
        # Verify connection
        self.client.ping()
        logger.info(f"Connected to Redis at {url}")
    
    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self.client.get(self._k(key))
            if raw is None:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Redis get failed for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            self.client.setex(self._k(key), ttl, json.dumps(value))
        except (redis.RedisError, TypeError) as e:
            logger.warning(f"Redis set failed for {key}: {e}")
    
    def delete(self, key: str) -> None:
        try:
            self.client.delete(self._k(key))
        except redis.RedisError as e:
            logger.warning(f"Redis delete failed: {e}")
    
    def clear(self) -> None:
        """Delete all keys with our prefix."""
        try:
            for k in self.client.scan_iter(f"{self.prefix}*"):
                self.client.delete(k)
        except redis.RedisError as e:
            logger.warning(f"Redis clear failed: {e}")
    
    def stats(self) -> dict:
        try:
            info = self.client.info("stats")
            keys = sum(1 for _ in self.client.scan_iter(f"{self.prefix}*"))
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            total = hits + misses
            return {
                "backend": "redis",
                "keys": keys,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 3) if total > 0 else 0.0,
            }
        except redis.RedisError as e:
            return {"backend": "redis", "error": str(e)}


# ===== Helpers =====

def make_cache_key(namespace: str, *args: Any, **kwargs: Any) -> str:
    """Create a stable cache key from arguments."""
    payload = {"args": args, "kwargs": kwargs}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"{namespace}:{digest}"


# ===== Singleton =====
_cache: Optional[CacheBackend] = None


def get_cache() -> CacheBackend:
    """Return the configured cache backend."""
    global _cache
    if _cache is not None:
        return _cache
    
    from config import get_settings
    settings = get_settings()
    
    if settings.cache_backend == "redis" and settings.redis_url:
        try:
            _cache = RedisCache(settings.redis_url)
        except (redis.RedisError, redis.ConnectionError) as e:
            logger.warning(f"Redis unavailable ({e}), falling back to in-memory")
            _cache = InMemoryCache(max_size=settings.cache_max_size)
    else:
        _cache = InMemoryCache(max_size=settings.cache_max_size)
        logger.info(f"Using in-memory cache (max_size={settings.cache_max_size})")
    
    return _cache
