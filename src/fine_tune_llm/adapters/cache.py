"""
Cache adapter for hexagonal architecture.

This module provides caching operations through the adapter pattern,
supporting multiple cache backends with TTL and eviction policies.
"""

import time
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import logging

from ..core.interfaces import CachePort
from ..core.exceptions import SystemError, DataError
from ..utils.resilience import circuit_breaker

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used  
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live only


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


class InMemoryCache:
    """Thread-safe in-memory cache implementation."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
    
    def _evict_if_needed(self):
        """Evict entries if cache is at capacity."""
        if len(self._cache) < self.max_size:
            return
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self._cache.items() 
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._stats['expired'] += 1
        
        # If still at capacity, apply eviction policy
        if len(self._cache) >= self.max_size:
            entries_to_remove = len(self._cache) - self.max_size + 1
            
            if self.eviction_policy == EvictionPolicy.LRU:
                # Sort by access time (oldest first)
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].accessed_at
                )
            elif self.eviction_policy == EvictionPolicy.LFU:
                # Sort by access count (least frequent first)
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].access_count
                )
            elif self.eviction_policy == EvictionPolicy.FIFO:
                # Sort by creation time (oldest first)
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
            else:  # TTL
                # Sort by creation time
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
            
            # Remove oldest entries
            for key in sorted_keys[:entries_to_remove]:
                del self._cache[key]
                self._stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._stats['expired'] += 1
                self._stats['misses'] += 1
                return None
            
            # Update access metadata
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            self._evict_if_needed()
            
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                access_count=1,
                ttl=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'eviction_policy': self.eviction_policy.value
            }


class CacheAdapter(CachePort):
    """
    Cache adapter implementing caching operations.
    
    Provides high-performance caching with multiple eviction policies,
    TTL support, and comprehensive statistics tracking.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 enable_serialization: bool = True,
                 enable_resilience: bool = True):
        """
        Initialize cache adapter.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
            eviction_policy: Cache eviction policy
            enable_serialization: Enable automatic serialization
            enable_resilience: Enable circuit breaker protection
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.enable_serialization = enable_serialization
        self.enable_resilience = enable_resilience
        
        # Initialize cache backend
        self._cache = InMemoryCache(
            max_size=max_size,
            default_ttl=default_ttl,
            eviction_policy=eviction_policy
        )
        
        logger.info(f"Initialized CacheAdapter with policy: {eviction_policy.value}, max_size: {max_size}")
    
    def _serialize_key(self, key: Union[str, tuple, dict]) -> str:
        """Convert key to string."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return json.dumps(sorted(key), ensure_ascii=False)
        elif isinstance(key, dict):
            return json.dumps(sorted(key.items()), ensure_ascii=False)
        else:
            return str(key)
    
    def _hash_key(self, key: str) -> str:
        """Generate hash of key for consistent storage."""
        return hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if not self.enable_serialization:
            return value
        
        try:
            # Try JSON first (faster)
            json_str = json.dumps(value, ensure_ascii=False)
            return json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not self.enable_serialization:
            return data
        
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    @circuit_breaker("cache_operations")
    def get(self, key: Union[str, tuple, dict]) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            cache_key = self._serialize_key(key)
            hashed_key = self._hash_key(cache_key)
            
            data = self._cache.get(hashed_key)
            
            if data is None:
                return None
            
            if self.enable_serialization:
                return self._deserialize_value(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    @circuit_breaker("cache_operations")
    def set(self, 
            key: Union[str, tuple, dict], 
            value: Any, 
            ttl: Optional[float] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        try:
            cache_key = self._serialize_key(key)
            hashed_key = self._hash_key(cache_key)
            
            if self.enable_serialization:
                serialized_value = self._serialize_value(value)
            else:
                serialized_value = value
            
            return self._cache.set(hashed_key, serialized_value, ttl)
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def delete(self, key: Union[str, tuple, dict]) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        try:
            cache_key = self._serialize_key(key)
            hashed_key = self._hash_key(cache_key)
            
            return self._cache.delete(hashed_key)
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    def exists(self, key: Union[str, tuple, dict]) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        try:
            self._cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def mget(self, keys: List[Union[str, tuple, dict]]) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs for found entries
        """
        result = {}
        
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[str(key)] = value
        
        return result
    
    def mset(self, items: Dict[str, Any], ttl: Optional[float] = None) -> int:
        """
        Set multiple key-value pairs.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time-to-live for all items
            
        Returns:
            Number of items successfully set
        """
        success_count = 0
        
        for key, value in items.items():
            if self.set(key, value, ttl):
                success_count += 1
        
        return success_count
    
    def get_or_set(self, 
                   key: Union[str, tuple, dict], 
                   factory: Callable[[], T], 
                   ttl: Optional[float] = None) -> T:
        """
        Get value from cache or set using factory function.
        
        Args:
            key: Cache key
            factory: Function to generate value if not in cache
            ttl: Time-to-live for new value
            
        Returns:
            Cached or newly generated value
        """
        value = self.get(key)
        
        if value is not None:
            return value
        
        # Generate new value
        new_value = factory()
        self.set(key, new_value, ttl)
        
        return new_value
    
    def increment(self, key: Union[str, tuple, dict], delta: int = 1) -> Optional[int]:
        """
        Increment numeric value in cache.
        
        Args:
            key: Cache key
            delta: Increment amount
            
        Returns:
            New value after increment or None if not numeric
        """
        current_value = self.get(key)
        
        if current_value is None:
            new_value = delta
        elif isinstance(current_value, (int, float)):
            new_value = current_value + delta
        else:
            logger.error(f"Cannot increment non-numeric value for key {key}")
            return None
        
        if self.set(key, new_value):
            return new_value
        return None
    
    def expire_at(self, key: Union[str, tuple, dict], timestamp: datetime) -> bool:
        """
        Set expiration time for existing key.
        
        Args:
            key: Cache key
            timestamp: Expiration timestamp
            
        Returns:
            True if successful
        """
        current_value = self.get(key)
        
        if current_value is None:
            return False
        
        ttl = (timestamp - datetime.now()).total_seconds()
        
        if ttl <= 0:
            return self.delete(key)
        
        return self.set(key, current_value, ttl)
    
    def get_ttl(self, key: Union[str, tuple, dict]) -> Optional[float]:
        """
        Get remaining time-to-live for key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds or None if not found/no TTL
        """
        try:
            cache_key = self._serialize_key(key)
            hashed_key = self._hash_key(cache_key)
            
            entry = self._cache._cache.get(hashed_key)
            
            if entry is None or entry.ttl is None:
                return None
            
            remaining = entry.ttl - (time.time() - entry.created_at)
            return max(0, remaining)
            
        except Exception as e:
            logger.error(f"Get TTL failed for key {key}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self._cache.get_stats()
        
        return {
            **stats,
            'default_ttl': self.default_ttl,
            'enable_serialization': self.enable_serialization,
            'enable_resilience': self.enable_resilience
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage stats
        """
        try:
            import sys
            
            total_size = 0
            entry_sizes = []
            
            for entry in self._cache._cache.values():
                try:
                    entry_size = sys.getsizeof(entry.value)
                    entry_sizes.append(entry_size)
                    total_size += entry_size
                except:
                    pass
            
            avg_size = sum(entry_sizes) / len(entry_sizes) if entry_sizes else 0
            
            return {
                'total_size_bytes': total_size,
                'average_entry_size': avg_size,
                'entries': len(entry_sizes),
                'size_distribution': {
                    'min': min(entry_sizes) if entry_sizes else 0,
                    'max': max(entry_sizes) if entry_sizes else 0,
                    'avg': avg_size
                }
            }
            
        except Exception as e:
            logger.error(f"Memory usage calculation failed: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close cache adapter."""
        try:
            self._cache.clear()
            logger.info("Cache adapter closed")
        except Exception as e:
            logger.error(f"Error closing cache adapter: {e}")