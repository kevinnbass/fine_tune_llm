"""Redis-based caching layer for predictions."""

import hashlib
import json
import pickle
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


class PredictionCache:
    """Cache for expensive predictions."""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        ttl_seconds: int = 3600,
        max_cache_size: int = 10000
    ):
        """
        Initialize cache.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            ttl_seconds: Time to live in seconds
            max_cache_size: Maximum number of cached items
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        
        self.redis_client = None
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'evictions': 0
        }
        
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = await redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False  # We'll handle encoding
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis cache")
    
    async def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except:
            return False
    
    def _generate_key(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Generate cache key from text and metadata.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Cache key
        """
        # Create a deterministic hash
        hasher = hashlib.sha256()
        hasher.update(text.encode('utf-8'))
        
        if metadata:
            # Sort metadata for consistent hashing
            sorted_meta = json.dumps(metadata, sort_keys=True)
            hasher.update(sorted_meta.encode('utf-8'))
        
        return f"pred:{hasher.hexdigest()[:16]}"
    
    async def get_prediction(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Cached prediction or None
        """
        if not self.redis_client:
            return None
        
        key = self._generate_key(text, metadata)
        
        try:
            # Get from cache
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                # Update stats
                self.stats['hits'] += 1
                
                # Update access time for LRU
                await self.redis_client.expire(key, self.ttl_seconds)
                
                # Deserialize
                result = pickle.loads(cached_data)
                
                logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                self.stats['misses'] += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats['errors'] += 1
            return None
    
    async def set_prediction(
        self,
        text: str,
        prediction: Dict[str, Any],
        metadata: Optional[Dict] = None
    ):
        """
        Cache prediction.
        
        Args:
            text: Input text
            prediction: Prediction result
            metadata: Optional metadata
        """
        if not self.redis_client:
            return
        
        key = self._generate_key(text, metadata)
        
        try:
            # Add timestamp
            prediction['cached_at'] = datetime.utcnow().isoformat()
            
            # Serialize
            serialized = pickle.dumps(prediction)
            
            # Check cache size
            current_size = await self.redis_client.dbsize()
            if current_size >= self.max_cache_size:
                # Evict oldest entries (simplified LRU)
                await self._evict_oldest()
            
            # Set with TTL
            await self.redis_client.setex(
                key,
                self.ttl_seconds,
                serialized
            )
            
            logger.debug(f"Cached prediction for key: {key}")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats['errors'] += 1
    
    async def _evict_oldest(self, n: int = 100):
        """Evict oldest cache entries."""
        try:
            # Get all keys with TTL
            keys = await self.redis_client.keys("pred:*")
            
            if not keys:
                return
            
            # Get TTLs for all keys
            ttls = []
            for key in keys[:1000]:  # Limit to avoid blocking
                ttl = await self.redis_client.ttl(key)
                ttls.append((key, ttl))
            
            # Sort by TTL (oldest first)
            ttls.sort(key=lambda x: x[1])
            
            # Delete oldest entries
            for key, _ in ttls[:n]:
                await self.redis_client.delete(key)
                self.stats['evictions'] += 1
            
            logger.info(f"Evicted {min(n, len(ttls))} cache entries")
            
        except Exception as e:
            logger.error(f"Eviction error: {e}")
    
    async def invalidate(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ):
        """
        Invalidate cached prediction.
        
        Args:
            text: Input text
            metadata: Optional metadata
        """
        if not self.redis_client:
            return
        
        key = self._generate_key(text, metadata)
        
        try:
            await self.redis_client.delete(key)
            logger.debug(f"Invalidated cache for key: {key}")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    async def clear(self):
        """Clear all cached predictions."""
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys("pred:*")
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")
            
            # Reset stats
            self.stats = {
                'hits': 0,
                'misses': 0,
                'errors': 0,
                'evictions': 0
            }
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0
        
        # Get current size
        if self.redis_client:
            try:
                stats['current_size'] = await self.redis_client.dbsize()
                
                # Get memory usage (if available)
                info = await self.redis_client.info('memory')
                stats['memory_used_mb'] = info.get('used_memory', 0) / (1024 * 1024)
                
            except:
                pass
        
        stats['max_size'] = self.max_cache_size
        stats['ttl_seconds'] = self.ttl_seconds
        
        return stats
    
    async def warm_up(self, predictions: List[Tuple[str, Dict[str, Any]]]):
        """
        Warm up cache with pre-computed predictions.
        
        Args:
            predictions: List of (text, prediction) tuples
        """
        logger.info(f"Warming up cache with {len(predictions)} predictions")
        
        for text, prediction in predictions:
            await self.set_prediction(text, prediction)
        
        logger.info("Cache warm-up complete")


class MultiLevelCache:
    """Multi-level cache with memory and Redis layers."""
    
    def __init__(
        self,
        memory_size: int = 100,
        redis_config: Optional[Dict] = None
    ):
        """
        Initialize multi-level cache.
        
        Args:
            memory_size: Size of in-memory cache
            redis_config: Redis configuration
        """
        self.memory_cache = {}
        self.memory_size = memory_size
        self.memory_access_times = {}
        
        # Initialize Redis cache
        redis_config = redis_config or {}
        self.redis_cache = PredictionCache(**redis_config)
        
    async def get(self, text: str, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Get from cache (memory first, then Redis)."""
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Check memory cache
        if key in self.memory_cache:
            self.memory_access_times[key] = datetime.utcnow()
            return self.memory_cache[key]
        
        # Check Redis cache
        result = await self.redis_cache.get_prediction(text, metadata)
        
        if result:
            # Add to memory cache
            self._add_to_memory(key, result)
        
        return result
    
    async def set(
        self,
        text: str,
        prediction: Dict,
        metadata: Optional[Dict] = None
    ):
        """Set in both cache levels."""
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Add to memory cache
        self._add_to_memory(key, prediction)
        
        # Add to Redis cache
        await self.redis_cache.set_prediction(text, prediction, metadata)
    
    def _add_to_memory(self, key: str, value: Dict):
        """Add to memory cache with LRU eviction."""
        # Check if need to evict
        if len(self.memory_cache) >= self.memory_size and key not in self.memory_cache:
            # Find least recently used
            lru_key = min(self.memory_access_times.keys(), 
                         key=lambda k: self.memory_access_times[k])
            
            del self.memory_cache[lru_key]
            del self.memory_access_times[lru_key]
        
        self.memory_cache[key] = value
        self.memory_access_times[key] = datetime.utcnow()