"""
AEGIS Performance Optimization
Optimizations for large-scale deployments and high-throughput scenarios
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking"""
    operation_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    memory_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MemoryOptimizer:
    """Memory optimization for large-scale deployments"""
    
    def __init__(self, max_cache_size: int = 10000, gc_interval: int = 1000):
        self.max_cache_size = max_cache_size
        self.gc_interval = gc_interval
        self.operation_count = 0
        
        # Caches with size limits
        self.lru_cache: Dict[str, Any] = {}
        self.cache_access_count: Dict[str, int] = defaultdict(int)
        self.cache_timestamps: Dict[str, float] = {}
        
        # Memory pools for frequently allocated objects
        self.object_pools: Dict[str, deque] = defaultdict(deque)
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    def get_from_pool(self, pool_name: str, factory_func: Callable) -> Any:
        """Get object from pool or create new one"""
        if self.object_pools[pool_name]:
            return self.object_pools[pool_name].popleft()
        return factory_func()
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to pool"""
        if len(self.object_pools[pool_name]) < 100:  # Limit pool size
            self.object_pools[pool_name].append(obj)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU eviction"""
        if key in self.lru_cache:
            self.cache_access_count[key] += 1
            self.cache_timestamps[key] = time.time()
            return self.lru_cache[key]
        return None
    
    def cache_put(self, key: str, value: Any, max_size: Optional[int] = None):
        """Put item in cache with size management"""
        max_size = max_size or self.max_cache_size
        
        # Evict if cache is full
        if len(self.lru_cache) >= max_size:
            self._evict_lru_items(max_size // 10)  # Evict 10% of items
        
        self.lru_cache[key] = value
        self.cache_access_count[key] = 1
        self.cache_timestamps[key] = time.time()
    
    def _evict_lru_items(self, count: int):
        """Evict least recently used items"""
        # Sort by access count and timestamp
        items_to_evict = sorted(
            self.cache_access_count.items(),
            key=lambda x: (x[1], self.cache_timestamps.get(x[0], 0))
        )[:count]
        
        for key, _ in items_to_evict:
            self.lru_cache.pop(key, None)
            self.cache_access_count.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def periodic_gc(self):
        """Periodic garbage collection"""
        self.operation_count += 1
        if self.operation_count % self.gc_interval == 0:
            collected = gc.collect()
            if collected > 0:
                logger.info(f"Garbage collected {collected} objects")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "cache_size": len(self.lru_cache),
            "pools": {name: len(pool) for name, pool in self.object_pools.items()},
            "total_accesses": sum(self.cache_access_count.values()),
            "operation_count": self.operation_count
        }


class ConcurrencyOptimizer:
    """Concurrency optimization for high-throughput scenarios"""
    
    def __init__(self, max_workers: int = 100, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Thread pools
        self.thread_pools: Dict[str, Any] = {}
        
        # Async task management
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        
        # Resource limiting
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Performance tracking
        self.task_metrics: Dict[str, int] = defaultdict(int)
    
    def create_semaphore(self, name: str, limit: int) -> asyncio.Semaphore:
        """Create a semaphore for resource limiting"""
        if name not in self.semaphores:
            self.semaphores[name] = asyncio.Semaphore(limit)
        return self.semaphores[name]
    
    async def run_with_limit(self, name: str, coro: Callable, *args, **kwargs):
        """Run coroutine with resource limiting"""
        semaphore = self.semaphores.get(name, asyncio.Semaphore(self.max_workers))
        
        async with semaphore:
            try:
                result = await coro(*args, **kwargs)
                self.task_metrics[f"{name}_success"] += 1
                return result
            except Exception as e:
                self.task_metrics[f"{name}_error"] += 1
                raise
    
    async def batch_process(self, items: List[Any], processor: Callable, batch_size: int = 100) -> List[Any]:
        """Process items in batches for better throughput"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[processor(item) for item in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
        
        return results
    
    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get concurrency statistics"""
        return {
            "running_tasks": len(self.running_tasks),
            "queue_size": self.task_queue.qsize(),
            "semaphores": {name: sem._value for name, sem in self.semaphores.items()},
            "task_metrics": dict(self.task_metrics)
        }


class NetworkOptimizer:
    """Network optimization for distributed systems"""
    
    def __init__(self, compression_threshold: int = 1024, max_retries: int = 3):
        self.compression_threshold = compression_threshold
        self.max_retries = max_retries
        
        # Connection pooling
        self.connection_pools: Dict[str, List[Any]] = defaultdict(list)
        self.active_connections: Dict[str, int] = defaultdict(int)
        
        # Message batching
        self.message_batches: Dict[str, List[Dict]] = defaultdict(list)
        self.batch_timeouts: Dict[str, float] = {}
        
        # Compression
        self.compression_enabled = True
        
        # Performance tracking
        self.network_metrics: Dict[str, PerformanceMetrics] = {}
    
    def compress_message(self, message: bytes) -> bytes:
        """Compress message if large enough"""
        if not self.compression_enabled or len(message) < self.compression_threshold:
            return message
        
        try:
            import zlib
            compressed = zlib.compress(message)
            logger.debug(f"Compressed {len(message)} -> {len(compressed)} bytes ({len(compressed)/len(message)*100:.1f}%)")
            return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return message
    
    def decompress_message(self, message: bytes) -> bytes:
        """Decompress message"""
        if not self.compression_enabled:
            return message
        
        try:
            import zlib
            return zlib.decompress(message)
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return message
    
    async def batch_messages(self, destination: str, message: Dict, max_batch_size: int = 50, timeout: float = 0.1) -> Optional[List[Dict]]:
        """Batch messages for efficient transmission"""
        self.message_batches[destination].append(message)
        
        # Set timeout for first message in batch
        if destination not in self.batch_timeouts:
            self.batch_timeouts[destination] = time.time() + timeout
        
        # Check if batch is ready
        batch_ready = (
            len(self.message_batches[destination]) >= max_batch_size or
            time.time() >= self.batch_timeouts.get(destination, 0)
        )
        
        if batch_ready:
            batch = self.message_batches[destination]
            self.message_batches[destination] = []
            self.batch_timeouts.pop(destination, None)
            return batch
        
        return None
    
    def get_connection_from_pool(self, pool_name: str) -> Optional[Any]:
        """Get connection from pool"""
        if self.connection_pools[pool_name]:
            connection = self.connection_pools[pool_name].pop()
            self.active_connections[pool_name] += 1
            return connection
        return None
    
    def return_connection_to_pool(self, pool_name: str, connection: Any):
        """Return connection to pool"""
        if len(self.connection_pools[pool_name]) < 10:  # Limit pool size
            self.connection_pools[pool_name].append(connection)
        self.active_connections[pool_name] = max(0, self.active_connections[pool_name] - 1)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network optimization statistics"""
        return {
            "connection_pools": {name: len(pool) for name, pool in self.connection_pools.items()},
            "active_connections": dict(self.active_connections),
            "message_batches": {name: len(batch) for name, batch in self.message_batches.items()},
            "compression_enabled": self.compression_enabled
        }


class ComputationalOptimizer:
    """Computational optimization for intensive operations"""
    
    def __init__(self, parallel_threshold: int = 1000):
        self.parallel_threshold = parallel_threshold
        
        # Parallel processing
        self.executor = None
        
        # Caching for expensive computations
        self.computation_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # JIT compilation hints
        self.jit_candidates: set = set()
    
    def parallel_map(self, func: Callable, items: List[Any], chunk_size: int = 100) -> List[Any]:
        """Parallel map operation"""
        if len(items) < self.parallel_threshold:
            return [func(item) for item in items]
        
        try:
            import concurrent.futures
            if self.executor is None:
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            
            # Process in chunks
            results = []
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                chunk_results = list(self.executor.map(func, chunk))
                results.extend(chunk_results)
            
            return results
        except Exception as e:
            logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
            return [func(item) for item in items]
    
    def cached_computation(self, key: str, compute_func: Callable, *args, **kwargs) -> Any:
        """Cached computation with automatic caching"""
        if key in self.computation_cache:
            self.cache_hits += 1
            return self.computation_cache[key]
        
        self.cache_misses += 1
        result = compute_func(*args, **kwargs)
        self.computation_cache[key] = result
        return result
    
    def should_jit_compile(self, func_name: str) -> bool:
        """Determine if function should be JIT compiled"""
        self.jit_candidates.add(func_name)
        # In practice, this would analyze usage patterns
        return len(self.jit_candidates) > 10
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computational optimization statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "jit_candidates": list(self.jit_candidates),
            "executor_active": self.executor is not None
        }


class PerformanceOptimizer:
    """Main performance optimizer coordinating all optimizations"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.concurrency_optimizer = ConcurrencyOptimizer()
        self.network_optimizer = NetworkOptimizer()
        self.computational_optimizer = ComputationalOptimizer()
        
        # Global performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.start_time = time.time()
    
    async def optimize_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """Optimize and execute an operation"""
        start_time = time.time()
        
        try:
            # Apply appropriate optimizations
            result = await operation_func(*args, **kwargs)
            
            # Track performance
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_metrics(operation_name, elapsed_ms, success=True)
            
            # Periodic cleanup
            self.memory_optimizer.periodic_gc()
            
            return result
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_metrics(operation_name, elapsed_ms, success=False)
            raise
    
    def _update_metrics(self, operation_name: str, elapsed_ms: float, success: bool):
        """Update performance metrics"""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = PerformanceMetrics(operation_name=operation_name)
        
        metrics = self.metrics[operation_name]
        metrics.total_calls += 1
        metrics.total_time_ms += elapsed_ms
        metrics.avg_time_ms = metrics.total_time_ms / metrics.total_calls
        metrics.min_time_ms = min(metrics.min_time_ms, elapsed_ms)
        metrics.max_time_ms = max(metrics.max_time_ms, elapsed_ms)
        
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "operation_metrics": {
                name: {
                    "total_calls": m.total_calls,
                    "avg_time_ms": m.avg_time_ms,
                    "min_time_ms": m.min_time_ms,
                    "max_time_ms": m.max_time_ms,
                    "success_rate": m.success_count / m.total_calls if m.total_calls > 0 else 0,
                    "error_rate": m.error_count / m.total_calls if m.total_calls > 0 else 0
                }
                for name, m in self.metrics.items()
            },
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "concurrency_stats": self.concurrency_optimizer.get_concurrency_stats(),
            "network_stats": self.network_optimizer.get_network_stats(),
            "computation_stats": self.computational_optimizer.get_computation_stats()
        }
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics.clear()
        self.start_time = time.time()
        self.memory_optimizer.operation_count = 0
        self.computational_optimizer.cache_hits = 0
        self.computational_optimizer.cache_misses = 0
