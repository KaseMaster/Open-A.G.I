"""
AEGIS Blockchain Performance Optimizer
Optimizations for consensus, block validation, and transaction processing
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlockCacheEntry:
    """Entry in the block cache"""
    block_hash: str
    block_data: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


@dataclass
class TransactionPool:
    """Optimized transaction pool with priority queue"""
    transactions: deque = field(default_factory=deque)
    tx_by_hash: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tx_by_sender: Dict[str, List[str]] = field(default_factory=dict)
    max_size: int = 10000
    total_size: int = 0


class BlockCache:
    """LRU cache for blockchain blocks"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, BlockCacheEntry] = {}
        self.access_order: deque = deque()
        self.total_memory = 0
        
        self.hits = 0
        self.misses = 0
    
    def get(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """Get block from cache"""
        if block_hash in self.cache:
            entry = self.cache[block_hash]
            entry.access_count += 1
            entry.timestamp = time.time()
            
            self.access_order.remove(block_hash)
            self.access_order.append(block_hash)
            
            self.hits += 1
            return entry.block_data
        
        self.misses += 1
        return None
    
    def put(self, block_hash: str, block_data: Dict[str, Any]):
        """Put block into cache"""
        size_bytes = len(json.dumps(block_data).encode())
        
        while (len(self.cache) >= self.max_size or 
               self.total_memory + size_bytes > self.max_memory_bytes):
            if not self.access_order:
                break
            self._evict_lru()
        
        if block_hash in self.cache:
            old_entry = self.cache[block_hash]
            self.total_memory -= old_entry.size_bytes
            self.access_order.remove(block_hash)
        
        entry = BlockCacheEntry(
            block_hash=block_hash,
            block_data=block_data,
            timestamp=time.time(),
            size_bytes=size_bytes
        )
        
        self.cache[block_hash] = entry
        self.access_order.append(block_hash)
        self.total_memory += size_bytes
    
    def _evict_lru(self):
        """Evict least recently used block"""
        if not self.access_order:
            return
        
        lru_hash = self.access_order.popleft()
        if lru_hash in self.cache:
            entry = self.cache[lru_hash]
            self.total_memory -= entry.size_bytes
            del self.cache[lru_hash]
    
    def invalidate(self, block_hash: str):
        """Invalidate cache entry"""
        if block_hash in self.cache:
            entry = self.cache[block_hash]
            self.total_memory -= entry.size_bytes
            del self.cache[block_hash]
            self.access_order.remove(block_hash)
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_order.clear()
        self.total_memory = 0
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_mb": self.total_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class TransactionPoolManager:
    """Optimized transaction pool with priority and batching"""
    
    def __init__(self, max_pool_size: int = 10000):
        self.pool = TransactionPool(max_size=max_pool_size)
        self.pending_count = 0
        self.processed_count = 0
        
        self.priority_weights = {
            "high": 10,
            "medium": 5,
            "low": 1
        }
    
    def add_transaction(
        self,
        tx_hash: str,
        tx_data: Dict[str, Any],
        priority: str = "medium"
    ) -> bool:
        """Add transaction to pool"""
        if tx_hash in self.pool.tx_by_hash:
            return False
        
        if self.pool.total_size >= self.pool.max_size:
            self._evict_lowest_priority()
        
        tx_data["priority"] = priority
        tx_data["timestamp"] = time.time()
        tx_data["priority_score"] = self.priority_weights.get(priority, 1)
        
        self.pool.transactions.append(tx_hash)
        self.pool.tx_by_hash[tx_hash] = tx_data
        
        sender = tx_data.get("sender", "unknown")
        if sender not in self.pool.tx_by_sender:
            self.pool.tx_by_sender[sender] = []
        self.pool.tx_by_sender[sender].append(tx_hash)
        
        self.pool.total_size += 1
        self.pending_count += 1
        
        return True
    
    def get_transactions_batch(
        self,
        batch_size: int = 100,
        min_priority: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get batch of transactions sorted by priority"""
        candidates = []
        
        for tx_hash in self.pool.transactions:
            if tx_hash not in self.pool.tx_by_hash:
                continue
            
            tx_data = self.pool.tx_by_hash[tx_hash]
            
            if min_priority:
                if tx_data.get("priority_score", 0) < self.priority_weights.get(min_priority, 0):
                    continue
            
            candidates.append((tx_hash, tx_data))
        
        candidates.sort(key=lambda x: x[1].get("priority_score", 0), reverse=True)
        
        batch = []
        for tx_hash, tx_data in candidates[:batch_size]:
            batch.append({
                "tx_hash": tx_hash,
                **tx_data
            })
        
        return batch
    
    def remove_transaction(self, tx_hash: str) -> bool:
        """Remove transaction from pool"""
        if tx_hash not in self.pool.tx_by_hash:
            return False
        
        tx_data = self.pool.tx_by_hash[tx_hash]
        sender = tx_data.get("sender", "unknown")
        
        del self.pool.tx_by_hash[tx_hash]
        self.pool.transactions.remove(tx_hash)
        
        if sender in self.pool.tx_by_sender:
            self.pool.tx_by_sender[sender].remove(tx_hash)
            if not self.pool.tx_by_sender[sender]:
                del self.pool.tx_by_sender[sender]
        
        self.pool.total_size -= 1
        self.pending_count -= 1
        self.processed_count += 1
        
        return True
    
    def _evict_lowest_priority(self):
        """Evict lowest priority transaction"""
        if not self.pool.transactions:
            return
        
        lowest_tx_hash = None
        lowest_score = float('inf')
        
        for tx_hash in self.pool.transactions:
            if tx_hash not in self.pool.tx_by_hash:
                continue
            
            tx_data = self.pool.tx_by_hash[tx_hash]
            score = tx_data.get("priority_score", 0)
            
            if score < lowest_score:
                lowest_score = score
                lowest_tx_hash = tx_hash
        
        if lowest_tx_hash:
            self.remove_transaction(lowest_tx_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        priority_counts = {"high": 0, "medium": 0, "low": 0}
        
        for tx_hash in self.pool.transactions:
            if tx_hash in self.pool.tx_by_hash:
                priority = self.pool.tx_by_hash[tx_hash].get("priority", "medium")
                if priority in priority_counts:
                    priority_counts[priority] += 1
        
        return {
            "pending": self.pending_count,
            "processed": self.processed_count,
            "total_size": self.pool.total_size,
            "max_size": self.pool.max_size,
            "priority_counts": priority_counts,
            "unique_senders": len(self.pool.tx_by_sender)
        }


class ConsensusOptimizer:
    """Optimizations for consensus protocol"""
    
    def __init__(self):
        self.message_cache: Dict[str, Any] = {}
        self.vote_aggregator: Dict[str, List[str]] = {}
        self.round_metrics: Dict[int, Dict[str, float]] = {}
        
        self.parallel_validation = True
        self.batch_verification = True
        self.early_abort = True
    
    async def parallel_message_validation(
        self,
        messages: List[Dict[str, Any]],
        validation_func: Any
    ) -> List[Tuple[bool, Optional[str]]]:
        """Validate messages in parallel"""
        if not self.parallel_validation or len(messages) < 2:
            results = []
            for msg in messages:
                try:
                    is_valid = validation_func(msg)
                    results.append((is_valid, None))
                except Exception as e:
                    results.append((False, str(e)))
            return results
        
        tasks = []
        for msg in messages:
            task = asyncio.create_task(self._validate_message_async(msg, validation_func))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append((False, str(result)))
            else:
                processed_results.append((result, None))
        
        return processed_results
    
    async def _validate_message_async(self, message: Dict[str, Any], validation_func: Any) -> bool:
        """Async wrapper for message validation"""
        return validation_func(message)
    
    def aggregate_votes(
        self,
        proposal_id: str,
        voter_id: str,
        vote: bool,
        threshold: float = 0.67
    ) -> Optional[bool]:
        """Aggregate votes with early decision"""
        if proposal_id not in self.vote_aggregator:
            self.vote_aggregator[proposal_id] = []
        
        vote_str = f"{voter_id}:{vote}"
        if vote_str not in self.vote_aggregator[proposal_id]:
            self.vote_aggregator[proposal_id].append(vote_str)
        
        votes = self.vote_aggregator[proposal_id]
        yes_votes = sum(1 for v in votes if v.endswith(":True"))
        no_votes = sum(1 for v in votes if v.endswith(":False"))
        total_votes = yes_votes + no_votes
        
        if total_votes == 0:
            return None
        
        yes_ratio = yes_votes / total_votes
        no_ratio = no_votes / total_votes
        
        if self.early_abort:
            if yes_ratio >= threshold:
                return True
            if no_ratio > (1 - threshold):
                return False
        
        return None
    
    def clear_votes(self, proposal_id: str):
        """Clear votes for a proposal"""
        if proposal_id in self.vote_aggregator:
            del self.vote_aggregator[proposal_id]
    
    def batch_verify_signatures(
        self,
        messages: List[Dict[str, Any]],
        verify_func: Any
    ) -> List[bool]:
        """Batch signature verification"""
        if not self.batch_verification:
            return [verify_func(msg) for msg in messages]
        
        results = []
        batch_size = 10
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            batch_results = [verify_func(msg) for msg in batch]
            results.extend(batch_results)
        
        return results
    
    def record_round_metrics(self, round_num: int, metrics: Dict[str, float]):
        """Record metrics for a consensus round"""
        self.round_metrics[round_num] = {
            **metrics,
            "timestamp": time.time()
        }
        
        if len(self.round_metrics) > 100:
            oldest_round = min(self.round_metrics.keys())
            del self.round_metrics[oldest_round]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get consensus performance statistics"""
        if not self.round_metrics:
            return {}
        
        recent_rounds = list(self.round_metrics.values())[-10:]
        
        avg_latency = sum(r.get("latency", 0) for r in recent_rounds) / len(recent_rounds)
        avg_message_count = sum(r.get("message_count", 0) for r in recent_rounds) / len(recent_rounds)
        
        return {
            "total_rounds": len(self.round_metrics),
            "recent_avg_latency_ms": avg_latency,
            "recent_avg_messages": avg_message_count,
            "optimizations_enabled": {
                "parallel_validation": self.parallel_validation,
                "batch_verification": self.batch_verification,
                "early_abort": self.early_abort
            }
        }


class BlockchainOptimizer:
    """Main optimizer coordinating all optimization components"""
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_memory_mb: int = 100,
        tx_pool_size: int = 10000
    ):
        self.block_cache = BlockCache(max_size=cache_size, max_memory_mb=cache_memory_mb)
        self.tx_pool = TransactionPoolManager(max_pool_size=tx_pool_size)
        self.consensus_optimizer = ConsensusOptimizer()
        
        self.start_time = time.time()
        self.blocks_processed = 0
        self.transactions_processed = 0
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "blocks_processed": self.blocks_processed,
            "transactions_processed": self.transactions_processed,
            "block_cache": self.block_cache.get_stats(),
            "tx_pool": self.tx_pool.get_stats(),
            "consensus": self.consensus_optimizer.get_performance_stats()
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.start_time = time.time()
        self.blocks_processed = 0
        self.transactions_processed = 0
        self.block_cache.clear()
