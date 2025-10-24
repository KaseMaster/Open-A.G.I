"""
AEGIS Performance Benchmarking Suite
Comprehensive benchmarks for all framework components
"""

import time
import asyncio
import json
import hashlib
import statistics
import sys
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark"""
    name: str
    description: str
    iterations: int
    duration_ms: float
    operations_per_second: float
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceBenchmarkSuite:
    """Comprehensive benchmarking suite for AEGIS components"""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[BenchmarkResult] = []
        
        # Import required modules
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
            logger.warning("psutil not available, memory/CPU metrics will be disabled")
    
    def _measure_performance(
        self,
        func: Callable,
        iterations: int = 1000,
        warmup: int = 100
    ) -> Dict[str, Any]:
        """Measure performance of a function"""
        # Warmup
        for _ in range(warmup):
            func()
        
        # Actual measurement
        times = []
        start_memory = 0
        end_memory = 0
        
        if self.psutil:
            start_memory = self.psutil.Process().memory_info().rss / 1024 / 1024
        
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            iter_start = time.perf_counter()
            func()
            iter_end = time.perf_counter()
            times.append((iter_end - iter_start) * 1000)  # Convert to ms
        
        end_time = time.perf_counter()
        
        if self.psutil:
            end_memory = self.psutil.Process().memory_info().rss / 1024 / 1024
        
        total_duration = (end_time - start_time) * 1000  # Convert to ms
        ops_per_sec = iterations / (total_duration / 1000)
        
        return {
            "iterations": iterations,
            "duration_ms": total_duration,
            "operations_per_second": ops_per_sec,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "memory_mb": end_memory - start_memory if self.psutil else None
        }
    
    def benchmark_hashing(self) -> List[BenchmarkResult]:
        """Benchmark cryptographic hashing operations"""
        results = []
        test_data = b"A" * 1024  # 1KB test data
        
        # SHA-256
        def sha256_test():
            return hashlib.sha256(test_data).hexdigest()
        
        metrics = self._measure_performance(sha256_test, iterations=10000)
        results.append(BenchmarkResult(
            name="hash_sha256_1kb",
            description="SHA-256 hash of 1KB data",
            **metrics
        ))
        
        # SHA-512
        def sha512_test():
            return hashlib.sha512(test_data).hexdigest()
        
        metrics = self._measure_performance(sha512_test, iterations=10000)
        results.append(BenchmarkResult(
            name="hash_sha512_1kb",
            description="SHA-512 hash of 1KB data",
            **metrics
        ))
        
        return results
    
    def benchmark_merkle_tree(self) -> List[BenchmarkResult]:
        """Benchmark Merkle tree operations"""
        results = []
        
        # Import Merkle tree implementation
        try:
            from ..blockchain.merkle_tree import MerkleTree
            
            # Create tree with 1000 leaves
            tree = MerkleTree()
            leaves = [f"data_{i}".encode() for i in range(1000)]
            
            def add_leaves():
                tree.add_leaf(leaves[0])
            
            metrics = self._measure_performance(add_leaves, iterations=1000)
            results.append(BenchmarkResult(
                name="merkle_add_leaf",
                description="Add leaf to Merkle tree",
                **metrics
            ))
            
            # Build tree
            tree = MerkleTree()
            for leaf in leaves[:100]:  # Use 100 leaves for build test
                tree.add_leaf(leaf)
            
            def build_tree():
                tree.build_tree()
            
            metrics = self._measure_performance(build_tree, iterations=100)
            results.append(BenchmarkResult(
                name="merkle_build_tree_100",
                description="Build Merkle tree with 100 leaves",
                **metrics
            ))
            
            # Generate proof
            tree.build_tree()
            def generate_proof():
                tree.generate_proof(0)
            
            metrics = self._measure_performance(generate_proof, iterations=1000)
            results.append(BenchmarkResult(
                name="merkle_generate_proof",
                description="Generate Merkle proof for leaf 0",
                **metrics
            ))
            
        except Exception as e:
            logger.error(f"Error benchmarking Merkle tree: {e}")
        
        return results
    
    def benchmark_consensus(self) -> List[BenchmarkResult]:
        """Benchmark consensus operations"""
        results = []
        
        try:
            from ..blockchain.consensus_protocol import HybridConsensus
            from cryptography.hazmat.primitives.asymmetric import ed25519
            
            # Generate keys
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            
            consensus = HybridConsensus(
                node_id="benchmark_node",
                private_key=private_key
            )
            
            # Benchmark proposal creation
            def create_proposal():
                consensus.create_proposal(
                    block_data={"test": "data"},
                    previous_hash="0" * 64
                )
            
            metrics = self._measure_performance(create_proposal, iterations=100)
            results.append(BenchmarkResult(
                name="consensus_create_proposal",
                description="Create consensus proposal",
                **metrics
            ))
            
            # Benchmark vote creation
            proposal = consensus.create_proposal(
                block_data={"test": "data"},
                previous_hash="0" * 64
            )
            
            def create_vote():
                consensus.create_vote(
                    proposal_id=proposal.proposal_id,
                    vote=True
                )
            
            metrics = self._measure_performance(create_vote, iterations=1000)
            results.append(BenchmarkResult(
                name="consensus_create_vote",
                description="Create consensus vote",
                **metrics
            ))
            
        except Exception as e:
            logger.error(f"Error benchmarking consensus: {e}")
        
        return results
    
    def benchmark_federated_learning(self) -> List[BenchmarkResult]:
        """Benchmark federated learning operations"""
        results = []
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            from ..ml.federated_learning import (
                FederatedClient,
                FederatedServer,
                FederatedConfig,
                AggregationStrategy
            )
            
            # Simple model
            class BenchmarkModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = BenchmarkModel()
            config = FederatedConfig(
                aggregation_strategy=AggregationStrategy.FED_AVG,
                local_epochs=1
            )
            
            # Benchmark client creation
            def create_client():
                client = FederatedClient(
                    client_id="benchmark_client",
                    model=BenchmarkModel(),
                    config=config
                )
                return client
            
            metrics = self._measure_performance(create_client, iterations=100)
            results.append(BenchmarkResult(
                name="fl_create_client",
                description="Create federated learning client",
                **metrics
            ))
            
            # Benchmark server creation
            def create_server():
                server = FederatedServer(
                    model=BenchmarkModel(),
                    config=config
                )
                return server
            
            metrics = self._measure_performance(create_server, iterations=100)
            results.append(BenchmarkResult(
                name="fl_create_server",
                description="Create federated learning server",
                **metrics
            ))
            
            # Benchmark client training (simplified)
            client = FederatedClient(
                client_id="benchmark_client",
                model=model,
                config=config
            )
            
            # Create dummy data
            X = torch.randn(100, 10)
            y = torch.randint(0, 2, (100,))
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32)
            
            def client_train():
                state = client.train(dataloader)
                return state
            
            metrics = self._measure_performance(client_train, iterations=10)
            results.append(BenchmarkResult(
                name="fl_client_train_epoch",
                description="Client training for 1 epoch",
                **metrics
            ))
            
        except Exception as e:
            logger.error(f"Error benchmarking federated learning: {e}")
        
        return results
    
    def benchmark_networking(self) -> List[BenchmarkResult]:
        """Benchmark networking operations"""
        results = []
        
        try:
            from ..networking.p2p_network import P2PNetworkManager
            
            # Benchmark message creation
            def create_message():
                message = {
                    "type": "test",
                    "sender": "benchmark",
                    "timestamp": time.time(),
                    "data": "test_data" * 10
                }
                return json.dumps(message)
            
            metrics = self._measure_performance(create_message, iterations=10000)
            results.append(BenchmarkResult(
                name="network_create_message",
                description="Create and serialize P2P message",
                **metrics
            ))
            
            # Benchmark message parsing
            test_message = json.dumps({
                "type": "test",
                "sender": "benchmark",
                "timestamp": time.time(),
                "data": "test_data" * 10
            })
            
            def parse_message():
                return json.loads(test_message)
            
            metrics = self._measure_performance(parse_message, iterations=10000)
            results.append(BenchmarkResult(
                name="network_parse_message",
                description="Parse and deserialize P2P message",
                **metrics
            ))
            
        except Exception as e:
            logger.error(f"Error benchmarking networking: {e}")
        
        return results
    
    def benchmark_security(self) -> List[BenchmarkResult]:
        """Benchmark security operations"""
        results = []
        
        try:
            from ..security.middleware import InputValidator, RateLimiter
            
            # Benchmark input validation
            validator = InputValidator()
            test_string = "test_string_123"
            
            def validate_string():
                return validator.validate_string(test_string, "alphanumeric")
            
            metrics = self._measure_performance(validate_string, iterations=10000)
            results.append(BenchmarkResult(
                name="security_validate_string",
                description="Validate alphanumeric string",
                **metrics
            ))
            
            # Benchmark rate limiting
            limiter = RateLimiter()
            
            def check_rate_limit():
                return limiter.check_rate_limit("test_client", "test_endpoint")
            
            metrics = self._measure_performance(check_rate_limit, iterations=10000)
            results.append(BenchmarkResult(
                name="security_rate_limit_check",
                description="Check rate limit for client",
                **metrics
            ))
            
        except Exception as e:
            logger.error(f"Error benchmarking security: {e}")
        
        return results
    
    def benchmark_optimization(self) -> List[BenchmarkResult]:
        """Benchmark optimization components"""
        results = []
        
        try:
            from ..blockchain.optimizer import BlockCache, TransactionPoolManager
            
            # Benchmark block cache
            cache = BlockCache(max_size=1000)
            test_block = {"data": "test" * 100}
            
            def cache_put():
                cache.put(f"block_{hash(str(test_block))}", test_block)
            
            metrics = self._measure_performance(cache_put, iterations=1000)
            results.append(BenchmarkResult(
                name="optimizer_cache_put",
                description="Put block in cache",
                **metrics
            ))
            
            def cache_get():
                return cache.get(f"block_{hash(str(test_block))}")
            
            metrics = self._measure_performance(cache_get, iterations=10000)
            results.append(BenchmarkResult(
                name="optimizer_cache_get",
                description="Get block from cache",
                **metrics
            ))
            
            # Benchmark transaction pool
            pool = TransactionPoolManager()
            
            def add_transaction():
                pool.add_transaction(
                    tx_hash=f"tx_{hash(str(time.time()))}",
                    tx_data={"test": "data"},
                    priority="medium"
                )
            
            metrics = self._measure_performance(add_transaction, iterations=1000)
            results.append(BenchmarkResult(
                name="optimizer_tx_pool_add",
                description="Add transaction to pool",
                **metrics
            ))
            
        except Exception as e:
            logger.error(f"Error benchmarking optimization: {e}")
        
        return results
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark suites"""
        logger.info("Starting comprehensive benchmark suite...")
        
        all_results = []
        
        # Run each benchmark suite
        benchmark_suites = [
            ("Hashing", self.benchmark_hashing),
            ("Merkle Tree", self.benchmark_merkle_tree),
            ("Consensus", self.benchmark_consensus),
            ("Federated Learning", self.benchmark_federated_learning),
            ("Networking", self.benchmark_networking),
            ("Security", self.benchmark_security),
            ("Optimization", self.benchmark_optimization)
        ]
        
        for suite_name, suite_func in benchmark_suites:
            logger.info(f"Running {suite_name} benchmarks...")
            try:
                results = suite_func()
                all_results.extend(results)
                logger.info(f"Completed {suite_name}: {len(results)} benchmarks")
            except Exception as e:
                logger.error(f"Error in {suite_name} benchmarks: {e}")
        
        self.results = all_results
        self.save_results()
        
        return all_results
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        results_data = []
        for result in self.results:
            result_dict = {
                "name": result.name,
                "description": result.description,
                "iterations": result.iterations,
                "duration_ms": result.duration_ms,
                "operations_per_second": result.operations_per_second,
                "min_time_ms": result.min_time_ms,
                "max_time_ms": result.max_time_ms,
                "avg_time_ms": result.avg_time_ms,
                "median_time_ms": result.median_time_ms,
                "std_dev_ms": result.std_dev_ms,
                "memory_mb": result.memory_mb,
                "cpu_percent": result.cpu_percent,
                "timestamp": result.timestamp
            }
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            logger.info("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("AEGIS FRAMEWORK - PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, results in categories.items():
            print(f"\n{category.upper()} BENCHMARKS:")
            print("-" * 50)
            
            for result in sorted(results, key=lambda x: x.operations_per_second, reverse=True):
                print(f"  {result.name:30} {result.operations_per_second:10.0f} ops/s "
                      f"({result.avg_time_ms:6.2f}ms avg)")
        
        # Overall statistics
        total_ops = sum(r.operations_per_second for r in self.results)
        avg_ops = total_ops / len(self.results) if self.results else 0
        
        print(f"\n" + "="*80)
        print(f"OVERALL STATISTICS:")
        print(f"  Total Benchmarks: {len(self.results)}")
        print(f"  Average Performance: {avg_ops:,.0f} ops/s")
        print(f"  Best Performance: {max(r.operations_per_second for r in self.results):,.0f} ops/s")
        print(f"  Worst Performance: {min(r.operations_per_second for r in self.results):,.0f} ops/s")
        print("="*80)


# Example usage
if __name__ == "__main__":
    # Run benchmarks
    benchmark_suite = PerformanceBenchmarkSuite()
    results = benchmark_suite.run_all_benchmarks()
    benchmark_suite.print_summary()
