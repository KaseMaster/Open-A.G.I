"""
Advanced Benchmark Tests for AEGIS Consensus Features
Performance and stress tests for dynamic validator selection and adaptive timeouts
"""

import asyncio
import time
import statistics
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.blockchain.advanced_consensus import (
    DynamicValidatorSelector,
    AdaptiveTimeoutManager,
    AdvancedConsensusFeatures
)
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from cryptography.hazmat.primitives.asymmetric import ed25519

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
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
    success_rate: float
    memory_mb: Optional[float] = None
    timestamp: float = time.time()


class ConsensusBenchmarkSuite:
    """Benchmark suite for advanced consensus features"""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[BenchmarkResult] = []
    
    def _measure_performance(
        self,
        func: callable,
        iterations: int = 1000,
        warmup: int = 100
    ) -> Dict[str, Any]:
        """Measure performance of a function"""
        # Warmup
        for _ in range(warmup):
            func()
        
        # Actual measurement
        times = []
        successes = 0
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            try:
                iter_start = time.perf_counter()
                func()
                iter_end = time.perf_counter()
                times.append((iter_end - iter_start) * 1000)  # Convert to ms
                successes += 1
            except Exception as e:
                logger.warning(f"Iteration {i} failed: {e}")
                # Still record time for failed iterations
                iter_end = time.perf_counter()
                times.append((iter_end - start_time) * 1000)
        
        end_time = time.perf_counter()
        
        total_duration = (end_time - start_time) * 1000  # Convert to ms
        ops_per_sec = iterations / (total_duration / 1000)
        success_rate = successes / iterations if iterations > 0 else 0
        
        return {
            "iterations": iterations,
            "duration_ms": total_duration,
            "operations_per_second": ops_per_sec,
            "min_time_ms": min(times) if times else 0,
            "max_time_ms": max(times) if times else 0,
            "avg_time_ms": statistics.mean(times) if times else 0,
            "median_time_ms": statistics.median(times) if times else 0,
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "success_rate": success_rate
        }
    
    def benchmark_validator_selection(self) -> List[BenchmarkResult]:
        """Benchmark dynamic validator selection"""
        results = []
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="benchmark_node", private_key=private_key)
        
        # Add mock nodes
        mock_nodes = [f"node_{i:03d}" for i in range(100)]
        for node_id in mock_nodes:
            consensus.pbft.known_nodes.add(node_id)
            consensus.pbft.node_reputations[node_id] = consensus.pbft._create_default_reputation(node_id)
        
        # Create validator selector
        selector = DynamicValidatorSelector(consensus)
        
        # Benchmark selection with different numbers of validators
        for num_validators in [5, 10, 20, 50]:
            def select_validators():
                return selector.select_validators(num_validators)
            
            metrics = self._measure_performance(select_validators, iterations=1000)
            results.append(BenchmarkResult(
                name=f"consensus_validator_selection_{num_validators}",
                description=f"Select {num_validators} validators from 100 nodes",
                **metrics
            ))
        
        return results
    
    def benchmark_timeout_management(self) -> List[BenchmarkResult]:
        """Benchmark adaptive timeout management"""
        results = []
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="benchmark_node", private_key=private_key)
        
        # Create timeout manager
        timeout_manager = AdaptiveTimeoutManager(consensus)
        
        # Benchmark timeout retrieval
        def get_timeout():
            return timeout_manager.get_timeout(consensus.pbft.state)
        
        metrics = self._measure_performance(get_timeout, iterations=10000)
        results.append(BenchmarkResult(
            name="consensus_timeout_get",
            description="Get adaptive timeout value",
            **metrics
        ))
        
        # Benchmark timeout adjustment
        def adjust_timeout():
            timeout_manager._adjust_timeout(consensus.pbft.state, random.uniform(1.0, 10.0))
        
        metrics = self._measure_performance(adjust_timeout, iterations=10000)
        results.append(BenchmarkResult(
            name="consensus_timeout_adjust",
            description="Adjust adaptive timeout value",
            **metrics
        ))
        
        return results
    
    def benchmark_performance_tracking(self) -> List[BenchmarkResult]:
        """Benchmark performance tracking and recording"""
        results = []
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="benchmark_node", private_key=private_key)
        
        # Create advanced features
        advanced_features = AdvancedConsensusFeatures(consensus)
        
        # Benchmark validation result recording
        def record_validation():
            advanced_features.record_validation_result(
                node_id=f"node_{random.randint(0, 99):03d}",
                success=random.choice([True, False]),
                response_time=random.uniform(0.1, 5.0),
                message_type=random.choice(list(advanced_features.consensus.pbft.MessageType))
            )
        
        metrics = self._measure_performance(record_validation, iterations=10000)
        results.append(BenchmarkResult(
            name="consensus_record_validation",
            description="Record validation result with performance tracking",
            **metrics
        ))
        
        # Benchmark message latency recording
        def record_latency():
            advanced_features.record_message_latency(
                message_type=random.choice(list(advanced_features.consensus.pbft.MessageType)),
                latency=random.uniform(0.1, 100.0)
            )
        
        metrics = self._measure_performance(record_latency, iterations=10000)
        results.append(BenchmarkResult(
            name="consensus_record_latency",
            description="Record message latency",
            **metrics
        ))
        
        return results
    
    def benchmark_stress_scenarios(self) -> List[BenchmarkResult]:
        """Benchmark stress scenarios with high load"""
        results = []
        
        # Create multiple consensus instances
        consensuses = []
        selectors = []
        
        for i in range(10):
            private_key = ed25519.Ed25519PrivateKey.generate()
            consensus = HybridConsensus(node_id=f"node_{i:02d}", private_key=private_key)
            
            # Add mock nodes
            mock_nodes = [f"node_{i}_{j:03d}" for j in range(50)]
            for node_id in mock_nodes:
                consensus.pbft.known_nodes.add(node_id)
                consensus.pbft.node_reputations[node_id] = consensus.pbft._create_default_reputation(node_id)
            
            consensuses.append(consensus)
            selectors.append(DynamicValidatorSelector(consensus))
        
        # Benchmark concurrent validator selection
        async def concurrent_selection():
            tasks = []
            for selector in selectors:
                task = asyncio.create_task(asyncio.get_event_loop().run_in_executor(
                    None, lambda s=selector: s.select_validators(10)
                ))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        def run_concurrent_selection():
            asyncio.run(concurrent_selection())
        
        metrics = self._measure_performance(run_concurrent_selection, iterations=100)
        results.append(BenchmarkResult(
            name="consensus_concurrent_selection",
            description="Concurrent validator selection across 10 nodes",
            **metrics
        ))
        
        return results
    
    def benchmark_recovery_scenarios(self) -> List[BenchmarkResult]:
        """Benchmark recovery scenarios"""
        results = []
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="benchmark_node", private_key=private_key)
        
        # Create advanced features
        advanced_features = AdvancedConsensusFeatures(consensus)
        
        # Simulate multiple validation results to build history
        for i in range(1000):
            advanced_features.record_validation_result(
                node_id=f"node_{i % 50:03d}",
                success=i % 10 != 0,  # 90% success rate
                response_time=random.uniform(0.1, 5.0),
                message_type=consensus.pbft.MessageType.PREPARE
            )
        
        # Benchmark performance report generation
        def generate_report():
            return advanced_features.get_performance_report()
        
        metrics = self._measure_performance(generate_report, iterations=100)
        results.append(BenchmarkResult(
            name="consensus_performance_report",
            description="Generate comprehensive performance report",
            **metrics
        ))
        
        return results
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all consensus benchmark suites"""
        logger.info("Starting advanced consensus benchmark suite...")
        
        all_results = []
        
        # Run each benchmark suite
        benchmark_suites = [
            ("Validator Selection", self.benchmark_validator_selection),
            ("Timeout Management", self.benchmark_timeout_management),
            ("Performance Tracking", self.benchmark_performance_tracking),
            ("Stress Scenarios", self.benchmark_stress_scenarios),
            ("Recovery Scenarios", self.benchmark_recovery_scenarios)
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
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"consensus_benchmark_results_{timestamp}.json"
        
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
                "success_rate": result.success_rate,
                "memory_mb": result.memory_mb,
                "timestamp": result.timestamp
            }
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Consensus benchmark results saved to {filepath}")
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            logger.info("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("AEGIS CONSENSUS - ADVANCED BENCHMARK SUMMARY")
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
                print(f"  {result.name:40} {result.operations_per_second:10.0f} ops/s "
                      f"({result.avg_time_ms:6.2f}ms avg) "
                      f"[{result.success_rate:5.1%} success]")
        
        # Overall statistics
        total_ops = sum(r.operations_per_second for r in self.results)
        avg_ops = total_ops / len(self.results) if self.results else 0
        avg_success = statistics.mean(r.success_rate for r in self.results) if self.results else 0
        
        print(f"\n" + "="*80)
        print(f"OVERALL STATISTICS:")
        print(f"  Total Benchmarks: {len(self.results)}")
        print(f"  Average Performance: {avg_ops:,.0f} ops/s")
        print(f"  Average Success Rate: {avg_success:.1%}")
        print(f"  Best Performance: {max(r.operations_per_second for r in self.results):,.0f} ops/s")
        print(f"  Worst Performance: {min(r.operations_per_second for r in self.results):,.0f} ops/s")
        print("="*80)


# Example usage
if __name__ == "__main__":
    # Run benchmarks
    benchmark_suite = ConsensusBenchmarkSuite()
    results = benchmark_suite.run_all_benchmarks()
    benchmark_suite.print_summary()
