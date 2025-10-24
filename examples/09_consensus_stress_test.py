"""
AEGIS Consensus Stress Test
Performance testing for advanced consensus features under high load
"""

import asyncio
import time
import random
import statistics
from typing import List, Dict, Any
import logging
from pathlib import Path
import sys
import multiprocessing
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.blockchain.advanced_consensus import AdvancedConsensusFeatures
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from src.aegis.core.error_handling import ErrorRecoveryManager
from src.aegis.security.middleware import SecurityMiddleware
from cryptography.hazmat.primitives.asymmetric import ed25519

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Results from a stress test"""
    test_name: str
    duration_seconds: float
    operations_completed: int
    operations_per_second: float
    average_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    success_rate: float
    errors_occurred: int
    memory_mb: float = 0.0


class ConsensusStressTester:
    """Stress tester for AEGIS consensus features"""
    
    def __init__(self, num_nodes: int = 100, test_duration: int = 30):
        self.num_nodes = num_nodes
        self.test_duration = test_duration
        self.results: List[StressTestResult] = []
        
    async def run_all_tests(self) -> List[StressTestResult]:
        """Run all stress tests"""
        logger.info(f"Starting consensus stress tests with {self.num_nodes} nodes")
        
        # Run individual tests
        tests = [
            self._test_validator_selection,
            self._test_timeout_management,
            self._test_performance_tracking,
            self._test_concurrent_operations,
            self._test_error_handling
        ]
        
        for test_func in tests:
            try:
                result = await test_func()
                self.results.append(result)
                logger.info(f"Completed {result.test_name}: {result.operations_per_second:.0f} ops/s")
            except Exception as e:
                logger.error(f"Test failed: {test_func.__name__} - {e}")
        
        return self.results
    
    async def _test_validator_selection(self) -> StressTestResult:
        """Test validator selection under load"""
        start_time = time.time()
        operations = 0
        latencies = []
        errors = 0
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="stress_test", private_key=private_key)
        advanced_features = AdvancedConsensusFeatures(consensus)
        
        # Add mock validators
        mock_validators = [f"node_{i:04d}" for i in range(1000)]
        
        # Pre-record some performance data
        for validator in mock_validators:
            for _ in range(10):
                advanced_features.record_validation_result(
                    node_id=validator,
                    success=random.choice([True, True, False]),
                    response_time=random.uniform(0.1, 2.0),
                    message_type="PREPARE"
                )
        
        # Perform validator selection operations
        while time.time() - start_time < self.test_duration:
            try:
                op_start = time.time()
                
                # Select validators
                selected = await advanced_features.select_optimal_validators(
                    num_validators=20,
                    exclude_nodes={"stress_test"}
                )
                
                op_latency = (time.time() - op_start) * 1000  # Convert to ms
                latencies.append(op_latency)
                operations += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Validator selection error: {e}")
        
        duration = time.time() - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) if (operations + errors) > 0 else 0
        
        return StressTestResult(
            test_name="Validator Selection",
            duration_seconds=duration,
            operations_completed=operations,
            operations_per_second=ops_per_sec,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            success_rate=success_rate,
            errors_occurred=errors
        )
    
    async def _test_timeout_management(self) -> StressTestResult:
        """Test timeout management under load"""
        start_time = time.time()
        operations = 0
        latencies = []
        errors = 0
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="stress_test", private_key=private_key)
        advanced_features = AdvancedConsensusFeatures(consensus)
        
        phases = ["PROPOSING", "PREPARING", "COMMITTING", "FINALIZING"]
        
        # Perform timeout operations
        while time.time() - start_time < self.test_duration:
            try:
                for phase in phases:
                    op_start = time.time()
                    
                    # Start and stop phase timers
                    advanced_features.start_consensus_phase_timer(phase)
                    await asyncio.sleep(0.0001)  # Micro delay
                    advanced_features.stop_consensus_phase_timer(phase, success=True)
                    
                    # Get timeout
                    timeout = advanced_features.get_adaptive_timeout(phase)
                    
                    op_latency = (time.time() - op_start) * 1000  # Convert to ms
                    latencies.append(op_latency)
                    operations += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Timeout management error: {e}")
        
        duration = time.time() - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) if (operations + errors) > 0 else 0
        
        return StressTestResult(
            test_name="Timeout Management",
            duration_seconds=duration,
            operations_completed=operations,
            operations_per_second=ops_per_sec,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            success_rate=success_rate,
            errors_occurred=errors
        )
    
    async def _test_performance_tracking(self) -> StressTestResult:
        """Test performance tracking under load"""
        start_time = time.time()
        operations = 0
        latencies = []
        errors = 0
        
        # Create mock consensus instance
        private_key = ed25519.Ed25519PrivateKey.generate()
        consensus = HybridConsensus(node_id="stress_test", private_key=private_key)
        advanced_features = AdvancedConsensusFeatures(consensus)
        
        # Perform performance tracking operations
        while time.time() - start_time < self.test_duration:
            try:
                op_start = time.time()
                
                # Record validation results
                for i in range(10):
                    advanced_features.record_validation_result(
                        node_id=f"node_{i:03d}",
                        success=random.choice([True, False]),
                        response_time=random.uniform(0.1, 5.0),
                        message_type="PREPARE"
                    )
                
                # Record message latencies
                for i in range(5):
                    advanced_features.record_message_latency(
                        message_type="PREPARE",
                        latency=random.uniform(10, 100)
                    )
                
                # Get performance report
                report = advanced_features.get_performance_report()
                
                op_latency = (time.time() - op_start) * 1000  # Convert to ms
                latencies.append(op_latency)
                operations += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Performance tracking error: {e}")
        
        duration = time.time() - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) if (operations + errors) > 0 else 0
        
        return StressTestResult(
            test_name="Performance Tracking",
            duration_seconds=duration,
            operations_completed=operations,
            operations_per_second=ops_per_sec,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            success_rate=success_rate,
            errors_occurred=errors
        )
    
    async def _test_concurrent_operations(self) -> StressTestResult:
        """Test concurrent operations"""
        start_time = time.time()
        operations = 0
        latencies = []
        errors = 0
        
        # Create multiple consensus instances
        consensus_instances = []
        for i in range(10):
            private_key = ed25519.Ed25519PrivateKey.generate()
            consensus = HybridConsensus(node_id=f"node_{i:02d}", private_key=private_key)
            advanced_features = AdvancedConsensusFeatures(consensus)
            consensus_instances.append(advanced_features)
        
        # Perform concurrent operations
        while time.time() - start_time < self.test_duration:
            try:
                op_start = time.time()
                
                # Create concurrent tasks
                tasks = []
                for advanced_features in consensus_instances:
                    task = asyncio.create_task(self._perform_concurrent_operation(advanced_features))
                    tasks.append(task)
                
                # Wait for all tasks
                await asyncio.gather(*tasks, return_exceptions=True)
                
                op_latency = (time.time() - op_start) * 1000  # Convert to ms
                latencies.append(op_latency)
                operations += len(tasks)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Concurrent operations error: {e}")
        
        duration = time.time() - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) if (operations + errors) > 0 else 0
        
        return StressTestResult(
            test_name="Concurrent Operations",
            duration_seconds=duration,
            operations_completed=operations,
            operations_per_second=ops_per_sec,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            success_rate=success_rate,
            errors_occurred=errors
        )
    
    async def _perform_concurrent_operation(self, advanced_features):
        """Perform a single concurrent operation"""
        try:
            # Record validation
            advanced_features.record_validation_result(
                node_id="concurrent_node",
                success=True,
                response_time=random.uniform(0.1, 1.0),
                message_type="PREPARE"
            )
            
            # Get timeout
            timeout = advanced_features.get_adaptive_timeout("PREPARING")
            
            # Select validators
            await advanced_features.select_optimal_validators(
                num_validators=5,
                exclude_nodes={"current_node"}
            )
            
        except Exception as e:
            logger.warning(f"Concurrent operation error: {e}")
    
    async def _test_error_handling(self) -> StressTestResult:
        """Test error handling under load"""
        start_time = time.time()
        operations = 0
        latencies = []
        errors = 0
        handled_errors = 0
        
        # Create error recovery manager
        error_manager = ErrorRecoveryManager()
        error_manager.register_component("stress_test")
        
        # Perform error handling operations
        while time.time() - start_time < self.test_duration:
            try:
                op_start = time.time()
                
                # Simulate errors
                for i in range(5):
                    try:
                        # Simulate various errors
                        error_type = random.choice([
                            ValueError("Test error"),
                            ConnectionError("Network error"),
                            RuntimeError("Runtime error")
                        ])
                        
                        # Handle error
                        await error_manager.handle_error(
                            component="stress_test",
                            error=error_type,
                            severity=random.choice([
                                "low", "medium", "high"
                            ])
                        )
                        
                        handled_errors += 1
                        
                    except Exception:
                        errors += 1
                
                op_latency = (time.time() - op_start) * 1000  # Convert to ms
                latencies.append(op_latency)
                operations += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error handling error: {e}")
        
        duration = time.time() - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        success_rate = (handled_errors / (handled_errors + errors)) if (handled_errors + errors) > 0 else 0
        
        return StressTestResult(
            test_name="Error Handling",
            duration_seconds=duration,
            operations_completed=operations,
            operations_per_second=ops_per_sec,
            average_latency_ms=statistics.mean(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            success_rate=success_rate,
            errors_occurred=errors
        )
    
    def print_results(self):
        """Print stress test results"""
        if not self.results:
            logger.info("No test results available")
            return
        
        print("\n" + "="*100)
        print("AEGIS CONSENSUS STRESS TEST RESULTS")
        print("="*100)
        
        # Sort results by operations per second
        sorted_results = sorted(self.results, key=lambda x: x.operations_per_second, reverse=True)
        
        print(f"{'Test Name':<25} {'Ops/Sec':<12} {'Avg Latency':<12} {'Success Rate':<12} {'Errors':<8}")
        print("-" * 100)
        
        for result in sorted_results:
            print(f"{result.test_name:<25} "
                  f"{result.operations_per_second:>10.0f} "
                  f"{result.average_latency_ms:>10.2f}ms "
                  f"{result.success_rate:>10.1%} "
                  f"{result.errors_occurred:>8}")
        
        print("-" * 100)
        
        # Summary statistics
        total_ops = sum(r.operations_per_second for r in self.results)
        avg_ops = total_ops / len(self.results) if self.results else 0
        avg_success = statistics.mean(r.success_rate for r in self.results) if self.results else 0
        
        print(f"Average Performance: {avg_ops:,.0f} ops/s")
        print(f"Average Success Rate: {avg_success:.1%}")
        print(f"Best Performance: {max(r.operations_per_second for r in self.results):,.0f} ops/s")
        print(f"Total Tests: {len(self.results)}")
        
        print("="*100)


async def run_stress_test():
    """Run the stress test"""
    logger.info("Starting AEGIS Consensus Stress Test")
    
    # Create stress tester
    tester = ConsensusStressTester(num_nodes=100, test_duration=15)
    
    # Run tests
    results = await tester.run_all_tests()
    
    # Print results
    tester.print_results()
    
    logger.info("Stress test completed")


if __name__ == "__main__":
    asyncio.run(run_stress_test())
