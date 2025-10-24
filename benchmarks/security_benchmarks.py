"""
Security benchmarks for AEGIS Advanced Security Features
Performance testing for zero-knowledge proofs, homomorphic encryption, and security operations
"""

import time
import statistics
import random
from typing import List, Dict, Any
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.advanced_crypto import (
    AdvancedSecurityManager,
    ZeroKnowledgeProver,
    HomomorphicEncryption,
    SecureMultiPartyComputation,
    DifferentialPrivacy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityBenchmarkResult:
    """Result of a security benchmark"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.times: List[float] = []
        self.memory_usage: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.total_operations = 0
    
    def add_result(self, elapsed_time: float, memory_mb: float = 0.0, success: bool = True):
        """Add a benchmark result"""
        self.times.append(elapsed_time)
        self.memory_usage.append(memory_mb)
        self.total_operations += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics"""
        if not self.times:
            return {}
        
        return {
            "name": self.name,
            "description": self.description,
            "operations": self.total_operations,
            "success_rate": self.success_count / self.total_operations if self.total_operations > 0 else 0,
            "avg_time_ms": statistics.mean(self.times) * 1000,
            "min_time_ms": min(self.times) * 1000,
            "max_time_ms": max(self.times) * 1000,
            "median_time_ms": statistics.median(self.times) * 1000,
            "std_dev_ms": statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0,
            "ops_per_second": self.total_operations / sum(self.times) if sum(self.times) > 0 else 0,
            "avg_memory_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0
        }


class SecurityBenchmarkSuite:
    """Benchmark suite for advanced security features"""
    
    def __init__(self):
        self.results: List[SecurityBenchmarkResult] = []
        self.security_manager = AdvancedSecurityManager()
    
    def benchmark_zk_proofs(self) -> SecurityBenchmarkResult:
        """Benchmark zero-knowledge proofs"""
        result = SecurityBenchmarkResult(
            "zk_proofs",
            "Zero-knowledge proof generation and verification"
        )
        
        # Benchmark proof generation
        for i in range(1000):
            start_time = time.time()
            
            try:
                proof = self.security_manager.create_zk_proof(
                    f"secret_{i}".encode(),
                    f"statement_{i}",
                    "benchmark_verifier"
                )
                
                # Verify the proof
                is_valid = self.security_manager.verify_zk_proof(
                    proof, f"statement_{i}"
                )
                
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=is_valid)
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=False)
                logger.warning(f"ZK proof benchmark error: {e}")
        
        return result
    
    def benchmark_homomorphic_encryption(self) -> SecurityBenchmarkResult:
        """Benchmark homomorphic encryption operations"""
        result = SecurityBenchmarkResult(
            "homomorphic_encryption",
            "Homomorphic encryption, decryption, and operations"
        )
        
        # Benchmark basic encryption/decryption
        for i in range(1000):
            start_time = time.time()
            
            try:
                # Encrypt
                encrypted = self.security_manager.encrypt_value(i, {"benchmark": True})
                
                # Decrypt
                decrypted = self.security_manager.decrypt_value(encrypted)
                
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=(decrypted == i))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=False)
                logger.warning(f"Homomorphic encryption benchmark error: {e}")
        
        # Benchmark homomorphic addition
        result_add = SecurityBenchmarkResult(
            "homomorphic_addition",
            "Homomorphic addition of encrypted values"
        )
        
        for i in range(500):
            start_time = time.time()
            
            try:
                # Encrypt two values
                a = self.security_manager.encrypt_value(i)
                b = self.security_manager.encrypt_value(i * 2)
                
                # Add homomorphically
                encrypted_sum = self.security_manager.add_encrypted_values(a, b)
                
                # Decrypt result
                decrypted_sum = self.security_manager.decrypt_value(encrypted_sum)
                expected_sum = i + (i * 2)
                
                elapsed_time = time.time() - start_time
                result_add.add_result(elapsed_time, success=(decrypted_sum == expected_sum))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result_add.add_result(elapsed_time, success=False)
                logger.warning(f"Homomorphic addition benchmark error: {e}")
        
        self.results.append(result_add)
        
        # Benchmark homomorphic multiplication
        result_mult = SecurityBenchmarkResult(
            "homomorphic_multiplication",
            "Homomorphic multiplication by scalar"
        )
        
        for i in range(500):
            start_time = time.time()
            
            try:
                # Encrypt value
                encrypted = self.security_manager.encrypt_value(i)
                
                # Multiply by scalar
                scalar = 3
                encrypted_product = self.security_manager.multiply_encrypted_by_scalar(
                    encrypted, scalar
                )
                
                # Decrypt result
                decrypted_product = self.security_manager.decrypt_value(encrypted_product)
                expected_product = i * scalar
                
                elapsed_time = time.time() - start_time
                result_mult.add_result(elapsed_time, success=(decrypted_product == expected_product))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result_mult.add_result(elapsed_time, success=False)
                logger.warning(f"Homomorphic multiplication benchmark error: {e}")
        
        self.results.append(result_mult)
        
        return result
    
    def benchmark_smc(self) -> SecurityBenchmarkResult:
        """Benchmark secure multi-party computation"""
        result = SecurityBenchmarkResult(
            "smc",
            "Secure multi-party computation operations"
        )
        
        # Add parties
        for i in range(10):
            self.security_manager.add_party_to_smc(f"party_{i:03d}")
        
        # Benchmark secret sharing
        for i in range(200):
            start_time = time.time()
            
            try:
                # Generate shares
                shares = self.security_manager.generate_secret_shares(i, threshold=3)
                
                # Reconstruct secret
                reconstructed = self.security_manager.reconstruct_secret_from_shares(shares)
                
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=(reconstructed == i))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=False)
                logger.warning(f"SMC benchmark error: {e}")
        
        return result
    
    def benchmark_differential_privacy(self) -> SecurityBenchmarkResult:
        """Benchmark differential privacy operations"""
        result = SecurityBenchmarkResult(
            "differential_privacy",
            "Differential privacy mechanisms"
        )
        
        # Benchmark count queries
        for i in range(500):
            start_time = time.time()
            
            try:
                private_count = self.security_manager.privatize_data(i, "count")
                
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=isinstance(private_count, float))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=False)
                logger.warning(f"Differential privacy benchmark error: {e}")
        
        # Benchmark sum queries
        result_sum = SecurityBenchmarkResult(
            "dp_sum",
            "Differential privacy sum queries"
        )
        
        for i in range(500):
            start_time = time.time()
            
            try:
                private_sum = self.security_manager.privatize_data(
                    float(i * 100), "sum", max_value=1000.0
                )
                
                elapsed_time = time.time() - start_time
                result_sum.add_result(elapsed_time, success=isinstance(private_sum, float))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result_sum.add_result(elapsed_time, success=False)
                logger.warning(f"DP sum benchmark error: {e}")
        
        self.results.append(result_sum)
        
        # Benchmark mean queries
        result_mean = SecurityBenchmarkResult(
            "dp_mean",
            "Differential privacy mean queries"
        )
        
        for i in range(100):
            start_time = time.time()
            
            try:
                values = [random.uniform(0, 100) for _ in range(10)]
                private_mean = self.security_manager.privatize_data(values, "mean")
                
                elapsed_time = time.time() - start_time
                result_mean.add_result(elapsed_time, success=isinstance(private_mean, float))
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result_mean.add_result(elapsed_time, success=False)
                logger.warning(f"DP mean benchmark error: {e}")
        
        self.results.append(result_mean)
        
        return result
    
    def benchmark_concurrent_operations(self) -> SecurityBenchmarkResult:
        """Benchmark concurrent security operations"""
        result = SecurityBenchmarkResult(
            "concurrent_operations",
            "Concurrent security operations"
        )
        
        # Perform multiple operations concurrently
        operations = []
        
        for i in range(100):
            start_time = time.time()
            
            try:
                # Mix of different operations
                op_type = i % 4
                
                if op_type == 0:
                    # ZK proof
                    proof = self.security_manager.create_zk_proof(
                        f"concurrent_secret_{i}".encode(),
                        f"concurrent_statement_{i}",
                        "concurrent_verifier"
                    )
                    success = self.security_manager.verify_zk_proof(
                        proof, f"concurrent_statement_{i}"
                    )
                elif op_type == 1:
                    # Homomorphic encryption
                    encrypted = self.security_manager.encrypt_value(i)
                    decrypted = self.security_manager.decrypt_value(encrypted)
                    success = (decrypted == i)
                elif op_type == 2:
                    # SMC
                    shares = self.security_manager.generate_secret_shares(i, 2)
                    reconstructed = self.security_manager.reconstruct_secret_from_shares(shares)
                    success = (reconstructed == i)
                else:
                    # Differential privacy
                    private_value = self.security_manager.privatize_data(i, "count")
                    success = isinstance(private_value, float)
                
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=success)
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                result.add_result(elapsed_time, success=False)
                logger.warning(f"Concurrent operations benchmark error: {e}")
        
        return result
    
    def run_all_benchmarks(self) -> List[SecurityBenchmarkResult]:
        """Run all security benchmarks"""
        logger.info("Starting AEGIS Security Benchmark Suite")
        
        # Run individual benchmarks
        benchmarks = [
            self.benchmark_zk_proofs,
            self.benchmark_homomorphic_encryption,
            self.benchmark_smc,
            self.benchmark_differential_privacy,
            self.benchmark_concurrent_operations
        ]
        
        for benchmark_func in benchmarks:
            try:
                result = benchmark_func()
                self.results.append(result)
                logger.info(f"Completed benchmark: {result.name}")
            except Exception as e:
                logger.error(f"Benchmark failed: {benchmark_func.__name__} - {e}")
        
        return self.results
    
    def print_report(self):
        """Print benchmark report"""
        if not self.results:
            logger.info("No benchmark results available")
            return
        
        print("\n" + "="*100)
        print("AEGIS SECURITY BENCHMARK RESULTS")
        print("="*100)
        
        # Sort results by operations per second
        sorted_results = sorted(
            [r for r in self.results if r.get_stats()],
            key=lambda x: x.get_stats().get("ops_per_second", 0),
            reverse=True
        )
        
        print(f"{'Benchmark':<30} {'Ops/Sec':<12} {'Avg Time':<12} {'Success Rate':<12} {'Operations':<10}")
        print("-" * 100)
        
        for result in sorted_results:
            stats = result.get_stats()
            print(f"{stats['name']:<30} "
                  f"{stats['ops_per_second']:>10.0f} "
                  f"{stats['avg_time_ms']:>10.2f}ms "
                  f"{stats['success_rate']:>10.1%} "
                  f"{stats['operations']:>10}")
        
        print("-" * 100)
        
        # Summary statistics
        total_ops = sum(r.total_operations for r in self.results)
        avg_success_rate = statistics.mean(
            [r.success_count / r.total_operations if r.total_operations > 0 else 0 
             for r in self.results]
        ) if self.results else 0
        
        print(f"Total Operations: {total_ops:,}")
        print(f"Average Success Rate: {avg_success_rate:.1%}")
        print(f"Best Performance: {max(r.get_stats().get('ops_per_second', 0) for r in self.results):,.0f} ops/s")
        print(f"Total Benchmarks: {len(self.results)}")
        
        print("="*100)
    
    def export_results(self, filename: str = None) -> str:
        """Export benchmark results to JSON file"""
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_benchmark_results_{timestamp}.json"
        
        results_data = []
        for result in self.results:
            stats = result.get_stats()
            if stats:
                results_data.append(stats)
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results_data,
            "total_operations": sum(r.total_operations for r in self.results),
            "average_success_rate": statistics.mean(
                [r.success_count / r.total_operations if r.total_operations > 0 else 0 
                 for r in self.results]
            ) if self.results else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Security benchmark results exported to {filename}")
        return filename


def run_security_benchmarks():
    """Run the security benchmark suite"""
    logger.info("Starting AEGIS Security Benchmarks")
    
    # Create benchmark suite
    benchmark_suite = SecurityBenchmarkSuite()
    
    # Run benchmarks
    results = benchmark_suite.run_all_benchmarks()
    
    # Print report
    benchmark_suite.print_report()
    
    # Export results
    filename = benchmark_suite.export_results()
    logger.info(f"Benchmark results saved to {filename}")
    
    return results


if __name__ == "__main__":
    run_security_benchmarks()
