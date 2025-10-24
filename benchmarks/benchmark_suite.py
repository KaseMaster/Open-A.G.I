#!/usr/bin/env python3
"""
AEGIS Framework - Automated Benchmark Suite
Comprehensive performance benchmarking for all core components
"""

import time
import asyncio
import statistics
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.crypto_framework import CryptoEngine, SecurityLevel, CryptoConfig
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from src.aegis.blockchain.merkle_tree import MerkleTree
from src.aegis.networking.p2p_network import NodeType


@dataclass
class BenchmarkResult:
    """Result of a single benchmark"""
    name: str
    description: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    ops_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkSuite:
    """Automated benchmark suite for AEGIS components"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.start_time = datetime.now()
        
    def benchmark(self, name: str, description: str, iterations: int = 1000):
        """Decorator for benchmarking functions"""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                times = []
                
                print(f"\n🔬 Benchmarking: {name}")
                print(f"   Description: {description}")
                print(f"   Iterations: {iterations}")
                
                for i in range(iterations):
                    start = time.perf_counter()
                    func(*args, **kwargs)
                    end = time.perf_counter()
                    times.append(end - start)
                    
                    if (i + 1) % (iterations // 10) == 0:
                        progress = (i + 1) / iterations * 100
                        print(f"   Progress: {progress:.0f}%", end='\r')
                
                total_time = sum(times)
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                median_time = statistics.median(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                ops_per_second = 1 / avg_time if avg_time > 0 else 0
                
                result = BenchmarkResult(
                    name=name,
                    description=description,
                    iterations=iterations,
                    total_time=total_time,
                    avg_time=avg_time,
                    min_time=min_time,
                    max_time=max_time,
                    median_time=median_time,
                    std_dev=std_dev,
                    ops_per_second=ops_per_second
                )
                
                self.results.append(result)
                
                print(f"\n   ✅ Average: {avg_time*1000:.3f} ms")
                print(f"   ⚡ Ops/sec: {ops_per_second:.2f}")
                print(f"   📊 Min/Max: {min_time*1000:.3f} / {max_time*1000:.3f} ms")
                
                return result
            return wrapper
        return decorator
    
    def run_crypto_benchmarks(self):
        """Benchmark cryptographic operations"""
        print("\n" + "="*60)
        print("🔐 CRYPTOGRAPHIC OPERATIONS BENCHMARKS")
        print("="*60)
        
        crypto = CryptoEngine()
        
        # Benchmark: Node Identity Generation
        @self.benchmark(
            "Crypto: Generate Node Identity",
            "Generate Ed25519 + X25519 key pairs",
            iterations=100
        )
        def bench_identity_generation():
            crypto.generate_node_identity(f"node_{time.time()}")
        
        bench_identity_generation()
        
        # Setup for signing benchmarks
        identity = crypto.generate_node_identity("bench_node")
        crypto.identity = identity
        test_data = b"benchmark_test_data" * 100  # 1.9KB
        
        # Benchmark: Data Signing
        @self.benchmark(
            "Crypto: Sign Data",
            "Sign 1.9KB data with Ed25519",
            iterations=1000
        )
        def bench_sign_data():
            crypto.sign_data(test_data)
        
        bench_sign_data()
        
        # Benchmark: Key Export
        @self.benchmark(
            "Crypto: Export Public Identity",
            "Serialize public keys for network",
            iterations=1000
        )
        def bench_export_identity():
            identity.export_public_identity()
        
        bench_export_identity()
    
    def run_merkle_tree_benchmarks(self):
        """Benchmark Merkle Tree operations"""
        print("\n" + "="*60)
        print("🌳 MERKLE TREE BENCHMARKS")
        print("="*60)
        
        # Benchmark: Add Single Leaf
        @self.benchmark(
            "Merkle: Add Leaf",
            "Add single leaf to tree",
            iterations=10000
        )
        def bench_add_leaf():
            tree = MerkleTree()
            tree.add_leaf(b"test_data")
        
        bench_add_leaf()
        
        # Benchmark: Build Tree (100 leaves)
        @self.benchmark(
            "Merkle: Build Tree (100 leaves)",
            "Build complete Merkle tree",
            iterations=100
        )
        def bench_build_tree():
            tree = MerkleTree()
            for i in range(100):
                tree.add_leaf(f"leaf_{i}".encode())
            tree.make_tree()
        
        bench_build_tree()
        
        # Benchmark: Generate Proof
        tree = MerkleTree()
        for i in range(100):
            tree.add_leaf(f"leaf_{i}".encode())
        tree.make_tree()
        
        @self.benchmark(
            "Merkle: Generate Proof",
            "Generate inclusion proof for leaf",
            iterations=1000
        )
        def bench_generate_proof():
            tree.get_proof(50)
        
        bench_generate_proof()
        
        # Skip verify proof benchmark if method doesn't exist
        if hasattr(tree, 'verify_proof'):
            proof = tree.get_proof(50)
            leaf = tree.leaves[50]
            root = tree.get_merkle_root()
            
            @self.benchmark(
                "Merkle: Verify Proof",
                "Verify Merkle proof validity",
                iterations=1000
            )
            def bench_verify_proof():
                tree.verify_proof(proof, leaf, root)
            
            bench_verify_proof()
    
    def run_consensus_benchmarks(self):
        """Benchmark consensus operations"""
        print("\n" + "="*60)
        print("⛓️  CONSENSUS PROTOCOL BENCHMARKS")
        print("="*60)
        
        from cryptography.hazmat.primitives.asymmetric import ed25519
        
        # Benchmark: Consensus Initialization
        @self.benchmark(
            "Consensus: Initialize",
            "Initialize HybridConsensus instance",
            iterations=100
        )
        def bench_consensus_init():
            private_key = ed25519.Ed25519PrivateKey.generate()
            HybridConsensus(
                node_id=f"node_{time.time()}",
                private_key=private_key
            )
        
        bench_consensus_init()
    
    def run_hashing_benchmarks(self):
        """Benchmark hashing operations"""
        print("\n" + "="*60)
        print("🔢 HASHING BENCHMARKS")
        print("="*60)
        
        import hashlib
        test_data = b"benchmark_data" * 100  # ~1.4KB
        
        # SHA-256
        @self.benchmark(
            "Hash: SHA-256",
            "Hash 1.4KB with SHA-256",
            iterations=10000
        )
        def bench_sha256():
            hashlib.sha256(test_data).digest()
        
        bench_sha256()
        
        # SHA-512
        @self.benchmark(
            "Hash: SHA-512",
            "Hash 1.4KB with SHA-512",
            iterations=10000
        )
        def bench_sha512():
            hashlib.sha512(test_data).digest()
        
        bench_sha512()
        
        # BLAKE2b
        @self.benchmark(
            "Hash: BLAKE2b",
            "Hash 1.4KB with BLAKE2b",
            iterations=10000
        )
        def bench_blake2b():
            hashlib.blake2b(test_data).digest()
        
        bench_blake2b()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "benchmark_suite": "AEGIS Framework Performance Benchmarks",
            "version": "2.0.0",
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": duration,
            "total_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        categories = {
            "Crypto": [r for r in self.results if r.name.startswith("Crypto")],
            "Merkle": [r for r in self.results if r.name.startswith("Merkle")],
            "Hash": [r for r in self.results if r.name.startswith("Hash")],
            "Consensus": [r for r in self.results if r.name.startswith("Consensus")]
        }
        
        summary = {}
        for category, results in categories.items():
            if results:
                summary[category] = {
                    "count": len(results),
                    "avg_ops_per_second": statistics.mean([r.ops_per_second for r in results]),
                    "total_operations": sum([r.iterations for r in results])
                }
        
        return summary
    
    def print_report(self):
        """Print formatted benchmark report"""
        print("\n" + "="*60)
        print("📊 BENCHMARK REPORT")
        print("="*60)
        
        report = self.generate_report()
        
        print(f"\n⏱️  Total Duration: {report['duration_seconds']:.2f} seconds")
        print(f"📈 Total Benchmarks: {report['total_benchmarks']}")
        
        print("\n🎯 Summary by Category:")
        for category, stats in report['summary'].items():
            print(f"\n   {category}:")
            print(f"      Benchmarks: {stats['count']}")
            print(f"      Avg Ops/sec: {stats['avg_ops_per_second']:.2f}")
            print(f"      Total Ops: {stats['total_operations']:,}")
        
        print("\n📋 Top 5 Fastest Operations:")
        sorted_results = sorted(self.results, key=lambda x: x.avg_time)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"   {i}. {result.name}: {result.avg_time*1000:.3f} ms ({result.ops_per_second:.0f} ops/s)")
        
        print("\n⚠️  Top 5 Slowest Operations:")
        for i, result in enumerate(sorted(self.results, key=lambda x: x.avg_time, reverse=True)[:5], 1):
            print(f"   {i}. {result.name}: {result.avg_time*1000:.3f} ms ({result.ops_per_second:.0f} ops/s)")
    
    def save_report(self, filename: str = "benchmark_results.json"):
        """Save report to JSON file"""
        report = self.generate_report()
        
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {output_path}")


def main():
    """Run all benchmarks"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         AEGIS FRAMEWORK BENCHMARK SUITE v2.0.0           ║
║                                                           ║
║  Automated Performance Testing for Core Components       ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    suite = BenchmarkSuite()
    
    # Run all benchmark categories
    suite.run_hashing_benchmarks()
    suite.run_crypto_benchmarks()
    suite.run_merkle_tree_benchmarks()
    suite.run_consensus_benchmarks()
    
    # Generate and display report
    suite.print_report()
    suite.save_report()
    
    print("\n✅ Benchmark suite completed successfully!\n")


if __name__ == "__main__":
    main()
