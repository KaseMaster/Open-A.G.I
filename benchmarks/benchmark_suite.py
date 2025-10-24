#!/usr/bin/env python3
"""
AEGIS Framework - Benchmark Suite
Suite completa de benchmarks de rendimiento
"""

import sys
import time
import statistics
from pathlib import Path
from typing import Callable, List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class BenchmarkRunner:
    """Ejecutor de benchmarks con estadísticas"""
    
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        self.results = {}
    
    def benchmark(self, name: str, func: Callable, warmup: int = 100) -> Dict:
        """
        Ejecuta un benchmark y retorna estadísticas
        
        Args:
            name: Nombre del benchmark
            func: Función a benchmarkear
            warmup: Iteraciones de calentamiento
        
        Returns:
            Dict con estadísticas
        """
        print(f"  ⏱️  {name}...", end='', flush=True)
        
        # Warmup
        for _ in range(warmup):
            func()
        
        # Medición
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        stats = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'p95': sorted(times)[int(len(times) * 0.95)],
            'p99': sorted(times)[int(len(times) * 0.99)],
            'iterations': self.iterations
        }
        
        print(f" {stats['mean']:.3f}ms (±{stats['stdev']:.3f}ms)")
        
        self.results[name] = stats
        return stats
    
    def print_summary(self):
        """Imprime resumen de todos los benchmarks"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Benchmark':<30} {'Mean':>10} {'Median':>10} {'P95':>10} {'P99':>10}")
        print("-"*70)
        
        for name, stats in self.results.items():
            print(f"{name:<30} {stats['mean']:>9.3f}ms {stats['median']:>9.3f}ms "
                  f"{stats['p95']:>9.3f}ms {stats['p99']:>9.3f}ms")
        
        print("="*70)


def main():
    print("🏃 AEGIS Framework - Performance Benchmarks")
    print("="*70)
    print()
    
    runner = BenchmarkRunner(iterations=1000)
    
    # 1. Merkle Tree Benchmarks
    print("1️⃣  Merkle Tree Operations")
    
    from aegis.blockchain.merkle_tree import MerkleTree
    
    def bench_merkle_add_leaf():
        tree = MerkleTree()
        tree.add_leaf(b"benchmark data")
    
    def bench_merkle_build_tree():
        tree = MerkleTree()
        for i in range(10):
            tree.add_leaf(f"tx{i}".encode())
        tree.make_tree()
    
    def bench_merkle_get_root():
        tree = MerkleTree()
        for i in range(10):
            tree.add_leaf(f"tx{i}".encode())
        tree.make_tree()
        tree.get_merkle_root()
    
    runner.benchmark("Merkle: Add Leaf", bench_merkle_add_leaf)
    runner.benchmark("Merkle: Build Tree (10 tx)", bench_merkle_build_tree)
    runner.benchmark("Merkle: Get Root", bench_merkle_get_root)
    
    print()
    
    # 2. Cryptography Benchmarks
    print("2️⃣  Cryptography Operations")
    
    # Use simpler crypto operations
    import hashlib
    
    data = b"benchmark data for crypto operations"
    
    def bench_hash_sha256():
        hashlib.sha256(data).digest()
    
    def bench_hash_sha3():
        hashlib.sha3_256(data).digest()
    
    runner.benchmark("Crypto: SHA-256 Hash", bench_hash_sha256)
    runner.benchmark("Crypto: SHA3-256 Hash", bench_hash_sha3)
    
    print()
    
    # 3. Config Manager Benchmarks
    print("3️⃣  Configuration Management")
    
    from aegis.core.config_manager import ConfigManager
    
    config = ConfigManager()
    
    def bench_config_get():
        config.get('app.log_level', 'INFO')
    
    def bench_config_set():
        config.set('test.key', 'value')
    
    runner.benchmark("Config: Get Value", bench_config_get)
    runner.benchmark("Config: Set Value", bench_config_set)
    
    print()
    
    # 4. Data Serialization Benchmarks
    print("4️⃣  Data Serialization")
    
    import json
    import pickle
    
    test_data = {
        'type': 'transaction',
        'sender': 'node_abc123',
        'timestamp': time.time(),
        'payload': {'amount': 100, 'receiver': 'node_def456'}
    }
    
    def bench_json_dumps():
        json.dumps(test_data)
    
    def bench_json_loads():
        json_str = json.dumps(test_data)
        json.loads(json_str)
    
    def bench_pickle_dumps():
        pickle.dumps(test_data)
    
    def bench_pickle_loads():
        pickled = pickle.dumps(test_data)
        pickle.loads(pickled)
    
    runner.benchmark("Serialize: JSON dumps", bench_json_dumps)
    runner.benchmark("Serialize: JSON loads", bench_json_loads)
    runner.benchmark("Serialize: Pickle dumps", bench_pickle_dumps)
    runner.benchmark("Serialize: Pickle loads", bench_pickle_loads)
    
    print()
    
    # Print summary
    runner.print_summary()
    
    # Performance targets
    print("\n📊 Performance Targets vs Actual")
    print("-"*70)
    
    targets = {
        "Merkle: Add Leaf": 0.1,  # Target: <0.1ms
        "Merkle: Build Tree (10 tx)": 1.0,  # Target: <1ms
        "Crypto: SHA-256 Hash": 0.5,  # Target: <0.5ms
        "Crypto: AES Encrypt": 0.5,  # Target: <0.5ms
        "Config: Get Value": 0.01,  # Target: <0.01ms
        "Serialize: JSON dumps": 0.1,  # Target: <0.1ms
    }
    
    all_passed = True
    for name, target in targets.items():
        if name in runner.results:
            actual = runner.results[name]['mean']
            status = "✅" if actual < target else "⚠️"
            all_passed = all_passed and (actual < target)
            print(f"{status} {name:<30} Target: {target:>6.3f}ms  Actual: {actual:>6.3f}ms")
    
    print("-"*70)
    
    if all_passed:
        print("\n✅ All performance targets met!")
    else:
        print("\n⚠️  Some targets not met (optimization needed)")
    
    print()
    
    # Save results
    results_file = project_root / "benchmarks" / "results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(runner.results, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
