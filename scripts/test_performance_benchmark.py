#!/usr/bin/env python3
"""
Performance benchmark test for quantum currency system
Measures TPS, latency, and resource usage
"""

import sys
import os
import time
import psutil
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate
from openagi.token_rules import validate_harmonic_tx, apply_token_effects

def measure_resources() -> Dict[str, float]:
    """Measure current CPU and memory usage"""
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "memory_percent": process.memory_percent()
    }

def run_performance_benchmark():
    """Run performance benchmark test"""
    print("⚡ Performance Benchmark Test")
    print("=" * 40)
    
    # Initial resource measurement
    initial_resources = measure_resources()
    print(f"📊 Initial Resources:")
    print(f"   CPU: {initial_resources['cpu_percent']:.1f}%")
    print(f"   Memory: {initial_resources['memory_mb']:.1f} MB ({initial_resources['memory_percent']:.1f}%)")
    
    # Generate test signals
    print("\n🔄 Generating test signals...")
    t = np.linspace(0, 0.5, 1024)
    
    # Create multiple snapshots for batch processing
    snapshots = []
    num_snapshots = 10
    
    start_time = time.time()
    for i in range(num_snapshots):
        # Create coherent signals
        freq = 50.0
        phase = i * 0.05
        values = np.sin(2 * np.pi * freq * t + phase) + 0.01 * np.random.randn(len(t))
        snapshot = make_snapshot(f"node-{i}", t.tolist(), values.tolist(), secret_key=f"key{i}")
        snapshots.append(snapshot)
    
    snapshot_generation_time = time.time() - start_time
    print(f"   Generated {num_snapshots} snapshots in {snapshot_generation_time:.3f}s")
    
    # Measure coherence calculation performance
    print("\n🔍 Calculating coherence scores...")
    start_time = time.time()
    coherence_scores = []
    
    for i, local_snapshot in enumerate(snapshots):
        remote_snapshots = [s for j, s in enumerate(snapshots) if i != j]
        if remote_snapshots:
            cs = compute_coherence_score(local_snapshot, remote_snapshots)
            coherence_scores.append(cs)
    
    coherence_calc_time = time.time() - start_time
    avg_coherence_time = coherence_calc_time / len(snapshots) if snapshots else 0
    print(f"   Calculated {len(coherence_scores)} coherence scores in {coherence_calc_time:.3f}s")
    print(f"   Average time per coherence calculation: {avg_coherence_time:.4f}s")
    
    # Measure recursive validation performance
    print("\n🔍 Performing recursive validation...")
    start_time = time.time()
    validations = []
    
    # Test with different bundle sizes
    for bundle_size in [3, 5, 7]:
        if len(snapshots) >= bundle_size:
            bundle = snapshots[:bundle_size]
            is_valid, proof_bundle = recursive_validate(bundle, threshold=0.1)  # Lower threshold for testing
            validations.append((bundle_size, is_valid, proof_bundle))
    
    validation_time = time.time() - start_time
    avg_validation_time = validation_time / len(validations) if validations else 0
    print(f"   Performed {len(validations)} validations in {validation_time:.3f}s")
    print(f"   Average time per validation: {avg_validation_time:.4f}s")
    
    # Measure transaction processing performance
    print("\n💰 Processing transactions...")
    ledger = {"balances": {}, "chr": {}}
    transactions = []
    
    # Create test transactions
    for i in range(20):
        tx = {
            "id": f"tx-{i}",
            "type": "harmonic",
            "action": "mint",
            "token": "FLX",
            "sender": f"node-{i % 3}",
            "receiver": f"node-{i % 3}",
            "amount": 100 + i,
            "aggregated_cs": 0.8,  # High coherence for testing
            "sender_chr": 0.8,
            "timestamp": time.time()
        }
        transactions.append(tx)
    
    # Process transactions
    start_time = time.time()
    processed_transactions = 0
    successful_transactions = 0
    
    for tx in transactions:
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        ok = validate_harmonic_tx(tx, config)
        if ok:
            apply_token_effects(ledger, tx)
            successful_transactions += 1
        processed_transactions += 1
    
    transaction_processing_time = time.time() - start_time
    tps = processed_transactions / transaction_processing_time if transaction_processing_time > 0 else 0
    
    print(f"   Processed {processed_transactions} transactions in {transaction_processing_time:.3f}s")
    print(f"   Successful transactions: {successful_transactions}/{processed_transactions}")
    print(f"   Transactions Per Second (TPS): {tps:.2f}")
    
    # Final resource measurement
    final_resources = measure_resources()
    print(f"\n📊 Final Resources:")
    print(f"   CPU: {final_resources['cpu_percent']:.1f}%")
    print(f"   Memory: {final_resources['memory_mb']:.1f} MB ({final_resources['memory_percent']:.1f}%)")
    
    # Resource usage differences
    cpu_diff = final_resources['cpu_percent'] - initial_resources['cpu_percent']
    memory_diff = final_resources['memory_mb'] - initial_resources['memory_mb']
    print(f"\n📈 Resource Usage Differences:")
    print(f"   CPU Change: {cpu_diff:+.1f}%")
    print(f"   Memory Change: {memory_diff:+.1f} MB")
    
    # Performance summary
    print(f"\n🏆 Performance Summary:")
    print(f"   Snapshot Generation: {snapshot_generation_time:.3f}s ({num_snapshots} snapshots)")
    print(f"   Coherence Calculation: {coherence_calc_time:.3f}s ({len(coherence_scores)} calculations)")
    print(f"   Recursive Validation: {validation_time:.3f}s ({len(validations)} validations)")
    print(f"   Transaction Processing: {transaction_processing_time:.3f}s ({processed_transactions} transactions)")
    print(f"   Overall TPS: {tps:.2f}")
    
    print("\n✅ Performance benchmark completed successfully!")

if __name__ == "__main__":
    run_performance_benchmark()