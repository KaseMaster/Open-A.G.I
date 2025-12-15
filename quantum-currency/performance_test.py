#!/usr/bin/env python3
"""
Simple Performance Test for HMN Components
"""

import sys
import time
sys.path.insert(0, 'src')

from network.hmn.full_node import FullNode

def run_performance_test():
    """Run a simple performance test"""
    print("Running HMN Performance Test...")
    
    # Initialize test node
    node = FullNode('perf-test-001', {
        'shard_count': 3, 
        'replication_factor': 2, 
        'validator_count': 3, 
        'metrics_port': 8000
    })
    
    # Run performance test
    start = time.time()
    iterations = 1000
    
    for i in range(iterations):
        node.get_node_stats()
        node.get_health_status()
    
    end = time.time()
    duration = end - start
    ops_per_sec = iterations / duration
    
    print("Performance Test Results:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Operations: {iterations}")
    print(f"  Rate: {ops_per_sec:.2f} ops/second")
    
    # Test individual component performance
    print("\nComponent Performance Tests:")
    
    # Test CAL Engine
    start = time.time()
    for i in range(100):
        node.cal_engine.get_current_state()
    cal_duration = time.time() - start
    print(f"  CAL Engine: {100/cal_duration:.2f} ops/second")
    
    # Test Memory Mesh
    start = time.time()
    for i in range(100):
        node.memory_mesh_service.get_memory_stats()
    memory_duration = time.time() - start
    print(f"  Memory Mesh: {100/memory_duration:.2f} ops/second")
    
    # Test Consensus
    start = time.time()
    for i in range(100):
        node.consensus_engine.get_consensus_stats()
    consensus_duration = time.time() - start
    print(f"  Consensus: {100/consensus_duration:.2f} ops/second")
    
    return ops_per_sec

if __name__ == "__main__":
    rate = run_performance_test()
    print(f"\n✅ Overall Performance: {rate:.2f} ops/second")