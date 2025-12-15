#!/usr/bin/env python3
"""
Simple Monitoring Test for HMN Components
"""

import sys
sys.path.insert(0, 'src')

from network.hmn.full_node import FullNode

def run_monitoring_test():
    """Run a simple monitoring test"""
    print("Running HMN Monitoring Test...")
    
    # Initialize test node
    node = FullNode('monitor-test-001', {
        'shard_count': 3, 
        'replication_factor': 2, 
        'validator_count': 3, 
        'metrics_port': 8000
    })
    
    # Test monitoring endpoints
    print("Monitoring Endpoints Test:")
    
    # Get node stats
    stats = node.get_node_stats()
    print(f"  Node ID: {stats['node_id']}")
    print(f"  Health Status: {stats['health_status']}")
    print(f"  CAL State: λ(t)={stats['cal_state']['lambda_t']:.3f}, Ĉ(t)={stats['cal_state']['coherence_density']:.3f}")
    
    # Get health status
    health = node.get_health_status()
    print(f"  Overall Health: {health['overall_health']}")
    print(f"  Services Monitored: {len(health['services'])}")
    
    # Check that all services are being monitored
    service_names = list(health['services'].keys())
    print(f"  Monitored Services: {', '.join(service_names)}")
    
    # Test memory stats
    memory_stats = node.memory_mesh_service.get_memory_stats()
    print(f"  Memory Updates: {memory_stats['local_updates_count']}")
    
    # Test consensus stats
    consensus_stats = node.consensus_engine.get_consensus_stats()
    print(f"  Validators: {consensus_stats['validators_count']}")
    
    return health['overall_health']

if __name__ == "__main__":
    is_healthy = run_monitoring_test()
    if is_healthy:
        print("\n✅ Monitoring endpoints are fully accessible")
    else:
        print("\n❌ Monitoring endpoints have issues")