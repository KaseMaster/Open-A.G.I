"""
Performance & Load Testing for HMN Components
"""

import sys
import os
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from network.hmn.full_node import FullNode
from network.hmn.memory_mesh_service import MemoryMeshService
from network.hmn.attuned_consensus import AttunedConsensus


class HMNPerformanceTester:
    """Performance tester for HMN components"""

    def __init__(self):
        self.node_id = "perf-test-node-001"
        self.network_config = {
            "shard_count": 3,
            "replication_factor": 2,
            "validator_count": 5,
            "metrics_port": 8000,
            "enable_tls": True,
            "discovery_enabled": True
        }
        
        # Initialize HMN components
        self.hmn_node = FullNode(self.node_id, self.network_config)
        
        # Add sample validators
        validators_data = [
            ("validator-1", 0.95, 10000.0),
            ("validator-2", 0.87, 8000.0),
            ("validator-3", 0.75, 12000.0),
            ("validator-4", 0.92, 9000.0),
            ("validator-5", 0.85, 7000.0),
        ]
        
        for validator_id, psi_score, stake in validators_data:
            self.hmn_node.consensus_engine.add_validator(validator_id, psi_score, stake)

    def measure_cpu_memory_usage(self):
        """Measure current CPU and memory usage"""
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        return {
            "cpu_percent": cpu_percent,
            "memory_rss": memory_info.rss / 1024 / 1024,  # MB
            "memory_vms": memory_info.vms / 1024 / 1024,  # MB
        }

    def performance_test_single_thread(self, iterations=100):
        """Run performance test with single thread"""
        print(f"Running single-thread performance test with {iterations} iterations...")
        
        start_time = time.time()
        start_resources = self.measure_cpu_memory_usage()
        
        # Test CAL engine operations
        cal_success = 0
        for i in range(iterations // 4):
            try:
                asyncio.run(self.hmn_node.run_cal_engine())
                cal_success += 1
            except Exception:
                pass
        
        # Test memory mesh operations
        memory_success = 0
        for i in range(iterations // 4):
            try:
                self.hmn_node.memory_mesh_service.get_memory_stats()
                memory_success += 1
            except Exception:
                pass
        
        # Test consensus operations
        consensus_success = 0
        for i in range(iterations // 4):
            try:
                network_state = self.hmn_node.cal_engine.get_current_state()
                self.hmn_node.consensus_engine.execute_consensus_round(network_state)
                consensus_success += 1
            except Exception:
                pass
        
        # Test mining operations
        mining_success = 0
        for i in range(iterations // 4):
            try:
                network_state = self.hmn_node.cal_engine.get_current_state()
                self.hmn_node.mining_agent.run_epoch(network_state)
                mining_success += 1
            except Exception:
                pass
        
        end_time = time.time()
        end_resources = self.measure_cpu_memory_usage()
        total_time = end_time - start_time
        
        print(f"  CAL Engine: {cal_success}/{iterations//4} successful")
        print(f"  Memory Mesh: {memory_success}/{iterations//4} successful")
        print(f"  Consensus: {consensus_success}/{iterations//4} successful")
        print(f"  Mining: {mining_success}/{iterations//4} successful")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Requests per second: {iterations/total_time:.2f}")
        print(f"  CPU Usage: {end_resources['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {end_resources['memory_rss']:.1f} MB")
        
        return {
            "total_time": total_time,
            "requests_per_second": iterations/total_time,
            "cpu_percent": end_resources['cpu_percent'],
            "memory_mb": end_resources['memory_rss'],
            "success_rate": (cal_success + memory_success + consensus_success + mining_success) / iterations
        }

    def performance_test_multi_thread(self, iterations=100, threads=10):
        """Run performance test with multiple threads"""
        print(f"Running multi-thread performance test with {iterations} iterations across {threads} threads...")
        
        start_time = time.time()
        start_resources = self.measure_cpu_memory_usage()
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            
            # Submit CAL engine tasks
            for i in range(iterations // 4):
                futures.append(executor.submit(lambda: asyncio.run(self.hmn_node.run_cal_engine())))
            
            # Submit memory mesh tasks
            for i in range(iterations // 4):
                futures.append(executor.submit(self.hmn_node.memory_mesh_service.get_memory_stats))
            
            # Submit consensus tasks
            for i in range(iterations // 4):
                def run_consensus():
                    network_state = self.hmn_node.cal_engine.get_current_state()
                    return self.hmn_node.consensus_engine.execute_consensus_round(network_state)
                futures.append(executor.submit(run_consensus))
            
            # Submit mining tasks
            for i in range(iterations // 4):
                def run_mining():
                    network_state = self.hmn_node.cal_engine.get_current_state()
                    return self.hmn_node.mining_agent.run_epoch(network_state)
                futures.append(executor.submit(run_mining))
            
            # Wait for all tasks to complete
            success_count = 0
            for future in as_completed(futures):
                try:
                    future.result()
                    success_count += 1
                except Exception:
                    pass
        
        end_time = time.time()
        end_resources = self.measure_cpu_memory_usage()
        total_time = end_time - start_time
        
        print(f"  Successful requests: {success_count}/{iterations}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Requests per second: {iterations/total_time:.2f}")
        print(f"  CPU Usage: {end_resources['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {end_resources['memory_rss']:.1f} MB")
        
        return {
            "total_time": total_time,
            "requests_per_second": iterations/total_time,
            "cpu_percent": end_resources['cpu_percent'],
            "memory_mb": end_resources['memory_rss'],
            "success_rate": success_count / iterations
        }

    def stress_test_high_frequency(self, duration=30):
        """Stress test with high-frequency interactions"""
        print(f"Running high-frequency stress test for {duration} seconds...")
        
        start_time = time.time()
        request_count = 0
        success_count = 0
        end_time = start_time + duration
        
        # Track resource usage over time
        resource_samples = []
        
        while time.time() < end_time:
            # Alternate between different types of requests
            action = request_count % 4
            
            try:
                if action == 0:
                    # CAL Engine
                    asyncio.run(self.hmn_node.run_cal_engine())
                elif action == 1:
                    # Memory Mesh
                    self.hmn_node.memory_mesh_service.get_memory_stats()
                elif action == 2:
                    # Consensus
                    network_state = self.hmn_node.cal_engine.get_current_state()
                    self.hmn_node.consensus_engine.execute_consensus_round(network_state)
                else:
                    # Mining
                    network_state = self.hmn_node.cal_engine.get_current_state()
                    self.hmn_node.mining_agent.run_epoch(network_state)
                
                success_count += 1
            except Exception:
                pass
            
            request_count += 1
            
            # Sample resources every 5 seconds
            if request_count % 50 == 0:
                resources = self.measure_cpu_memory_usage()
                resources["timestamp"] = time.time() - start_time
                resource_samples.append(resources)
        
        total_time = time.time() - start_time
        
        print(f"  Total requests: {request_count}")
        print(f"  Successful requests: {success_count}")
        print(f"  Success rate: {success_count/request_count*100:.2f}%")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Requests per second: {request_count/total_time:.2f}")
        
        # Analyze resource usage
        if resource_samples:
            avg_cpu = sum(sample["cpu_percent"] for sample in resource_samples) / len(resource_samples)
            max_cpu = max(sample["cpu_percent"] for sample in resource_samples)
            avg_memory = sum(sample["memory_rss"] for sample in resource_samples) / len(resource_samples)
            max_memory = max(sample["memory_rss"] for sample in resource_samples)
            
            print(f"  Avg CPU Usage: {avg_cpu:.1f}%")
            print(f"  Max CPU Usage: {max_cpu:.1f}%")
            print(f"  Avg Memory Usage: {avg_memory:.1f} MB")
            print(f"  Max Memory Usage: {max_memory:.1f} MB")
        
        # Check for system stability
        stability = success_count/request_count >= 0.95
        if stability:
            print("  ✅ System remained stable under high load")
        else:
            print("  ❌ System showed instability under high load")
        
        return {
            "total_requests": request_count,
            "successful_requests": success_count,
            "success_rate": success_count/request_count,
            "total_time": total_time,
            "requests_per_second": request_count/total_time,
            "stability": stability,
            "resource_samples": resource_samples
        }

    def test_memory_mesh_throughput(self, update_count=1000):
        """Test memory mesh throughput with bulk updates"""
        print(f"Testing memory mesh throughput with {update_count} updates...")
        
        from network.hmn.memory_mesh_service import MemoryUpdate
        start_time = time.time()
        start_resources = self.measure_cpu_memory_usage()
        
        # Create bulk memory updates
        updates = []
        for i in range(update_count):
            update = MemoryUpdate(
                id=f"bulk-update-{i}",
                content={"data": f"content_{i}", "timestamp": time.time()},
                timestamp=time.time(),
                rphiv_score=0.5 + (i % 100) / 200,  # Varying RΦV scores
                node_id=self.node_id,
                shard_id=f"shard-{i % 3}"
            )
            updates.append(update)
        
        # Index all updates
        self.hmn_node.memory_mesh_service.index_updates(updates)
        
        # Participate in gossip
        network_state = self.hmn_node.cal_engine.get_current_state()
        self.hmn_node.memory_mesh_service.participate_in_gossip(network_state)
        
        end_time = time.time()
        end_resources = self.measure_cpu_memory_usage()
        total_time = end_time - start_time
        
        # Get final stats
        memory_stats = self.hmn_node.memory_mesh_service.get_memory_stats()
        
        print(f"  Indexed Updates: {len(self.hmn_node.memory_mesh_service.local_memory)}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Updates per second: {update_count/total_time:.2f}")
        print(f"  CPU Usage: {end_resources['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {end_resources['memory_rss']:.1f} MB")
        
        return {
            "indexed_updates": len(self.hmn_node.memory_mesh_service.local_memory),
            "total_time": total_time,
            "updates_per_second": update_count/total_time,
            "cpu_percent": end_resources['cpu_percent'],
            "memory_mb": end_resources['memory_rss'],
            "gossip_messages_sent": memory_stats["metrics"]["gossip_messages_sent"]
        }

    def test_consensus_scalability(self, validator_counts=[3, 5, 10, 20]):
        """Test consensus scalability with varying validator counts"""
        print("Testing consensus scalability...")
        
        results = []
        
        for validator_count in validator_counts:
            print(f"  Testing with {validator_count} validators...")
            
            # Reinitialize consensus engine with new validator count
            consensus_engine = AttunedConsensus(self.node_id, {**self.network_config, "validator_count": validator_count})
            
            # Add validators
            for i in range(validator_count):
                consensus_engine.add_validator(f"validator-{i}", 0.8 + (i % 20) / 100, 1000.0 + i * 100)
            
            start_time = time.time()
            
            # Run consensus rounds
            rounds = 10
            for _ in range(rounds):
                network_state = {"lambda_t": 0.7, "coherence_density": 0.8, "psi_score": 0.9}
                consensus_engine.execute_consensus_round(network_state)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Get stats
            consensus_stats = consensus_engine.get_consensus_stats()
            
            result = {
                "validator_count": validator_count,
                "rounds": rounds,
                "total_time": total_time,
                "rounds_per_second": rounds/total_time,
                "validators_count": consensus_stats["validators_count"]
            }
            
            results.append(result)
            print(f"    Time: {total_time:.2f}s, Rounds/sec: {rounds/total_time:.2f}")
        
        return results

    def run_all_performance_tests(self):
        """Run all performance tests"""
        print("Running HMN Performance & Load Tests")
        print("=" * 50)
        
        results = {}
        
        # Single-thread test
        print("\n1. Single-thread performance test:")
        results["single_thread"] = self.performance_test_single_thread(50)
        
        # Multi-thread test
        print("\n2. Multi-thread performance test:")
        results["multi_thread"] = self.performance_test_multi_thread(50, 5)
        
        # High-frequency stress test
        print("\n3. High-frequency stress test:")
        results["stress_test"] = self.stress_test_high_frequency(15)
        
        # Memory mesh throughput test
        print("\n4. Memory mesh throughput test:")
        results["memory_mesh"] = self.test_memory_mesh_throughput(500)
        
        # Consensus scalability test
        print("\n5. Consensus scalability test:")
        results["consensus_scalability"] = self.test_consensus_scalability([3, 5, 10])
        
        # Summary
        print("\n" + "=" * 50)
        print("Performance Test Summary:")
        print("=" * 50)
        
        print(f"Single-thread: {results['single_thread']['requests_per_second']:.2f} req/s")
        print(f"Multi-thread: {results['multi_thread']['requests_per_second']:.2f} req/s")
        print(f"Stress test success rate: {results['stress_test']['success_rate']:.2%}")
        print(f"Memory mesh throughput: {results['memory_mesh']['updates_per_second']:.2f} updates/s")
        print(f"Consensus scalability: {len(results['consensus_scalability'])} configurations tested")
        
        # Save results to file
        timestamp = int(time.time())
        filename = f"performance_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")
        
        return results


if __name__ == "__main__":
    tester = HMNPerformanceTester()
    results = tester.run_all_performance_tests()
    
    # Print final summary
    print("\n" + "=" * 50)
    print("HMN PERFORMANCE TEST COMPLETE")
    print("=" * 50)
    
    overall_success = all([
        results["stress_test"]["stability"],
        results["single_thread"]["success_rate"] > 0.9,
        results["multi_thread"]["success_rate"] > 0.9
    ])
    
    if overall_success:
        print("🎉 All performance tests PASSED!")
    else:
        print("❌ Some performance tests FAILED")
    
    sys.exit(0 if overall_success else 1)