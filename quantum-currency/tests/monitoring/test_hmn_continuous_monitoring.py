"""
Continuous Monitoring for HMN Components
"""

import sys
import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List
import psutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from network.hmn.full_node import FullNode


class HMNContinuousMonitor:
    """Continuous monitor for HMN components"""

    def __init__(self, monitoring_duration=60):
        self.node_id = "monitoring-test-node-001"
        self.network_config = {
            "shard_count": 3,
            "replication_factor": 2,
            "validator_count": 5,
            "metrics_port": 8000,
            "enable_tls": True,
            "discovery_enabled": True
        }
        self.monitoring_duration = monitoring_duration
        self.monitoring_data = []
        self.running = False

    def collect_system_metrics(self):
        """Collect system-level metrics"""
        process = psutil.Process(os.getpid())
        
        return {
            "timestamp": time.time(),
            "cpu_percent": process.cpu_percent(),
            "memory_rss_mb": process.memory_info().rss / 1024 / 1024,
            "memory_vms_mb": process.memory_info().vms / 1024 / 1024,
            "threads": process.num_threads(),
            "connections": len(process.connections())
        }

    def collect_hmn_metrics(self, node):
        """Collect HMN-specific metrics"""
        try:
            # Get node stats
            node_stats = node.get_node_stats()
            
            # Get health status
            health_status = node.get_health_status()
            
            return {
                "timestamp": time.time(),
                "node_stats": {
                    "running": node_stats.get("running", False),
                    "health_status": node_stats.get("health_status", False),
                    "ledger_transaction_count": node_stats.get("ledger_transaction_count", 0),
                    "mining_epoch_count": node_stats.get("mining_epoch_count", 0)
                },
                "health_status": {
                    "overall_health": health_status.get("overall_health", False),
                    "services_healthy": sum(1 for service in health_status.get("services", {}).values() if service.get("healthy", False)),
                    "total_services": len(health_status.get("services", {}))
                },
                "cal_state": node_stats.get("cal_state", {}),
                "memory_stats": node_stats.get("memory_stats", {}),
                "consensus_stats": node_stats.get("consensus_stats", {})
            }
        except Exception as e:
            return {
                "timestamp": time.time(),
                "error": str(e)
            }

    def monitor_continuously(self):
        """Monitor HMN components continuously"""
        print(f"Starting continuous monitoring for {self.monitoring_duration} seconds...")
        
        # Initialize a test node
        node = FullNode(self.node_id, self.network_config)
        
        start_time = time.time()
        self.running = True
        
        while self.running and (time.time() - start_time) < self.monitoring_duration:
            try:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                
                # Collect HMN metrics
                hmn_metrics = self.collect_hmn_metrics(node)
                
                # Combine metrics
                monitoring_point = {
                    "system": system_metrics,
                    "hmn": hmn_metrics
                }
                
                self.monitoring_data.append(monitoring_point)
                
                # Print summary
                if len(self.monitoring_data) % 10 == 0:
                    print(f"  Collected {len(self.monitoring_data)} monitoring points")
                
                # Wait before next collection
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error during monitoring: {e}")
                time.sleep(1)
        
        self.running = False
        print(f"Monitoring completed. Collected {len(self.monitoring_data)} data points.")

    def analyze_monitoring_data(self):
        """Analyze collected monitoring data"""
        print("Analyzing monitoring data...")
        
        if not self.monitoring_data:
            print("❌ No monitoring data to analyze")
            return False
        
        # Analyze system metrics
        cpu_values = [point["system"]["cpu_percent"] for point in self.monitoring_data]
        memory_values = [point["system"]["memory_rss_mb"] for point in self.monitoring_data]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        
        # Analyze HMN metrics
        health_points = [point["hmn"].get("health_status", {}).get("overall_health", False) for point in self.monitoring_data]
        health_success_rate = sum(1 for healthy in health_points if healthy) / len(health_points)
        
        # Check for any errors
        error_points = [point for point in self.monitoring_data if "error" in point["hmn"]]
        error_rate = len(error_points) / len(self.monitoring_data)
        
        print(f"System Metrics Analysis:")
        print(f"  Average CPU Usage: {avg_cpu:.2f}%")
        print(f"  Maximum CPU Usage: {max_cpu:.2f}%")
        print(f"  Average Memory Usage: {avg_memory:.2f} MB")
        print(f"  Maximum Memory Usage: {max_memory:.2f} MB")
        print(f"  Health Success Rate: {health_success_rate:.2%}")
        print(f"  Error Rate: {error_rate:.2%}")
        
        # Determine overall system health
        system_healthy = (
            avg_cpu < 80.0 and  # CPU usage below 80%
            avg_memory < 500.0 and  # Memory usage below 500 MB
            health_success_rate > 0.95 and  # Health success rate above 95%
            error_rate < 0.05  # Error rate below 5%
        )
        
        if system_healthy:
            print("✅ System appears healthy during monitoring period")
        else:
            print("❌ System showed issues during monitoring period")
        
        return system_healthy

    def generate_monitoring_report(self):
        """Generate a monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"continuous_monitoring_report_{timestamp}.json"
        
        report = {
            "report_timestamp": datetime.now().isoformat() + "Z",
            "monitoring_duration": self.monitoring_duration,
            "data_points_collected": len(self.monitoring_data),
            "analysis": {},
            "raw_data": self.monitoring_data
        }
        
        # Add analysis if data exists
        if self.monitoring_data:
            cpu_values = [point["system"]["cpu_percent"] for point in self.monitoring_data]
            memory_values = [point["system"]["memory_rss_mb"] for point in self.monitoring_data]
            
            report["analysis"] = {
                "system_metrics": {
                    "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                    "max_cpu_percent": max(cpu_values),
                    "avg_memory_mb": sum(memory_values) / len(memory_values),
                    "max_memory_mb": max(memory_values)
                },
                "hmn_metrics": {
                    "health_points": len([point for point in self.monitoring_data if point["hmn"].get("health_status", {}).get("overall_health", False)]),
                    "total_points": len(self.monitoring_data)
                }
            }
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Monitoring report saved to: {report_filename}")
        return report_filename

    def test_dynamic_interval_adjustment(self):
        """Test dynamic service interval adjustment"""
        print("Testing dynamic interval adjustment...")
        
        # Initialize a test node
        node = FullNode(self.node_id, self.network_config)
        
        # Test with different network states
        test_states = [
            {"lambda_t": 0.9, "coherence_density": 0.95, "psi_score": 0.9},  # High instability
            {"lambda_t": 0.5, "coherence_density": 0.75, "psi_score": 0.7},   # Medium stability
            {"lambda_t": 0.2, "coherence_density": 0.5, "psi_score": 0.5},    # Low instability
        ]
        
        original_intervals = node.intervals.copy()
        results = []
        
        for i, state in enumerate(test_states):
            # Adjust intervals based on network state
            node.adjust_service_intervals(state)
            
            # Check that intervals were adjusted
            adjustments = {}
            for service, original_interval in original_intervals.items():
                new_interval = node.intervals[service]
                adjustment_factor = original_interval / new_interval if new_interval != 0 else 0
                adjustments[service] = {
                    "original": original_interval,
                    "new": new_interval,
                    "factor": adjustment_factor
                }
            
            results.append({
                "test_case": i + 1,
                "network_state": state,
                "adjustments": adjustments
            })
            
            print(f"  Test case {i+1}: λ(t)={state['lambda_t']:.2f}")
            print(f"    Ledger interval: {adjustments['ledger']['original']:.2f}s → {adjustments['ledger']['new']:.2f}s")
            print(f"    CAL Engine interval: {adjustments['cal_engine']['original']:.2f}s → {adjustments['cal_engine']['new']:.2f}s")
        
        # Verify that higher λ(t) results in shorter intervals
        first_case_intervals = results[0]["adjustments"]
        last_case_intervals = results[-1]["adjustments"]
        
        # High λ(t) should have shorter intervals than low λ(t)
        high_lambda_shorter = (
            first_case_intervals["ledger"]["new"] < last_case_intervals["ledger"]["new"] and
            first_case_intervals["cal_engine"]["new"] < last_case_intervals["cal_engine"]["new"]
        )
        
        if high_lambda_shorter:
            print("✅ Dynamic interval adjustment works correctly")
            return True
        else:
            print("❌ Dynamic interval adjustment may not be working correctly")
            return False

    def test_service_health_monitoring(self):
        """Test service health monitoring capabilities"""
        print("Testing service health monitoring...")
        
        # Initialize a test node
        node = FullNode(self.node_id, self.network_config)
        
        # Get initial health status
        initial_health = node.get_health_status()
        print(f"  Initial health status: {initial_health['overall_health']}")
        
        # Simulate a service failure
        node._update_service_health("cal_engine", False, "Simulated failure")
        
        # Check that health status reflects the failure
        updated_health = node.get_health_status()
        cal_engine_healthy = updated_health["services"]["cal_engine"]["healthy"]
        
        if not cal_engine_healthy:
            print("✅ Service health monitoring detected failure correctly")
        else:
            print("❌ Service health monitoring failed to detect failure")
            return False
        
        # Test auto-restart functionality
        node.auto_restart_failed_services()
        
        # Check that restart count was incremented
        final_health = node.get_health_status()
        restart_count = final_health["services"]["cal_engine"]["restart_count"]
        
        if restart_count > 0:
            print("✅ Auto-restart functionality works correctly")
        else:
            print("❌ Auto-restart functionality may not be working")
            return False
        
        return True

    def run_continuous_monitoring_tests(self):
        """Run all continuous monitoring tests"""
        print("Running HMN Continuous Monitoring Tests")
        print("=" * 50)
        
        tests = [
            ("Dynamic Interval Adjustment", self.test_dynamic_interval_adjustment),
            ("Service Health Monitoring", self.test_service_health_monitoring)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                results.append((test_name, result))
                print()  # Add spacing between tests
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
                print()
        
        # Run continuous monitoring
        print("Continuous Monitoring:")
        try:
            self.monitor_continuously()
            monitoring_healthy = self.analyze_monitoring_data()
            results.append(("Continuous Monitoring", monitoring_healthy))
            
            # Generate report
            self.generate_monitoring_report()
            
        except Exception as e:
            print(f"❌ Continuous monitoring failed with exception: {e}")
            results.append(("Continuous Monitoring", False))
        
        # Summary
        print("\n" + "=" * 50)
        print("Continuous Monitoring Test Summary:")
        passed = 0
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {test_name}: {status}")
            if success:
                passed += 1
        
        total = len(results)
        print(f"\nOverall: {passed}/{total} tests PASSED")
        
        if passed == total:
            print("\n🎉 All continuous monitoring tests PASSED!")
            print("✅ HMN is ready for continuous production monitoring!")
            return True
        else:
            print("\n❌ Some continuous monitoring tests FAILED")
            print("❌ HMN requires further monitoring improvements")
            return False


if __name__ == "__main__":
    monitor = HMNContinuousMonitor(monitoring_duration=30)  # 30 seconds for testing
    success = monitor.run_continuous_monitoring_tests()
    sys.exit(0 if success else 1)