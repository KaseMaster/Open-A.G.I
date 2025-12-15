#!/usr/bin/env python3
"""
Stress tests for the Lambda Attunement Controller
"""

import unittest
import numpy as np
import time
import sys
import os
import threading
from unittest.mock import Mock, patch

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, src_path)

# Import the modules dynamically
lambda_attunement_path = os.path.join(src_path, 'core', 'lambda_attunement.py')
if os.path.exists(lambda_attunement_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lambda_attunement", lambda_attunement_path)
    if spec is not None and spec.loader is not None:
        lambda_attunement = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lambda_attunement)
        
        LambdaAttunementController = lambda_attunement.LambdaAttunementController
        CoherenceDensityMeter = lambda_attunement.CoherenceDensityMeter
        AttunementConfig = lambda_attunement.AttunementConfig
    else:
        raise ImportError("Could not import lambda_attunement module")
else:
    raise ImportError(f"Lambda attunement module not found at {lambda_attunement_path}")

class HighEntropyCALEngine:
    """Mock CAL engine that simulates high entropy conditions"""
    def __init__(self):
        self.alpha_multiplier = 1.0
        self.call_count = 0
        # Generate complex omega samples for stress testing
        self.omega_samples = []
        for i in range(50):  # Many nodes
            node_samples = []
            for j in range(20):  # Many dimensions
                # Create complex varying patterns
                value = (np.sin(i * 0.1) * np.cos(j * 0.2) * 
                        np.sin(time.time() * 0.01) * 10.0)
                node_samples.append(value)
            self.omega_samples.append(node_samples)
        
    def sample_omega_snapshot(self):
        self.call_count += 1
        # Add time-based variation for dynamic stress
        variation = 0.5 * np.sin(self.call_count * 0.01)
        samples = []
        for sample in self.omega_samples:
            varied_sample = [x + variation + (np.random.random() - 0.5) * 0.1 for x in sample]
            samples.append(varied_sample)
        return samples
        
    def normalize_C(self, C):
        # Normalize C to [0,1] range with more complex logic
        return min(1.0, max(0.0, C / 100.0))  # Adjust for larger values
        
    def set_alpha_multiplier(self, alpha):
        self.alpha_multiplier = alpha
        
    def get_lambda(self):
        return 0.5 + 0.1 * np.sin(time.time() * 0.1)  # Varying lambda
        
    def get_entropy_rate(self):
        # High entropy for stress testing
        return 0.0015 + 0.0005 * np.sin(time.time() * 0.1)
        
    def get_h_internal(self):
        # Varying internal coherence
        return 0.96 + 0.02 * np.sin(time.time() * 0.05)
        
    def get_m_t_bounds(self):
        return (-2.0, 2.0)  # Wider bounds for stress

class TestAttunementStress(unittest.TestCase):
    """Stress tests for LambdaAttunementController"""
    
    def setUp(self):
        self.cal_engine = HighEntropyCALEngine()
        config = {
            "alpha_initial": 1.0,
            "alpha_min": 0.5,  # Wider range for stress
            "alpha_max": 1.5,
            "lr": 0.001,  # Smaller learning rate for stability under stress
            "momentum": 0.9,
            "epsilon": 1e-5,
            "settle_delay": 0.01,  # Shorter delays for faster testing
            "cycle_sleep": 0.01,
            "gradient_averaging_window": 5,  # Larger window for smoothing
            "delta_alpha_max": 0.01,  # Smaller delta for stability
            "safety": {
                "entropy_max": 0.003,  # Higher threshold for stress testing
                "h_internal_min": 0.90,  # Lower threshold for stress testing
                "revert_on_failure": True
            }
        }
        self.controller = LambdaAttunementController(self.cal_engine, config)
        
    def test_high_entanglement_density(self):
        """Test attunement under high entanglement density"""
        print("Testing attunement under high entanglement density...")
        
        # Run many cycles to stress test
        start_time = time.time()
        cycle_count = 50  # Many cycles for stress testing
        
        accept_count = 0
        revert_count = 0
        
        for i in range(cycle_count):
            result = self.controller.update()
            status = self.controller.get_status()
            
            if result:
                accept_count += 1
            else:
                revert_count += 1
                
            # Check that we're not consuming excessive CPU
            if i % 10 == 0:  # Check every 10 cycles
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed > 10.0:  # Cap test time
                    print(f"Test taking too long, stopping early at cycle {i}")
                    break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Completed {cycle_count} cycles in {total_time:.2f} seconds")
        print(f"Rate: {cycle_count/total_time:.2f} cycles/second")
        print(f"Accepts: {accept_count}, Reverts: {revert_count}")
        
        # Verify reasonable performance
        self.assertGreater(cycle_count/total_time, 1.0, "Should achieve at least 1 cycle/second")
        self.assertLess(total_time, 30.0, "Should complete within reasonable time")
        
        # Verify system stability
        status = self.controller.get_status()
        self.assertGreaterEqual(status['alpha'], 0.5)
        self.assertLessEqual(status['alpha'], 1.5)
        
    def test_concurrent_controller_access(self):
        """Test concurrent access to the controller"""
        print("Testing concurrent controller access...")
        
        # Create multiple threads that access the controller simultaneously
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                for i in range(10):
                    # Mix of different operations
                    if i % 3 == 0:
                        result = self.controller.update()
                        results.append((thread_id, i, result))
                    elif i % 3 == 1:
                        status = self.controller.get_status()
                        results.append((thread_id, i, 'status'))
                    else:
                        ledger = self.controller.get_audit_ledger()
                        results.append((thread_id, i, 'ledger'))
                    time.sleep(0.001)  # Very short delay
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        thread_count = 5
        for i in range(thread_count):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout
            
        print(f"Completed concurrent access test with {thread_count} threads")
        print(f"Total operations: {len(results)}")
        print(f"Errors: {len(errors)}")
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Should have no errors, got: {errors}")
        
        # Verify reasonable number of operations completed
        self.assertGreater(len(results), 20, "Should complete reasonable number of operations")
        
    def test_performance_overhead_measurement(self):
        """Measure performance overhead of attunement"""
        print("Measuring performance overhead...")
        
        # Measure baseline performance
        baseline_start = time.time()
        for i in range(100):
            # Just calling the engine methods directly
            _ = self.cal_engine.sample_omega_snapshot()
            _ = self.cal_engine.get_entropy_rate()
            _ = self.cal_engine.get_h_internal()
        baseline_end = time.time()
        baseline_time = baseline_end - baseline_start
        
        # Measure performance with attunement
        attunement_start = time.time()
        for i in range(100):
            # Calling through the attunement controller
            _ = self.controller.meter.compute_C_hat()
            _ = self.controller._check_safety_constraints()
            self.controller._set_alpha(1.0)
        attunement_end = time.time()
        attunement_time = attunement_end - attunement_start
        
        overhead = attunement_time - baseline_time
        overhead_percentage = (overhead / baseline_time) * 100 if baseline_time > 0 else 0
        
        print(f"Baseline time: {baseline_time:.4f}s")
        print(f"Attunement time: {attunement_time:.4f}s")
        print(f"Overhead: {overhead:.4f}s ({overhead_percentage:.1f}%)")
        
        # Verify overhead is reasonable (less than 500%)
        self.assertLess(overhead_percentage, 500.0, 
                       f"Overhead should be reasonable, got {overhead_percentage:.1f}%")
        
    def test_stability_under_extreme_conditions(self):
        """Test stability under extreme conditions"""
        print("Testing stability under extreme conditions...")
        
        # Configure controller for extreme testing
        extreme_config = {
            "alpha_initial": 1.0,
            "alpha_min": 0.1,  # Very wide range
            "alpha_max": 2.0,
            "lr": 0.01,  # Higher learning rate
            "momentum": 0.95,
            "epsilon": 1e-6,
            "settle_delay": 0.001,  # Very short delays
            "cycle_sleep": 0.001,
            "delta_alpha_max": 0.05,  # Larger delta
            "safety": {
                "entropy_max": 0.01,  # Much higher threshold
                "h_internal_min": 0.5,  # Much lower threshold
                "revert_on_failure": True
            }
        }
        
        extreme_engine = HighEntropyCALEngine()
        extreme_controller = LambdaAttunementController(extreme_engine, extreme_config)
        
        # Run cycles under extreme conditions
        mode_changes = []
        for i in range(100):
            old_mode = extreme_controller.mode
            result = extreme_controller.update()
            new_mode = extreme_controller.mode
            
            if old_mode != new_mode:
                mode_changes.append((i, old_mode, new_mode))
                
            # Check for stability violations
            status = extreme_controller.get_status()
            self.assertGreaterEqual(status['alpha'], 0.1, "Alpha should not go below minimum")
            self.assertLessEqual(status['alpha'], 2.0, "Alpha should not exceed maximum")
            
        print(f"Extreme conditions test completed with {len(mode_changes)} mode changes")
        
        # Verify that emergency mode is only entered when appropriate
        emergency_entries = [change for change in mode_changes if change[2] == 3]
        print(f"Emergency mode entries: {len(emergency_entries)}")
        
        # Should not enter emergency mode too frequently
        self.assertLess(len(emergency_entries), 10, "Should not enter emergency mode too frequently")

class TestMemoryAndResourceUsage(unittest.TestCase):
    """Test memory and resource usage"""
    
    def test_memory_leak_prevention(self):
        """Test that memory usage doesn't grow unbounded"""
        print("Testing memory leak prevention...")
        
        cal_engine = HighEntropyCALEngine()
        config = {
            "alpha_initial": 1.0,
            "alpha_min": 0.8,
            "alpha_max": 1.2,
            "lr": 0.001,
            "momentum": 0.9,
            "epsilon": 1e-5,
            "settle_delay": 0.01,
            "cycle_sleep": 0.01,
            "gradient_averaging_window": 3,
            "safety": {
                "entropy_max": 0.002,
                "h_internal_min": 0.95,
                "revert_on_failure": True
            }
        }
        controller = LambdaAttunementController(cal_engine, config)
        
        # Run many cycles and monitor history size
        initial_history_size = len(controller.history)
        initial_ledger_size = len(controller.audit_ledger)
        
        for i in range(200):  # Many cycles
            controller.update()
            
            # Check that history doesn't grow unbounded
            if i % 50 == 0:
                self.assertLessEqual(len(controller.history), 10, 
                                   "History should not grow unbounded")
        
        final_history_size = len(controller.history)
        final_ledger_size = len(controller.audit_ledger)
        
        print(f"History size: {initial_history_size} -> {final_history_size}")
        print(f"Audit ledger size: {initial_ledger_size} -> {final_ledger_size}")
        
        # Verify history is bounded
        self.assertLessEqual(final_history_size, 10, "History should be bounded")
        
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        print("Testing resource cleanup...")
        
        # Create controller in a local scope
        def create_and_use_controller():
            cal_engine = HighEntropyCALEngine()
            config = {
                "alpha_initial": 1.0,
                "alpha_min": 0.8,
                "alpha_max": 1.2,
                "lr": 0.001,
                "momentum": 0.9,
                "epsilon": 1e-5,
                "settle_delay": 0.01,
                "cycle_sleep": 0.01,
                "safety": {
                    "entropy_max": 0.002,
                    "h_internal_min": 0.95,
                    "revert_on_failure": True
                }
            }
            controller = LambdaAttunementController(cal_engine, config)
            
            # Use controller
            for i in range(10):
                controller.update()
                
            # Return some data to prevent premature garbage collection
            return controller.get_status()
        
        # Run multiple times to test cleanup
        for i in range(5):
            status = create_and_use_controller()
            self.assertIsNotNone(status)
            
        print("Resource cleanup test completed")

def run_stress_tests():
    """Run all stress tests"""
    print("🏋️ Running Lambda Attunement Stress Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAttunementStress))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMemoryAndResourceUsage))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("STRESS TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_stress_tests()
    sys.exit(0 if success else 1)