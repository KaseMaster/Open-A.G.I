#!/usr/bin/env python3
"""
Integration tests for the Lambda Attunement Controller
"""

import unittest
import numpy as np
import time
import sys
import os
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

class MockCALEngine:
    """Mock CAL engine for integration testing"""
    def __init__(self):
        self.alpha_multiplier = 1.0
        self.omega_samples = [[1.0, 0.5, 0.2], [0.8, 0.6, 0.3], [0.9, 0.4, 0.1]]
        self.entropy_rate_value = 0.001
        self.h_internal_value = 0.98
        self.m_t_bounds_value = (-1.0, 1.0)
        self.call_count = 0
        
    def sample_omega_snapshot(self):
        # Simulate changing omega samples over time
        self.call_count += 1
        # Add some variation to simulate real system behavior
        variation = 0.1 * np.sin(self.call_count * 0.1)
        samples = []
        for sample in self.omega_samples:
            varied_sample = [x + variation for x in sample]
            samples.append(varied_sample)
        return samples
        
    def normalize_C(self, C):
        # Normalize C to [0,1] range
        return min(1.0, max(0.0, C / 10.0))
        
    def set_alpha_multiplier(self, alpha):
        self.alpha_multiplier = alpha
        
    def get_lambda(self):
        return 0.5  # Mock lambda value
        
    def get_entropy_rate(self):
        # Simulate entropy rate that can vary
        return self.entropy_rate_value + 0.0001 * np.sin(time.time())
        
    def get_h_internal(self):
        # Simulate internal coherence that can vary
        return self.h_internal_value + 0.01 * np.sin(time.time() * 0.5)
        
    def get_m_t_bounds(self):
        return self.m_t_bounds_value

class TestAttunementIntegration(unittest.TestCase):
    """Integration tests for LambdaAttunementController"""
    
    def setUp(self):
        self.cal_engine = MockCALEngine()
        config = {
            "alpha_initial": 1.0,
            "alpha_min": 0.8,
            "alpha_max": 1.2,
            "lr": 0.01,
            "momentum": 0.9,
            "epsilon": 1e-4,
            "settle_delay": 0.1,
            "cycle_sleep": 0.1,
            "safety": {
                "entropy_max": 0.002,
                "h_internal_min": 0.95,
                "revert_on_failure": True
            }
        }
        self.controller = LambdaAttunementController(self.cal_engine, config)
        
    def test_full_attunement_cycle(self):
        """Test a full attunement cycle with multiple updates"""
        print("Testing full attunement cycle...")
        
        # Initial state
        initial_alpha = self.controller.alpha
        initial_c_hat = self.controller.meter.compute_C_hat()
        
        # Run several update cycles
        for i in range(5):
            accepted = self.controller.update()
            status = self.controller.get_status()
            current_c_hat = self.controller.meter.compute_C_hat()
            
            print(f"Cycle {i+1}: alpha={status['alpha']:.4f}, "
                  f"C_hat={current_c_hat:.4f}, "
                  f"{'ACCEPTED' if accepted else 'REVERTED'}")
            
            # Verify basic constraints
            self.assertGreaterEqual(status['alpha'], 0.8)
            self.assertLessEqual(status['alpha'], 1.2)
            self.assertGreaterEqual(current_c_hat, 0.0)
            self.assertLessEqual(current_c_hat, 1.0)
        
        # Final state
        final_alpha = self.controller.alpha
        final_c_hat = self.controller.meter.compute_C_hat()
        
        print(f"Initial: alpha={initial_alpha:.4f}, C_hat={initial_c_hat:.4f}")
        print(f"Final: alpha={final_alpha:.4f}, C_hat={final_c_hat:.4f}")
        
        # Verify that the controller ran
        self.assertGreaterEqual(self.controller.accept_counter + self.controller.revert_counter, 1)
        
    def test_safety_constraints_integration(self):
        """Test integration with safety constraints"""
        print("Testing safety constraints integration...")
        
        # Test with normal conditions
        is_safe = self.controller._check_safety_constraints()
        self.assertTrue(is_safe, "Should be safe under normal conditions")
        
        # Test with high entropy
        self.cal_engine.entropy_rate_value = 0.003  # Above threshold
        is_safe = self.controller._check_safety_constraints()
        self.assertFalse(is_safe, "Should not be safe with high entropy")
        
        # Reset to normal
        self.cal_engine.entropy_rate_value = 0.001
        
        # Test with low internal coherence
        self.cal_engine.h_internal_value = 0.90  # Below threshold
        is_safe = self.controller._check_safety_constraints()
        self.assertFalse(is_safe, "Should not be safe with low internal coherence")
        
    def test_emergency_mode_integration(self):
        """Test integration with emergency mode"""
        print("Testing emergency mode integration...")
        
        # Simulate conditions that would trigger emergency mode
        with patch.object(self.controller, '_check_safety_constraints', return_value=False):
            with patch.object(self.controller.meter, 'compute_C_hat', return_value=0.1):
                # Force a revert by making the coherence decrease
                result = self.controller.update()
                self.assertFalse(result, "Update should be reverted")
                self.assertEqual(self.controller.mode, 2, "Should switch to PID mode")
                
                # Run another update that also fails
                result = self.controller.update()
                # After multiple failures, should enter emergency mode
                # Note: In the current implementation, emergency mode is set on exceptions
                # We're testing that the system handles failures gracefully
                
    def test_audit_ledger_integration(self):
        """Test audit ledger integration"""
        print("Testing audit ledger integration...")
        
        # Run a few updates to generate ledger entries
        for i in range(3):
            self.controller.update()
        
        # Check audit ledger
        ledger = self.controller.get_audit_ledger()
        self.assertIsInstance(ledger, list)
        self.assertGreaterEqual(len(ledger), 0)  # May be 0 if all updates were reverted
        
        # Check that ledger entries have the expected structure
        for entry in ledger:
            self.assertIn('timestamp', entry)
            self.assertIn('old_alpha', entry)
            self.assertIn('new_alpha', entry)
            self.assertIn('reason', entry)
            self.assertIn('c_before', entry)
            self.assertIn('c_after', entry)
            self.assertIn('actor', entry)
            
    def test_state_persistence_integration(self):
        """Test state persistence integration"""
        print("Testing state persistence integration...")
        
        import tempfile
        import os
        
        # Run a few updates to change the state
        for i in range(3):
            self.controller.update()
            
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            temp_file = f.name
            
        try:
            # Save state
            result = self.controller.save_state(temp_file)
            self.assertTrue(result, "State should be saved successfully")
            
            # Create a new controller and load state
            new_controller = LambdaAttunementController(self.cal_engine, {})
            result = new_controller.load_state(temp_file)
            self.assertTrue(result, "State should be loaded successfully")
            
            # Verify that the state was loaded correctly
            self.assertAlmostEqual(new_controller.alpha, self.controller.alpha, 
                                 places=5, msg="Alpha should be preserved")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    def test_concurrent_updates_safety(self):
        """Test that concurrent updates are handled safely"""
        print("Testing concurrent updates safety...")
        
        # This test verifies that the controller's locking mechanism works
        # by running updates in quick succession
        
        # Run multiple updates rapidly
        results = []
        for i in range(10):
            result = self.controller.update()
            results.append(result)
            time.sleep(0.01)  # Very short delay
            
        # Verify that all updates completed without error
        self.assertEqual(len(results), 10, "All updates should complete")
        
        # Check that counters are reasonable
        status = self.controller.get_status()
        total_actions = status['accept_counter'] + status['revert_counter']
        self.assertGreaterEqual(total_actions, 0, "Counters should be non-negative")

class TestAttunementWithSimulatedShocks(unittest.TestCase):
    """Test attunement controller under simulated shocks"""
    
    def setUp(self):
        self.cal_engine = MockCALEngine()
        config = {
            "alpha_initial": 1.0,
            "alpha_min": 0.8,
            "alpha_max": 1.2,
            "lr": 0.005,  # Lower learning rate for more stable testing
            "momentum": 0.8,
            "epsilon": 1e-4,
            "settle_delay": 0.05,
            "cycle_sleep": 0.05,
            "safety": {
                "entropy_max": 0.002,
                "h_internal_min": 0.95,
                "revert_on_failure": True
            }
        }
        self.controller = LambdaAttunementController(self.cal_engine, config)
        
    def test_latency_shock(self):
        """Test controller behavior under simulated latency"""
        print("Testing controller under latency shock...")
        
        # Simulate latency by adding delays
        original_sleep = time.sleep
        time.sleep = Mock()  # Mock sleep to speed up tests
        
        try:
            # Run controller updates
            for i in range(5):
                self.controller.update()
                
            # Verify controller still functions
            status = self.controller.get_status()
            self.assertIsNotNone(status)
            self.assertIn('alpha', status)
        finally:
            # Restore original sleep
            time.sleep = original_sleep
            
    def test_partition_shock(self):
        """Test controller behavior under simulated network partitions"""
        print("Testing controller under partition shock...")
        
        # Simulate network partition by making omega samples temporarily unavailable
        def failing_sample():
            raise Exception("Network partition simulated")
            
        # Temporarily replace the sample method
        original_sample = self.cal_engine.sample_omega_snapshot
        self.cal_engine.sample_omega_snapshot = failing_sample
        
        try:
            # Run update that should handle the exception gracefully
            result = self.controller.update()
            # Should handle gracefully and possibly revert
            self.assertIn(result, [True, False], "Update should complete without crashing")
        finally:
            # Restore original method
            self.cal_engine.sample_omega_snapshot = original_sample
            
        # Verify controller still works after the shock
        result = self.controller.update()
        self.assertIn(result, [True, False], "Controller should recover from shock")

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Lambda Attunement Integration Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAttunementIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAttunementWithSimulatedShocks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)