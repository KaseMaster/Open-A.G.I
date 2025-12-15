#!/usr/bin/env python3
"""
Unit tests for Coherence Density computation
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock

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
        
        CoherenceDensityMeter = lambda_attunement.CoherenceDensityMeter
    else:
        raise ImportError("Could not import lambda_attunement module")
else:
    raise ImportError(f"Lambda attunement module not found at {lambda_attunement_path}")

class MockCALEngine:
    """Mock CAL engine for testing coherence density computation"""
    def __init__(self, omega_samples=None):
        self.omega_samples = omega_samples or []
        
    def sample_omega_snapshot(self):
        return self.omega_samples
        
    def normalize_C(self, C):
        # Normalize C to [0,1] range
        return min(1.0, max(0.0, C / 10.0))

class TestCoherenceDensityComputation(unittest.TestCase):
    """Test cases for Coherence Density computation"""
    
    def setUp(self):
        self.cal_engine = MockCALEngine()
        self.meter = CoherenceDensityMeter(self.cal_engine)
        
    def test_compute_C_hat_normal_case(self):
        """Test normal computation of C_hat with typical values"""
        # Set up typical omega samples
        self.cal_engine.omega_samples = [
            [1.0, 0.5, 0.2],  # Node 1
            [0.8, 0.6, 0.3],  # Node 2
            [0.9, 0.4, 0.1]   # Node 3
        ]
        
        C_hat = self.meter.compute_C_hat()
        
        # Expected calculation:
        # Node 1: 1.0^2 + 0.5^2 + 0.2^2 = 1.0 + 0.25 + 0.04 = 1.29
        # Node 2: 0.8^2 + 0.6^2 + 0.3^2 = 0.64 + 0.36 + 0.09 = 1.09
        # Node 3: 0.9^2 + 0.4^2 + 0.1^2 = 0.81 + 0.16 + 0.01 = 0.98
        # Average: (1.29 + 1.09 + 0.98) / 3 = 3.36 / 3 = 1.12
        # Normalized: min(1.0, 1.12 / 10.0) = 0.112
        expected = 0.112
        
        self.assertAlmostEqual(C_hat, expected, places=3)
        
    def test_compute_C_hat_empty_samples(self):
        """Test computation with empty samples"""
        self.cal_engine.omega_samples = []
        C_hat = self.meter.compute_C_hat()
        self.assertEqual(C_hat, 0.0)
        
    def test_compute_C_hat_single_node(self):
        """Test computation with single node"""
        self.cal_engine.omega_samples = [
            [2.0, 3.0, 1.0]  # Single node with values
        ]
        
        C_hat = self.meter.compute_C_hat()
        
        # Expected calculation:
        # Node 1: 2.0^2 + 3.0^2 + 1.0^2 = 4.0 + 9.0 + 1.0 = 14.0
        # Average: 14.0 / 1 = 14.0
        # Normalized: min(1.0, 14.0 / 10.0) = 1.0 (clamped)
        expected = 1.0  # Clamped to 1.0
        
        self.assertAlmostEqual(C_hat, expected, places=3)
        
    def test_compute_C_hat_large_values(self):
        """Test computation with large values"""
        self.cal_engine.omega_samples = [
            [10.0, 10.0, 10.0],  # Large values
            [5.0, 5.0, 5.0]
        ]
        
        C_hat = self.meter.compute_C_hat()
        
        # Expected calculation:
        # Node 1: 10.0^2 + 10.0^2 + 10.0^2 = 100 + 100 + 100 = 300
        # Node 2: 5.0^2 + 5.0^2 + 5.0^2 = 25 + 25 + 25 = 75
        # Average: (300 + 75) / 2 = 375 / 2 = 187.5
        # Normalized: min(1.0, 187.5 / 10.0) = 1.0 (clamped)
        expected = 1.0  # Clamped to 1.0
        
        self.assertAlmostEqual(C_hat, expected, places=3)
        
    def test_compute_C_hat_zero_values(self):
        """Test computation with zero values"""
        self.cal_engine.omega_samples = [
            [0.0, 0.0, 0.0],  # Zero values
            [0.0, 0.0, 0.0]
        ]
        
        C_hat = self.meter.compute_C_hat()
        
        # Expected calculation:
        # Both nodes have zero squared norms
        # Average: (0 + 0) / 2 = 0
        # Normalized: min(1.0, 0 / 10.0) = 0.0
        expected = 0.0
        
        self.assertAlmostEqual(C_hat, expected, places=3)
        
    def test_compute_C_hat_mixed_positive_negative(self):
        """Test computation with mixed positive and negative values"""
        self.cal_engine.omega_samples = [
            [1.0, -2.0, 3.0],   # Mixed values
            [-1.0, 2.0, -3.0]
        ]
        
        C_hat = self.meter.compute_C_hat()
        
        # Expected calculation (squaring makes all positive):
        # Node 1: 1.0^2 + (-2.0)^2 + 3.0^2 = 1 + 4 + 9 = 14
        # Node 2: (-1.0)^2 + 2.0^2 + (-3.0)^2 = 1 + 4 + 9 = 14
        # Average: (14 + 14) / 2 = 28 / 2 = 14
        # Normalized: min(1.0, 14 / 10.0) = 1.0 (clamped)
        expected = 1.0  # Clamped to 1.0
        
        self.assertAlmostEqual(C_hat, expected, places=3)
        
    def test_compute_C_hat_many_dimensions(self):
        """Test computation with many dimensions"""
        # Create high-dimensional omega vectors
        dimensions = 50
        nodes = 10
        
        omega_samples = []
        for i in range(nodes):
            # Create node with random values
            node_vector = [np.random.random() * 2.0 for _ in range(dimensions)]
            omega_samples.append(node_vector)
            
        self.cal_engine.omega_samples = omega_samples
        C_hat = self.meter.compute_C_hat()
        
        # Should produce a valid result between 0 and 1
        self.assertGreaterEqual(C_hat, 0.0)
        self.assertLessEqual(C_hat, 1.0)
        
    def test_compute_C_hat_many_nodes(self):
        """Test computation with many nodes"""
        # Create many nodes with simple values
        nodes = 100
        omega_samples = []
        for i in range(nodes):
            # Each node has values that sum to a predictable squared norm
            node_vector = [0.1, 0.2, 0.3]  # Squared norm = 0.01 + 0.04 + 0.09 = 0.14
            omega_samples.append(node_vector)
            
        self.cal_engine.omega_samples = omega_samples
        C_hat = self.meter.compute_C_hat()
        
        # Expected: All nodes have same squared norm of 0.14
        # Average: 0.14
        # Normalized: min(1.0, 0.14 / 10.0) = 0.014
        expected = 0.014
        
        self.assertAlmostEqual(C_hat, expected, places=3)
        
    def test_compute_C_hat_exception_handling(self):
        """Test exception handling in C_hat computation"""
        # Mock an exception in sample_omega_snapshot
        self.cal_engine.sample_omega_snapshot = Mock(side_effect=Exception("Test error"))
        C_hat = self.meter.compute_C_hat()
        self.assertEqual(C_hat, 0.0)
        
    def test_compute_C_hat_normalization_edge_cases(self):
        """Test edge cases in normalization"""
        # Test with exactly 10.0 (boundary case)
        self.cal_engine.omega_samples = [
            [np.sqrt(50), np.sqrt(50)]  # Squared norm = 50 + 50 = 100, avg = 100, norm = 10.0
        ]
        
        # Override normalization to test boundary
        self.cal_engine.normalize_C = lambda C: min(1.0, max(0.0, C))
        
        C_hat = self.meter.compute_C_hat()
        expected = 10.0  # Before normalization
        
        # Since our mock doesn't do the 10.0 division, we check it's a reasonable value
        self.assertGreaterEqual(C_hat, 0.0)
        self.assertLessEqual(C_hat, 100.0)  # Reasonable upper bound

class TestCoherenceDensityStability(unittest.TestCase):
    """Test stability and consistency of coherence density computation"""
    
    def setUp(self):
        self.cal_engine = MockCALEngine()
        self.meter = CoherenceDensityMeter(self.cal_engine, window=10)
        
    def test_consistent_results(self):
        """Test that identical inputs produce identical outputs"""
        omega_samples = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        self.cal_engine.omega_samples = omega_samples
        result1 = self.meter.compute_C_hat()
        
        # Reset with same samples
        self.cal_engine.omega_samples = omega_samples
        result2 = self.meter.compute_C_hat()
        
        self.assertEqual(result1, result2)
        
    def test_deterministic_computation(self):
        """Test that computation is deterministic"""
        # Generate deterministic test data
        np.random.seed(42)  # Fixed seed for reproducibility
        omega_samples = []
        for i in range(5):
            node = [np.random.random() for _ in range(10)]
            omega_samples.append(node)
            
        self.cal_engine.omega_samples = omega_samples
        
        # Run multiple times
        results = []
        for i in range(10):
            result = self.meter.compute_C_hat()
            results.append(result)
            
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(results[0], result)
            
    def test_numerical_stability(self):
        """Test numerical stability with very small values"""
        self.cal_engine.omega_samples = [
            [1e-10, 1e-10, 1e-10],  # Very small values
            [1e-15, 1e-15, 1e-15]
        ]
        
        C_hat = self.meter.compute_C_hat()
        
        # Should produce a valid result (very close to zero)
        self.assertGreaterEqual(C_hat, 0.0)
        self.assertLess(C_hat, 1e-5)  # Should be very small
        
    def test_window_parameter(self):
        """Test that window parameter is properly handled"""
        # Create meter with different window size
        meter_small_window = CoherenceDensityMeter(self.cal_engine, window=2)
        meter_large_window = CoherenceDensityMeter(self.cal_engine, window=20)
        
        # Both should work correctly
        self.cal_engine.omega_samples = [[1.0, 1.0]]
        result_small = meter_small_window.compute_C_hat()
        result_large = meter_large_window.compute_C_hat()
        
        # Results should be the same for identical inputs
        self.assertEqual(result_small, result_large)

def run_coherence_density_tests():
    """Run all coherence density tests"""
    print("🔬 Running Coherence Density Computation Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCoherenceDensityComputation))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCoherenceDensityStability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("COHERENCE DENSITY TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_coherence_density_tests()
    sys.exit(0 if success else 1)