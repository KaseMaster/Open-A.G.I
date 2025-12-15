#!/usr/bin/env python3
"""
Unit tests for the Lambda Attunement Controller
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

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
    """Mock CAL engine for testing"""
    def __init__(self):
        self.alpha_multiplier = 1.0
        self.omega_samples = [[1.0, 0.5, 0.2], [0.8, 0.6, 0.3], [0.9, 0.4, 0.1]]
        self.entropy_rate_value = 0.001
        self.h_internal_value = 0.98
        self.m_t_bounds_value = (-1.0, 1.0)
        
    def sample_omega_snapshot(self):
        return self.omega_samples
        
    def normalize_C(self, C):
        return min(1.0, max(0.0, C / 10.0))
        
    def set_alpha_multiplier(self, alpha):
        self.alpha_multiplier = alpha
        
    def get_lambda(self):
        return 0.5
        
    def get_entropy_rate(self):
        return self.entropy_rate_value
        
    def get_h_internal(self):
        return self.h_internal_value
        
    def get_m_t_bounds(self):
        return self.m_t_bounds_value

class TestCoherenceDensityMeter(unittest.TestCase):
    """Test cases for CoherenceDensityMeter"""
    
    def setUp(self):
        self.cal_engine = MockCALEngine()
        self.meter = CoherenceDensityMeter(self.cal_engine)
        
    def test_compute_C_hat_normal_case(self):
        """Test normal computation of C_hat"""
        C_hat = self.meter.compute_C_hat()
        # Expected: sum of squared norms / number of samples
        # Sample 1: 1.0^2 + 0.5^2 + 0.2^2 = 1.29
        # Sample 2: 0.8^2 + 0.6^2 + 0.3^2 = 1.09
        # Sample 3: 0.9^2 + 0.4^2 + 0.1^2 = 0.98
        # Average: (1.29 + 1.09 + 0.98) / 3 = 1.12
        # Normalized: min(1.0, 1.12 / 10.0) = 0.112
        self.assertAlmostEqual(C_hat, 0.112, places=3)
        
    def test_compute_C_hat_empty_samples(self):
        """Test computation with empty samples"""
        self.cal_engine.omega_samples = []
        C_hat = self.meter.compute_C_hat()
        self.assertEqual(C_hat, 0.0)
        
    def test_compute_C_hat_exception_handling(self):
        """Test exception handling in C_hat computation"""
        # Mock an exception in sample_omega_snapshot
        self.cal_engine.sample_omega_snapshot = Mock(side_effect=Exception("Test error"))
        C_hat = self.meter.compute_C_hat()
        self.assertEqual(C_hat, 0.0)

class TestLambdaAttunementController(unittest.TestCase):
    """Test cases for LambdaAttunementController"""
    
    def setUp(self):
        self.cal_engine = MockCALEngine()
        config = {
            "alpha_initial": 1.0,
            "alpha_min": 0.8,
            "alpha_max": 1.2,
            "lr": 0.01,
            "momentum": 0.9,
            "epsilon": 1e-4
        }
        self.controller = LambdaAttunementController(self.cal_engine, config)
        
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.alpha, 1.0)
        self.assertEqual(self.controller.velocity, 0.0)
        self.assertFalse(self.controller.running)
        self.assertIsInstance(self.controller.meter, CoherenceDensityMeter)
        
    def test_set_alpha_within_bounds(self):
        """Test setting alpha within bounds"""
        new_alpha = self.controller._set_alpha(1.1)
        self.assertEqual(new_alpha, 1.1)
        self.assertEqual(self.cal_engine.alpha_multiplier, 1.1)
        
    def test_set_alpha_clamping_high(self):
        """Test alpha clamping at upper bound"""
        new_alpha = self.controller._set_alpha(1.5)
        self.assertEqual(new_alpha, 1.2)  # Should be clamped to alpha_max
        self.assertEqual(self.cal_engine.alpha_multiplier, 1.2)
        
    def test_set_alpha_clamping_low(self):
        """Test alpha clamping at lower bound"""
        new_alpha = self.controller._set_alpha(0.5)
        self.assertEqual(new_alpha, 0.8)  # Should be clamped to alpha_min
        self.assertEqual(self.cal_engine.alpha_multiplier, 0.8)
        
    def test_estimate_gradient(self):
        """Test gradient estimation"""
        grad, C0 = self.controller._estimate_gradient(1e-4)
        self.assertIsInstance(grad, float)
        self.assertIsInstance(C0, float)
        self.assertGreaterEqual(C0, 0.0)
        self.assertLessEqual(C0, 1.0)
        
    def test_check_safety_constraints_safe(self):
        """Test safety constraints with safe values"""
        is_safe = self.controller._check_safety_constraints()
        self.assertTrue(is_safe)
        
    def test_check_safety_constraints_high_entropy(self):
        """Test safety constraints with high entropy"""
        self.cal_engine.entropy_rate_value = 0.003  # Above threshold
        is_safe = self.controller._check_safety_constraints()
        self.assertFalse(is_safe)
        
    def test_check_safety_constraints_low_coherence(self):
        """Test safety constraints with low internal coherence"""
        self.cal_engine.h_internal_value = 0.90  # Below threshold
        is_safe = self.controller._check_safety_constraints()
        self.assertFalse(is_safe)
        
    def test_update_accepted_change(self):
        """Test update with accepted change"""
        # Set up conditions for an accepted change
        self.controller.meter.compute_C_hat = Mock(return_value=0.5)  # Initial C_hat
        with patch.object(self.controller, '_estimate_gradient', return_value=(1.0, 0.5)):
            result = self.controller.update()
            self.assertTrue(result)  # Change should be accepted
            self.assertEqual(self.controller.mode, 1)  # Gradient mode
            self.assertEqual(self.controller.accept_counter, 1)
            self.assertEqual(self.controller.revert_counter, 0)
            
    def test_update_reverted_change(self):
        """Test update with reverted change"""
        # Set up conditions for a reverted change
        self.controller.meter.compute_C_hat = Mock(return_value=0.5)  # Initial C_hat
        with patch.object(self.controller, '_estimate_gradient', return_value=(-1.0, 0.5)):
            # Mock the meter to return a lower C_hat after the update
            self.controller.meter.compute_C_hat = Mock(return_value=0.3)  # Lower C_hat
            result = self.controller.update()
            self.assertFalse(result)  # Change should be reverted
            self.assertEqual(self.controller.mode, 2)  # PID mode fallback
            self.assertEqual(self.controller.accept_counter, 0)
            self.assertEqual(self.controller.revert_counter, 1)
            
    def test_get_status(self):
        """Test getting controller status"""
        status = self.controller.get_status()
        self.assertIn("alpha", status)
        self.assertIn("velocity", status)
        self.assertIn("mode", status)
        self.assertIn("accept_counter", status)
        self.assertIn("revert_counter", status)
        self.assertIn("history_length", status)
        self.assertIn("enabled", status)
        
    def test_get_audit_ledger(self):
        """Test getting audit ledger"""
        ledger = self.controller.get_audit_ledger()
        self.assertIsInstance(ledger, list)
        
    def test_save_and_load_state(self):
        """Test saving and loading state"""
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            temp_file = f.name
            
        try:
            # Save state
            result = self.controller.save_state(temp_file)
            self.assertTrue(result)
            
            # Load state
            new_controller = LambdaAttunementController(self.cal_engine, {})
            result = new_controller.load_state(temp_file)
            self.assertTrue(result)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestAttunementConfig(unittest.TestCase):
    """Test cases for AttunementConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = AttunementConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.alpha_initial, 1.0)
        self.assertEqual(config.alpha_min, 0.8)
        self.assertEqual(config.alpha_max, 1.2)
        self.assertIsInstance(config.safety, dict)
        self.assertIsInstance(config.logging, dict)
        
    def test_custom_config(self):
        """Test custom configuration"""
        custom_config = {
            "alpha_initial": 0.9,
            "alpha_min": 0.7,
            "alpha_max": 1.3,
            "safety": {
                "entropy_max": 0.003,
                "h_internal_min": 0.90,
                "revert_on_failure": False
            }
        }
        config = AttunementConfig(**custom_config)
        self.assertEqual(config.alpha_initial, 0.9)
        self.assertEqual(config.alpha_min, 0.7)
        self.assertEqual(config.alpha_max, 1.3)
        self.assertEqual(config.safety["entropy_max"], 0.003)
        self.assertEqual(config.safety["h_internal_min"], 0.90)
        self.assertFalse(config.safety["revert_on_failure"])

if __name__ == '__main__':
    unittest.main()