#!/usr/bin/env python3
"""
Edge Case Test Suite for Dimensional Observer Agent
Tests boundary behaviors, null states, and coherence collapse handling.
"""

import sys
import os
import unittest
import asyncio
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Handle relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.monitoring.observer_agent import DimensionalObserverAgent, TelemetryData, AnomalyDetectionResult

class TestObserverAgentEdgeCases(unittest.TestCase):
    """Edge case test suite for Dimensional Observer Agent"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.observer = DimensionalObserverAgent("test-network-edge-cases")
        
    def test_null_state_handling(self):
        """Test handling of null/empty states"""
        # Test with empty attention spectrum
        telemetry = asyncio.run(self.observer._collect_telemetry())
        
        # Verify telemetry structure even with minimal data
        self.assertIsInstance(telemetry, TelemetryData)
        self.assertGreater(telemetry.timestamp, 0)
        self.assertIsInstance(telemetry.omega_state, dict)
        self.assertGreaterEqual(telemetry.psi_score, 0.0)
        self.assertLessEqual(telemetry.psi_score, 1.0)
        
    def test_boundary_value_validation(self):
        """Test validation with boundary values"""
        # Test with extreme values
        extreme_values = [
            {"token": 1000.0, "sentiment": 100.0, "semantic": 10.0, "attention": [100.0] * 5},
            {"token": -1000.0, "sentiment": -100.0, "semantic": -10.0, "attention": [-100.0] * 5},
            {"token": 0.0, "sentiment": 0.0, "semantic": 0.0, "attention": [0.0] * 5},
            {"token": 1e-10, "sentiment": 1e-10, "semantic": 1e-10, "attention": [1e-10] * 5}
        ]
        
        for i, case in enumerate(extreme_values):
            # Update field statistics with extreme values
            self.observer._update_field_statistics("semantic_shift", case["semantic"])
            self.observer._update_field_statistics("sentiment_energy", case["sentiment"])
            
            # Verify statistics are still valid
            semantic_stats = self.observer.field_stats["semantic_shift"]
            sentiment_stats = self.observer.field_stats["sentiment_energy"]
            
            self.assertGreaterEqual(semantic_stats["count"], 1)
            self.assertGreaterEqual(sentiment_stats["count"], 1)
        
    def test_coherence_collapse_scenarios(self):
        """Test handling of coherence collapse scenarios"""
        # Simulate coherence collapse by creating highly inconsistent data
        for i in range(50):
            # Alternate between very high and very low values to create instability
            if i % 2 == 0:
                value = 100.0  # Very high
            else:
                value = -100.0  # Very low
                
            self.observer._update_field_statistics("semantic_shift", value)
            self.observer._update_field_statistics("sentiment_energy", value * 0.5)
        
        # Check that statistics can handle extreme variations
        semantic_stats = self.observer.field_stats["semantic_shift"]
        sentiment_stats = self.observer.field_stats["sentiment_energy"]
        
        # Stats should still be valid numbers
        self.assertIsInstance(semantic_stats["mean"], float)
        self.assertIsInstance(semantic_stats["std"], float)
        self.assertIsInstance(sentiment_stats["mean"], float)
        self.assertIsInstance(sentiment_stats["std"], float)
        
        # Standard deviation should be high due to extreme variations
        self.assertGreater(semantic_stats["std"], 0)
        self.assertGreater(sentiment_stats["std"], 0)
        
    def test_empty_attention_spectrum(self):
        """Test handling of empty attention spectrum"""
        # Create telemetry with empty attention spectrum
        omega_state = {
            "token_rate": 5.0,
            "sentiment_energy": 0.7,
            "semantic_shift": 0.3,
            "meta_attention_spectrum": [],  # Empty spectrum
            "coherence_score": 0.8,
            "modulator": 1.0,
            "time_delay": 0.5,
            "integrated_feedback": 2.0
        }
        
        telemetry = TelemetryData(
            timestamp=1234567890.0,
            omega_state=omega_state,
            psi_score=0.8,
            modulator=1.0,
            time_delay=0.5,
            integrated_feedback=2.0,
            network_health="green"
        )
        
        # Detect anomalies with empty spectrum
        anomalies = asyncio.run(self.observer._detect_anomalies(telemetry))
        
        # Should still detect anomalies for other fields
        self.assertIsInstance(anomalies, list)
        self.assertGreaterEqual(len(anomalies), 1)  # At least semantic_shift and sentiment_energy
        
        for anomaly in anomalies:
            self.assertIsInstance(anomaly, AnomalyDetectionResult)
            self.assertIn(anomaly.field_name, ["semantic_shift", "sentiment_energy"])
            
    def test_single_data_point_statistics(self):
        """Test statistics with single data point"""
        # Reset statistics
        self.observer.field_stats = {
            "semantic_shift": {"mean": 0.0, "std": 1.0, "count": 0},
            "sentiment_energy": {"mean": 0.0, "std": 1.0, "count": 0}
        }
        
        # Add single data point
        self.observer._update_field_statistics("semantic_shift", 5.0)
        
        # Check statistics
        stats = self.observer.field_stats["semantic_shift"]
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["mean"], 5.0)
        # Std should be 1.0 for single point (default)
        self.assertEqual(stats["std"], 1.0)
        
    def test_very_large_dataset_handling(self):
        """Test handling of very large datasets"""
        # Generate large dataset
        large_values = np.random.normal(0, 10, 10000)  # 10,000 random values
        
        # Update statistics with large dataset
        for value in large_values:
            self.observer._update_field_statistics("semantic_shift", value)
        
        # Check that statistics are still valid
        stats = self.observer.field_stats["semantic_shift"]
        self.assertEqual(stats["count"], 10000)
        self.assertIsInstance(stats["mean"], float)
        self.assertIsInstance(stats["std"], float)
        
        # Mean should be close to 0 (normal distribution)
        self.assertGreater(abs(stats["mean"]), 0)  # Should not be exactly 0
        self.assertGreater(stats["std"], 0)  # Should be positive

if __name__ == '__main__':
    unittest.main()