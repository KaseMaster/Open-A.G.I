#!/usr/bin/env python3
"""
Test suite for Dimensional Observer Agent
Tests telemetry collection, anomaly detection, and network health monitoring.
"""

import sys
import os
import unittest
import asyncio
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Handle relative imports
try:
    from monitoring.observer_agent import DimensionalObserverAgent, TelemetryData, AnomalyDetectionResult
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'monitoring'))
    from observer_agent import DimensionalObserverAgent, TelemetryData, AnomalyDetectionResult

class TestObserverAgent(unittest.TestCase):
    """Test suite for Dimensional Observer Agent"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.observer = DimensionalObserverAgent("test-network")
        
    def test_initialization(self):
        """Test observer agent initialization"""
        self.assertEqual(self.observer.network_id, "test-network")
        self.assertIsNotNone(self.observer.cal_engine)
        self.assertIsNotNone(self.observer.coherence_layer)
        self.assertEqual(len(self.observer.telemetry_history), 0)
        self.assertEqual(len(self.observer.anomaly_history), 0)
        
        print("✅ Observer agent initialization test passed")
        
    def test_telemetry_collection(self):
        """Test telemetry data collection"""
        # Collect telemetry
        telemetry = asyncio.run(self.observer._collect_telemetry())
        
        # Verify telemetry structure
        self.assertIsInstance(telemetry, TelemetryData)
        self.assertGreater(telemetry.timestamp, 0)
        self.assertIsInstance(telemetry.omega_state, dict)
        self.assertGreaterEqual(telemetry.psi_score, 0.0)
        self.assertLessEqual(telemetry.psi_score, 1.0)
        self.assertIn(telemetry.network_health, ["green", "yellow", "red", "critical", "unknown"])
        
        print(f"✅ Telemetry collection test passed: Ψ={telemetry.psi_score:.4f}")
        
    def test_anomaly_detection(self):
        """Test anomaly detection functionality"""
        # Create sample telemetry data
        omega_state = {
            "token_rate": 5.0,
            "sentiment_energy": 0.7,
            "semantic_shift": 0.3,
            "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
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
        
        # Detect anomalies
        anomalies = asyncio.run(self.observer._detect_anomalies(telemetry))
        
        # Verify anomaly detection results
        self.assertIsInstance(anomalies, list)
        self.assertEqual(len(anomalies), 2)  # semantic_shift and sentiment_energy
        
        for anomaly in anomalies:
            self.assertIsInstance(anomaly, AnomalyDetectionResult)
            self.assertGreaterEqual(anomaly.timestamp, 0)
            self.assertIn(anomaly.field_name, ["semantic_shift", "sentiment_energy"])
            self.assertIsInstance(anomaly.value, float)
            self.assertIsInstance(anomaly.z_score, float)
            self.assertIsInstance(anomaly.is_anomaly, bool)
            self.assertIn(anomaly.severity, ["low", "medium", "high"])
            self.assertIsInstance(anomaly.description, str)
        
        print(f"✅ Anomaly detection test passed: {len(anomalies)} anomalies detected")
        
    def test_field_statistics_update(self):
        """Test field statistics updating"""
        # Test updating statistics for semantic_shift
        self.observer._update_field_statistics("semantic_shift", 0.5)
        self.observer._update_field_statistics("semantic_shift", 0.6)
        self.observer._update_field_statistics("semantic_shift", 0.4)
        
        # Verify statistics were updated
        stats = self.observer.field_stats["semantic_shift"]
        self.assertEqual(stats["count"], 3)
        self.assertGreater(stats["mean"], 0)
        
        # Test updating statistics for sentiment_energy
        self.observer._update_field_statistics("sentiment_energy", 0.7)
        self.observer._update_field_statistics("sentiment_energy", 0.8)
        
        stats = self.observer.field_stats["sentiment_energy"]
        self.assertEqual(stats["count"], 2)
        self.assertGreater(stats["mean"], 0)
        
        print("✅ Field statistics update test passed")
        
    def test_network_health_check(self):
        """Test network health checking"""
        # Test with no telemetry data
        health = asyncio.run(self.observer._check_network_health())
        self.assertEqual(health, "unknown")
        
        # Add some telemetry data with different health statuses
        for i in range(10):
            omega_state = {
                "token_rate": 5.0,
                "sentiment_energy": 0.7,
                "semantic_shift": 0.3,
                "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
                "coherence_score": 0.85 if i < 8 else 0.4,  # Mostly green, some red
                "modulator": 1.0,
                "time_delay": 0.5,
                "integrated_feedback": 2.0
            }
            
            telemetry = TelemetryData(
                timestamp=1234567890.0 + i,
                omega_state=omega_state,
                psi_score=0.85 if i < 8 else 0.4,
                modulator=1.0,
                time_delay=0.5,
                integrated_feedback=2.0,
                network_health="green" if i < 8 else "red"
            )
            
            self.observer.telemetry_history.append(telemetry)
        
        # Check health with mixed data
        health = asyncio.run(self.observer._check_network_health())
        self.assertIn(health, ["healthy", "degraded", "unstable", "critical"])
        
        print(f"✅ Network health check test passed: {health}")
        
    def test_recent_data_retrieval(self):
        """Test retrieval of recent telemetry and anomalies"""
        # Add some telemetry data
        for i in range(15):
            omega_state = {
                "token_rate": 5.0,
                "sentiment_energy": 0.7,
                "semantic_shift": 0.3,
                "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
                "coherence_score": 0.8,
                "modulator": 1.0,
                "time_delay": 0.5,
                "integrated_feedback": 2.0
            }
            
            telemetry = TelemetryData(
                timestamp=1234567890.0 + i,
                omega_state=omega_state,
                psi_score=0.8,
                modulator=1.0,
                time_delay=0.5,
                integrated_feedback=2.0,
                network_health="green"
            )
            
            self.observer.telemetry_history.append(telemetry)
        
        # Add some anomalies
        for i in range(5):
            anomaly = AnomalyDetectionResult(
                timestamp=1234567890.0 + i,
                field_name="semantic_shift",
                value=0.5,
                z_score=3.5,
                is_anomaly=True,
                severity="medium",
                description="Test anomaly"
            )
            self.observer.anomaly_history.append(anomaly)
        
        # Test recent telemetry retrieval
        recent_telemetry = self.observer.get_recent_telemetry(10)
        self.assertEqual(len(recent_telemetry), 10)
        self.assertIsInstance(recent_telemetry[0], TelemetryData)
        
        # Test recent anomalies retrieval
        recent_anomalies = self.observer.get_recent_anomalies(3)
        self.assertEqual(len(recent_anomalies), 3)
        self.assertIsInstance(recent_anomalies[0], AnomalyDetectionResult)
        
        print("✅ Recent data retrieval test passed")
        
    def test_network_health_summary(self):
        """Test network health summary generation"""
        # Test with no data
        summary = self.observer.get_network_health_summary()
        self.assertEqual(summary["status"], "no_data")
        
        # Add some telemetry data
        for i in range(20):
            omega_state = {
                "token_rate": 5.0,
                "sentiment_energy": 0.7,
                "semantic_shift": 0.3,
                "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
                "coherence_score": 0.75 + (i * 0.01),  # Improving trend
                "modulator": 1.0,
                "time_delay": 0.5,
                "integrated_feedback": 2.0
            }
            
            telemetry = TelemetryData(
                timestamp=1234567890.0 + i,
                omega_state=omega_state,
                psi_score=0.75 + (i * 0.01),
                modulator=1.0,
                time_delay=0.5,
                integrated_feedback=2.0,
                network_health="yellow" if i < 15 else "green"
            )
            
            self.observer.telemetry_history.append(telemetry)
        
        # Add some anomalies
        for i in range(3):
            anomaly = AnomalyDetectionResult(
                timestamp=1234567890.0 + i,
                field_name="semantic_shift",
                value=0.5,
                z_score=3.5,
                is_anomaly=True,
                severity="medium",
                description="Test anomaly"
            )
            self.observer.anomaly_history.append(anomaly)
        
        # Test summary generation
        summary = self.observer.get_network_health_summary()
        self.assertEqual(summary["status"], "operational")
        self.assertIn(summary["current_health"], ["green", "yellow", "red", "critical", "unknown"])
        self.assertIsInstance(summary["psi_stats"], dict)
        self.assertGreaterEqual(summary["anomaly_count"], 0)
        self.assertGreater(summary["total_telemetry_points"], 0)
        
        # Verify psi_stats structure
        psi_stats = summary["psi_stats"]
        self.assertIn("mean", psi_stats)
        self.assertIn("std", psi_stats)
        self.assertIn("min", psi_stats)
        self.assertIn("max", psi_stats)
        self.assertIn("trend", psi_stats)
        self.assertIn(psi_stats["trend"], ["stable", "improving", "declining", "unknown"])
        
        print(f"✅ Network health summary test passed: trend={psi_stats['trend']}, anomalies={summary['anomaly_count']}")

if __name__ == '__main__':
    unittest.main()