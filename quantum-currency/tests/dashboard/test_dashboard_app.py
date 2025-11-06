"""
Unit tests for the Quantum Currency Dashboard
"""

import unittest
import sys
import os
import tempfile
import json
import time

# Add the quantum-currency directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dashboard.dashboard_app import QuantumCurrencyDashboard, DashboardMetrics


class TestQuantumCurrencyDashboard(unittest.TestCase):
    """Test suite for QuantumCurrencyDashboard"""

    def setUp(self):
        """Set up test fixtures"""
        self.dashboard = QuantumCurrencyDashboard(network_id="test-dashboard-001")

    def test_initialization(self):
        """Test dashboard initialization"""
        self.assertEqual(self.dashboard.network_id, "test-dashboard-001")
        self.assertIsNotNone(self.dashboard.cal)
        self.assertEqual(len(self.dashboard.metrics_history), 1)  # Initial sample data
        self.assertFalse(self.dashboard.is_running)
        
        # Check initial metrics
        initial_metrics = self.dashboard.metrics_history[0]
        self.assertIsInstance(initial_metrics, DashboardMetrics)
        self.assertIsInstance(initial_metrics.timestamp, float)
        self.assertIsInstance(initial_metrics.network_coherence, float)
        self.assertIsInstance(initial_metrics.omega_state_metrics, dict)
        self.assertIsInstance(initial_metrics.token_balances, dict)
        self.assertEqual(initial_metrics.active_validators, 5)
        self.assertIsInstance(initial_metrics.recent_events, list)
        self.assertIn(initial_metrics.health_status, ["green", "yellow", "red", "critical"])

    def test_health_status_calculation(self):
        """Test health status calculation"""
        # Test green status (high coherence)
        self.assertEqual(self.dashboard._get_health_status(0.9), "green")
        self.assertEqual(self.dashboard._get_health_status(0.85), "green")
        
        # Test yellow status (medium coherence)
        self.assertEqual(self.dashboard._get_health_status(0.7), "yellow")
        self.assertEqual(self.dashboard._get_health_status(0.65), "yellow")
        
        # Test red status (low coherence)
        self.assertEqual(self.dashboard._get_health_status(0.5), "red")
        self.assertEqual(self.dashboard._get_health_status(0.35), "red")
        
        # Test critical status (very low coherence)
        self.assertEqual(self.dashboard._get_health_status(0.2), "critical")
        self.assertEqual(self.dashboard._get_health_status(0.0), "critical")

    def test_metrics_collection(self):
        """Test metrics collection"""
        metrics = self.dashboard._collect_current_metrics()
        
        self.assertIsInstance(metrics, DashboardMetrics)
        self.assertIsInstance(metrics.timestamp, float)
        self.assertIsInstance(metrics.network_coherence, float)
        self.assertGreaterEqual(metrics.network_coherence, 0.0)
        self.assertLessEqual(metrics.network_coherence, 1.0)
        self.assertIsInstance(metrics.omega_state_metrics, dict)
        self.assertIsInstance(metrics.token_balances, dict)
        self.assertEqual(metrics.active_validators, 5)
        self.assertIsInstance(metrics.recent_events, list)
        self.assertIn(metrics.health_status, ["green", "yellow", "red", "critical"])

    def test_metrics_update(self):
        """Test metrics update functionality"""
        initial_history_length = len(self.dashboard.metrics_history)
        
        # Update metrics
        self.dashboard.update_metrics()
        
        # Check that history grew
        self.assertEqual(len(self.dashboard.metrics_history), initial_history_length + 1)
        
        # Check that metrics are reasonable
        latest_metrics = self.dashboard.metrics_history[-1]
        self.assertIsInstance(latest_metrics.network_coherence, float)
        self.assertGreaterEqual(latest_metrics.network_coherence, 0.0)
        self.assertLessEqual(latest_metrics.network_coherence, 1.0)

    def test_current_view_generation(self):
        """Test current view generation"""
        view_data = self.dashboard.get_current_view()
        
        self.assertIsInstance(view_data, dict)
        self.assertIn("timestamp", view_data)
        self.assertIn("network_coherence", view_data)
        self.assertIn("health_status", view_data)
        self.assertIn("omega_metrics", view_data)
        self.assertIn("token_balances", view_data)
        self.assertIn("active_validators", view_data)
        self.assertIn("recent_events", view_data)
        self.assertIn("historical_data", view_data)
        
        # Check data types
        self.assertIsInstance(view_data["timestamp"], str)
        self.assertIsInstance(view_data["network_coherence"], float)
        self.assertIn(view_data["health_status"], ["green", "yellow", "red", "critical"])
        self.assertIsInstance(view_data["omega_metrics"], dict)
        self.assertIsInstance(view_data["token_balances"], dict)
        self.assertIsInstance(view_data["active_validators"], int)
        self.assertIsInstance(view_data["recent_events"], list)
        self.assertIsInstance(view_data["historical_data"], dict)

    def test_health_summary(self):
        """Test health summary generation"""
        # Update metrics a few times to have history
        for i in range(3):
            self.dashboard.update_metrics()
            time.sleep(0.1)  # Small delay to ensure different timestamps
        
        summary = self.dashboard.get_health_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("current_coherence", summary)
        self.assertIn("health_status", summary)
        self.assertIn("avg_recent_coherence", summary)
        self.assertIn("coherence_trend", summary)
        self.assertIn("active_validators", summary)
        self.assertIn("timestamp", summary)
        
        # Check data types
        self.assertIsInstance(summary["current_coherence"], float)
        self.assertIn(summary["health_status"], ["green", "yellow", "red", "critical"])
        self.assertIsInstance(summary["avg_recent_coherence"], float)
        self.assertIn(summary["coherence_trend"], ["stable", "improving", "declining"])
        self.assertIsInstance(summary["active_validators"], int)
        self.assertIsInstance(summary["timestamp"], str)

    def test_metrics_export(self):
        """Test metrics export functionality"""
        # Update metrics a few times
        for i in range(3):
            self.dashboard.update_metrics()
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.dashboard.export_metrics(temp_file)
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Read and verify data
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            self.assertIn("network_id", data)
            self.assertIn("export_timestamp", data)
            self.assertIn("metrics_history", data)
            self.assertEqual(data["network_id"], "test-dashboard-001")
            self.assertIsInstance(data["export_timestamp"], (int, float))
            self.assertIsInstance(data["metrics_history"], list)
            self.assertGreater(len(data["metrics_history"]), 0)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_history_limiting(self):
        """Test that history is properly limited"""
        # Add more metrics than the limit
        for i in range(110):  # More than max_history of 100
            self.dashboard.update_metrics()
        
        # Check that history is limited
        self.assertLessEqual(len(self.dashboard.metrics_history), 100)
        
        # Check that we still have the most recent data
        self.assertEqual(len(self.dashboard.metrics_history), 100)


if __name__ == '__main__':
    unittest.main()