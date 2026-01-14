#!/usr/bin/env python3
"""
Test for HMN Metrics Endpoint
"""

import unittest
import asyncio
import threading
import time
import requests
from prometheus_client import generate_latest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network.hmn.full_node import FullNode

class TestHMNMetrics(unittest.TestCase):
    """Test HMN Metrics functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.node_id = "test-node-001"
        self.network_config = {
            "shard_count": 3,
            "replication_factor": 2,
            "validator_count": 3,
            "metrics_port": 8000
        }
        self.node = FullNode(self.node_id, self.network_config)
        
        # Add sample validators
        validators_data = [
            ("validator-1", 0.95, 10000.0),
            ("validator-2", 0.87, 8000.0),
            ("validator-3", 0.75, 12000.0),
        ]
        
        for validator_id, psi_score, stake in validators_data:
            self.node.consensus_engine.add_validator(validator_id, psi_score, stake)
    
    def test_prometheus_metrics_generation(self):
        """Test that Prometheus metrics can be generated"""
        # Run some services to generate metrics
        asyncio.run(self.node.run_cal_engine())
        
        # Generate metrics
        metrics_output = generate_latest()
        metrics_text = metrics_output.decode('utf-8')
        
        # Check that our custom metrics are present
        self.assertIn("hmn_node_health_status", metrics_text)
        self.assertIn("hmn_node_lambda_t", metrics_text)
        self.assertIn("hmn_node_coherence_density", metrics_text)
        self.assertIn("hmn_node_psi_score", metrics_text)
    
    def test_service_health_metrics(self):
        """Test service health metrics"""
        # Check initial health status
        health_status = self.node.check_service_health()
        self.assertIsInstance(health_status, bool)
        
        # Get detailed health status
        health_details = self.node.get_health_status()
        self.assertIn("overall_health", health_details)
        self.assertIn("services", health_details)
        self.assertEqual(health_details["node_id"], self.node_id)
    
    def test_node_stats_collection(self):
        """Test node statistics collection"""
        stats = self.node.get_node_stats()
        
        # Check that all expected fields are present
        expected_fields = [
            "node_id", "running", "health_status", "cal_state", 
            "memory_stats", "consensus_stats", "mining_epoch_count",
            "ledger_transaction_count", "service_health"
        ]
        
        for field in expected_fields:
            self.assertIn(field, stats)
        
        # Check nested structures
        self.assertIn("lambda_t", stats["cal_state"])
        self.assertIn("coherence_density", stats["cal_state"])
        self.assertIn("psi_score", stats["cal_state"])

if __name__ == "__main__":
    unittest.main()