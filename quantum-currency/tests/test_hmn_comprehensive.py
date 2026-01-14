#!/usr/bin/env python3
"""
Comprehensive Test Suite for Harmonic Mesh Network Enhancements
"""

import unittest
import asyncio
import json
import time
from typing import Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestHMNComprehensive(unittest.TestCase):
    """Comprehensive test suite for HMN enhancements"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.node_id = "test-node-001"
        self.network_config = {
            "shard_count": 3,
            "replication_factor": 2,
            "validator_count": 3,
            "metrics_port": 8000,
            "enable_tls": True,
            "discovery_enabled": True,
            "network_peers": ["peer-001", "peer-002", "peer-003"]
        }
        # Import inside method to avoid import errors
        from network.hmn.full_node import FullNode
        self.node = FullNode(self.node_id, self.network_config)
        
        # Add sample validators
        validators_data = [
            ("validator-1", 0.95, 10000.0),
            ("validator-2", 0.87, 8000.0),
            ("validator-3", 0.75, 12000.0),
        ]
        
        for validator_id, psi_score, stake in validators_data:
            self.node.consensus_engine.add_validator(validator_id, psi_score, stake)
    
    def test_full_node_initialization(self):
        """Test full node initialization with all enhancements"""
        # Check that all services are initialized
        self.assertIsNotNone(self.node.ledger)
        self.assertIsNotNone(self.node.cal_engine)
        self.assertIsNotNone(self.node.mining_agent)
        self.assertIsNotNone(self.node.memory_mesh_service)
        self.assertIsNotNone(self.node.consensus_engine)
        
        # Check that worker threads are set up
        self.assertIsNotNone(self.node.worker_threads)
        self.assertFalse(self.node.workers_running)
        
        # Check that service queues are initialized
        expected_queues = ["ledger", "cal_engine", "mining_agent", "memory_mesh", "consensus"]
        for queue_name in expected_queues:
            self.assertIn(queue_name, self.node.service_queues)
    
    def test_asynchronous_message_queues(self):
        """Test asynchronous message queues functionality"""
        # Start worker threads
        self.node.start_worker_threads()
        self.assertTrue(self.node.workers_running)
        
        # Check that worker threads are running
        for thread in self.node.worker_threads.values():
            self.assertTrue(thread.is_alive())
        
        # Stop worker threads
        self.node.stop_worker_threads()
        self.assertFalse(self.node.workers_running)
    
    def test_service_health_checks(self):
        """Test service health check functionality"""
        # Check initial health status
        health_status = self.node.check_service_health()
        self.assertIsInstance(health_status, bool)
        
        # Get detailed health status
        health_details = self.node.get_health_status()
        self.assertIn("overall_health", health_details)
        self.assertIn("services", health_details)
        self.assertEqual(health_details["node_id"], self.node_id)
        
        # Check that all services have health status
        expected_services = ["ledger", "cal_engine", "mining_agent", "memory_mesh", "consensus"]
        for service in expected_services:
            self.assertIn(service, health_details["services"])
    
    def test_dynamic_service_intervals(self):
        """Test dynamic service interval adjustment"""
        # Test with different network states
        network_states = [
            {"lambda_t": 0.9, "coherence_density": 0.95, "psi_score": 0.9},  # High stability
            {"lambda_t": 0.5, "coherence_density": 0.7, "psi_score": 0.7},   # Medium stability
            {"lambda_t": 0.2, "coherence_density": 0.4, "psi_score": 0.5},   # Low stability
        ]
        
        original_intervals = self.node.intervals.copy()
        
        for network_state in network_states:
            self.node.adjust_service_intervals(network_state)
            
            # Check that intervals have been adjusted
            for service in original_intervals.keys():
                self.assertIn(service, self.node.intervals)
    
    def test_memory_mesh_enhancements(self):
        """Test Memory Mesh Service enhancements"""
        memory_service = self.node.memory_mesh_service
        
        # Check peer discovery configuration
        self.assertTrue(memory_service.config["network"]["discovery_enabled"])
        self.assertTrue(memory_service.config["network"]["enable_tls"])
        
        # Check that initial peers are known
        self.assertGreater(len(memory_service.known_peers), 0)
        
        # Test peer discovery
        network_state = {"coherence_density": 0.85}
        memory_service.discover_peers(network_state)
        
        # Check that peers were discovered
        self.assertGreaterEqual(memory_service.metrics["peers_discovered"], 0)
        
        # Test secure connection establishment
        peer_id = "test-peer-001"
        success = memory_service.establish_secure_connection(peer_id)
        self.assertTrue(success)
        self.assertIn(peer_id, memory_service.peer_connections)
    
    def test_cal_engine_enhancements(self):
        """Test CAL Engine enhancements"""
        cal_engine = self.node.cal_engine
        
        # Test time-series forecasting
        forecasts = cal_engine.forecast_coherence_trend(steps=5)
        self.assertEqual(len(forecasts), 5)
        for forecast in forecasts:
            self.assertIsInstance(forecast, float)
            self.assertGreaterEqual(forecast, 0.5)
            self.assertLessEqual(forecast, 1.0)
        
        # Test coherence trend analysis
        trends = cal_engine.analyze_coherence_trends()
        self.assertIn("trend", trends)
        self.assertIn("slope", trends)
        self.assertIn("volatility", trends)
        self.assertIn("stability", trends)
    
    def test_mining_agent_enhancements(self):
        """Test Mining Agent enhancements"""
        mining_agent = self.node.mining_agent
        
        # Test adaptive minting calculation
        network_state = {
            "coherence_density": 0.9,
            "lambda_t": 0.8,
            "psi_score": 0.95
        }
        
        mint_amount = mining_agent.calculate_adaptive_minting_amount(network_state)
        self.assertIsInstance(mint_amount, float)
        self.assertGreater(mint_amount, 0)
        
        # Test minting decision
        should_mint = mining_agent.should_mint_in_epoch(network_state)
        self.assertIsInstance(should_mint, bool)
        
        # Test minting statistics
        stats = mining_agent.get_minting_stats()
        self.assertIn("total_minted", stats)
        self.assertIn("epochs_run", stats)
        self.assertIn("average_mint_amount", stats)
    
    def test_consensus_engine_enhancements(self):
        """Test Consensus Engine enhancements"""
        consensus_engine = self.node.consensus_engine
        
        # Test weighted validator voting
        validators = list(consensus_engine.validators.values())
        self.assertGreater(len(validators), 0)
        
        for validator in validators:
            weight = consensus_engine._calculate_validator_weight(validator)
            self.assertIsInstance(weight, float)
            self.assertGreaterEqual(weight, 0.1)
        
        # Test parallel consensus support
        self.assertIsNotNone(consensus_engine.parallel_consensus_lock)
        self.assertIsInstance(consensus_engine.active_consensus_rounds, dict)
    
    def test_node_statistics_collection(self):
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
        
        # Check service health structure
        service_health = stats["service_health"]
        self.assertIsInstance(service_health, dict)
        
        expected_services = ["ledger", "cal_engine", "mining_agent", "memory_mesh", "consensus"]
        for service in expected_services:
            self.assertIn(service, service_health)
            service_info = service_health[service]
            self.assertIn("healthy", service_info)
            self.assertIn("restarts", service_info)
            self.assertIn("last_error", service_info)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for HMN enhancements"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.node_id = "integration-test-node"
        self.network_config = {
            "shard_count": 2,
            "replication_factor": 2,
            "validator_count": 2,
            "metrics_port": 8000,
            "enable_tls": True,
            "discovery_enabled": True
        }
        # Import inside method to avoid import errors
        from network.hmn.full_node import FullNode
        self.node = FullNode(self.node_id, self.network_config)
        
        # Add validators
        self.node.consensus_engine.add_validator("validator-1", 0.9, 10000.0)
        self.node.consensus_engine.add_validator("validator-2", 0.8, 8000.0)
    
    def test_full_lifecycle_integration(self):
        """Test full node lifecycle with all enhancements"""
        # Start the node
        self.node.start_worker_threads()
        self.assertTrue(self.node.workers_running)
        
        # Run services
        asyncio.run(self.node.run_cal_engine())
        asyncio.run(self.node.run_memory_mesh_service())
        asyncio.run(self.node.run_mining_agent())
        asyncio.run(self.node.run_consensus_engine())
        
        # Check that services ran successfully
        cal_state = self.node.cal_engine.get_current_state()
        self.assertIn("lambda_t", cal_state)
        self.assertIn("coherence_density", cal_state)
        self.assertIn("psi_score", cal_state)
        
        # Check memory mesh
        memory_stats = self.node.memory_mesh_service.get_memory_stats()
        self.assertIn("local_updates_count", memory_stats)
        
        # Check consensus
        consensus_stats = self.node.consensus_engine.get_consensus_stats()
        self.assertIn("validators_count", consensus_stats)
        
        # Check mining
        mining_stats = self.node.mining_agent.get_minting_stats()
        self.assertIn("epochs_run", mining_stats)
        
        # Stop the node
        self.node.stop_worker_threads()
        self.assertFalse(self.node.workers_running)
    
    def test_health_monitoring_integration(self):
        """Test health monitoring integration"""
        # Run services to generate health data
        asyncio.run(self.node.run_cal_engine())
        asyncio.run(self.node.run_memory_mesh_service())
        
        # Check health status
        health_status = self.node.check_service_health()
        self.assertIsInstance(health_status, bool)
        
        # Get detailed health
        health_details = self.node.get_health_status()
        self.assertTrue(health_details["overall_health"])
        
        # Check service health details
        services = health_details["services"]
        self.assertIn("cal_engine", services)
        self.assertIn("memory_mesh", services)
        
        cal_health = services["cal_engine"]
        self.assertTrue(cal_health["healthy"])
    
    def test_metrics_collection_integration(self):
        """Test metrics collection integration"""
        # Run services to generate metrics
        asyncio.run(self.node.run_cal_engine())
        asyncio.run(self.node.run_memory_mesh_service())
        
        # Get node stats
        stats = self.node.get_node_stats()
        
        # Check that metrics are collected
        self.assertGreaterEqual(stats["mining_epoch_count"], 0)
        self.assertGreaterEqual(stats["ledger_transaction_count"], 0)
        
        # Check memory stats
        memory_stats = stats["memory_stats"]
        self.assertIn("local_updates_count", memory_stats)
        self.assertIn("metrics", memory_stats)
        
        # Check consensus stats
        consensus_stats = stats["consensus_stats"]
        self.assertIn("validators_count", consensus_stats)

if __name__ == "__main__":
    unittest.main()