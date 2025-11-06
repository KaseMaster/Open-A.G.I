"""
Unit tests for the Multi-Node Simulator
"""

import unittest
import sys
import os
import tempfile
import json

# Add the quantum-currency directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.simulation.multi_node_simulator import MultiNodeSimulator, NodeState


class TestMultiNodeSimulator(unittest.TestCase):
    """Test suite for MultiNodeSimulator"""

    def setUp(self):
        """Set up test fixtures"""
        self.simulator = MultiNodeSimulator(num_nodes=5, network_id="test-network-001")

    def test_initialization(self):
        """Test simulator initialization"""
        self.assertEqual(self.simulator.num_nodes, 5)
        self.assertEqual(self.simulator.network_id, "test-network-001")
        self.assertEqual(len(self.simulator.nodes), 5)
        self.assertEqual(len(self.simulator.metrics_history), 0)
        
        # Check that all nodes are properly initialized
        for i, node in enumerate(self.simulator.nodes):
            self.assertIsInstance(node, NodeState)
            self.assertEqual(node.node_id, f"sim-node-{i:03d}")
            self.assertIsNotNone(node.cal)
            self.assertIsNone(node.snapshot)
            self.assertIsNone(node.omega_state)
            self.assertEqual(len(node.coherence_history), 0)
            self.assertTrue(node.is_active)
            
            # Check token balances
            expected_balances = {"FLX": 1000.0, "CHR": 500.0, "PSY": 200.0, "ATR": 300.0, "RES": 50.0}
            self.assertEqual(node.token_balances, expected_balances)

    def test_single_round_simulation(self):
        """Test running a single simulation round"""
        result = self.simulator.run_simulation_round(0)
        
        # Check result structure
        self.assertIn("round", result)
        self.assertIn("network_coherence", result)
        self.assertIn("aggregated_cs", result)
        self.assertIn("is_valid", result)
        self.assertIn("node_results", result)
        self.assertIn("metrics", result)
        self.assertIn("duration", result)
        
        # Check values
        self.assertEqual(result["round"], 0)
        self.assertIsInstance(result["network_coherence"], float)
        self.assertIsInstance(result["aggregated_cs"], float)
        self.assertIsInstance(result["is_valid"], bool)
        self.assertIsInstance(result["duration"], float)
        self.assertGreaterEqual(result["duration"], 0)
        
        # Check node results
        self.assertEqual(len(result["node_results"]), 5)
        for node_id, node_result in result["node_results"].items():
            self.assertIn("coherence", node_result)
            self.assertIn("details", node_result)
            self.assertIsInstance(node_result["coherence"], float)
            self.assertIsInstance(node_result["details"], dict)

    def test_multiple_rounds_simulation(self):
        """Test running multiple simulation rounds"""
        results = []
        for i in range(3):
            result = self.simulator.run_simulation_round(i)
            results.append(result)
        
        # Check that we have results for all rounds
        self.assertEqual(len(results), 3)
        
        # Check that metrics history is populated
        self.assertEqual(len(self.simulator.metrics_history), 3)
        
        # Check that nodes have coherence history
        for node in self.simulator.nodes:
            self.assertEqual(len(node.coherence_history), 3)
            for coherence in node.coherence_history:
                self.assertIsInstance(coherence, float)
                self.assertGreaterEqual(coherence, 0.0)
                self.assertLessEqual(coherence, 1.0)

    def test_network_shock_simulation(self):
        """Test network shock simulation"""
        # Run a round first to establish baseline
        self.simulator.run_simulation_round(0)
        
        # Get baseline coherences
        baseline_coherences = [
            node.omega_state.coherence_score 
            for node in self.simulator.nodes 
            if node.omega_state is not None
        ]
        
        # Apply shock
        self.simulator.simulate_network_shock(shock_magnitude=0.3, affected_nodes=0.6)
        
        # Check that some nodes were affected
        affected_coherences = [
            node.omega_state.coherence_score 
            for node in self.simulator.nodes 
            if node.omega_state is not None
        ]
        
        # At least some coherences should be lower
        self.assertLess(sum(affected_coherences), sum(baseline_coherences))

    def test_node_failure_simulation(self):
        """Test node failure simulation"""
        # Check initial state
        initial_active = sum(1 for node in self.simulator.nodes if node.is_active)
        self.assertEqual(initial_active, 5)
        
        # Simulate failures
        self.simulator.simulate_node_failure(failure_rate=0.4)
        
        # Check that some nodes failed
        final_active = sum(1 for node in self.simulator.nodes if node.is_active)
        self.assertLess(final_active, initial_active)
        
        # Should be 3 active nodes (60% of 5 = 3)
        self.assertEqual(final_active, 3)

    def test_performance_report_generation(self):
        """Test performance report generation"""
        # Run a few rounds first
        for i in range(3):
            self.simulator.run_simulation_round(i)
        
        # Generate report
        report = self.simulator.generate_performance_report()
        
        # Check report structure
        self.assertIn("simulation_summary", report)
        self.assertIn("coherence_metrics", report)
        self.assertIn("validation_metrics", report)
        self.assertIn("network_stability", report)
        self.assertIn("token_distribution", report)
        
        # Check key metrics
        self.assertEqual(report["simulation_summary"]["total_rounds"], 3)
        self.assertEqual(report["simulation_summary"]["total_nodes"], 5)
        self.assertGreaterEqual(report["coherence_metrics"]["avg_network_coherence"], 0.0)
        self.assertLessEqual(report["coherence_metrics"]["avg_network_coherence"], 1.0)

    def test_data_saving(self):
        """Test saving simulation data"""
        # Run a round
        self.simulator.run_simulation_round(0)
        
        # Save data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save data
            results = [self.simulator.run_simulation_round(0)]
            self.simulator.save_simulation_data(results, temp_file)
            
            # Check that file was created and contains data
            self.assertTrue(os.path.exists(temp_file))
            
            # Read and verify data
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            self.assertIn("simulation_config", data)
            self.assertIn("network_id", data)
            self.assertIn("total_nodes", data)
            self.assertIn("results", data)
            self.assertIn("metrics_history", data)
            self.assertIn("final_report", data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()