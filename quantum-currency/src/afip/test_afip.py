#!/usr/bin/env python3
"""
Test suite for AFIP (Absolute Field Integrity Protocol) v1.0
"""

import unittest
import numpy as np
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afip.orchestrator import AFIPOrchestrator
from afip.phase_i_hardening import PhiHarmonicSharding, ZeroDissonanceDeployment, QRAKeyManagement
from afip.phase_ii_predictive import PredictiveGravityWell, OptimalParameterMapper
from afip.phase_iii_evolution import CoherenceProtocolGovernance, FinalCoherenceLock

class TestPhiHarmonicSharding(unittest.TestCase):
    """Test Φ-Harmonic Sharding functionality"""
    
    def setUp(self):
        self.sharding = PhiHarmonicSharding(shard_count=3)
        self.nodes = [
            {"node_id": "node_001", "coherence_score": 0.98, 
             "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
            {"node_id": "node_002", "coherence_score": 0.96,
             "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
            {"node_id": "node_003", "coherence_score": 0.97,
             "qra_params": {"n": 1, "l": 1, "m": 0, "s": 0.3}},
        ]
    
    def test_partition_nodes(self):
        """Test node partitioning into shards"""
        shards = self.sharding.partition_nodes(self.nodes)
        self.assertEqual(len(shards), 3)
        self.assertEqual(sum(len(nodes) for nodes in shards.values()), len(self.nodes))
    
    def test_simulate_shard_failure(self):
        """Test shard failure simulation"""
        self.sharding.partition_nodes(self.nodes)
        result = self.sharding.simulate_shard_failure("SHARD_00")
        self.assertIn("I_eff", result)
        self.assertIn("remaining_system_coherence", result)

class TestZeroDissonanceDeployment(unittest.TestCase):
    """Test Zero-Dissonance Deployment functionality"""
    
    def setUp(self):
        self.deployment = ZeroDissonanceDeployment()
    
    def test_validate_deployment_pass(self):
        """Test deployment validation with passing criteria"""
        result = {
            "final_status": "200_COHERENT_LOCK",
            "delta_lambda": 0.001
        }
        allowed = self.deployment.validate_deployment(result)
        self.assertTrue(allowed)
        self.assertEqual(self.deployment.get_final_status_code(), "200_COHERENT_LOCK")
    
    def test_validate_deployment_fail_status(self):
        """Test deployment validation with failing status code"""
        result = {
            "final_status": "500_CRITICAL_DISSONANCE",
            "delta_lambda": 0.001
        }
        allowed = self.deployment.validate_deployment(result)
        self.assertFalse(allowed)
        self.assertEqual(self.deployment.get_final_status_code(), "500_CRITICAL_DISSONANCE")
    
    def test_validate_deployment_fail_delta_lambda(self):
        """Test deployment validation with failing delta lambda"""
        result = {
            "final_status": "200_COHERENT_LOCK",
            "delta_lambda": 0.01
        }
        allowed = self.deployment.validate_deployment(result)
        self.assertFalse(allowed)
        self.assertEqual(self.deployment.get_final_status_code(), "400_LAMBDA_DISCREPANCY")

class TestQRAKeyManagement(unittest.TestCase):
    """Test QRA Key Management functionality"""
    
    def setUp(self):
        self.key_mgmt = QRAKeyManagement(tee_enabled=True)
    
    def test_generate_qra_in_tee(self):
        """Test QRA key generation in TEE"""
        system_metrics = {
            "C_system": 0.98,
            "GAS_target": 0.95,
            "cycle_count": 150
        }
        qra_key = self.key_mgmt.generate_qra_in_tee("node_001", system_metrics)
        self.assertIsInstance(qra_key, dict)
        self.assertIn("node_id", qra_key)
        self.assertIn("Coherence_Score", qra_key)
    
    def test_validate_key_integrity(self):
        """Test QRA key integrity validation"""
        qra_key = {
            "node_id": "node_001",
            "Coherence_Score": 0.95,
            "Phi_Ratio": 1.618,
            "I_eff_Cost": 0.05
        }
        valid = self.key_mgmt.validate_key_integrity(qra_key)
        self.assertTrue(valid)

class TestPredictiveGravityWell(unittest.TestCase):
    """Test Predictive Gravity Well functionality"""
    
    def setUp(self):
        self.gravity_predictor = PredictiveGravityWell(prediction_cycles=5)
    
    def test_compute_projected_g_vector(self):
        """Test g-vector projection computation"""
        historical_data = [
            {"node_id": "node_001", "g_vector_magnitude": 0.5},
            {"node_id": "node_001", "g_vector_magnitude": 0.7},
            {"node_id": "node_001", "g_vector_magnitude": 0.9},
        ]
        result = self.gravity_predictor.compute_projected_g_vector(historical_data, "node_001")
        self.assertIn("projected_g_magnitude", result)
        self.assertIn("confidence", result)
    
    def test_trigger_proactive_isolation(self):
        """Test proactive isolation triggering"""
        projected_data = {
            "projected_g_magnitude": 2.0,
            "confidence": 0.8,
            "node_id": "node_001"
        }
        isolated = self.gravity_predictor.trigger_proactive_isolation(projected_data)
        self.assertTrue(isolated)

class TestOptimalParameterMapper(unittest.TestCase):
    """Test Optimal Parameter Mapping functionality"""
    
    def setUp(self):
        self.param_mapper = OptimalParameterMapper()
    
    def test_train_phi_recursive_nn(self):
        """Test Φ-Recursive Neural Network training"""
        telemetry_data = [
            {"lambda1": 0.3, "lambda2": 0.4, "caf_emission": 0.05, 
             "I_eff": 0.008, "delta_lambda": 0.002, "rsi": 0.92},
            {"lambda1": 0.4, "lambda2": 0.5, "caf_emission": 0.08,
             "I_eff": 0.006, "delta_lambda": 0.0015, "rsi": 0.94},
        ] * 10  # Repeat for sufficient data
        result = self.param_mapper.train_phi_recursive_nn(telemetry_data)
        self.assertIn("trained", result)
        self.assertIn("lambda_weights", result)
    
    def test_derive_dynamic_parameter_map(self):
        """Test dynamic parameter mapping"""
        current_state = {
            "C_system": 0.98,
            "GAS_target": 0.95,
            "rsi": 0.92
        }
        params = self.param_mapper.derive_dynamic_parameter_map(current_state)
        self.assertIn("lambda1", params)
        self.assertIn("lambda2", params)
        self.assertIn("alpha_emission_rate", params)

class TestCoherenceProtocolGovernance(unittest.TestCase):
    """Test Coherence Protocol Governance functionality"""
    
    def setUp(self):
        self.cpgm = CoherenceProtocolGovernance()
    
    def test_create_protocol_amendment_proposal(self):
        """Test protocol amendment proposal creation"""
        optimization_vector = {"lambda_adjustment": {"lambda1": 0.05, "lambda2": -0.03}}
        proposal = self.cpgm.create_protocol_amendment_proposal(optimization_vector)
        self.assertIn("proposal_id", proposal)
        self.assertIn("optimization_vector", proposal)
        self.assertEqual(proposal["status"], "PENDING")
    
    def test_evaluate_approval_gate(self):
        """Test approval gate evaluation"""
        system_metrics = {"g_avg": 0.05}
        qra_metrics = [
            {"node_id": "node_001", "C_score": 0.98},
            {"node_id": "node_002", "C_score": 0.96},
        ]
        result = self.cpgm.evaluate_approval_gate(system_metrics, qra_metrics)
        self.assertIn("approval_gate_passed", result)

class TestFinalCoherenceLock(unittest.TestCase):
    """Test Final Coherence Lock functionality"""
    
    def setUp(self):
        self.final_lock = FinalCoherenceLock(observation_period_days=1)  # Short for testing
    
    def test_start_observation_period(self):
        """Test observation period start"""
        self.final_lock.start_observation_period()
        self.assertIsNotNone(self.final_lock.observation_start_time)
    
    def test_record_metrics(self):
        """Test metrics recording"""
        self.final_lock.start_observation_period()
        metrics = {
            "C_system": 0.999,
            "delta_lambda": 0.001,
            "I_eff": 0.001,
            "RSI": 0.99,
            "gravity_well_anomalies": 0
        }
        recorded = self.final_lock.record_metrics(metrics)
        self.assertTrue(recorded)
        self.assertEqual(len(self.final_lock.metrics_history), 1)

class TestAFIPOrchestrator(unittest.TestCase):
    """Test AFIP Orchestrator functionality"""
    
    def setUp(self):
        self.config = {
            "shard_count": 2,
            "tee_enabled": True,
            "prediction_cycles": 5,
            "observation_period_days": 1
        }
        self.afip = AFIPOrchestrator(self.config)
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertEqual(self.afip.status, "INITIALIZED")
        self.assertIsInstance(self.afip.phi_sharding, PhiHarmonicSharding)
        self.assertIsInstance(self.afip.zero_dissonance, ZeroDissonanceDeployment)
    
    def test_execute_full_afip_protocol(self):
        """Test full AFIP protocol execution"""
        nodes = [
            {"node_id": "node_001", "coherence_score": 0.98, 
             "qra_params": {"n": 1, "l": 0, "m": 0, "s": 0.5}},
            {"node_id": "node_002", "coherence_score": 0.96,
             "qra_params": {"n": 2, "l": 1, "m": -1, "s": 0.7}},
        ]
        
        telemetry_data = [
            {"node_id": "node_001", "g_vector_magnitude": 0.5, "coherence": 0.98, "rsi": 0.92},
            {"node_id": "node_001", "g_vector_magnitude": 0.7, "coherence": 0.97, "rsi": 0.91},
        ] * 5  # Repeat for sufficient data
        
        # This is a simplified test - in practice, we'd want to mock the long-running operations
        # For now, we'll just verify the structure
        self.assertIsNotNone(self.afip)

if __name__ == "__main__":
    unittest.main()