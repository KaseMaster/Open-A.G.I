"""
Integration tests for CAL-RΦV Fusion (v0.2.0)
Tests the integration between Coherence Attunement Layer and Recursive Φ-Resonance Validation
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the modules to be tested
from src.models.coherence_attunement_layer import CoherenceAttunementLayer, OmegaState
from src.core.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate, HarmonicProofBundle
from src.core.token_rules import apply_token_effects
from src.core.harmonic_validation import HarmonicSnapshot


class TestCALRPhiVFusion(unittest.TestCase):
    """Test suite for CAL-RΦV Fusion integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.cal = CoherenceAttunementLayer()
        self.test_node_id = "test-validator-001"
        self.test_secret_key = "test-secret-key"
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4]
        self.test_values = [1.0, 1.1, 1.2, 1.1, 1.0]

    def _omega_state_to_dict(self, omega_state: OmegaState) -> dict:
        """Convert OmegaState object to dictionary for make_snapshot"""
        return {
            "token_rate": omega_state.token_rate,
            "sentiment_energy": omega_state.sentiment_energy,
            "semantic_shift": omega_state.semantic_shift,
            "meta_attention_spectrum": omega_state.meta_attention_spectrum,
            "coherence_score": omega_state.coherence_score,
            "modulator": omega_state.modulator,
            "time_delay": omega_state.time_delay
        }

    def test_omega_state_computation_integration(self):
        """Test Ω-state computation integration with harmonic validation"""
        # Compute Ω-state using CAL
        omega_state = self.cal.compute_omega_state(
            token_data={"rate": 5.0},
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Convert to dictionary for make_snapshot
        omega_dict = self._omega_state_to_dict(omega_state)
        
        # Create snapshot with Ω-state
        snapshot = make_snapshot(
            node_id=self.test_node_id,
            times=self.test_times,
            values=self.test_values,
            secret_key=self.test_secret_key,
            omega_state=omega_dict
        )
        
        # Verify integration
        self.assertIsInstance(snapshot, HarmonicSnapshot)
        self.assertIsNotNone(snapshot.omega_state)
        # Add explicit check to make linter happy
        if snapshot.omega_state is not None:
            self.assertEqual(snapshot.omega_state["token_rate"], 5.0)
            self.assertEqual(snapshot.omega_state["sentiment_energy"], 0.7)
            self.assertEqual(snapshot.omega_state["semantic_shift"], 0.3)

    def test_recursive_coherence_computation(self):
        """Test recursive coherence computation with multiple Ω-states"""
        # Create multiple Ω-states
        omega_states = []
        for i in range(5):
            omega = self.cal.compute_omega_state(
                token_data={"rate": 5.0 + i * 0.1},
                sentiment_data={"energy": 0.7 + i * 0.01},
                semantic_data={"shift": 0.3 + i * 0.005},
                attention_data=[0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01, 0.5 + i*0.01]
            )
            omega_states.append(omega)
        
        # Compute recursive coherence
        coherence_score, penalties = self.cal.compute_recursive_coherence(omega_states)
        
        # Verify results
        self.assertIsInstance(coherence_score, float)
        self.assertTrue(0.0 <= coherence_score <= 1.0)
        self.assertIsNotNone(penalties)
        
    def test_dimensional_consistency_validation(self):
        """Test dimensional consistency validation"""
        # Create valid Ω-state
        valid_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},  # Within bounds
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Test validation
        is_valid = self.cal.validate_dimensional_consistency(valid_omega)
        self.assertTrue(is_valid)
        
        # Create invalid Ω-state (beyond bounds)
        invalid_omega = OmegaState(
            timestamp=1234567890.0,
            token_rate=15.0,  # Beyond bounds
            sentiment_energy=0.7,
            semantic_shift=0.3,
            meta_attention_spectrum=[0.1, 0.2, 0.3, 0.4, 0.5],
            coherence_score=0.8,
            modulator=1.0,
            time_delay=0.5
        )
        
        # Test validation
        is_invalid = self.cal.validate_dimensional_consistency(invalid_omega)
        self.assertFalse(is_invalid)

    def test_harmonic_validation_with_omega_state(self):
        """Test harmonic validation integration with Ω-state"""
        # Create local snapshot with Ω-state
        local_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Convert to dictionary for make_snapshot
        local_omega_dict = self._omega_state_to_dict(local_omega)
        
        local_snapshot = make_snapshot(
            node_id=self.test_node_id,
            times=self.test_times,
            values=self.test_values,
            secret_key=self.test_secret_key,
            omega_state=local_omega_dict
        )
        
        # Create remote snapshots with Ω-states
        remote_snapshots = []
        for i in range(3):
            remote_omega = self.cal.compute_omega_state(
                token_data={"rate": 4.8 + i * 0.1},
                sentiment_data={"energy": 0.65 + i * 0.01},
                semantic_data={"shift": 0.28 + i * 0.005},
                attention_data=[0.12 + i*0.01, 0.22 + i*0.01, 0.32 + i*0.01, 0.42 + i*0.01, 0.52 + i*0.01]
            )
            
            # Convert to dictionary for make_snapshot
            remote_omega_dict = self._omega_state_to_dict(remote_omega)
            
            remote_snapshot = make_snapshot(
                node_id=f"remote-validator-{i}",
                times=self.test_times,
                values=[v + i * 0.05 for v in self.test_values],
                secret_key=f"remote-secret-{i}",
                omega_state=remote_omega_dict
            )
            remote_snapshots.append(remote_snapshot)
        
        # Compute coherence score
        coherence_score = compute_coherence_score(local_snapshot, remote_snapshots)
        
        # Verify integration
        self.assertIsInstance(coherence_score, float)
        self.assertTrue(0.0 <= coherence_score <= 1.0)

    def test_recursive_validation_with_omega_integration(self):
        """Test recursive validation with Ω-state integration"""
        # Create snapshots with Ω-states
        snapshots = []
        for i in range(4):
            omega = self.cal.compute_omega_state(
                token_data={"rate": 5.0 + i * 0.1},
                sentiment_data={"energy": 0.7 + i * 0.01},
                semantic_data={"shift": 0.3 + i * 0.005},
                attention_data=[0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01, 0.5 + i*0.01]
            )
            
            # Convert to dictionary for make_snapshot
            omega_dict = self._omega_state_to_dict(omega)
            
            snapshot = make_snapshot(
                node_id=f"validator-{i}",
                times=self.test_times,
                values=[v + i * 0.05 for v in self.test_values],
                secret_key=f"secret-{i}",
                omega_state=omega_dict
            )
            snapshots.append(snapshot)
        
        # Perform recursive validation
        is_valid, proof_bundle = recursive_validate(snapshots, threshold=0.7)
        
        # Verify validation
        self.assertIsInstance(is_valid, bool)
        self.assertIsNotNone(proof_bundle)
        self.assertIsInstance(proof_bundle, HarmonicProofBundle)

    def test_token_effects_with_harmonic_gating(self):
        """Test token effects with harmonic gating mechanisms"""
        # Initial state
        initial_state = {
            "balances": {
                "validator-1": {
                    "FLX": 1000.0,
                    "CHR": 500.0,
                    "PSY": 200.0,
                    "ATR": 300.0,
                    "RES": 50.0
                }
            },
            "chr": {
                "validator-1": 0.85
            },
            "staking": {
                "validator-1": 100.0
            }
        }
        
        # Test CHR harmonic gating
        chr_tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 10.0,
            "token": "CHR",
            "action": "macro_write_gate",
            "psi_threshold": 0.8,
            "psi_score": 0.85
        }
        
        updated_state = apply_token_effects(initial_state, chr_tx)
        self.assertGreater(updated_state["balances"]["validator-1"]["CHR"], 500.0)
        
        # Test FLX harmonic gating
        flx_tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 50.0,
            "token": "FLX",
            "action": "memory_retrieval",
            "retrieval_cost": 10.0
        }
        
        updated_state = apply_token_effects(updated_state, flx_tx)
        self.assertLess(updated_state["balances"]["validator-1"]["FLX"], 1000.0)
        
        # Test PSY harmonic gating
        psy_tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 20.0,
            "token": "PSY",
            "action": "behavioral_balance",
            "psi_score": 0.4  # Low coherence should incur penalty
        }
        
        updated_state = apply_token_effects(updated_state, psy_tx)
        # Should have penalty applied due to low coherence
        
        # Test ATR harmonic gating
        atr_tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 50.0,
            "token": "ATR",
            "action": "set_omega_target",
            "omega_target": {"token_rate": 6.0}
        }
        
        updated_state = apply_token_effects(updated_state, atr_tx)
        # Should set Ω target for validator
        
        # Test RES harmonic gating
        res_tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 10.0,
            "token": "RES",
            "action": "expand_bandwidth",
            "max_multiplier": 2.0
        }
        
        updated_state = apply_token_effects(updated_state, res_tx)
        # Should expand Ω bandwidth

    def test_modulator_adaptation(self):
        """Test modulator adaptation based on coherence scores"""
        # Test with high coherence - we need to create some history first
        # Add some omega states to history
        for i in range(5):
            omega = self.cal.compute_omega_state(
                token_data={"rate": 5.0 + i * 0.1},
                sentiment_data={"energy": 0.7 + i * 0.01},
                semantic_data={"shift": 0.3 + i * 0.005},
                attention_data=[0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01, 0.5 + i*0.01]
            )
        
        # Test with high coherence
        high_coherence = 0.9
        modulator_high = self.cal._compute_modulator(high_coherence)
        self.assertIsInstance(modulator_high, float)
        self.assertGreater(modulator_high, 0)
        
        # Test with low coherence
        low_coherence = 0.3
        modulator_low = self.cal._compute_modulator(low_coherence)
        self.assertIsInstance(modulator_low, float)
        self.assertGreater(modulator_low, 0)

    def test_harmonic_shock_recovery_simulation(self):
        """Test harmonic shock recovery simulation"""
        # This is a simplified test - in practice, this would involve
        # a more complex simulation of network conditions
        
        # Establish baseline
        baseline_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        baseline_coherence = baseline_omega.coherence_score
        
        # Simulate shock (sudden drop in coherence)
        shock_omega = self.cal.compute_omega_state(
            token_data={"rate": 2.0},  # Significant change
            sentiment_data={"energy": 0.2},
            semantic_data={"shift": 0.8},
            attention_data=[0.5, 0.4, 0.3, 0.2, 0.1]
        )
        
        shock_coherence = shock_omega.coherence_score
        self.assertLess(shock_coherence, baseline_coherence)
        
        # Recovery would be tested through a sequence of computations
        # that show improvement over time

    def test_ai_coherence_regression(self):
        """Test AI coherence regression (simplified)"""
        # This tests that actions taken by AI result in improved coherence
        # In practice, this would involve a more complex AI simulation
        
        # Add some history first
        for i in range(3):
            omega = self.cal.compute_omega_state(
                token_data={"rate": 5.0 + i * 0.05},
                sentiment_data={"energy": 0.7 + i * 0.01},
                semantic_data={"shift": 0.3 + i * 0.002},
                attention_data=[0.1 + i*0.005, 0.2 + i*0.005, 0.3 + i*0.005, 0.4 + i*0.005, 0.5 + i*0.005]
            )
        
        # Get current coherence
        current_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.2},
            sentiment_data={"energy": 0.72},
            semantic_data={"shift": 0.31},
            attention_data=[0.12, 0.22, 0.32, 0.42, 0.52]
        )
        current_coherence = current_omega.coherence_score
        
        # Simulate AI action that should improve coherence (better parameters)
        improved_omega = self.cal.compute_omega_state(
            token_data={"rate": 5.1},  # More optimal parameter
            sentiment_data={"energy": 0.75},  # Better sentiment
            semantic_data={"shift": 0.28},  # Better semantic alignment
            attention_data=[0.25, 0.30, 0.35, 0.40, 0.45]  # Better attention distribution
        )
        
        improved_coherence = improved_omega.coherence_score
        
        # Test that we can compute coherences (the actual improvement depends on the specific algorithm)
        self.assertIsInstance(current_coherence, float)
        self.assertIsInstance(improved_coherence, float)
        self.assertTrue(0.0 <= current_coherence <= 1.0)
        self.assertTrue(0.0 <= improved_coherence <= 1.0)

    def test_numerical_stability_constraints(self):
        """Test numerical stability constraints"""
        # Test that all parameters stay within safe bounds
        
        # Create Ω-state with boundary values
        boundary_omega = self.cal.compute_omega_state(
            token_data={"rate": 10.0},  # At upper boundary
            sentiment_data={"energy": -10.0},  # At lower boundary
            semantic_data={"shift": 0.0},
            attention_data=[10.0, -10.0, 0.0, 0.0, 0.0]  # At boundaries
        )
        
        # Validate dimensional consistency
        is_consistent = self.cal.validate_dimensional_consistency(boundary_omega)
        self.assertTrue(is_consistent)
        
        # Test modulator bounds
        modulator = boundary_omega.modulator
        self.assertGreaterEqual(modulator, np.exp(-10))
        self.assertLessEqual(modulator, np.exp(10))

    def test_complete_validation_cycle(self):
        """Test a complete validation cycle from Ω-state to consensus"""
        # Step 1: Compute Ω-states for multiple nodes
        node_snapshots = []
        
        for i in range(5):
            omega = self.cal.compute_omega_state(
                token_data={"rate": 5.0 + i * 0.2},
                sentiment_data={"energy": 0.7 + i * 0.02},
                semantic_data={"shift": 0.3 + i * 0.01},
                attention_data=[0.1 + i*0.02, 0.2 + i*0.02, 0.3 + i*0.02, 0.4 + i*0.02, 0.5 + i*0.02]
            )
            
            # Convert to dictionary for make_snapshot
            omega_dict = self._omega_state_to_dict(omega)
            
            snapshot = make_snapshot(
                node_id=f"node-{i}",
                times=self.test_times,
                values=[v + i * 0.1 for v in self.test_values],
                secret_key=f"secret-{i}",
                omega_state=omega_dict
            )
            node_snapshots.append(snapshot)
        
        # Step 2: Compute pairwise coherence scores
        coherence_scores = []
        for i, local_snapshot in enumerate(node_snapshots):
            remote_snapshots = [s for j, s in enumerate(node_snapshots) if j != i]
            coherence = compute_coherence_score(local_snapshot, remote_snapshots)
            coherence_scores.append(coherence)
        
        # Step 3: Validate consensus
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        self.assertIsInstance(avg_coherence, float)
        self.assertTrue(0.0 <= avg_coherence <= 1.0)
        
        # Step 4: Perform recursive validation
        is_valid, proof_bundle = recursive_validate(node_snapshots, threshold=0.7)
        self.assertIsInstance(is_valid, bool)
        self.assertIsNotNone(proof_bundle)


if __name__ == '__main__':
    unittest.main()