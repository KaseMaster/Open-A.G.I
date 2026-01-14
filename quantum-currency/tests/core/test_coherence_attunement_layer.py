#!/usr/bin/env python3
"""
Test suite for Coherence Attunement Layer (CAL) integration
"""

import sys
import os
import unittest
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.coherence_attunement_layer import CoherenceAttunementLayer, OmegaState
from core.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate

class TestCoherenceAttunementLayer(unittest.TestCase):
    """Test suite for Coherence Attunement Layer integration"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cal = CoherenceAttunementLayer()
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
    def test_omega_state_computation(self):
        """Test Ω-state vector computation"""
        omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},  # Within bounds
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Verify Ω-state components
        self.assertIsInstance(omega, OmegaState)
        self.assertGreaterEqual(omega.coherence_score, 0.0)
        self.assertLessEqual(omega.coherence_score, 1.0)
        self.assertGreater(omega.modulator, 0.0)
        self.assertGreater(omega.time_delay, 0.0)
        
        print(f"Ω-state computed: Ψ={omega.coherence_score:.4f}, m_t={omega.modulator:.4f}")
        
    def test_recursive_coherence_with_penalties(self):
        """Test recursive coherence computation with all three penalty components"""
        # Create multiple Ω-states
        omega_states = []
        for i in range(3):
            omega = self.cal.compute_omega_state(
                token_data={"rate": 3.0 + i * 2},  # Within bounds
                sentiment_data={"energy": 0.5 + i * 0.1},
                semantic_data={"shift": 0.2 + i * 0.05},
                attention_data=[0.1 + i*0.1, 0.2 + i*0.1, 0.3 + i*0.1, 0.4 + i*0.1, 0.5 + i*0.1]
            )
            omega_states.append(omega)
        
        # Compute recursive coherence with penalties
        coherence, penalties = self.cal.compute_recursive_coherence(omega_states)
        
        # Verify coherence score and penalties
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
        self.assertGreaterEqual(penalties.cosine_penalty, 0.0)
        self.assertGreaterEqual(penalties.entropy_penalty, 0.0)
        self.assertGreaterEqual(penalties.variance_penalty, 0.0)
        
        print(f"Recursive coherence: {coherence:.4f}")
        print(f"Penalties: cos={penalties.cosine_penalty:.4f}, "
              f"ent={penalties.entropy_penalty:.4f}, "
              f"var={penalties.variance_penalty:.4f}")
        
    def test_dimensional_consistency_validation(self):
        """Test dimensional consistency validation"""
        # Create a valid Ω-state with values within bounds
        omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},  # Within bounds
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Validate dimensional consistency
        is_consistent = self.cal.validate_dimensional_consistency(omega)
        self.assertTrue(is_consistent)
        
        print(f"Dimensional consistency: {is_consistent}")
        
    def test_harmonic_shock_recovery(self):
        """Test harmonic shock recovery simulation"""
        # Create some Ω-states first
        for i in range(5):
            self.cal.compute_omega_state(
                token_data={"rate": 3.0 + i * 1},  # Within bounds
                sentiment_data={"energy": 0.5 + i * 0.1},
                semantic_data={"shift": 0.2 + i * 0.05},
                attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
        
        # Test recovery
        recovered, steps = self.cal.simulate_harmonic_shock_recovery(0.5)
        self.assertIsInstance(recovered, bool)
        self.assertIsInstance(steps, int)
        self.assertGreaterEqual(steps, 0)
        
        print(f"Harmonic shock recovery: {recovered} in {steps} steps")
        
    def test_coherence_health_indicator(self):
        """Test coherence health indicator"""
        # Create some Ω-states
        for i in range(3):
            self.cal.compute_omega_state(
                token_data={"rate": 3.0 + i * 1},  # Within bounds
                sentiment_data={"energy": 0.5 + i * 0.1},
                semantic_data={"shift": 0.2 + i * 0.05},
                attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
        
        # Get health indicator
        health = self.cal.get_coherence_health_indicator()
        self.assertIn(health, ["green", "yellow", "red", "critical", "unknown"])
        
        print(f"Coherence health: {health}")
        
    def test_integration_with_harmonic_validation(self):
        """Test integration with harmonic validation module"""
        # Create Ω-state
        omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},  # Within bounds
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Create snapshot with Ω-state
        snapshot = make_snapshot(
            node_id="test-node",
            times=self.test_times,
            values=self.test_values,
            omega_state={
                "token_rate": omega.token_rate,
                "sentiment_energy": omega.sentiment_energy,
                "semantic_shift": omega.semantic_shift,
                "meta_attention_spectrum": omega.meta_attention_spectrum,
                "coherence_score": omega.coherence_score,
                "modulator": omega.modulator,
                "time_delay": omega.time_delay
            }
        )
        
        # Verify snapshot has Ω-state
        self.assertIsNotNone(snapshot.omega_state)
        self.assertEqual(snapshot.omega_state["token_rate"], omega.token_rate)
        self.assertEqual(snapshot.omega_state["coherence_score"], omega.coherence_score)
        
        print(f"Snapshot with Ω-state: Ψ={snapshot.omega_state['coherence_score']:.4f}")
        
    def test_cosine_similarity_computation(self):
        """Test cosine similarity computation"""
        # Create test vectors
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])  # Same vector
        vec3 = np.array([3.0, 2.0, 1.0])  # Different vector
        
        # Test same vectors (should be 1.0)
        similarity_same = self.cal._cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity_same, 1.0, places=4)
        
        # Test different vectors
        similarity_diff = self.cal._cosine_similarity(vec1, vec3)
        self.assertGreaterEqual(similarity_diff, 0.0)
        self.assertLessEqual(similarity_diff, 1.0)
        
        print(f"Cosine similarity (same): {similarity_same:.4f}")
        print(f"Cosine similarity (diff): {similarity_diff:.4f}")

if __name__ == '__main__':
    unittest.main()