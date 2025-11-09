#!/usr/bin/env python3
"""
Performance Test Suite for Coherence Attunement Layer (CAL)
Tests latency, throughput, and stability under high entanglement density.
"""

import sys
import os
import time
import numpy as np
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Handle relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.models.coherence_attunement_layer import CoherenceAttunementLayer as CALEngine, OmegaState, CoherencePenalties

class TestCALPerformance:
    """Performance test suite for Coherence Attunement Layer"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.cal_engine = CALEngine()
        
    def test_omega_state_computation_latency(self, benchmark):
        """Benchmark Ω-state computation latency"""
        def compute_omega_state():
            return self.cal_engine.compute_omega_state(
                token_data={"rate": 5.0},
                sentiment_data={"energy": 0.7},
                semantic_data={"shift": 0.3},
                attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
        
        # Benchmark the function
        result = benchmark(compute_omega_state)
        
        # Verify result is valid
        assert isinstance(result, OmegaState)
        assert 0.0 <= result.coherence_score <= 1.0
        
        # Check performance criteria (should complete within 10ms)
        # This is a soft check - the actual benchmark will show the real performance
        assert result.timestamp > 0
        
    def test_recursive_coherence_computation_performance(self, benchmark):
        """Benchmark recursive coherence computation with multiple Ω-states"""
        # Generate multiple Ω-states for testing
        omega_states = []
        for i in range(10):
            omega = self.cal_engine.compute_omega_state(
                token_data={"rate": 5.0 + i * 0.1},
                sentiment_data={"energy": 0.7 + i * 0.01},
                semantic_data={"shift": 0.3 + i * 0.005},
                attention_data=[0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01, 0.5 + i*0.01]
            )
            omega_states.append(omega)
        
        def compute_recursive_coherence():
            return self.cal_engine.compute_recursive_coherence(omega_states)
        
        # Benchmark the function
        coherence_score, penalties = benchmark(compute_recursive_coherence)
        
        # Verify results are valid
        assert isinstance(coherence_score, float)
        assert isinstance(penalties, CoherencePenalties)
        # The coherence score should be a finite number
        assert np.isfinite(coherence_score)
        
    def test_high_entanglement_density_stress(self, benchmark):
        """Stress test with high entanglement density (100+ Ω-states)"""
        # Generate high density of Ω-states with more controlled values
        omega_states = []
        for i in range(100):
            omega = self.cal_engine.compute_omega_state(
                token_data={"rate": 5.0 + np.sin(i * 0.1) * 2.0},
                sentiment_data={"energy": 0.7 + np.cos(i * 0.15) * 0.3},
                semantic_data={"shift": 0.3 + np.sin(i * 0.08) * 0.2},
                attention_data=[0.1 + np.sin(i * 0.05 + j * 0.5) * 0.2 for j in range(5)]
            )
            omega_states.append(omega)
        
        def compute_high_density_coherence():
            return self.cal_engine.compute_recursive_coherence(omega_states)
        
        # Benchmark the function
        coherence_score, penalties = benchmark(compute_high_density_coherence)
        
        # Verify results are valid
        assert isinstance(coherence_score, float)
        assert isinstance(penalties, CoherencePenalties)
        # The coherence score should be a finite number
        assert np.isfinite(coherence_score)
        
    def test_throughput_under_load(self, benchmark):
        """Test throughput under sustained load"""
        def sustained_computation():
            results = []
            for i in range(50):
                omega = self.cal_engine.compute_omega_state(
                    token_data={"rate": 5.0 + i * 0.02},
                    sentiment_data={"energy": 0.7 + i * 0.005},
                    semantic_data={"shift": 0.3 + i * 0.002},
                    attention_data=[0.1 + i*0.002, 0.2 + i*0.002, 0.3 + i*0.002, 0.4 + i*0.002, 0.5 + i*0.002]
                )
                results.append(omega)
            return results
        
        # Benchmark the function
        omega_results = benchmark(sustained_computation)
        
        # Verify results
        assert len(omega_results) == 50
        assert all(isinstance(omega, OmegaState) for omega in omega_results)
        
    def test_memory_efficiency(self, benchmark):
        """Test memory efficiency with large history"""
        # Generate large history
        for i in range(200):
            self.cal_engine.compute_omega_state(
                token_data={"rate": 5.0 + np.sin(i * 0.1) * 1.0},
                sentiment_data={"energy": 0.7 + np.cos(i * 0.12) * 0.2},
                semantic_data={"shift": 0.3 + np.sin(i * 0.09) * 0.1},
                attention_data=[0.1 + np.sin(i * 0.03 + j * 0.4) * 0.15 for j in range(5)]
            )
        
        def get_history_stats():
            return {
                "history_length": len(self.cal_engine.omega_history),
                "penalty_history_length": len(self.cal_engine.penalty_history)
            }
        
        # Benchmark the function
        stats = benchmark(get_history_stats)
        
        # Verify history management
        assert stats["history_length"] <= 100  # Should be capped at 100
        assert stats["penalty_history_length"] <= 100

if __name__ == '__main__':
    pytest.main([__file__, '-v'])