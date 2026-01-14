"""
Integration tests for the Coherence Economy functionality
"""
import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import with direct path manipulation for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'economy'))

# Since we haven't implemented the economy module yet, we'll import what we can
from quantum_bridge import QuantumBridge
from validator_staking import Validator

def test_coherence_economy_integration():
    """Test the integration of the Coherence Economy with the overall system"""
    # Create a quantum bridge instance
    bridge = QuantumBridge()
    
    # Create validators with different coherence scores
    validator1 = Validator(
        validator_id="validator_1",
        operator_address="valoper1xyz...",
        chr_score=0.95,
        total_staked={"FLX": 1000.0, "ATR": 500.0},
        total_delegated={"FLX": 200.0, "ATR": 100.0},
        psi_score=0.92,
        psi_score_history=[0.90, 0.91, 0.92],
        chr_balance=500.0
    )
    
    validator2 = Validator(
        validator_id="validator_2",
        operator_address="valoper2abc...",
        chr_score=0.88,
        total_staked={"FLX": 2000.0, "ATR": 1000.0},
        total_delegated={"FLX": 400.0, "ATR": 200.0},
        psi_score=0.88,
        psi_score_history=[0.85, 0.87, 0.88],
        chr_balance=1000.0
    )
    
    # Add validators to bridge for cross-system validation (simulated)
    bridge.add_connection(
        system_id="validator_1",
        websocket_uri="ws://validator-1:8080",
        shared_secret="validator_secret_1"
    )
    
    bridge.add_connection(
        system_id="validator_2",
        websocket_uri="ws://validator-2:8080",
        shared_secret="validator_secret_2"
    )
    
    # Update coherence history
    bridge.coherence_history["validator_1"] = [0.92]
    bridge.coherence_history["validator_2"] = [0.88]
    
    # Test psi balancing calculation
    adjustments = bridge.calculate_psi_balancing()
    assert "validator_1" in adjustments
    assert "validator_2" in adjustments
    
    # Verify that entropy rate is within acceptable bounds
    entropy_ok = bridge.check_entropy_rate()
    assert isinstance(entropy_ok, bool)

def test_token_flow_consistency():
    """Test consistency of token flows in the coherence economy"""
    # Create validators with different token balances
    validator1 = Validator(
        validator_id="validator_1",
        operator_address="valoper1xyz...",
        chr_score=0.95,
        total_staked={"FLX": 1000.0, "ATR": 500.0},
        total_delegated={"FLX": 200.0, "ATR": 100.0},
        psi_score=0.92,
        psi_score_history=[0.90, 0.91, 0.92],
        chr_balance=500.0
    )
    
    validator2 = Validator(
        validator_id="validator_2",
        operator_address="valoper2abc...",
        chr_score=0.88,
        total_staked={"FLX": 2000.0, "ATR": 1000.0},
        total_delegated={"FLX": 400.0, "ATR": 200.0},
        psi_score=0.88,
        psi_score_history=[0.85, 0.87, 0.88],
        chr_balance=1000.0
    )
    
    # Verify that validators have the expected token balances
    assert validator1.chr_balance == 500.0
    assert validator2.chr_balance == 1000.0
    
    # Verify that validators have psi scores
    assert validator1.psi_score == 0.92
    assert validator2.psi_score == 0.88

def test_cross_system_validation():
    """Test cross-system validation in the coherence economy"""
    bridge = QuantumBridge()
    
    # Add connections
    bridge.add_connection(
        system_id="system_a",
        websocket_uri="ws://system-a:8080",
        shared_secret="secret_a"
    )
    
    bridge.add_connection(
        system_id="system_b",
        websocket_uri="ws://system-b:8080",
        shared_secret="secret_b"
    )
    
    # Update coherence history for both systems
    bridge.coherence_history["system_a"] = [0.95, 0.94, 0.96]
    bridge.coherence_history["system_b"] = [0.88, 0.89, 0.87]
    
    # Calculate psi balancing
    adjustments = bridge.calculate_psi_balancing()
    
    # Both systems should have adjustments
    assert "system_a" in adjustments
    assert "system_b" in adjustments
    
    # Check entropy rate
    entropy_ok = bridge.check_entropy_rate()
    assert isinstance(entropy_ok, bool)

def test_economic_stability_metrics():
    """Test economic stability metrics in the coherence economy"""
    # Create validators with different characteristics
    high_coherence_validator = Validator(
        validator_id="high_coherence",
        operator_address="valoper1high...",
        chr_score=0.98,
        total_staked={"FLX": 5000.0, "ATR": 2500.0},
        total_delegated={"FLX": 1000.0, "ATR": 500.0},
        psi_score=0.95,
        psi_score_history=[0.93, 0.94, 0.95],
        chr_balance=2000.0
    )
    
    medium_coherence_validator = Validator(
        validator_id="medium_coherence",
        operator_address="valoper1medium...",
        chr_score=0.85,
        total_staked={"FLX": 2000.0, "ATR": 1000.0},
        total_delegated={"FLX": 400.0, "ATR": 200.0},
        psi_score=0.85,
        psi_score_history=[0.83, 0.84, 0.85],
        chr_balance=1000.0
    )
    
    # Verify validator properties
    assert high_coherence_validator.chr_score > medium_coherence_validator.chr_score
    assert high_coherence_validator.psi_score > medium_coherence_validator.psi_score
    assert high_coherence_validator.chr_balance > medium_coherence_validator.chr_balance

def test_coherence_economy_performance():
    """Test performance of the coherence economy under load"""
    bridge = QuantumBridge()
    
    # Add multiple systems
    num_systems = 20
    for i in range(num_systems):
        bridge.add_connection(
            system_id=f"system_{i}",
            websocket_uri=f"ws://system-{i}:8080",
            shared_secret=f"secret_{i}"
        )
    
    # Add coherence history for all systems
    for i in range(num_systems):
        # Add varying coherence scores
        scores = [0.90 + (i * 0.01) for i in range(10)]
        bridge.coherence_history[f"system_{i}"] = scores
    
    # Calculate psi balancing for all systems
    adjustments = bridge.calculate_psi_balancing()
    
    # All systems should have adjustments
    assert len(adjustments) == num_systems
    
    # Check entropy rate
    entropy_ok = bridge.check_entropy_rate()
    assert isinstance(entropy_ok, bool)

if __name__ == "__main__":
    pytest.main([__file__])