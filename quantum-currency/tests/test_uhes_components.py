#!/usr/bin/env python3
"""
Test script for UHES components
Verifies that all components of the Unified Harmonic Economic System are working correctly
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.cal_engine import CALEngine
from src.models.quantum_memory import UnifiedFieldMemory, QuantumPacket
from src.models.coherent_db import CoherentDatabase
from src.models.entropy_monitor import EntropyMonitor
from src.models.ai_governance import AIGovernance

def test_cal_engine():
    """Test the CAL Engine components"""
    print("Testing CAL Engine...")
    
    # Create CAL engine
    cal_engine = CALEngine()
    
    # Test λ(L) computation
    lambda_value = cal_engine.lambda_L("LΦ", 0.85)
    assert isinstance(lambda_value, float), "λ(L) should return a float"
    assert lambda_value > 0, "λ(L) should be positive"
    print(f"  ✓ λ(L) computation: {lambda_value:.4f}")
    
    # Test modulator computation
    I_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    modulator_value = cal_engine.modulator(I_vector, lambda_value)
    assert isinstance(modulator_value, float), "Modulator should return a float"
    assert modulator_value > 0, "Modulator should be positive"
    print(f"  ✓ Modulator computation: {modulator_value:.4f}")
    
    # Test dimensional stability
    is_stable = cal_engine.validate_dimensional_stability(modulator_value)
    assert isinstance(is_stable, bool), "Dimensional stability check should return boolean"
    print(f"  ✓ Dimensional stability: {is_stable}")
    
    # Test Ω state update
    features = [1.0, 2.0, 3.0, 4.0, 5.0]
    I_vector = [0.1, 0.15, 0.2, 0.25, 0.3]
    updated_omega, m_t = cal_engine.update_omega(features, I_vector, "LΦ")
    assert isinstance(updated_omega, np.ndarray), "Updated Ω should be numpy array"
    assert isinstance(m_t, float), "Modulator should be float"
    print(f"  ✓ Ω state update: norm={np.linalg.norm(updated_omega):.4f}")
    
    print("✓ CAL Engine tests passed\n")

def test_quantum_memory():
    """Test the Quantum Memory components"""
    print("Testing Quantum Memory...")
    
    # Create UFM
    ufm = UnifiedFieldMemory()
    
    # Test packet creation
    packet = ufm.create_quantum_packet(
        omega_vector=[1.0, 0.5, 0.2],
        psi_score=0.85,
        scale_level="LΦ",
        data_payload="Test data"
    )
    assert isinstance(packet, QuantumPacket), "Should create QuantumPacket"
    assert packet.id is not None, "Packet should have an ID"
    assert packet.compression_ratio > 1.0, "Compression ratio should be > 1.0"
    print(f"  ✓ Packet creation: {packet.id}")
    
    # Test packet storage
    success = ufm.store_packet(packet)
    assert success, "Packet storage should succeed"
    print(f"  ✓ Packet storage: {success}")
    
    # Test packet retrieval
    retrieved = ufm.retrieve_packet(packet.id)
    assert retrieved is not None, "Should retrieve packet"
    assert retrieved.id == packet.id, "Retrieved packet should have same ID"
    print(f"  ✓ Packet retrieval: {retrieved.id}")
    
    # Test layer statistics
    stats = ufm.get_layer_statistics()
    assert isinstance(stats, dict), "Statistics should be dictionary"
    assert "LΦ" in stats, "Should have LΦ layer statistics"
    print(f"  ✓ Layer statistics: {len(stats)} layers")
    
    print("✓ Quantum Memory tests passed\n")

def test_coherent_database():
    """Test the Coherent Database components"""
    print("Testing Coherent Database...")
    
    # Create components
    ufm = UnifiedFieldMemory()
    cdb = CoherentDatabase(ufm)
    
    # Test packet addition
    packet = ufm.create_quantum_packet(
        omega_vector=[1.0, 0.5, 0.2],
        psi_score=0.85,
        scale_level="LΦ"
    )
    success = ufm.store_packet(packet)
    assert success, "Packet storage should succeed"
    
    # Test CDB sync
    nodes_added, edges_added = cdb.sync_with_ufm()
    assert nodes_added >= 1, "Should add at least one node"
    print(f"  ✓ CDB sync: {nodes_added} nodes, {edges_added} edges")
    
    # Test wave propagation query
    results = cdb.wave_propagation_query([1.0, 0.5, 0.2])
    assert isinstance(results, list), "Wave query should return list"
    print(f"  ✓ Wave propagation query: {len(results)} results")
    
    # Test priority indexing
    if packet.id is not None:
        priority = cdb.get_priority_index(packet.id)
        assert isinstance(priority, float), "Priority should be float"
        assert priority >= 0, "Priority should be non-negative"
        print(f"  ✓ Priority indexing: {priority:.4f}")
    
    # Test database statistics
    stats = cdb.get_database_statistics()
    assert isinstance(stats, dict), "Statistics should be dictionary"
    print(f"  ✓ Database statistics: {stats['node_count']} nodes")
    
    print("✓ Coherent Database tests passed\n")

def test_entropy_monitor():
    """Test the Entropy Monitor components"""
    print("Testing Entropy Monitor...")
    
    # Create components
    ufm = UnifiedFieldMemory()
    entropy_monitor = EntropyMonitor(ufm)
    
    # Test packet creation and storage
    packet = ufm.create_quantum_packet(
        omega_vector=[1.0, 0.5, 0.2],
        psi_score=0.85,
        scale_level="LΦ",
        attention_spectrum=[0.3, 0.4, 0.3]
    )
    success = ufm.store_packet(packet)
    assert success, "Packet storage should succeed"
    
    # Test entropy metrics computation
    metrics = entropy_monitor.compute_entropy_metrics(packet)
    assert hasattr(metrics, 'spectral_entropy'), "Should have spectral_entropy"
    assert hasattr(metrics, 'temporal_variance'), "Should have temporal_variance"
    print(f"  ✓ Entropy metrics: {metrics.spectral_entropy:.4f}")
    
    # Test high entropy detection
    is_high = entropy_monitor.detect_high_entropy(packet.id or "", metrics)
    assert isinstance(is_high, bool), "High entropy detection should return boolean"
    print(f"  ✓ High entropy detection: {is_high}")
    
    # Test monitoring
    actions = entropy_monitor.monitor_all_packets()
    assert isinstance(actions, dict), "Monitor should return dictionary"
    print(f"  ✓ Entropy monitoring: {len(actions['monitored'])} packets monitored")
    
    print("✓ Entropy Monitor tests passed\n")

def test_ai_governance():
    """Test the AI Governance components"""
    print("Testing AI Governance...")
    
    # Create components
    ufm = UnifiedFieldMemory()
    cdb = CoherentDatabase(ufm)
    governance = AIGovernance(cdb, ufm)
    
    # Test validator registration
    success = governance.register_validator("validator_1", 1000.0, 0.85)
    assert success, "Validator registration should succeed"
    print(f"  ✓ Validator registration: {len(governance.validators)} validators")
    
    # Test reward function
    reward = governance.compute_reward_function(0.8, 0.85)
    assert isinstance(reward, float), "Reward should be float"
    assert abs(reward) <= 0.1, "Reward should be clipped to ±0.1"
    print(f"  ✓ Reward function: {reward:.4f}")
    
    # Test proposal creation
    proposal_target = {"token_rate": 1.2}
    proposal_id = governance.create_proposal(
        "validator_1",
        "Test Proposal",
        "Test description",
        proposal_target
    )
    assert proposal_id is not None, "Should create proposal"
    print(f"  ✓ Proposal creation: {proposal_id}")
    
    # Test voting
    success = governance.vote_on_proposal("validator_1", proposal_id, 1.0)
    assert success, "Voting should succeed"
    print(f"  ✓ Voting: {len(governance.proposals[proposal_id].votes)} votes")
    
    # Test proposal evaluation
    is_approved, score = governance.evaluate_proposal(proposal_id)
    assert isinstance(is_approved, bool), "Approval should be boolean"
    assert isinstance(score, float), "Score should be float"
    print(f"  ✓ Proposal evaluation: approved={is_approved}, score={score:.4f}")
    
    # Test dimensional stability test
    is_stable = governance.dimensional_stability_test(1.5)
    assert isinstance(is_stable, bool), "Stability test should return boolean"
    print(f"  ✓ Dimensional stability: {is_stable}")
    
    # Test network topology
    topology = governance.get_network_topology()
    assert isinstance(topology, dict), "Topology should be dictionary"
    print(f"  ✓ Network topology: {topology['validators']['total']} validators")
    
    print("✓ AI Governance tests passed\n")

def run_all_tests():
    """Run all UHES component tests"""
    print("🧪 Running UHES Component Tests\n")
    
    try:
        test_cal_engine()
        test_quantum_memory()
        test_coherent_database()
        test_entropy_monitor()
        test_ai_governance()
        
        print("🎉 All UHES Component Tests Passed!")
        print("\nSummary:")
        print("  ✓ CAL Engine - Coherence Attunement Layer")
        print("  ✓ Quantum Memory - Unified Field Memory")
        print("  ✓ Coherent Database - Graph Database")
        print("  ✓ Entropy Monitor - Self-Healing System")
        print("  ✓ AI Governance - Harmonic Regulation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)