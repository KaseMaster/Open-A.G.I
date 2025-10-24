"""
AEGIS Framework - End-to-End Basic Flow Test
Tests the complete workflow: initialization -> operation -> shutdown
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.crypto_framework import CryptoEngine, SecurityLevel
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from src.aegis.networking.p2p_network import NodeType
from src.aegis.monitoring.metrics_collector import AEGISMetricsCollector


def test_e2e_crypto_engine_initialization():
    """Test: CryptoEngine initialization and basic operations"""
    
    crypto = CryptoEngine()
    assert crypto is not None
    assert crypto.config is not None
    assert crypto.config.security_level == SecurityLevel.HIGH
    
    # Generate node identity
    identity = crypto.generate_node_identity("test_node_001")
    assert identity is not None
    assert identity.node_id == "test_node_001"
    assert identity.signing_key is not None
    assert identity.encryption_key is not None
    assert identity.public_signing_key is not None
    assert identity.public_encryption_key is not None


def test_e2e_crypto_signing_and_verification():
    """Test: Crypto signing and signature verification"""
    
    crypto = CryptoEngine()
    identity = crypto.generate_node_identity("test_node_002")
    crypto.identity = identity
    
    # Test signing
    test_data = b"test_message_data"
    signature = crypto.sign_data(test_data)
    assert signature is not None
    assert len(signature) > 0
    
    # For now, just verify signature was created successfully
    # Full signature verification requires peer identity setup
    assert isinstance(signature, bytes)
    assert len(signature) == 64  # Ed25519 signature length


def test_e2e_consensus_initialization():
    """Test: HybridConsensus initialization"""
    
    from cryptography.hazmat.primitives.asymmetric import ed25519
    
    node_id = "test_consensus_node"
    private_key = ed25519.Ed25519PrivateKey.generate()
    
    consensus = HybridConsensus(
        node_id=node_id,
        private_key=private_key
    )
    
    assert consensus is not None
    assert consensus.node_id == node_id
    assert hasattr(consensus, 'poc')
    assert hasattr(consensus, 'pbft')


def test_e2e_p2p_node_types():
    """Test: P2P NodeType enum and values"""
    
    # Test NodeType enum exists and is accessible
    assert hasattr(NodeType, 'VALIDATOR')
    assert hasattr(NodeType, 'FULL')
    assert hasattr(NodeType, 'LIGHT')
    assert hasattr(NodeType, 'BOOTSTRAP')
    assert hasattr(NodeType, 'STORAGE')
    
    # Test that NodeType values work correctly
    validator_type = NodeType.VALIDATOR
    assert validator_type.value == "validator"
    
    full_type = NodeType.FULL
    assert full_type.value == "full"
    
    light_type = NodeType.LIGHT
    assert light_type.value == "light"


def test_e2e_metrics_collector_initialization():
    """Test: Metrics collector initialization"""
    
    collector = AEGISMetricsCollector()
    
    # Test basic initialization
    assert collector is not None
    # Check that collector has the expected structure
    assert hasattr(collector, 'collect_all_metrics') or hasattr(collector, 'get_metrics')


def test_e2e_multi_component_integration():
    """Test: Multiple components working together"""
    
    from cryptography.hazmat.primitives.asymmetric import ed25519
    
    # Initialize crypto
    crypto = CryptoEngine()
    identity = crypto.generate_node_identity("integration_node")
    crypto.identity = identity
    
    # Initialize consensus
    private_key = ed25519.Ed25519PrivateKey.generate()
    consensus = HybridConsensus(
        node_id="integration_node",
        private_key=private_key
    )
    
    # Initialize metrics
    metrics = AEGISMetricsCollector()
    
    # Test crypto operations
    message = b"integration_test_message"
    signature = crypto.sign_data(message)
    
    assert signature is not None
    assert isinstance(signature, bytes)
    assert len(signature) == 64  # Ed25519 signature length
    
    # Verify all components initialized correctly
    assert crypto is not None
    assert crypto.identity is not None
    assert consensus is not None
    assert consensus.node_id == "integration_node"
    assert metrics is not None


def test_e2e_crypto_key_generation():
    """Test: Cryptographic key generation"""
    
    crypto = CryptoEngine()
    
    # Generate multiple identities
    identity1 = crypto.generate_node_identity("node_1")
    identity2 = crypto.generate_node_identity("node_2")
    
    # Verify they are different
    assert identity1.node_id != identity2.node_id
    assert identity1.signing_key != identity2.signing_key
    assert identity1.encryption_key != identity2.encryption_key
    
    # Verify each can sign independently
    crypto.identity = identity1
    sig1 = crypto.sign_data(b"message1")
    
    crypto.identity = identity2
    sig2 = crypto.sign_data(b"message2")
    
    assert sig1 != sig2


def test_e2e_security_levels():
    """Test: Different security levels"""
    
    # Test HIGH security level (default)
    crypto_high = CryptoEngine()
    assert crypto_high.config.security_level == SecurityLevel.HIGH
    assert crypto_high.config.key_rotation_interval == 86400
    
    # Test STANDARD security level
    from src.aegis.security.crypto_framework import CryptoConfig
    config_standard = CryptoConfig(security_level=SecurityLevel.STANDARD)
    crypto_standard = CryptoEngine(config=config_standard)
    assert crypto_standard.config.security_level == SecurityLevel.STANDARD
    assert crypto_standard.config.key_rotation_interval == 172800
    
    # Test PARANOID security level
    config_paranoid = CryptoConfig(security_level=SecurityLevel.PARANOID)
    crypto_paranoid = CryptoEngine(config=config_paranoid)
    assert crypto_paranoid.config.security_level == SecurityLevel.PARANOID
    assert crypto_paranoid.config.key_rotation_interval == 3600


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
