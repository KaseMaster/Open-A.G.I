#!/usr/bin/env python3
"""
Test suite for Quantum Bridge
"""

import sys
import os
import pytest
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from network.quantum_bridge import QuantumBridge, CoherenceMessage

class TestQuantumBridge:
    """Test cases for QuantumBridge"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.bridge = QuantumBridge("test-system")
        
    def test_add_connection(self):
        """Test adding a connection"""
        success = self.bridge.add_connection(
            system_id="external-system-001",
            websocket_uri="ws://external-system-001:8765",
            shared_secret="shared_secret_123"
        )
        
        assert success == True
        assert "external-system-001" in self.bridge.connections
        assert "external-system-001" in self.bridge.coherence_history
        
    def test_create_coherence_message(self):
        """Test creating a coherence message"""
        # Add a connection first
        self.bridge.add_connection(
            system_id="external-system-001",
            websocket_uri="ws://external-system-001:8765",
            shared_secret="shared_secret_123"
        )
        
        omega_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        psi_score = 0.85
        
        message = self.bridge.create_coherence_message(
            receiver_id="external-system-001",
            omega_vector=omega_vector,
            psi_score=psi_score,
            payload={"test": "data", "value": 42}
        )
        
        assert message is not None
        assert message.sender_id == "test-system"
        assert message.receiver_id == "external-system-001"
        assert message.omega_vector == omega_vector
        assert message.psi_score == psi_score
        assert len(message.message_id) == 32
        assert len(message.signature) == 32
        
    def test_validate_message_integrity(self):
        """Test message integrity validation"""
        # Add a connection first
        self.bridge.add_connection(
            system_id="external-system-001",
            websocket_uri="ws://external-system-001:8765",
            shared_secret="shared_secret_123"
        )
        
        omega_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        psi_score = 0.85
        
        # Create a valid message
        message = self.bridge.create_coherence_message(
            receiver_id="external-system-001",
            omega_vector=omega_vector,
            psi_score=psi_score
        )
        
        # Validate the message
        is_valid = self.bridge.validate_message_integrity(message)
        assert is_valid == True
        
        # Test with tampered message
        tampered_message = CoherenceMessage(
            message_id=message.message_id,
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            omega_vector=[0.2, 0.3, 0.4, 0.5, 0.6],  # Tampered
            psi_score=message.psi_score,
            timestamp=message.timestamp,
            signature=message.signature
        )
        
        is_valid_tampered = self.bridge.validate_message_integrity(tampered_message)
        assert is_valid_tampered == False
        
    def test_send_coherence_message(self):
        """Test sending a coherence message"""
        # Add a connection first
        self.bridge.add_connection(
            system_id="external-system-001",
            websocket_uri="ws://external-system-001:8765",
            shared_secret="shared_secret_123"
        )
        
        # Mark connection as active
        self.bridge.connections["external-system-001"].is_active = True
        
        omega_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        psi_score = 0.85
        
        message = self.bridge.create_coherence_message(
            receiver_id="external-system-001",
            omega_vector=omega_vector,
            psi_score=psi_score
        )
        
        # Send the message
        sent = self.bridge.send_coherence_message(message)
        assert sent == True
        assert len(self.bridge.message_queue) == 1
        
        # Check coherence history was updated
        assert len(self.bridge.coherence_history["test-system"]) == 1
        assert self.bridge.coherence_history["test-system"][0] == psi_score
        
    def test_receive_coherence_message(self):
        """Test receiving a coherence message"""
        # Add a connection first
        self.bridge.add_connection(
            system_id="external-system-001",
            websocket_uri="ws://external-system-001:8765",
            shared_secret="shared_secret_123"
        )
        
        # Create a message dictionary (simulating received data)
        message_dict = {
            "message_id": "test-message-id-1234567890123456",
            "sender_id": "external-system-001",
            "receiver_id": "test-system",
            "omega_vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "psi_score": 0.85,
            "timestamp": 1234567890.0,
            "signature": "test-signature-1234567890123456"
        }
        
        # Process the message
        processed_message = self.bridge.receive_coherence_message(message_dict)
        
        # For this test, we expect None because the signature validation will fail
        # In a real implementation, we would create a proper signature
        assert processed_message is None
        
    def test_calculate_psi_balancing(self):
        """Test Ψ-balancing calculations"""
        # Add some coherence history
        self.bridge.coherence_history["system-001"] = [0.8, 0.85, 0.9, 0.87, 0.83]
        self.bridge.coherence_history["system-002"] = [0.7, 0.75, 0.72, 0.78, 0.74]
        
        adjustments = self.bridge.calculate_psi_balancing()
        
        assert "system-001" in adjustments
        assert "system-002" in adjustments
        # Adjustments should be between -0.1 and 0.1
        assert -0.1 <= adjustments["system-001"] <= 0.1
        assert -0.1 <= adjustments["system-002"] <= 0.1
        
    def test_check_entropy_rate(self):
        """Test entropy rate checking"""
        # Add some coherence history with low variance
        self.bridge.coherence_history["system-001"] = [0.85, 0.86, 0.84, 0.85, 0.86]
        
        entropy_ok = self.bridge.check_entropy_rate()
        assert isinstance(entropy_ok, bool)
        
    def test_get_bridge_status(self):
        """Test getting bridge status"""
        # Add a connection
        self.bridge.add_connection(
            system_id="external-system-001",
            websocket_uri="ws://external-system-001:8765",
            shared_secret="shared_secret_123"
        )
        
        status = self.bridge.get_bridge_status()
        
        assert status["local_system_id"] == "test-system"
        assert status["total_connections"] == 1
        assert status["active_connections"] == 0  # Not active yet
        assert status["message_queue_size"] == 0
        assert "entropy_rate" in status
        assert "connected_systems" in status

if __name__ == "__main__":
    pytest.main([__file__, "-v"])