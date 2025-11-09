#!/usr/bin/env python3
"""
Quantum Bridge for Inter-System Integration
Implements cross-system quantum integration layer for coherence economy activation

This module provides:
1. Quantum Integration Daemon (QID) for inter-network coherence signaling
2. Ω-Security Exchange Module for secure Ω-vector data transfer
3. Cross-system validation and coherence reflection consistency
"""

import sys
import os
import json
import time
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class CoherenceMessage:
    """Represents a coherence message exchanged between systems"""
    message_id: str
    sender_id: str
    receiver_id: str
    omega_vector: List[float]  # Ω-state vector
    psi_score: float  # Coherence score
    timestamp: float
    signature: str  # Digital signature
    encrypted_content: Optional[str] = None  # Encrypted payload

@dataclass
class BridgeConnection:
    """Represents a connection to another quantum system"""
    system_id: str
    websocket_uri: str
    shared_secret: str
    last_heartbeat: float = 0.0
    is_active: bool = False
    encryption_key: Optional[bytes] = None

class QuantumBridge:
    """
    Quantum Bridge for Inter-System Integration
    Enables secure Ω-vector data transfer and cross-system coherence validation
    """
    
    def __init__(self, local_system_id: str = "quantum-currency-001"):
        self.local_system_id = local_system_id
        self.connections: Dict[str, BridgeConnection] = {}
        self.message_queue: List[CoherenceMessage] = []
        self.coherence_history: Dict[str, List[float]] = {}  # system_id -> list of Ψ scores
        self.entropy_rate = 0.0  # Current entropy rate
        self.max_entropy_threshold = 0.002  # Maximum allowed entropy
        
    def add_connection(self, system_id: str, websocket_uri: str, shared_secret: str) -> bool:
        """
        Add a connection to another quantum system
        
        Args:
            system_id: ID of the remote system
            websocket_uri: WebSocket URI for the connection
            shared_secret: Shared secret for encryption
            
        Returns:
            bool: True if connection added successfully
        """
        try:
            # Derive encryption key from shared secret
            salt = b'quantum_bridge_salt'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(shared_secret.encode()))
            
            # Create connection
            connection = BridgeConnection(
                system_id=system_id,
                websocket_uri=websocket_uri,
                shared_secret=shared_secret,
                encryption_key=key
            )
            
            self.connections[system_id] = connection
            self.coherence_history[system_id] = []
            
            return True
        except Exception as e:
            print(f"Failed to add connection to {system_id}: {e}")
            return False
    
    def encrypt_message(self, connection: BridgeConnection, message: str) -> str:
        """
        Encrypt a message using the connection's encryption key
        
        Args:
            connection: Bridge connection
            message: Message to encrypt
            
        Returns:
            str: Encrypted message
        """
        if connection.encryption_key is None:
            raise ValueError("No encryption key available")
            
        f = Fernet(connection.encryption_key)
        encrypted_message = f.encrypt(message.encode())
        return base64.urlsafe_b64encode(encrypted_message).decode()
    
    def decrypt_message(self, connection: BridgeConnection, encrypted_message: str) -> str:
        """
        Decrypt a message using the connection's encryption key
        
        Args:
            connection: Bridge connection
            encrypted_message: Encrypted message
            
        Returns:
            str: Decrypted message
        """
        if connection.encryption_key is None:
            raise ValueError("No encryption key available")
            
        f = Fernet(connection.encryption_key)
        decoded_message = base64.urlsafe_b64decode(encrypted_message.encode())
        decrypted_message = f.decrypt(decoded_message)
        return decrypted_message.decode()
    
    def create_coherence_message(self, receiver_id: str, omega_vector: List[float], 
                               psi_score: float, payload: Optional[Dict[str, Any]] = None) -> CoherenceMessage:
        """
        Create a coherence message for transmission
        
        Args:
            receiver_id: ID of the receiving system
            omega_vector: Ω-state vector
            psi_score: Coherence score
            payload: Optional additional payload data
            
        Returns:
            CoherenceMessage: Created message
        """
        message_id = hashlib.sha256(f"{self.local_system_id}{receiver_id}{time.time()}".encode()).hexdigest()[:32]
        
        # Create message
        message = CoherenceMessage(
            message_id=message_id,
            sender_id=self.local_system_id,
            receiver_id=receiver_id,
            omega_vector=omega_vector,
            psi_score=psi_score,
            timestamp=time.time(),
            signature=""  # Will be set below
        )
        
        # Add signature (simplified for demo)
        message_data = f"{message_id}{self.local_system_id}{receiver_id}{omega_vector}{psi_score}"
        message.signature = hashlib.sha256(message_data.encode()).hexdigest()[:32]
        
        # Encrypt payload if provided
        if payload and receiver_id in self.connections:
            connection = self.connections[receiver_id]
            payload_json = json.dumps(payload)
            message.encrypted_content = self.encrypt_message(connection, payload_json)
        
        return message
    
    def validate_message_integrity(self, message: CoherenceMessage) -> bool:
        """
        Validate the integrity of a received message
        
        Args:
            message: Message to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Recreate signature
        message_data = f"{message.message_id}{message.sender_id}{message.receiver_id}{message.omega_vector}{message.psi_score}"
        expected_signature = hashlib.sha256(message_data.encode()).hexdigest()[:32]
        
        # Check signature
        if message.signature != expected_signature:
            print(f"Invalid signature for message {message.message_id}")
            return False
            
        # Check timestamp (within 5 minutes)
        if abs(time.time() - message.timestamp) > 300:
            print(f"Message {message.message_id} timestamp too old")
            return False
            
        return True
    
    def send_coherence_message(self, message: CoherenceMessage) -> bool:
        """
        Send a coherence message to a connected system
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if message.receiver_id not in self.connections:
            print(f"No connection to system {message.receiver_id}")
            return False
            
        connection = self.connections[message.receiver_id]
        if not connection.is_active:
            print(f"Connection to {message.receiver_id} is not active")
            return False
            
        try:
            # Convert message to JSON
            message_dict = {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "omega_vector": message.omega_vector,
                "psi_score": message.psi_score,
                "timestamp": message.timestamp,
                "signature": message.signature,
                "encrypted_content": message.encrypted_content
            }
            
            # In a real implementation, this would send via WebSocket
            # For now, we'll just queue the message
            self.message_queue.append(message)
            
            # Update coherence history
            if message.sender_id not in self.coherence_history:
                self.coherence_history[message.sender_id] = []
            self.coherence_history[message.sender_id].append(message.psi_score)
            
            # Keep only recent history (last 100 scores)
            if len(self.coherence_history[message.sender_id]) > 100:
                self.coherence_history[message.sender_id] = self.coherence_history[message.sender_id][-100:]
            
            return True
        except Exception as e:
            print(f"Failed to send message to {message.receiver_id}: {e}")
            return False
    
    def receive_coherence_message(self, message_dict: Dict[str, Any]) -> Optional[CoherenceMessage]:
        """
        Receive and process a coherence message
        
        Args:
            message_dict: Dictionary representation of the message
            
        Returns:
            CoherenceMessage: Processed message or None if invalid
        """
        try:
            # Create message object
            message = CoherenceMessage(
                message_id=message_dict["message_id"],
                sender_id=message_dict["sender_id"],
                receiver_id=message_dict["receiver_id"],
                omega_vector=message_dict["omega_vector"],
                psi_score=message_dict["psi_score"],
                timestamp=message_dict["timestamp"],
                signature=message_dict["signature"],
                encrypted_content=message_dict.get("encrypted_content")
            )
            
            # Validate message integrity
            if not self.validate_message_integrity(message):
                return None
                
            # Decrypt content if present
            if message.encrypted_content and message.sender_id in self.connections:
                connection = self.connections[message.sender_id]
                try:
                    decrypted_content = self.decrypt_message(connection, message.encrypted_content)
                    print(f"Decrypted content from {message.sender_id}: {decrypted_content}")
                except Exception as e:
                    print(f"Failed to decrypt content from {message.sender_id}: {e}")
            
            # Update coherence history
            if message.sender_id not in self.coherence_history:
                self.coherence_history[message.sender_id] = []
            self.coherence_history[message.sender_id].append(message.psi_score)
            
            # Keep only recent history (last 100 scores)
            if len(self.coherence_history[message.sender_id]) > 100:
                self.coherence_history[message.sender_id] = self.coherence_history[message.sender_id][-100:]
            
            return message
        except Exception as e:
            print(f"Failed to process received message: {e}")
            return None
    
    def calculate_psi_balancing(self) -> Dict[str, float]:
        """
        Calculate Ψ-balancing heuristics across connected systems
        
        Returns:
            Dict mapping system IDs to balancing adjustments
        """
        adjustments = {}
        
        for system_id, psi_scores in self.coherence_history.items():
            if not psi_scores:
                adjustments[system_id] = 0.0
                continue
                
            # Calculate average Ψ score
            avg_psi = sum(psi_scores) / len(psi_scores)
            
            # Calculate variance
            variance = sum((score - avg_psi) ** 2 for score in psi_scores) / len(psi_scores)
            
            # Adjustment based on variance (simplified)
            adjustment = max(-0.1, min(0.1, -variance * 10))
            adjustments[system_id] = adjustment
            
        return adjustments
    
    def check_entropy_rate(self) -> bool:
        """
        Check if the current entropy rate is within acceptable bounds
        
        Returns:
            bool: True if entropy rate is acceptable, False otherwise
        """
        # Calculate current entropy rate (simplified)
        total_variance = 0.0
        count = 0
        
        for psi_scores in self.coherence_history.values():
            if len(psi_scores) < 2:
                continue
                
            # Calculate variance of recent scores
            recent_scores = psi_scores[-10:] if len(psi_scores) >= 10 else psi_scores
            avg = sum(recent_scores) / len(recent_scores)
            variance = sum((score - avg) ** 2 for score in recent_scores) / len(recent_scores)
            total_variance += variance
            count += 1
            
        if count > 0:
            self.entropy_rate = total_variance / count
            
        return self.entropy_rate <= self.max_entropy_threshold
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get the current status of the quantum bridge
        
        Returns:
            Dict with bridge status information
        """
        active_connections = sum(1 for conn in self.connections.values() if conn.is_active)
        
        return {
            "local_system_id": self.local_system_id,
            "total_connections": len(self.connections),
            "active_connections": active_connections,
            "message_queue_size": len(self.message_queue),
            "entropy_rate": self.entropy_rate,
            "max_entropy_threshold": self.max_entropy_threshold,
            "entropy_within_bounds": self.entropy_rate <= self.max_entropy_threshold,
            "connected_systems": list(self.connections.keys())
        }

# Example usage and testing
if __name__ == "__main__":
    # Create quantum bridge
    bridge = QuantumBridge("quantum-currency-local")
    
    # Add a connection
    success = bridge.add_connection(
        system_id="external-system-001",
        websocket_uri="ws://external-system-001:8765",
        shared_secret="shared_secret_123"
    )
    
    print(f"Connection added: {success}")
    
    # Create a coherence message
    omega_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    psi_score = 0.85
    
    message = bridge.create_coherence_message(
        receiver_id="external-system-001",
        omega_vector=omega_vector,
        psi_score=psi_score,
        payload={"test": "data", "value": 42}
    )
    
    print(f"Created message: {message.message_id}")
    
    # Send the message
    sent = bridge.send_coherence_message(message)
    print(f"Message sent: {sent}")
    
    # Check bridge status
    status = bridge.get_bridge_status()
    print(f"Bridge status: {status}")
    
    # Check entropy rate
    entropy_ok = bridge.check_entropy_rate()
    print(f"Entropy rate within bounds: {entropy_ok}")
    
    # Calculate Ψ balancing
    adjustments = bridge.calculate_psi_balancing()
    print(f"Ψ balancing adjustments: {adjustments}")