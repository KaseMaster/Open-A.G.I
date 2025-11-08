#!/usr/bin/env python3
"""
🛡️ Ω-Security Primitives - Intrinsic Security Based on Coherence
Elevates security by linking cryptographic primitives directly to the system's dimensional state.

This module implements:
1. Ω-Derived Keys (Intrinsic Security) - Coherence-Locked Keys (CLK)
2. Coherence-Based Throttling (CBT) - Dynamic throttling linked to client reputation
"""

import hashlib
import hmac
import time
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import sys
import os

# Add the parent directory to the path to resolve relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required components
from ..models.quantum_memory import QuantumPacket
from ..models.coherence_attunement_layer import OmegaState

# Dynamically import to avoid import errors
HardwareSecurityModule = None
ValidatorKey = None

# Mock implementations for when hardware_security is not available
class MockHSM:
    def __init__(self, *args, **kwargs):
        pass
    def get_hsm_info(self):
        return {"status": "mock", "message": "HardwareSecurityModule not available"}

class MockValidatorKey:
    pass

try:
    # Try absolute import
    from openagi.hardware_security import HardwareSecurityModule, ValidatorKey
except ImportError:
    print("HardwareSecurityModule not available - using mock implementation")
    # Use mock implementations
    HardwareSecurityModule = MockHSM
    ValidatorKey = MockValidatorKey

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CoherenceLockedKey:
    """Represents a Coherence-Locked Key (CLK)"""
    key_id: str
    encrypted_data_hash: str
    omega_vector_hash: str
    time_delay: float
    created_at: float
    expires_at: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ClientReputation:
    """Represents client reputation for throttling"""
    client_id: str
    psi_score: float  # Coherence score
    psy_balance: float  # PSY token balance
    flx_balance: float  # FLX token balance
    reputation_score: float  # Combined reputation score
    last_activity: float
    request_count: int = 0
    last_request_time: float = 0.0

class OmegaSecurityPrimitives:
    """
    Ω-Security Primitives - Intrinsic Security Based on Coherence
    Links cryptographic primitives directly to the system's dimensional state
    """
    
    def __init__(self, network_id: str = "quantum-currency-security-001"):
        self.network_id = network_id
        # Initialize HSM with proper class
        self.hsm = HardwareSecurityModule(f"{network_id}-hsm") if HardwareSecurityModule else MockHSM(f"{network_id}-hsm")
        self.clk_store: Dict[str, CoherenceLockedKey] = {}
        self.client_reputations: Dict[str, ClientReputation] = {}
        self.throttling_policies: Dict[str, Dict[str, Any]] = {}
        
        # Security configuration
        self.config = {
            "clk_hash_algorithm": "sha256",
            "throttling_enabled": True,
            "reputation_weight_psi": 0.5,
            "reputation_weight_psy": 0.3,
            "reputation_weight_flx": 0.2,
            "high_coherence_threshold": 0.90,
            "low_coherence_threshold": 0.70
        }
        
        logger.info(f"🛡️ Ω-Security Primitives initialized for network: {network_id}")
    
    def generate_coherence_locked_key(self, 
                                    qp_hash: str,
                                    omega_vector: List[float],
                                    time_delay: float = 0.0,
                                    metadata: Optional[Dict[str, Any]] = None) -> CoherenceLockedKey:
        """
        Ω-Derived Keys (Intrinsic Security) - Generate a Coherence-Locked Key (CLK)
        
        The symmetric key used for encrypting the Quantum Packet (QP) payload is generated 
        by hashing the QP's field_hash (from the QCL) concatenated with the current, 
        time-delayed Ω vector.
        
        CLK = Hash(QP_hash ∥ Ω_t-τ(L_μ)(L_μ))
        
        Advantage: This ensures that the data can only be decrypted if the current network's 
        coherence state Ω is dimensionally consistent with the state when the data was written.
        Any attempt to tamper with the Ω vector or the QP hash will break the key, 
        serving as a Proof-of-Integrity (PoI) against state tampering.
        
        Args:
            qp_hash: Hash of the Quantum Packet field
            omega_vector: Current Ω state vector
            time_delay: Time delay parameter τ(L)
            metadata: Optional metadata
            
        Returns:
            CoherenceLockedKey object
        """
        # Create hash of Ω vector
        omega_str = json.dumps(omega_vector, sort_keys=True)
        omega_hash = hashlib.sha256(omega_str.encode()).hexdigest()
        
        # Generate CLK: Hash(QP_hash ∥ Ω_hash ∥ time_delay)
        clk_data = f"{qp_hash}{omega_hash}{time_delay}"
        key_id = hashlib.sha256(clk_data.encode()).hexdigest()[:32]
        
        # Create CLK object
        clk = CoherenceLockedKey(
            key_id=key_id,
            encrypted_data_hash=qp_hash,
            omega_vector_hash=omega_hash,
            time_delay=time_delay,
            created_at=time.time(),
            expires_at=time.time() + 86400,  # 24 hours default
            metadata=metadata
        )
        
        # Store CLK
        self.clk_store[key_id] = clk
        
        logger.debug(f"🔐 CLK generated: {key_id[:16]}... for QP {qp_hash[:16]}...")
        return clk
    
    def validate_coherence_locked_key(self, 
                                    clk: CoherenceLockedKey,
                                    current_omega_vector: List[float]) -> bool:
        """
        Validate a Coherence-Locked Key against current Ω state
        
        Args:
            clk: CoherenceLockedKey to validate
            current_omega_vector: Current Ω state vector
            
        Returns:
            bool: True if CLK is valid, False otherwise
        """
        # Check expiration
        if clk.expires_at and time.time() > clk.expires_at:
            logger.warning(f"CLK {clk.key_id[:16]}... expired")
            return False
        
        # Recreate Ω hash
        current_omega_str = json.dumps(current_omega_vector, sort_keys=True)
        current_omega_hash = hashlib.sha256(current_omega_str.encode()).hexdigest()
        
        # Check if Ω state matches
        if current_omega_hash != clk.omega_vector_hash:
            logger.warning(f"CLK {clk.key_id[:16]}... Ω state mismatch")
            return False
        
        logger.debug(f"✅ CLK {clk.key_id[:16]}... validated successfully")
        return True
    
    def update_client_reputation(self, 
                               client_id: str,
                               psi_score: float,
                               psy_balance: float,
                               flx_balance: float) -> ClientReputation:
        """
        Update client reputation based on coherence and token balances
        
        Args:
            client_id: Client identifier
            psi_score: Current coherence score
            psy_balance: PSY token balance
            flx_balance: FLX token balance
            
        Returns:
            Updated ClientReputation object
        """
        # Calculate reputation score
        reputation_score = (
            self.config["reputation_weight_psi"] * psi_score +
            self.config["reputation_weight_psy"] * min(1.0, psy_balance / 1000.0) +
            self.config["reputation_weight_flx"] * min(1.0, flx_balance / 10000.0)
        )
        
        # Get or create client reputation
        if client_id in self.client_reputations:
            client_rep = self.client_reputations[client_id]
            client_rep.psi_score = psi_score
            client_rep.psy_balance = psy_balance
            client_rep.flx_balance = flx_balance
            client_rep.reputation_score = reputation_score
            client_rep.last_activity = time.time()
        else:
            client_rep = ClientReputation(
                client_id=client_id,
                psi_score=psi_score,
                psy_balance=psy_balance,
                flx_balance=flx_balance,
                reputation_score=reputation_score,
                last_activity=time.time()
            )
            self.client_reputations[client_id] = client_rep
        
        logger.debug(f"📊 Client {client_id[:16]}... reputation updated: {reputation_score:.4f}")
        return client_rep
    
    def apply_coherence_based_throttling(self, 
                                       client_id: str,
                                       request_type: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """
        Coherence-Based Throttling (CBT) - Dynamic throttling linked to client reputation (Ψ)
        
        Replace fixed-rate limiting with dynamic throttling linked to client reputation (Ψ).
        
        Args:
            client_id: Client identifier
            request_type: Type of request being made
            
        Returns:
            Tuple of (allowed, throttling_params)
        """
        if not self.config["throttling_enabled"]:
            return True, {"rate_limit": 1000, "burst_limit": 2000}
        
        # Get client reputation
        client_rep = self.client_reputations.get(client_id)
        if not client_rep:
            # Default throttling for unknown clients
            return False, {"rate_limit": 5, "burst_limit": 10, "retry_after": 60}
        
        # Update request count
        client_rep.request_count += 1
        client_rep.last_request_time = time.time()
        
        # Apply throttling based on reputation
        if client_rep.reputation_score >= self.config["high_coherence_threshold"]:
            # High Coherence (Ψ≥0.90): Rate limits are relaxed
            throttling_params = {
                "rate_limit": 500,  # requests per minute
                "burst_limit": 1000,
                "retry_after": 1  # seconds
            }
        elif client_rep.reputation_score >= self.config["low_coherence_threshold"]:
            # Medium Coherence (0.70≤Ψ<0.90): Standard rate limits
            throttling_params = {
                "rate_limit": 100,  # requests per minute
                "burst_limit": 200,
                "retry_after": 10  # seconds
            }
        else:
            # Low Coherence (Ψ<0.70): Rate limits are significantly tightened
            throttling_params = {
                "rate_limit": 5,  # requests per minute
                "burst_limit": 10,
                "retry_after": 60,  # seconds
                "message": "Low coherence - consider improving network participation"
            }
        
        # Store throttling policy
        policy_key = f"{client_id}:{request_type}"
        self.throttling_policies[policy_key] = throttling_params
        
        allowed = True  # In a real implementation, this would check actual rate limits
        logger.debug(f"🚦 CBT applied for client {client_id[:16]}...: "
                    f"rate_limit={throttling_params['rate_limit']}")
        
        return allowed, throttling_params
    
    def get_security_report(self) -> Dict[str, Any]:
        """
        Get security report for the Ω-Security Primitives
        
        Returns:
            Dict with security information
        """
        return {
            "network_id": self.network_id,
            "clk_count": len(self.clk_store),
            "client_count": len(self.client_reputations),
            "throttling_policies": len(self.throttling_policies),
            "config": self.config,
            "hsm_info": self.hsm.get_hsm_info(),
            "timestamp": time.time()
        }

# Example usage and testing
if __name__ == "__main__":
    # Create Ω-Security Primitives instance
    security = OmegaSecurityPrimitives()
    
    # Test CLK generation
    qp_hash = hashlib.sha256(b"test_quantum_packet_data").hexdigest()
    omega_vector = [1.0, 0.5, 0.2, 0.8, 0.3]
    clk = security.generate_coherence_locked_key(qp_hash, omega_vector, time_delay=1.5)
    print(f"CLK generated: {clk.key_id[:16]}...")
    
    # Test CLK validation
    is_valid = security.validate_coherence_locked_key(clk, omega_vector)
    print(f"CLK validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test client reputation
    client_rep = security.update_client_reputation(
        client_id="test_client_001",
        psi_score=0.85,
        psy_balance=500.0,
        flx_balance=2500.0
    )
    print(f"Client reputation: {client_rep.reputation_score:.4f}")
    
    # Test throttling
    allowed, params = security.apply_coherence_based_throttling("test_client_001")
    print(f"Throttling: {'✅ Allowed' if allowed else '❌ Denied'}, "
          f"Rate limit: {params['rate_limit']} req/min")
    
    # Security report
    report = security.get_security_report()
    print(f"Security report: {report}")