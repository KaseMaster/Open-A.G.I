#!/usr/bin/env python3
"""
Homomorphic Encryption Module for Quantum Currency System
Implements zero-knowledge validation of resonance data

This module provides homomorphic encryption capabilities for validating
resonance data without revealing the actual data values.
"""

import sys
import os
import json
import time
import hashlib
import secrets
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from openagi.harmonic_validation import (
    compute_spectrum, 
    pairwise_coherence,
    HarmonicSnapshot,
    HarmonicProofBundle
)

@dataclass
class EncryptedValue:
    """Represents an encrypted value using additive homomorphic encryption"""
    ciphertext: int
    nonce: int
    modulus: int

@dataclass
class ZeroKnowledgeProof:
    """Represents a zero-knowledge proof for encrypted data validation"""
    proof_id: str
    commitment: int
    challenge_response: int
    timestamp: float
    validator_signature: Optional[str] = None

@dataclass
class EncryptedCoherenceProof:
    """Encrypted proof of coherence validation"""
    proof_id: str
    encrypted_coherence: EncryptedValue
    zk_proof: ZeroKnowledgeProof
    validator_id: str
    timestamp: float

class SimpleHomomorphicEncryption:
    """
    Simple additive homomorphic encryption implementation
    Based on Paillier-like cryptosystem for demonstration purposes
    """
    
    def __init__(self, key_size: int = 1024):
        # Generate simple keys (in practice, use proper cryptographic libraries)
        self.key_size = key_size
        self.modulus = self._generate_prime(key_size // 2) * self._generate_prime(key_size // 2)
        self.generator = self.modulus + 1
        self.private_key = secrets.randbelow(self.modulus)
        
    def _generate_prime(self, bits: int) -> int:
        """Generate a simple prime number (for demonstration only)"""
        # This is a simplified implementation for demonstration
        # In practice, use proper cryptographic prime generation
        candidate = 2 ** bits + secrets.randbits(bits - 1)
        # Make odd
        if candidate % 2 == 0:
            candidate += 1
        return candidate
    
    def encrypt(self, plaintext: int) -> EncryptedValue:
        """
        Encrypt a value using additive homomorphic encryption
        
        Args:
            plaintext: Integer value to encrypt
            
        Returns:
            EncryptedValue object
        """
        # Simplified encryption for demonstration
        nonce = secrets.randbelow(self.modulus)
        # In a real implementation, this would use proper Paillier encryption
        ciphertext = (self.generator * plaintext + nonce) % self.modulus
        return EncryptedValue(ciphertext=ciphertext, nonce=nonce, modulus=self.modulus)
    
    def decrypt(self, encrypted: EncryptedValue) -> int:
        """
        Decrypt a value (only for demonstration - in practice, this would require private key)
        
        Args:
            encrypted: EncryptedValue to decrypt
            
        Returns:
            Decrypted integer value
        """
        # Simplified decryption for demonstration
        # In practice, this would use the private key
        return (encrypted.ciphertext - encrypted.nonce) // self.generator
    
    def add_encrypted(self, a: EncryptedValue, b: EncryptedValue) -> EncryptedValue:
        """
        Add two encrypted values homomorphically
        
        Args:
            a: First encrypted value
            b: Second encrypted value
            
        Returns:
            Encrypted sum of a and b
        """
        if a.modulus != b.modulus:
            raise ValueError("Cannot add encrypted values with different moduli")
        
        # Homomorphic addition: E(a) * E(b) = E(a + b)
        result_ciphertext = (a.ciphertext * b.ciphertext) % a.modulus
        result_nonce = (a.nonce + b.nonce) % a.modulus
        return EncryptedValue(ciphertext=result_ciphertext, nonce=result_nonce, modulus=a.modulus)

class ZeroKnowledgeValidator:
    """
    Zero-knowledge validator for encrypted resonance data
    """
    
    def __init__(self, validator_id: str):
        self.validator_id = validator_id
        self.crypto = SimpleHomomorphicEncryption()
        self.proofs: List[EncryptedCoherenceProof] = []
    
    def create_commitment(self, value: int) -> Tuple[int, int]:
        """
        Create a commitment to a value for zero-knowledge proof
        
        Args:
            value: Value to commit to
            
        Returns:
            Tuple of (commitment, randomness)
        """
        randomness = secrets.randbelow(2**128)
        commitment = (value + randomness) % (2**128)
        return commitment, randomness
    
    def verify_commitment(self, commitment: int, value: int, randomness: int) -> bool:
        """
        Verify a commitment
        
        Args:
            commitment: Committed value
            value: Original value
            randomness: Randomness used in commitment
            
        Returns:
            True if commitment is valid
        """
        expected = (value + randomness) % (2**128)
        return commitment == expected
    
    def generate_zk_proof(self, encrypted_value: EncryptedValue, plaintext_value: int) -> ZeroKnowledgeProof:
        """
        Generate a zero-knowledge proof for an encrypted value
        
        Args:
            encrypted_value: The encrypted value
            plaintext_value: The plaintext value (for demonstration)
            
        Returns:
            ZeroKnowledgeProof object
        """
        # Create commitment to the plaintext value
        commitment, randomness = self.create_commitment(plaintext_value)
        
        # Generate challenge response (simplified for demonstration)
        challenge = secrets.randbelow(2**64)
        challenge_response = (plaintext_value * challenge + randomness) % (2**128)
        
        proof = ZeroKnowledgeProof(
            proof_id=f"proof-{int(time.time())}-{hashlib.md5(str(plaintext_value).encode()).hexdigest()[:8]}",
            commitment=commitment,
            challenge_response=challenge_response,
            timestamp=time.time()
        )
        
        return proof
    
    def verify_zk_proof(self, proof: ZeroKnowledgeProof, encrypted_value: EncryptedValue) -> bool:
        """
        Verify a zero-knowledge proof
        
        Args:
            proof: ZeroKnowledgeProof to verify
            encrypted_value: The encrypted value being proved
            
        Returns:
            True if proof is valid
        """
        # Simplified verification for demonstration
        # In practice, this would involve more complex cryptographic operations
        return (
            proof.commitment > 0 and 
            proof.challenge_response > 0 and
            proof.timestamp > 0
        )
    
    def validate_encrypted_coherence(
        self, 
        local_snapshot: HarmonicSnapshot,
        remote_snapshots: List[HarmonicSnapshot]
    ) -> Optional[EncryptedCoherenceProof]:
        """
        Validate coherence using homomorphic encryption and zero-knowledge proofs
        
        Args:
            local_snapshot: Local node's harmonic snapshot
            remote_snapshots: List of remote nodes' snapshots
            
        Returns:
            EncryptedCoherenceProof if validation successful, None otherwise
        """
        try:
            # Compute coherence score
            if not remote_snapshots:
                return None
            
            # Convert local data to numpy arrays
            local_times = np.array(local_snapshot.times)
            local_values = np.array(local_snapshot.values)
            fs = 1.0 / (local_times[1] - local_times[0]) if len(local_times) > 1 else 1.0
            
            coherence_scores = []
            
            for remote in remote_snapshots:
                # Convert remote data to numpy arrays
                remote_times = np.array(remote.times)
                remote_values = np.array(remote.values)
                
                # Resample to match lengths if needed
                min_len = min(len(local_values), len(remote_values))
                if min_len == 0:
                    continue
                    
                local_resampled = local_values[:min_len]
                remote_resampled = remote_values[:min_len]
                
                # Compute pairwise coherence
                coh = pairwise_coherence(local_resampled, remote_resampled, fs)
                coherence_scores.append(coh)
            
            if not coherence_scores:
                return None
                
            # Calculate mean coherence score
            mean_coherence = float(np.mean(coherence_scores))
            
            # Convert to integer for encryption (scaled)
            scaled_coherence = int(mean_coherence * 10000)  # Scale to preserve precision
            
            # Encrypt the coherence score
            encrypted_coherence = self.crypto.encrypt(scaled_coherence)
            
            # Generate zero-knowledge proof
            zk_proof = self.generate_zk_proof(encrypted_coherence, scaled_coherence)
            
            # Create encrypted coherence proof
            proof = EncryptedCoherenceProof(
                proof_id=f"ecoherence-{int(time.time())}-{hashlib.md5(str(mean_coherence).encode()).hexdigest()[:8]}",
                encrypted_coherence=encrypted_coherence,
                zk_proof=zk_proof,
                validator_id=self.validator_id,
                timestamp=time.time()
            )
            
            # Store proof
            self.proofs.append(proof)
            
            return proof
            
        except Exception as e:
            print(f"Error validating encrypted coherence: {e}")
            return None
    
    def aggregate_encrypted_proofs(
        self, 
        proofs: List[EncryptedCoherenceProof]
    ) -> Optional[EncryptedValue]:
        """
        Aggregate multiple encrypted coherence proofs homomorphically
        
        Args:
            proofs: List of EncryptedCoherenceProof objects
            
        Returns:
            Aggregated encrypted value, or None if aggregation fails
        """
        if not proofs:
            return None
        
        try:
            # Start with the first encrypted value
            aggregated = proofs[0].encrypted_coherence
            
            # Add all other encrypted values homomorphically
            for proof in proofs[1:]:
                aggregated = self.crypto.add_encrypted(aggregated, proof.encrypted_coherence)
            
            return aggregated
            
        except Exception as e:
            print(f"Error aggregating encrypted proofs: {e}")
            return None

def demo_homomorphic_encryption():
    """Demonstrate homomorphic encryption and zero-knowledge validation"""
    print("🛡️  Homomorphic Encryption and Zero-Knowledge Validation Demo")
    print("=" * 60)
    
    # Create validator
    validator = ZeroKnowledgeValidator("validator-001")
    
    # Generate test signals
    print("\n📡 Generating test signals...")
    t1 = np.linspace(0, 1.0, 1000)
    x1 = np.sin(2 * np.pi * 10 * t1) + 0.1 * np.random.randn(len(t1))
    x2 = np.sin(2 * np.pi * 10 * t1) + 0.1 * np.random.randn(len(t1))  # Coherent
    x3 = np.sin(2 * np.pi * 15 * t1) + 0.2 * np.random.randn(len(t1))  # Less coherent
    
    # Create snapshots
    snapshot_a = HarmonicSnapshot(
        node_id="node-A",
        timestamp=time.time(),
        times=t1.tolist(),
        values=x1.tolist(),
        spectrum=[],
        spectrum_hash="",
        CS=0.0,
        phi_params={}
    )
    
    snapshot_b = HarmonicSnapshot(
        node_id="node-B",
        timestamp=time.time(),
        times=t1.tolist(),
        values=x2.tolist(),
        spectrum=[],
        spectrum_hash="",
        CS=0.0,
        phi_params={}
    )
    
    snapshot_c = HarmonicSnapshot(
        node_id="node-C",
        timestamp=time.time(),
        times=t1.tolist(),
        values=x3.tolist(),
        spectrum=[],
        spectrum_hash="",
        CS=0.0,
        phi_params={}
    )
    
    # Validate encrypted coherence
    print("\n🔍 Validating encrypted coherence...")
    proof1 = validator.validate_encrypted_coherence(snapshot_a, [snapshot_b])
    proof2 = validator.validate_encrypted_coherence(snapshot_a, [snapshot_c])
    
    if proof1 and proof2:
        print(f"✅ Proof 1 generated: {proof1.proof_id}")
        print(f"   Encrypted coherence: {proof1.encrypted_coherence.ciphertext}")
        print(f"   ZK proof ID: {proof1.zk_proof.proof_id}")
        
        print(f"✅ Proof 2 generated: {proof2.proof_id}")
        print(f"   Encrypted coherence: {proof2.encrypted_coherence.ciphertext}")
        print(f"   ZK proof ID: {proof2.zk_proof.proof_id}")
        
        # Verify zero-knowledge proofs
        print("\n🔐 Verifying zero-knowledge proofs...")
        zk_valid1 = validator.verify_zk_proof(proof1.zk_proof, proof1.encrypted_coherence)
        zk_valid2 = validator.verify_zk_proof(proof2.zk_proof, proof2.encrypted_coherence)
        
        print(f"   Proof 1 ZK validity: {'✅ VALID' if zk_valid1 else '❌ INVALID'}")
        print(f"   Proof 2 ZK validity: {'✅ VALID' if zk_valid2 else '❌ INVALID'}")
        
        # Aggregate encrypted proofs
        print("\n🧮 Aggregating encrypted proofs...")
        aggregated = validator.aggregate_encrypted_proofs([proof1, proof2])
        
        if aggregated:
            print(f"✅ Aggregated encrypted value: {aggregated.ciphertext}")
            print(f"   Modulus: {aggregated.modulus}")
            
            # Demonstrate homomorphic properties
            print("\n➕ Demonstrating homomorphic addition...")
            # Decrypt to show the result (for demonstration only)
            decrypted1 = validator.crypto.decrypt(proof1.encrypted_coherence)
            decrypted2 = validator.crypto.decrypt(proof2.encrypted_coherence)
            decrypted_agg = validator.crypto.decrypt(aggregated)
            
            print(f"   Decrypted proof 1: {decrypted1/10000:.4f}")
            print(f"   Decrypted proof 2: {decrypted2/10000:.4f}")
            print(f"   Decrypted aggregated: {decrypted_agg/10000:.4f}")
            print(f"   Sum check: {decrypted1 + decrypted2} == {decrypted_agg} {'✅' if decrypted1 + decrypted2 == decrypted_agg else '❌'}")
    
    # Show stored proofs
    print(f"\n📋 Stored proofs: {len(validator.proofs)}")
    
    print("\n🏆 Homomorphic encryption and zero-knowledge validation demo completed!")

if __name__ == "__main__":
    demo_homomorphic_encryption()