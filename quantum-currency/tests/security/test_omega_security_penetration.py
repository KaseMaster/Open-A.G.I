#!/usr/bin/env python3
"""
Penetration Test Suite for Ω-Security Primitives
Simulates attacks, injection vectors, and entropy-based exploits.
"""

import sys
import os
import unittest
import hashlib
import time
import json
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Handle relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.security.omega_security import OmegaSecurityPrimitives, CoherenceLockedKey, ClientReputation

class TestOmegaSecurityPenetration(unittest.TestCase):
    """Penetration test suite for Ω-Security Primitives"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.security = OmegaSecurityPrimitives("test-security-network")
        
    def test_clk_manipulation_attack(self):
        """Test resistance to CLK manipulation attacks"""
        # Generate a legitimate CLK
        qp_hash = hashlib.sha256(b"legitimate_data").hexdigest()
        omega_vector = [1.0, 0.5, 0.2, 0.8, 0.3]
        clk = self.security.generate_coherence_locked_key(qp_hash, omega_vector, time_delay=1.5)
        
        # Try to manipulate the CLK with a different Ω vector (this should fail)
        manipulated_clk = CoherenceLockedKey(
            key_id=clk.key_id,  # Same ID
            encrypted_data_hash=clk.encrypted_data_hash,  # Same data hash
            omega_vector_hash="manipulated_omega_hash",  # Different omega hash
            time_delay=clk.time_delay,  # Same time delay
            created_at=clk.created_at,  # Same creation time
            expires_at=clk.expires_at,  # Same expiration
            metadata=clk.metadata  # Same metadata
        )
        
        # Validate both CLKs
        legitimate_valid = self.security.validate_coherence_locked_key(clk, omega_vector)
        # Use a different omega vector for validation of manipulated CLK
        different_omega = [2.0, 1.0, 0.4, 1.6, 0.6]  # Different values
        manipulated_valid = self.security.validate_coherence_locked_key(manipulated_clk, different_omega)
        
        # Legitimate CLK should be valid
        self.assertTrue(legitimate_valid, "Legitimate CLK should be valid")
        
        # Manipulated CLK should be invalid due to Ω vector mismatch
        self.assertFalse(manipulated_valid, "Manipulated CLK should be invalid")
        
    def test_omega_vector_tampering(self):
        """Test resistance to Ω-vector tampering"""
        # Generate CLK with original Ω-vector
        qp_hash = hashlib.sha256(b"test_data").hexdigest()
        original_omega = [1.0, 0.5, 0.2, 0.8, 0.3]
        clk = self.security.generate_coherence_locked_key(qp_hash, original_omega, time_delay=1.0)
        
        # Try to validate with tampered Ω-vector
        tampered_omega = [2.0, 1.0, 0.4, 1.6, 0.6]  # Doubled values
        tampered_valid = self.security.validate_coherence_locked_key(clk, tampered_omega)
        
        # Original should be valid
        original_valid = self.security.validate_coherence_locked_key(clk, original_omega)
        self.assertTrue(original_valid, "CLK should be valid with original Ω-vector")
        
        # Tampered should be invalid
        self.assertFalse(tampered_valid, "CLK should be invalid with tampered Ω-vector")
        
    def test_replay_attack_resistance(self):
        """Test resistance to replay attacks"""
        # Generate CLK
        qp_hash = hashlib.sha256(b"replay_test_data").hexdigest()
        omega_vector = [0.8, 0.3, 0.6, 0.4, 0.9]
        clk = self.security.generate_coherence_locked_key(qp_hash, omega_vector, time_delay=0.5)
        
        # Validate initially
        initial_valid = self.security.validate_coherence_locked_key(clk, omega_vector)
        self.assertTrue(initial_valid, "CLK should be valid initially")
        
        # Simulate time passing beyond expiration
        # Manually expire the CLK
        expired_clk = CoherenceLockedKey(
            key_id=clk.key_id,
            encrypted_data_hash=clk.encrypted_data_hash,
            omega_vector_hash=clk.omega_vector_hash,
            time_delay=clk.time_delay,
            created_at=clk.created_at,
            expires_at=time.time() - 3600,  # Expired 1 hour ago
            metadata=clk.metadata
        )
        
        # Try to validate expired CLK
        expired_valid = self.security.validate_coherence_locked_key(expired_clk, omega_vector)
        self.assertFalse(expired_valid, "Expired CLK should be invalid")
        
    def test_json_injection_in_metadata(self):
        """Test resistance to JSON injection in metadata"""
        # Create malicious metadata
        malicious_metadata = {
            "normal_field": "normal_value",
            "injected_script": "<script>alert('xss')</script>",
            "sql_injection": "'; DROP TABLE users; --",
            "command_injection": "$(rm -rf /)"
        }
        
        # Generate CLK with malicious metadata
        qp_hash = hashlib.sha256(b"safe_data").hexdigest()
        omega_vector = [0.5, 0.4, 0.3, 0.2, 0.1]
        clk = self.security.generate_coherence_locked_key(
            qp_hash, omega_vector, time_delay=1.0, metadata=malicious_metadata
        )
        
        # Verify CLK is still valid
        is_valid = self.security.validate_coherence_locked_key(clk, omega_vector)
        self.assertTrue(is_valid, "CLK should be valid even with 'malicious' metadata")
        
        # Verify metadata is stored correctly
        stored_metadata = self.security.clk_store[clk.key_id].metadata
        self.assertEqual(stored_metadata, malicious_metadata)
        
    def test_entropy_based_attack_simulation(self):
        """Simulate entropy-based attacks on the security system"""
        # Create client with very low coherence score (potential attack client)
        low_coherence_client = self.security.update_client_reputation(
            client_id="attack_client_001",
            psi_score=0.1,  # Very low coherence
            psy_balance=100.0,  # Lower token balance
            flx_balance=100.0  # Lower token balance
        )
        
        # Check that reputation is still low despite high token balances
        self.assertLess(low_coherence_client.reputation_score, 0.5)
        self.assertEqual(low_coherence_client.psi_score, 0.1)
        
        # Apply throttling
        allowed, params = self.security.apply_coherence_based_throttling("attack_client_001")
        
        # Should be heavily throttled due to low coherence
        self.assertTrue(allowed)  # Still allowed but with strict limits
        self.assertLessEqual(params["rate_limit"], 10)  # Very low rate limit
        self.assertIn("message", params)  # Should have warning message
        
    def test_hash_collision_resistance(self):
        """Test resistance to hash collision attacks"""
        # Generate multiple CLKs with different data
        test_data = [
            (b"data1", [1.0, 0.5, 0.2]),
            (b"data2", [0.5, 1.0, 0.3]),
            (b"data3", [0.2, 0.3, 1.0]),
        ]
        
        clks = []
        for data, omega in test_data:
            qp_hash = hashlib.sha256(data).hexdigest()
            clk = self.security.generate_coherence_locked_key(qp_hash, omega, time_delay=1.0)
            clks.append(clk)
        
        # Verify all CLKs have unique IDs
        clk_ids = [clk.key_id for clk in clks]
        unique_ids = set(clk_ids)
        
        # Should have same number of unique IDs as CLKs
        self.assertEqual(len(unique_ids), len(clks))
        
        # Validate all CLKs
        for i, (data, omega) in enumerate(test_data):
            qp_hash = hashlib.sha256(data).hexdigest()
            is_valid = self.security.validate_coherence_locked_key(clks[i], omega)
            self.assertTrue(is_valid, f"CLK {i} should be valid")

if __name__ == '__main__':
    unittest.main()