"""
Integration tests for AEGIS Advanced Security Features
Tests zero-knowledge proofs, homomorphic encryption, and security integration
"""

import pytest
import asyncio
import time
import random
from typing import Dict, List, Any
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.advanced_crypto import (
    AdvancedSecurityManager,
    ZeroKnowledgeProver,
    HomomorphicEncryption,
    SecureMultiPartyComputation,
    DifferentialPrivacy,
    SecurityFeature
)
from src.aegis.security.middleware import SecurityMiddleware


class TestAdvancedSecurityIntegration:
    
    @pytest.fixture
    def security_manager(self):
        return AdvancedSecurityManager()
    
    @pytest.fixture
    def security_middleware(self):
        return SecurityMiddleware()
    
    def test_zero_knowledge_proof_integration(self, security_manager):
        """Test zero-knowledge proof generation and verification"""
        # Test basic ZK proof
        secret = b"test_secret_123"
        statement = "knowledge_of_secret"
        verifier_id = "test_verifier"
        
        # Generate proof
        proof = security_manager.create_zk_proof(secret, statement, verifier_id)
        
        # Verify proof
        is_valid = security_manager.verify_zk_proof(proof, statement)
        assert is_valid, "ZK proof should be valid"
        
        # Test invalid proof
        invalid_proof = security_manager.create_zk_proof(
            b"different_secret", 
            "different_statement", 
            verifier_id
        )
        is_invalid = security_manager.verify_zk_proof(invalid_proof, statement)
        assert not is_invalid, "Invalid ZK proof should be rejected"
        
        # Test range proof
        value = 42
        range_proof = security_manager.zk_prover.create_range_proof(
            value=value,
            min_val=0,
            max_val=100,
            verifier_id=verifier_id
        )
        
        # Verify range proof
        range_valid = security_manager.zk_prover.verify_range_proof(
            range_proof, 0, 100
        )
        assert range_valid, "Range proof should be valid"
        
        # Test out-of-range proof
        out_of_range_valid = security_manager.zk_prover.verify_range_proof(
            range_proof, 0, 40  # Value 42 is not in [0,40]
        )
        assert not out_of_range_valid, "Out-of-range proof should be invalid"
    
    def test_homomorphic_encryption_integration(self, security_manager):
        """Test homomorphic encryption operations"""
        # Test basic encryption/decryption
        original_value = 123
        encrypted = security_manager.encrypt_value(
            original_value, 
            {"test": "metadata"}
        )
        decrypted = security_manager.decrypt_value(encrypted)
        assert decrypted == original_value, "Encryption/decryption should preserve value"
        
        # Test homomorphic addition
        value1 = 15
        value2 = 25
        
        encrypted1 = security_manager.encrypt_value(value1)
        encrypted2 = security_manager.encrypt_value(value2)
        
        encrypted_sum = security_manager.add_encrypted_values(encrypted1, encrypted2)
        decrypted_sum = security_manager.decrypt_value(encrypted_sum)
        
        assert decrypted_sum == (value1 + value2), "Homomorphic addition should work"
        
        # Test homomorphic multiplication
        scalar = 3
        encrypted_value = security_manager.encrypt_value(10)
        encrypted_product = security_manager.multiply_encrypted_by_scalar(
            encrypted_value, scalar
        )
        decrypted_product = security_manager.decrypt_value(encrypted_product)
        
        assert decrypted_product == (10 * scalar), "Homomorphic multiplication should work"
    
    def test_secure_multi_party_computation_integration(self, security_manager):
        """Test secure multi-party computation"""
        # Add parties
        parties = ["party_001", "party_002", "party_003"]
        for party in parties:
            security_manager.add_party_to_smc(party)
        
        # Test secret sharing
        secret = 98765
        shares = security_manager.generate_secret_shares(secret, threshold=2)
        
        assert len(shares) >= 2, "Should generate at least threshold shares"
        
        # Test secret reconstruction
        reconstructed = security_manager.reconstruct_secret_from_shares(shares)
        assert reconstructed == secret, "Secret reconstruction should work"
        
        # Test reconstruction with subset
        subset_shares = dict(list(shares.items())[:2])
        subset_reconstructed = security_manager.reconstruct_secret_from_shares(subset_shares)
        assert subset_reconstructed == secret, "Subset reconstruction should work"
    
    def test_differential_privacy_integration(self, security_manager):
        """Test differential privacy mechanisms"""
        # Test count query
        true_count = 1000
        private_count = security_manager.privatize_data(true_count, "count")
        assert isinstance(private_count, float), "Private count should be float"
        
        # Test sum query
        true_sum = 50000.0
        private_sum = security_manager.privatize_data(
            true_sum, "sum", max_value=1000.0
        )
        assert isinstance(private_sum, float), "Private sum should be float"
        
        # Test mean query
        values = [10, 20, 30, 40, 50]
        private_mean = security_manager.privatize_data(values, "mean")
        assert isinstance(private_mean, float), "Private mean should be float"
        
        # Test with different epsilon values
        dp_high_privacy = DifferentialPrivacy(epsilon=0.01)
        dp_low_privacy = DifferentialPrivacy(epsilon=1.0)
        
        high_private = dp_high_privacy.add_laplace_noise(42.0, 1.0)
        low_private = dp_low_privacy.add_laplace_noise(42.0, 1.0)
        
        # High privacy should add more noise
        assert abs(high_private - 42.0) >= abs(low_private - 42.0), \
            "High privacy should add more noise"
    
    def test_security_manager_feature_management(self, security_manager):
        """Test security manager feature enablement"""
        # Test all features enabled by default
        for feature in SecurityFeature:
            assert security_manager.enabled_features[feature], \
                f"Feature {feature} should be enabled by default"
        
        # Test disabling features
        security_manager.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF] = False
        
        # Test that disabled features raise errors
        with pytest.raises(RuntimeError):
            security_manager.create_zk_proof(b"secret", "statement", "verifier")
        
        # Test that other features still work
        encrypted = security_manager.encrypt_value(42)
        decrypted = security_manager.decrypt_value(encrypted)
        assert decrypted == 42, "Other features should still work when one is disabled"
        
        # Re-enable feature
        security_manager.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF] = True
        proof = security_manager.create_zk_proof(b"secret", "statement", "verifier")
        assert proof is not None, "Feature should work after re-enabling"
    
    def test_security_integration_with_middleware(self, security_manager, security_middleware):
        """Test integration between advanced security and middleware"""
        # Test rate limiting with security operations
        client_id = "test_client"
        endpoint = "/security/zkp"
        
        # Perform multiple security operations with rate limiting
        for i in range(50):  # Within rate limit
            allowed, message = security_middleware.check_request_security(
                client_id=client_id,
                endpoint=endpoint,
                params={"operation": f"zkp_{i}"}
            )
            
            assert allowed, f"Request {i} should be allowed: {message}"
            
            # Perform security operation
            secret = f"secret_{i}".encode()
            proof = security_manager.create_zk_proof(
                secret, f"statement_{i}", "verifier"
            )
            assert proof is not None, f"ZK proof {i} should be generated"
        
        # Test that rate limiting still works
        for i in range(100):  # Exceed rate limit
            allowed, message = security_middleware.check_request_security(
                client_id=client_id,
                endpoint=endpoint,
                params={"operation": f"excess_{i}"}
            )
            
            if not allowed:
                # Rate limiting should kick in
                assert "rate limit" in message.lower() or "blocked" in message.lower()
                break
        
        # Test input validation with security data
        test_inputs = [
            ("valid_secret_123", "alphanumeric"),
            ("test@example.com", "email"),
            ("abc123def456", "hex"),
        ]
        
        for value, field_type in test_inputs:
            is_valid, error = security_middleware.input_validator.validate_string(
                value, field_type
            )
            assert is_valid, f"Valid input {value} should pass validation: {error}"
            
            # Use validated input in security operations
            if field_type == "alphanumeric":
                proof = security_manager.create_zk_proof(
                    value.encode(), "validated_statement", "verifier"
                )
                assert proof is not None, "Proof should be generated with validated input"
    
    def test_performance_security_integration(self, security_manager):
        """Test performance of integrated security operations"""
        # Test ZK proof performance
        start_time = time.time()
        for i in range(100):
            proof = security_manager.create_zk_proof(
                f"secret_{i}".encode(), f"statement_{i}", "verifier"
            )
            assert security_manager.verify_zk_proof(proof, f"statement_{i}")
        zk_time = time.time() - start_time
        
        # Test homomorphic encryption performance
        start_time = time.time()
        for i in range(100):
            encrypted = security_manager.encrypt_value(i)
            decrypted = security_manager.decrypt_value(encrypted)
            assert decrypted == i
        he_time = time.time() - start_time
        
        # Test SMC performance
        start_time = time.time()
        security_manager.add_party_to_smc("party1")
        security_manager.add_party_to_smc("party2")
        for i in range(50):
            shares = security_manager.generate_secret_shares(i, 2)
            reconstructed = security_manager.reconstruct_secret_from_shares(shares)
            assert reconstructed == i
        smc_time = time.time() - start_time
        
        # Log performance metrics
        print(f"ZK Proof Performance: {zk_time:.2f}s for 100 operations")
        print(f"Homomorphic Encryption Performance: {he_time:.2f}s for 100 operations")
        print(f"SMC Performance: {smc_time:.2f}s for 50 operations")
        
        # All operations should complete in reasonable time
        assert zk_time < 10.0, "ZK proofs should be fast"
        assert he_time < 5.0, "Homomorphic encryption should be fast"
        assert smc_time < 5.0, "SMC should be fast"
    
    def test_security_statistics_and_monitoring(self, security_manager):
        """Test security statistics and monitoring"""
        # Perform various security operations
        for i in range(10):
            # ZK proofs
            proof = security_manager.create_zk_proof(
                f"secret_{i}".encode(), f"statement_{i}", "verifier"
            )
            security_manager.verify_zk_proof(proof, f"statement_{i}")
            
            # Homomorphic encryption
            encrypted = security_manager.encrypt_value(i)
            security_manager.decrypt_value(encrypted)
            
            # SMC
            security_manager.add_party_to_smc(f"party_{i:03d}")
        
        # Get security statistics
        stats = security_manager.get_security_stats()
        
        # Verify statistics
        assert "enabled_features" in stats
        assert "zk_proofs_generated" in stats
        assert "parties_in_smc" in stats
        assert "privacy_parameters" in stats
        
        # Check specific values
        assert stats["zk_proofs_generated"] >= 10
        assert stats["parties_in_smc"] >= 10
        
        # Check feature enablement
        enabled_features = stats["enabled_features"]
        assert enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF.value]
        assert enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION.value]
        assert enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION.value]
        assert enabled_features[SecurityFeature.DIFFERENTIAL_PRIVACY.value]
        
        # Check privacy parameters
        privacy_params = stats["privacy_parameters"]
        assert "epsilon" in privacy_params
        assert "delta" in privacy_params
        assert privacy_params["epsilon"] > 0
        assert privacy_params["delta"] > 0
