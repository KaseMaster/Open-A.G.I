"""
AEGIS Advanced Security Features
Zero-knowledge proofs and homomorphic encryption for enhanced privacy
"""

import asyncio
import time
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Try to import advanced crypto libraries
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, rsa, ed25519, x25519
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logging.warning("cryptography library not available, some features will be disabled")

# Try to import numpy for homomorphic operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
    logging.warning("numpy not available, homomorphic encryption will be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityFeature(Enum):
    """Advanced security features"""
    ZERO_KNOWLEDGE_PROOF = "zkp"
    HOMOMORPHIC_ENCRYPTION = "homomorphic"
    SECURE_MULTI_PARTY_COMPUTATION = "smc"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


@dataclass
class ZKProof:
    """Zero-knowledge proof data structure"""
    proof_id: str
    statement: str
    commitment: bytes
    challenge: bytes
    response: bytes
    timestamp: float
    verifier: str


@dataclass
class HomomorphicEncryptedValue:
    """Homomorphically encrypted value"""
    ciphertext: bytes
    public_key: bytes
    encryption_nonce: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZeroKnowledgeProver:
    """Zero-knowledge proof system for AEGIS"""
    
    def __init__(self, security_bits: int = 128):
        self.security_bits = security_bits
        self.proofs: Dict[str, ZKProof] = {}
        
    def generate_proof(self, secret: bytes, statement: str, verifier_id: str) -> ZKProof:
        """Generate a zero-knowledge proof"""
        proof_id = hashlib.sha256(f"{secret.hex()}{statement}{verifier_id}{time.time()}".encode()).hexdigest()[:16]
        
        # Simple commitment scheme (in practice, would use more sophisticated ZKP)
        commitment = hashlib.sha256(secret + statement.encode()).digest()
        
        # Generate challenge
        challenge = secrets.token_bytes(32)
        
        # Generate response (simplified - real ZKP would be more complex)
        response = hashlib.sha256(secret + challenge).digest()
        
        proof = ZKProof(
            proof_id=proof_id,
            statement=statement,
            commitment=commitment,
            challenge=challenge,
            response=response,
            timestamp=time.time(),
            verifier=verifier_id
        )
        
        self.proofs[proof_id] = proof
        return proof
    
    def verify_proof(self, proof: ZKProof, public_statement: str) -> bool:
        """Verify a zero-knowledge proof"""
        try:
            # Check if proof exists
            # Note: In a real implementation, we'd verify the cryptographic proof
            # For this demo, we'll do a simplified verification
            
            # Check statement matches
            if proof.statement != public_statement:
                return False
            
            # Check timestamp (not too old)
            if time.time() - proof.timestamp > 300:  # 5 minutes
                return False
            
            # In a real ZKP system, we'd verify:
            # 1. The commitment was computed correctly
            # 2. The response was computed correctly from the challenge
            # 3. The challenge was properly generated
            
            # For demo purposes, we'll accept the proof if it exists and is recent
            return proof.proof_id in self.proofs and self.proofs[proof.proof_id] == proof
            
        except Exception as e:
            logger.error(f"Error verifying ZK proof: {e}")
            return False
    
    def create_range_proof(self, value: int, min_val: int, max_val: int, verifier_id: str) -> ZKProof:
        """Create a range proof that value is in [min_val, max_val]"""
        # Convert value to bytes
        value_bytes = value.to_bytes(8, byteorder='big')
        
        # Create statement
        statement = f"range_proof:{min_val}:{max_val}"
        
        # Generate proof
        return self.generate_proof(value_bytes, statement, verifier_id)
    
    def verify_range_proof(self, proof: ZKProof, min_val: int, max_val: int) -> bool:
        """Verify a range proof"""
        statement = f"range_proof:{min_val}:{max_val}"
        return self.verify_proof(proof, statement)


class HomomorphicEncryption:
    """Homomorphic encryption system for privacy-preserving computations"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair for homomorphic encryption"""
        if HAS_CRYPTOGRAPHY:
            try:
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=self.key_size
                )
                self.public_key = self.private_key.public_key()
            except Exception as e:
                logger.error(f"Error generating RSA keys: {e}")
    
    def encrypt(self, value: int, metadata: Optional[Dict[str, Any]] = None) -> HomomorphicEncryptedValue:
        """Encrypt an integer value"""
        if not HAS_CRYPTOGRAPHY or not self.public_key:
            # Fallback for when cryptography is not available
            nonce = secrets.token_bytes(12)
            # Simple XOR encryption for demo purposes
            value_bytes = value.to_bytes(8, byteorder='big')
            ciphertext = bytes(a ^ b for a, b in zip(value_bytes, nonce * 2))
            
            return HomomorphicEncryptedValue(
                ciphertext=ciphertext,
                public_key=b'demo_key',
                encryption_nonce=nonce,
                metadata=metadata or {}
            )
        
        try:
            # Convert integer to bytes
            value_bytes = value.to_bytes((value.bit_length() + 7) // 8, byteorder='big') or b'\x00'
            
            # Generate nonce
            nonce = secrets.token_bytes(12)
            
            # Encrypt with RSA-OAEP (not truly homomorphic, but for demo)
            from cryptography.hazmat.primitives.asymmetric import padding
            ciphertext = self.public_key.encrypt(
                value_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Export public key
            public_key_bytes = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return HomomorphicEncryptedValue(
                ciphertext=ciphertext,
                public_key=public_key_bytes,
                encryption_nonce=nonce,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error encrypting value: {e}")
            raise
    
    def decrypt(self, encrypted_value: HomomorphicEncryptedValue) -> int:
        """Decrypt an encrypted value"""
        if not HAS_CRYPTOGRAPHY or not self.private_key:
            # Fallback decryption
            value_bytes = bytes(a ^ b for a, b in zip(encrypted_value.ciphertext, encrypted_value.encryption_nonce * 2))
            return int.from_bytes(value_bytes, byteorder='big')
        
        try:
            # Decrypt with RSA-OAEP
            from cryptography.hazmat.primitives.asymmetric import padding
            decrypted_bytes = self.private_key.decrypt(
                encrypted_value.ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return int.from_bytes(decrypted_bytes, byteorder='big')
            
        except Exception as e:
            logger.error(f"Error decrypting value: {e}")
            raise
    
    def add_encrypted(self, a: HomomorphicEncryptedValue, b: HomomorphicEncryptedValue) -> HomomorphicEncryptedValue:
        """Add two encrypted values (homomorphic addition)"""
        # Note: RSA is not additively homomorphic. This is a simplified implementation.
        # In practice, you would use Paillier or other additively homomorphic schemes.
        
        if HAS_NUMPY:
            # Convert to integers and add
            a_val = self.decrypt(a)
            b_val = self.decrypt(b)
            result_val = a_val + b_val
            
            # Re-encrypt result
            return self.encrypt(result_val, {
                "operation": "add",
                "operands": [a.metadata, b.metadata]
            })
        else:
            # Fallback without numpy
            a_val = self.decrypt(a)
            b_val = self.decrypt(b)
            result_val = a_val + b_val
            return self.encrypt(result_val, {
                "operation": "add",
                "operands": [a.metadata, b.metadata]
            })
    
    def multiply_encrypted(self, encrypted_value: HomomorphicEncryptedValue, scalar: int) -> HomomorphicEncryptedValue:
        """Multiply encrypted value by scalar (homomorphic multiplication)"""
        # Note: RSA is not multiplicatively homomorphic in this form.
        # This is a simplified implementation for demonstration.
        
        decrypted_value = self.decrypt(encrypted_value)
        result_value = decrypted_value * scalar
        
        return self.encrypt(result_value, {
            "operation": "multiply",
            "scalar": scalar,
            "operand": encrypted_value.metadata
        })


class SecureMultiPartyComputation:
    """Secure multi-party computation protocols"""
    
    def __init__(self):
        self.parties = set()
        self.shared_secrets = {}
        self.computation_results = {}
    
    def add_party(self, party_id: str):
        """Add a party to the computation"""
        self.parties.add(party_id)
        self.shared_secrets[party_id] = secrets.token_bytes(32)
    
    def generate_shares(self, secret: int, threshold: int) -> Dict[str, Tuple[int, int]]:
        """Generate secret shares using Shamir's Secret Sharing"""
        if not HAS_NUMPY:
            logger.warning("numpy not available, using simplified sharing")
            # Simple 2-out-of-2 sharing for demo
            share1 = secrets.randbits(64)
            share2 = secret - share1
            return {
                list(self.parties)[0]: (1, share1),
                list(self.parties)[1]: (2, share2)
            } if len(self.parties) >= 2 else {}
        
        # Shamir's Secret Sharing implementation
        try:
            # Generate polynomial coefficients
            coefficients = [secret] + [secrets.randbelow(2**64) for _ in range(threshold - 1)]
            
            shares = {}
            for i, party_id in enumerate(list(self.parties)[:threshold]):
                x = i + 1
                y = sum(coef * (x ** j) for j, coef in enumerate(coefficients))
                shares[party_id] = (x, y)
            
            return shares
        except Exception as e:
            logger.error(f"Error generating shares: {e}")
            return {}
    
    def reconstruct_secret(self, shares: Dict[str, Tuple[int, int]]) -> int:
        """Reconstruct secret from shares using Lagrange interpolation"""
        if not HAS_NUMPY or len(shares) < 2:
            # Simple reconstruction for 2 shares
            shares_list = list(shares.values())
            if len(shares_list) >= 2:
                return shares_list[0][1] + shares_list[1][1]
            return 0
        
        try:
            # Lagrange interpolation
            x_values = [share[0] for share in shares.values()]
            y_values = [share[1] for share in shares.values()]
            
            # Reconstruct using Lagrange basis polynomials
            secret = 0
            for i, (xi, yi) in enumerate(zip(x_values, y_values)):
                numerator = 1
                denominator = 1
                for j, xj in enumerate(x_values):
                    if i != j:
                        numerator *= -xj
                        denominator *= (xi - xj)
                if denominator != 0:
                    lagrange_coefficient = numerator // denominator
                    secret += yi * lagrange_coefficient
            
            return secret
        except Exception as e:
            logger.error(f"Error reconstructing secret: {e}")
            return 0


class DifferentialPrivacy:
    """Differential privacy mechanisms for data protection"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy"""
        if not HAS_NUMPY:
            # Simple noise generation
            import random
            scale = sensitivity / self.epsilon
            # Approximate Laplace noise using uniform distribution
            noise = random.uniform(-scale, scale)
            return value + noise
        
        try:
            # Generate Laplace noise
            scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, scale)
            return value + noise
        except Exception as e:
            logger.error(f"Error adding Laplace noise: {e}")
            return value
    
    def add_gaussian_noise(self, value: float, sensitivity: float) -> float:
        """Add Gaussian noise for differential privacy"""
        if not HAS_NUMPY:
            # Simple noise generation
            import random
            sigma = (sensitivity * (2 * np.log(1.25 / self.delta)) ** 0.5) / self.epsilon if np else 1.0
            noise = random.gauss(0, sigma)
            return value + noise
        
        try:
            # Calculate sigma for Gaussian mechanism
            sigma = (sensitivity * (2 * np.log(1.25 / self.delta)) ** 0.5) / self.epsilon
            noise = np.random.normal(0, sigma)
            return value + noise
        except Exception as e:
            logger.error(f"Error adding Gaussian noise: {e}")
            return value
    
    def privatize_count(self, count: int) -> float:
        """Privatize a count query"""
        return self.add_laplace_noise(float(count), sensitivity=1.0)
    
    def privatize_sum(self, sum_value: float, max_value: float) -> float:
        """Privatize a sum query"""
        return self.add_laplace_noise(sum_value, sensitivity=max_value)
    
    def privatize_mean(self, values: List[float]) -> float:
        """Privatize a mean query"""
        if not values:
            return 0.0
        
        mean_value = sum(values) / len(values)
        # Sensitivity for mean is max_value / n
        max_value = max(abs(v) for v in values) if values else 1.0
        sensitivity = max_value / len(values)
        return self.add_laplace_noise(mean_value, sensitivity=sensitivity)


class AdvancedSecurityManager:
    """Main manager for advanced security features"""
    
    def __init__(self):
        self.zk_prover = ZeroKnowledgeProver()
        self.homomorphic_encryption = HomomorphicEncryption()
        self.smc = SecureMultiPartyComputation()
        self.differential_privacy = DifferentialPrivacy()
        
        self.enabled_features = {
            SecurityFeature.ZERO_KNOWLEDGE_PROOF: True,
            SecurityFeature.HOMOMORPHIC_ENCRYPTION: True,
            SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION: True,
            SecurityFeature.DIFFERENTIAL_PRIVACY: True
        }
    
    def create_zk_proof(self, secret: bytes, statement: str, verifier_id: str) -> ZKProof:
        """Create a zero-knowledge proof"""
        if not self.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF]:
            raise RuntimeError("Zero-knowledge proofs are disabled")
        
        return self.zk_prover.generate_proof(secret, statement, verifier_id)
    
    def verify_zk_proof(self, proof: ZKProof, public_statement: str) -> bool:
        """Verify a zero-knowledge proof"""
        if not self.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF]:
            return False
        
        return self.zk_prover.verify_proof(proof, public_statement)
    
    def encrypt_value(self, value: int, metadata: Optional[Dict[str, Any]] = None) -> HomomorphicEncryptedValue:
        """Encrypt a value"""
        if not self.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION]:
            raise RuntimeError("Homomorphic encryption is disabled")
        
        return self.homomorphic_encryption.encrypt(value, metadata)
    
    def decrypt_value(self, encrypted_value: HomomorphicEncryptedValue) -> int:
        """Decrypt a value"""
        if not self.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION]:
            raise RuntimeError("Homomorphic encryption is disabled")
        
        return self.homomorphic_encryption.decrypt(encrypted_value)
    
    def add_encrypted_values(self, a: HomomorphicEncryptedValue, b: HomomorphicEncryptedValue) -> HomomorphicEncryptedValue:
        """Add two encrypted values"""
        if not self.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION]:
            raise RuntimeError("Homomorphic encryption is disabled")
        
        return self.homomorphic_encryption.add_encrypted(a, b)
    
    def multiply_encrypted_by_scalar(self, encrypted_value: HomomorphicEncryptedValue, scalar: int) -> HomomorphicEncryptedValue:
        """Multiply encrypted value by scalar"""
        if not self.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION]:
            raise RuntimeError("Homomorphic encryption is disabled")
        
        return self.homomorphic_encryption.multiply_encrypted(encrypted_value, scalar)
    
    def add_party_to_smc(self, party_id: str):
        """Add party to secure multi-party computation"""
        if not self.enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION]:
            raise RuntimeError("Secure multi-party computation is disabled")
        
        self.smc.add_party(party_id)
    
    def generate_secret_shares(self, secret: int, threshold: int) -> Dict[str, Tuple[int, int]]:
        """Generate secret shares"""
        if not self.enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION]:
            raise RuntimeError("Secure multi-party computation is disabled")
        
        return self.smc.generate_shares(secret, threshold)
    
    def reconstruct_secret_from_shares(self, shares: Dict[str, Tuple[int, int]]) -> int:
        """Reconstruct secret from shares"""
        if not self.enabled_features[SecurityFeature.SECURE_MULTI_PARTY_COMPUTATION]:
            raise RuntimeError("Secure multi-party computation is disabled")
        
        return self.smc.reconstruct_secret(shares)
    
    def privatize_data(self, data: Any, query_type: str = "count", **kwargs) -> Any:
        """Apply differential privacy to data"""
        if not self.enabled_features[SecurityFeature.DIFFERENTIAL_PRIVACY]:
            raise RuntimeError("Differential privacy is disabled")
        
        if query_type == "count":
            return self.differential_privacy.privatize_count(int(data))
        elif query_type == "sum":
            max_value = kwargs.get("max_value", 1.0)
            return self.differential_privacy.privatize_sum(float(data), max_value)
        elif query_type == "mean":
            if isinstance(data, list):
                return self.differential_privacy.privatize_mean([float(x) for x in data])
            else:
                return float(data)
        else:
            return data
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "enabled_features": {feature.value: enabled for feature, enabled in self.enabled_features.items()},
            "zk_proofs_generated": len(self.zk_prover.proofs),
            "parties_in_smc": len(self.smc.parties),
            "privacy_parameters": {
                "epsilon": self.differential_privacy.epsilon,
                "delta": self.differential_privacy.delta
            }
        }
