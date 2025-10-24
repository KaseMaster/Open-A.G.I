# AEGIS Framework - Advanced Security Documentation

## Table of Contents
1. [Zero-Knowledge Proofs](#zero-knowledge-proofs)
2. [Homomorphic Encryption](#homomorphic-encryption)
3. [Secure Multi-Party Computation](#secure-multi-party-computation)
4. [Differential Privacy](#differential-privacy)
5. [Security Manager](#security-manager)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)

---

## Zero-Knowledge Proofs

Zero-knowledge proofs (ZKPs) allow one party to prove to another that they know a value without revealing the value itself.

### Key Features

1. **Proof Generation**: Create proofs without revealing secrets
2. **Proof Verification**: Verify proofs without learning the underlying data
3. **Range Proofs**: Prove values are within specific ranges
4. **Statement Verification**: Verify complex mathematical statements

### Usage Example

```python
from aegis.security.advanced_crypto import AdvancedSecurityManager

# Create security manager
security = AdvancedSecurityManager()

# Generate a zero-knowledge proof
secret = b"my_secret_data"
proof = security.create_zk_proof(
    secret=secret,
    statement="I know the secret",
    verifier_id="verifier_001"
)

# Verify the proof
is_valid = security.verify_zk_proof(proof, "I know the secret")
print(f"Proof valid: {is_valid}")
```

### Range Proofs

```python
# Create a range proof
value = 42
proof = security.zk_prover.create_range_proof(
    value=value,
    min_val=0,
    max_val=100,
    verifier_id="verifier_001"
)

# Verify the range proof
is_valid = security.zk_prover.verify_range_proof(proof, 0, 100)
print(f"Range proof valid: {is_valid}")
```

### Configuration

```python
# Default configuration
zkp_config = {
    "security_bits": 128,  # Security level in bits
    "proof_timeout": 300,  # Proof validity in seconds (5 minutes)
    "max_proofs_cached": 1000  # Maximum proofs to cache
}
```

---

## Homomorphic Encryption

Homomorphic encryption allows computations to be performed on encrypted data without decrypting it first.

### Key Features

1. **Additive Homomorphism**: Add encrypted values
2. **Multiplicative Homomorphism**: Multiply encrypted values by scalars
3. **Privacy-Preserving Computations**: Perform calculations on sensitive data
4. **Key Management**: Secure key generation and storage

### Usage Example

```python
# Encrypt values
value1 = 15
value2 = 25

encrypted1 = security.encrypt_value(value1, {"type": "salary", "employee": "001"})
encrypted2 = security.encrypt_value(value2, {"type": "bonus", "employee": "001"})

# Perform homomorphic addition
encrypted_sum = security.add_encrypted_values(encrypted1, encrypted2)

# Decrypt result
result = security.decrypt_value(encrypted_sum)
print(f"Encrypted sum: {result}")  # Output: 40
```

### Scalar Multiplication

```python
# Encrypt a value
encrypted_value = security.encrypt_value(10)

# Multiply by scalar
encrypted_result = security.multiply_encrypted_by_scalar(encrypted_value, 5)

# Decrypt result
result = security.decrypt_value(encrypted_result)
print(f"Encrypted multiplication: {result}")  # Output: 50
```

### Configuration

```python
# Homomorphic encryption configuration
homomorphic_config = {
    "key_size": 2048,  # RSA key size in bits
    "encryption_algorithm": "RSA-OAEP",  # Encryption algorithm
    "max_plaintext_bits": 64,  # Maximum plaintext size
    "nonce_size": 12  # Nonce size in bytes
}
```

---

## Secure Multi-Party Computation

Secure multi-party computation (SMC) enables multiple parties to jointly compute a function over their inputs while keeping those inputs private.

### Key Features

1. **Secret Sharing**: Distribute secrets among parties
2. **Shamir's Secret Sharing**: Mathematical secret sharing scheme
3. **Secure Reconstruction**: Reconstruct secrets from shares
4. **Party Management**: Add and manage computation parties

### Usage Example

```python
# Add parties to computation
security.add_party_to_smc("party_001")
security.add_party_to_smc("party_002")
security.add_party_to_smc("party_003")

# Generate secret shares
secret = 12345
shares = security.generate_secret_shares(secret, threshold=2)

print(f"Generated {len(shares)} shares")

# Reconstruct secret from shares
reconstructed = security.reconstruct_secret_from_shares(shares)
print(f"Reconstructed secret: {reconstructed}")  # Output: 12345
```

### Threshold Cryptography

```python
# Generate shares with specific threshold
shares = security.generate_secret_shares(secret=98765, threshold=3)

# Reconstruct with subset of shares
subset_shares = dict(list(shares.items())[:3])
reconstructed = security.reconstruct_secret_from_shares(subset_shares)
print(f"Reconstructed with threshold: {reconstructed}")
```

### Configuration

```python
# SMC configuration
smc_config = {
    "min_parties": 2,  # Minimum number of parties
    "max_parties": 100,  # Maximum number of parties
    "default_threshold": 2,  # Default threshold for secret sharing
    "share_validity": 3600  # Share validity in seconds (1 hour)
}
```

---

## Differential Privacy

Differential privacy adds mathematical guarantees to protect individual privacy in statistical databases.

### Key Features

1. **Laplace Mechanism**: Add Laplace noise for privacy
2. **Gaussian Mechanism**: Add Gaussian noise for privacy
3. **Query Privacy**: Protect count, sum, and mean queries
4. **Privacy Budget**: Control privacy-utility tradeoff

### Usage Example

```python
# Privatize a count query
true_count = 1000
private_count = security.privatize_data(true_count, query_type="count")
print(f"True count: {true_count}, Private count: {private_count}")

# Privatize a sum query
true_sum = 50000.0
private_sum = security.privatize_data(true_sum, query_type="sum", max_value=1000.0)
print(f"True sum: {true_sum}, Private sum: {private_sum}")

# Privatize a mean query
values = [10, 20, 30, 40, 50]
private_mean = security.privatize_data(values, query_type="mean")
print(f"True mean: {sum(values)/len(values)}, Private mean: {private_mean}")
```

### Privacy Parameters

```python
# Configure differential privacy
from aegis.security.advanced_crypto import DifferentialPrivacy

# Create with custom privacy parameters
dp = DifferentialPrivacy(epsilon=0.1, delta=1e-6)

# Higher epsilon = less privacy, more accuracy
# Lower epsilon = more privacy, less accuracy

# Add noise to sensitive data
sensitive_value = 42.0
noisy_value = dp.add_laplace_noise(sensitive_value, sensitivity=1.0)
print(f"Noisy value: {noisy_value}")
```

### Configuration

```python
# Differential privacy configuration
dp_config = {
    "epsilon": 1.0,  # Privacy parameter (lower = more private)
    "delta": 1e-5,   # Failure probability
    "max_sensitivity": 1.0,  # Maximum query sensitivity
    "noise_mechanism": "laplace"  # "laplace" or "gaussian"
}
```

---

## Security Manager

The `AdvancedSecurityManager` provides a unified interface for all advanced security features.

### Key Features

1. **Feature Management**: Enable/disable specific security features
2. **Performance Monitoring**: Track security operations performance
3. **Resource Management**: Efficient resource utilization
4. **Error Handling**: Comprehensive error handling

### Usage Example

```python
from aegis.security.advanced_crypto import AdvancedSecurityManager, SecurityFeature

# Create security manager
security = AdvancedSecurityManager()

# Check enabled features
stats = security.get_security_stats()
print("Enabled features:")
for feature, enabled in stats["enabled_features"].items():
    print(f"  {feature}: {enabled}")

# Disable specific features if needed
security.enabled_features[SecurityFeature.DIFFERENTIAL_PRIVACY] = False

# Perform secure operations
try:
    # Zero-knowledge proof
    proof = security.create_zk_proof(b"secret", "statement", "verifier")
    
    # Homomorphic encryption
    encrypted = security.encrypt_value(42)
    
    # SMC operations
    security.add_party_to_smc("party1")
    
    # Differential privacy
    private_data = security.privatize_data(100, "count")
    
except RuntimeError as e:
    print(f"Security feature disabled: {e}")
```

### Performance Monitoring

```python
# Get security statistics
stats = security.get_security_stats()
print("Security Statistics:")
print(f"  ZK Proofs Generated: {stats['zk_proofs_generated']}")
print(f"  SMC Parties: {stats['parties_in_smc']}")
print(f"  Privacy Epsilon: {stats['privacy_parameters']['epsilon']}")
```

---

## Integration Examples

### Federated Learning with Privacy

```python
import asyncio
from aegis.security.advanced_crypto import AdvancedSecurityManager
from aegis.ml.federated_learning import FederatedClient, FederatedConfig

async def privacy_preserving_fl():
    """Federated learning with differential privacy"""
    security = AdvancedSecurityManager()
    
    # Create federated learning config
    config = FederatedConfig()
    client = FederatedClient("client_001", model=None, config=config)
    
    # Simulate training data
    gradients = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Add differential privacy to gradients
    private_gradients = [
        security.privatize_data(grad, "sum", max_value=1.0)
        for grad in gradients
    ]
    
    print(f"Original gradients: {gradients}")
    print(f"Private gradients: {private_gradients}")
    
    return private_gradients

# Run example
asyncio.run(privacy_preserving_fl())
```

### Secure Data Aggregation

```python
def secure_data_aggregation():
    """Secure aggregation using homomorphic encryption"""
    security = AdvancedSecurityManager()
    
    # Simulate data from multiple parties
    party_data = [100, 200, 300, 400, 500]
    
    # Encrypt each party's data
    encrypted_data = [
        security.encrypt_value(value, {"party": f"party_{i:03d}"})
        for i, value in enumerate(party_data)
    ]
    
    # Homomorphically add all encrypted values
    result = encrypted_data[0]
    for encrypted_value in encrypted_data[1:]:
        result = security.add_encrypted_values(result, encrypted_value)
    
    # Decrypt final result
    total = security.decrypt_value(result)
    expected = sum(party_data)
    
    print(f"Secure sum: {total}")
    print(f"Expected sum: {expected}")
    print(f"Match: {total == expected}")

# Run example
secure_data_aggregation()
```

### Zero-Knowledge Authentication

```python
def zk_authentication():
    """Zero-knowledge authentication example"""
    security = AdvancedSecurityManager()
    
    # User's secret credential
    user_secret = b"user_password_123"
    user_id = "user_001"
    
    # Generate proof of knowledge
    proof = security.create_zk_proof(
        secret=user_secret,
        statement=f"authenticate:{user_id}",
        verifier_id="auth_server"
    )
    
    # Server verifies without learning the password
    is_authenticated = security.verify_zk_proof(
        proof=proof,
        public_statement=f"authenticate:{user_id}"
    )
    
    print(f"Authentication successful: {is_authenticated}")
    
    # Proof cannot be reused (timestamp-based expiration)
    time.sleep(301)  # Wait for proof to expire
    
    is_expired = security.verify_zk_proof(
        proof=proof,
        public_statement=f"authenticate:{user_id}"
    )
    
    print(f"Expired proof valid: {is_expired}")

# Run example
zk_authentication()
```

---

## Best Practices

### Security Configuration

1. **Feature Selection**
   ```python
   # Enable only required features
   security = AdvancedSecurityManager()
   security.enabled_features[SecurityFeature.ZERO_KNOWLEDGE_PROOF] = True
   security.enabled_features[SecurityFeature.HOMOMORPHIC_ENCRYPTION] = False
   ```

2. **Parameter Tuning**
   ```python
   # Adjust privacy parameters based on use case
   dp = DifferentialPrivacy(epsilon=0.01, delta=1e-9)  # High privacy
   dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)   # Balanced privacy
   ```

3. **Performance Considerations**
   ```python
   # Monitor performance impact
   stats = security.get_security_stats()
   if stats["zk_proofs_generated"] > 1000:
       print("Consider optimizing ZK proof generation")
   ```

### Error Handling

```python
try:
    # Secure operation
    result = security.encrypt_value(sensitive_data)
except RuntimeError as e:
    # Feature disabled
    print(f"Security feature not available: {e}")
except Exception as e:
    # Other errors
    print(f"Security operation failed: {e}")
```

### Resource Management

```python
# Clean up resources when done
def cleanup_security_resources():
    # In practice, this would clean up keys, connections, etc.
    pass
```

### Testing

```python
def test_security_features():
    """Test all security features"""
    security = AdvancedSecurityManager()
    
    # Test ZK proofs
    assert security.verify_zk_proof(
        security.create_zk_proof(b"test", "test", "verifier"),
        "test"
    )
    
    # Test homomorphic encryption
    encrypted = security.encrypt_value(42)
    decrypted = security.decrypt_value(encrypted)
    assert decrypted == 42
    
    print("All security tests passed!")

# Run tests
test_security_features()
```

---

## API Reference

### ZeroKnowledgeProver

#### Methods
- `generate_proof(secret, statement, verifier_id)` - Generate ZK proof
- `verify_proof(proof, public_statement)` - Verify ZK proof
- `create_range_proof(value, min_val, max_val, verifier_id)` - Create range proof
- `verify_range_proof(proof, min_val, max_val)` - Verify range proof

### HomomorphicEncryption

#### Methods
- `encrypt(value, metadata)` - Encrypt integer value
- `decrypt(encrypted_value)` - Decrypt value
- `add_encrypted(a, b)` - Add encrypted values
- `multiply_encrypted(encrypted_value, scalar)` - Multiply by scalar

### SecureMultiPartyComputation

#### Methods
- `add_party(party_id)` - Add party to computation
- `generate_shares(secret, threshold)` - Generate secret shares
- `reconstruct_secret(shares)` - Reconstruct secret

### DifferentialPrivacy

#### Methods
- `add_laplace_noise(value, sensitivity)` - Add Laplace noise
- `add_gaussian_noise(value, sensitivity)` - Add Gaussian noise
- `privatize_count(count)` - Privatize count query
- `privatize_sum(sum_value, max_value)` - Privatize sum query
- `privatize_mean(values)` - Privatize mean query

### AdvancedSecurityManager

#### Methods
- `create_zk_proof(secret, statement, verifier_id)` - Create ZK proof
- `verify_zk_proof(proof, public_statement)` - Verify ZK proof
- `encrypt_value(value, metadata)` - Encrypt value
- `decrypt_value(encrypted_value)` - Decrypt value
- `add_encrypted_values(a, b)` - Add encrypted values
- `multiply_encrypted_by_scalar(encrypted_value, scalar)` - Multiply by scalar
- `add_party_to_smc(party_id)` - Add SMC party
- `generate_secret_shares(secret, threshold)` - Generate shares
- `reconstruct_secret_from_shares(shares)` - Reconstruct secret
- `privatize_data(data, query_type, **kwargs)` - Apply differential privacy
- `get_security_stats()` - Get security statistics

---

## Troubleshooting

### Common Issues

1. **Performance Degradation**
   - Monitor feature usage statistics
   - Disable unused features
   - Optimize privacy parameters

2. **Memory Usage**
   - Clean up old proofs and shares
   - Use connection pooling
   - Monitor cache sizes

3. **Security Errors**
   - Check feature enablement
   - Verify parameter ranges
   - Review error messages

### Performance Tuning

1. **Zero-Knowledge Proofs**
   - Cache frequently used proofs
   - Optimize proof generation algorithms
   - Use appropriate security parameters

2. **Homomorphic Encryption**
   - Batch operations when possible
   - Use efficient key sizes
   - Monitor memory usage

3. **Differential Privacy**
   - Balance epsilon/delta parameters
   - Choose appropriate noise mechanisms
   - Consider query composition

---

## Changelog

### Version 2.2.0 (2025-10-24)
- Initial release of advanced security features
- Zero-knowledge proofs implementation
- Homomorphic encryption support
- Secure multi-party computation
- Differential privacy mechanisms
- Unified security manager interface
- Comprehensive documentation and examples
