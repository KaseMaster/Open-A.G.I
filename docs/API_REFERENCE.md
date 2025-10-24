# AEGIS Framework - API Reference

**Version**: 2.0.0  
**Last Updated**: October 24, 2025

Complete API documentation for AEGIS Framework core components.

---

## Table of Contents

- [Security Layer](#security-layer)
  - [CryptoEngine](#cryptoengine)
- [Blockchain Layer](#blockchain-layer)
  - [HybridConsensus](#hybridconsensus)
  - [MerkleTree](#merkletree)
- [Networking Layer](#networking-layer)
  - [P2PNetworkManager](#p2pnetworkmanager)
- [Monitoring Layer](#monitoring-layer)
  - [AEGISMetricsCollector](#aegismetricscollector)

---

## Security Layer

### CryptoEngine

**Module**: `src.aegis.security.crypto_framework`

Cryptographic engine providing end-to-end encryption, digital signatures, and key management.

#### Constructor

```python
from src.aegis.security.crypto_framework import CryptoEngine, CryptoConfig, SecurityLevel

# Default configuration (HIGH security)
crypto = CryptoEngine()

# Custom configuration
config = CryptoConfig(
    security_level=SecurityLevel.PARANOID,
    key_rotation_interval=3600,  # 1 hour
    max_message_age=60,  # 1 minute
    pbkdf2_iterations=200000
)
crypto = CryptoEngine(config=config)
```

**Parameters**:
- `config` (CryptoConfig, optional): Cryptographic configuration. Defaults to HIGH security level.

#### Methods

##### `generate_node_identity(node_id: str = None) -> NodeIdentity`

Generate a new cryptographic identity for a node.

```python
identity = crypto.generate_node_identity("node_001")

# Access keys
signing_key = identity.signing_key  # Ed25519 private key
encryption_key = identity.encryption_key  # X25519 private key
public_signing_key = identity.public_signing_key
public_encryption_key = identity.public_encryption_key
```

**Returns**: `NodeIdentity` object with signing and encryption key pairs.

##### `sign_data(data: bytes) -> bytes`

Sign data with the node's signing key.

```python
crypto.identity = identity  # Set active identity
message = b"Important message"
signature = crypto.sign_data(message)
```

**Parameters**:
- `data` (bytes): Data to sign

**Returns**: Ed25519 signature (64 bytes)

**Raises**: `ValueError` if no identity is set

##### `verify_signature(data: bytes, signature: bytes, public_key: Ed25519PublicKey) -> bool`

Verify a signature against data and public key.

```python
is_valid = crypto.verify_signature(
    data=message,
    signature=signature,
    public_key=identity.public_signing_key
)
```

**Returns**: `True` if signature is valid, `False` otherwise

#### Example: Complete Workflow

```python
from src.aegis.security.crypto_framework import CryptoEngine

# Initialize crypto engine
crypto = CryptoEngine()

# Generate identity for this node
my_identity = crypto.generate_node_identity("my_node")
crypto.identity = my_identity

# Sign a message
message = b"Hello, AEGIS Network!"
signature = crypto.sign_data(message)

# Verify signature
is_valid = crypto.verify_signature(
    message,
    signature,
    my_identity.public_signing_key
)

print(f"Signature valid: {is_valid}")  # True

# Export public identity for sharing
public_data = my_identity.export_public_identity()
```

---

## Blockchain Layer

### HybridConsensus

**Module**: `src.aegis.blockchain.consensus_protocol`

Hybrid consensus combining Proof of Computation (PoC) and Practical Byzantine Fault Tolerance (PBFT).

#### Constructor

```python
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from cryptography.hazmat.primitives.asymmetric import ed25519

# Generate signing key
private_key = ed25519.Ed25519PrivateKey.generate()

# Initialize consensus
consensus = HybridConsensus(
    node_id="validator_001",
    private_key=private_key,
    network_manager=None  # Optional P2P network manager
)
```

**Parameters**:
- `node_id` (str): Unique identifier for this node
- `private_key` (Ed25519PrivateKey): Private key for signing consensus messages
- `network_manager` (optional): P2P network manager for message broadcasting

#### Attributes

- `node_id` (str): This node's identifier
- `poc` (ProofOfComputation): PoC algorithm instance
- `pbft` (PBFTConsensus): PBFT protocol instance
- `peers` (Dict): Connected peer nodes

#### Example: Basic Consensus

```python
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from cryptography.hazmat.primitives.asymmetric import ed25519

# Setup
private_key = ed25519.Ed25519PrivateKey.generate()
consensus = HybridConsensus("node_1", private_key)

# Check consensus status
print(f"Node ID: {consensus.node_id}")
print(f"PoC enabled: {consensus.poc is not None}")
print(f"PBFT enabled: {consensus.pbft is not None}")
```

---

### MerkleTree

**Module**: `src.aegis.blockchain.merkle_tree`

Native Merkle Tree implementation for transaction verification.

#### Constructor

```python
from src.aegis.blockchain.merkle_tree import MerkleTree

tree = MerkleTree(hash_type='sha256')
```

**Parameters**:
- `hash_type` (str, optional): Hash algorithm ('sha256', 'sha3_256', 'sha512', 'blake2b'). Default: 'sha256'

#### Methods

##### `add_leaf(data: bytes)`

Add a leaf node to the tree.

```python
tree.add_leaf(b"transaction_1")
tree.add_leaf(b"transaction_2")
tree.add_leaf(b"transaction_3")
```

##### `make_tree()`

Build the complete Merkle tree from leaves.

```python
tree.make_tree()
```

##### `get_merkle_root() -> bytes`

Get the root hash of the tree.

```python
root = tree.get_merkle_root()
print(f"Merkle root: {root.hex()}")
```

##### `get_proof(index: int) -> List[Tuple[bytes, str]]`

Generate inclusion proof for a leaf at given index.

```python
proof = tree.get_proof(1)  # Proof for transaction_2
```

**Returns**: List of (hash, direction) tuples

#### Example: Complete Merkle Tree Workflow

```python
from src.aegis.blockchain.merkle_tree import MerkleTree

# Create tree
tree = MerkleTree()

# Add transactions
transactions = [
    b"tx1: Alice -> Bob: 10 AEGIS",
    b"tx2: Bob -> Charlie: 5 AEGIS",
    b"tx3: Charlie -> Dave: 3 AEGIS",
    b"tx4: Dave -> Alice: 2 AEGIS"
]

for tx in transactions:
    tree.add_leaf(tx)

# Build tree
tree.make_tree()

# Get root
root = tree.get_merkle_root()
print(f"Merkle Root: {root.hex()}")

# Generate proof for tx2
proof = tree.get_proof(1)
print(f"Proof for tx2: {len(proof)} hashes")

# In production, you would send (tx, proof, root) to verify
```

---

## Networking Layer

### P2PNetworkManager

**Module**: `src.aegis.networking.p2p_network`

Decentralized peer-to-peer network manager with automatic discovery.

#### NodeType Enum

```python
from src.aegis.networking.p2p_network import NodeType

# Available node types
NodeType.BOOTSTRAP  # Bootstrap node for network entry
NodeType.FULL       # Full node with complete data
NodeType.LIGHT      # Light node with minimal data
NodeType.VALIDATOR  # Validator node for consensus
NodeType.STORAGE    # Storage node for data persistence
```

#### Example: Working with NodeTypes

```python
from src.aegis.networking.p2p_network import NodeType

# Define node configuration
node_config = {
    "type": NodeType.VALIDATOR,
    "host": "0.0.0.0",
    "port": 8080
}

print(f"Node type: {node_config['type'].value}")  # "validator"

# Check node capabilities
if node_config['type'] == NodeType.VALIDATOR:
    print("This node participates in consensus")
```

---

## Monitoring Layer

### AEGISMetricsCollector

**Module**: `src.aegis.monitoring.metrics_collector`

Comprehensive metrics collection system for monitoring AEGIS nodes.

#### Constructor

```python
from src.aegis.monitoring.metrics_collector import AEGISMetricsCollector

collector = AEGISMetricsCollector()
```

#### Example: Basic Metrics Collection

```python
from src.aegis.monitoring.metrics_collector import AEGISMetricsCollector

# Initialize collector
collector = AEGISMetricsCollector()

# Collector automatically gathers system metrics
# Access via the collector instance
print(f"Metrics collector initialized: {collector is not None}")
```

---

## Complete Integration Example

Here's a complete example integrating multiple AEGIS components:

```python
#!/usr/bin/env python3
"""
Complete AEGIS Framework Integration Example
"""

import asyncio
from src.aegis.security.crypto_framework import CryptoEngine, SecurityLevel
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from src.aegis.blockchain.merkle_tree import MerkleTree
from src.aegis.networking.p2p_network import NodeType
from src.aegis.monitoring.metrics_collector import AEGISMetricsCollector
from cryptography.hazmat.primitives.asymmetric import ed25519


async def main():
    print("🚀 Initializing AEGIS Node...\n")
    
    # 1. Initialize Crypto Engine
    print("1️⃣  Setting up cryptography...")
    crypto = CryptoEngine()
    identity = crypto.generate_node_identity("aegis_node_001")
    crypto.identity = identity
    print(f"   ✅ Node identity created: {identity.node_id}\n")
    
    # 2. Initialize Consensus
    print("2️⃣  Setting up consensus...")
    private_key = ed25519.Ed25519PrivateKey.generate()
    consensus = HybridConsensus(
        node_id="aegis_node_001",
        private_key=private_key
    )
    print(f"   ✅ Hybrid consensus initialized\n")
    
    # 3. Create Merkle Tree for transactions
    print("3️⃣  Building Merkle tree...")
    tree = MerkleTree()
    
    transactions = [
        b"tx1: Alice -> Bob: 10 AEGIS",
        b"tx2: Bob -> Charlie: 5 AEGIS",
        b"tx3: Charlie -> Dave: 3 AEGIS"
    ]
    
    for tx in transactions:
        tree.add_leaf(tx)
    
    tree.make_tree()
    root = tree.get_merkle_root()
    print(f"   ✅ Merkle root: {root.hex()[:16]}...\n")
    
    # 4. Sign the Merkle root
    print("4️⃣  Signing Merkle root...")
    signature = crypto.sign_data(root)
    print(f"   ✅ Signature: {signature.hex()[:16]}...\n")
    
    # 5. Initialize Metrics
    print("5️⃣  Starting metrics collection...")
    metrics = AEGISMetricsCollector()
    print(f"   ✅ Metrics collector running\n")
    
    # 6. Verify signature
    print("6️⃣  Verifying signature...")
    is_valid = crypto.verify_signature(
        root,
        signature,
        identity.public_signing_key
    )
    print(f"   ✅ Signature valid: {is_valid}\n")
    
    print("✨ AEGIS Node fully initialized and operational!")
    print(f"\nNode Configuration:")
    print(f"  - Node ID: {consensus.node_id}")
    print(f"  - Node Type: {NodeType.VALIDATOR.value}")
    print(f"  - Transactions: {len(transactions)}")
    print(f"  - Security Level: {crypto.config.security_level.value}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
🚀 Initializing AEGIS Node...

1️⃣  Setting up cryptography...
   ✅ Node identity created: aegis_node_001

2️⃣  Setting up consensus...
   ✅ Hybrid consensus initialized

3️⃣  Building Merkle tree...
   ✅ Merkle root: a1b2c3d4e5f6g7h8...

4️⃣  Signing Merkle root...
   ✅ Signature: 9f8e7d6c5b4a3210...

5️⃣  Starting metrics collection...
   ✅ Metrics collector running

6️⃣  Verifying signature...
   ✅ Signature valid: True

✨ AEGIS Node fully initialized and operational!

Node Configuration:
  - Node ID: aegis_node_001
  - Node Type: validator
  - Transactions: 3
  - Security Level: high
```

---

## Best Practices

### Security

1. **Always set an identity before signing**:
   ```python
   crypto.identity = identity  # Required!
   signature = crypto.sign_data(data)
   ```

2. **Use appropriate security levels**:
   - `STANDARD`: Development and testing
   - `HIGH`: Production (default)
   - `PARANOID`: High-security environments

3. **Rotate keys regularly**:
   ```python
   config = CryptoConfig(
       security_level=SecurityLevel.HIGH,
       key_rotation_interval=86400  # 24 hours
   )
   ```

### Performance

1. **Batch Merkle tree operations**:
   ```python
   # Good: Add all leaves then build once
   for tx in transactions:
       tree.add_leaf(tx)
   tree.make_tree()
   
   # Bad: Rebuilding after each leaf
   for tx in transactions:
       tree.add_leaf(tx)
       tree.make_tree()  # Inefficient!
   ```

2. **Reuse crypto engine instances**:
   ```python
   # Good: Single instance
   crypto = CryptoEngine()
   
   # Bad: Creating multiple instances
   for _ in range(100):
       crypto = CryptoEngine()  # Wasteful!
   ```

---

## Error Handling

```python
from src.aegis.security.crypto_framework import CryptoEngine

crypto = CryptoEngine()

try:
    # Attempt to sign without identity
    signature = crypto.sign_data(b"data")
except ValueError as e:
    print(f"Error: {e}")
    # Set identity first
    identity = crypto.generate_node_identity("node")
    crypto.identity = identity
    signature = crypto.sign_data(b"data")
```

---

## Additional Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [Quick Start Guide](../README.md#quick-start)
- [Examples](../examples/)
- [Benchmark Results](../benchmarks/benchmark_results.json)

---

**For more information or support**, contact: kasemaster@protonmail.com
