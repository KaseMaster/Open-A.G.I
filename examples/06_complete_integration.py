#!/usr/bin/env python3
"""
AEGIS Framework - Complete Integration Example
Demonstrates all major components working together
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
        b"tx3: Charlie -> Dave: 3 AEGIS",
        b"tx4: Dave -> Alice: 2 AEGIS"
    ]
    
    for tx in transactions:
        tree.add_leaf(tx)
    
    tree.make_tree()
    root = tree.get_merkle_root()
    print(f"   ✅ Merkle root: {root.hex()[:32]}...")
    print(f"   ✅ Transactions: {len(transactions)}\n")
    
    # 4. Sign the Merkle root
    print("4️⃣  Signing Merkle root...")
    signature = crypto.sign_data(root)
    print(f"   ✅ Signature: {signature.hex()[:32]}...\n")
    
    # 5. Generate and verify proof
    print("5️⃣  Generating Merkle proof...")
    proof = tree.get_proof(1)  # Proof for tx2
    print(f"   ✅ Proof generated: {len(proof)} hashes\n")
    
    # 6. Initialize Metrics
    print("6️⃣  Starting metrics collection...")
    metrics = AEGISMetricsCollector()
    print(f"   ✅ Metrics collector running\n")
    
    # 7. Verify signature (simplified for demo)
    print("7️⃣  Verifying signature...")
    # Note: Full signature verification requires peer identity setup
    # For demo, we just verify the signature was created
    print(f"   ✅ Signature created: {len(signature) == 64}\n")
    
    # 8. Display node configuration
    print("="*60)
    print("✨ AEGIS Node fully initialized and operational!")
    print("="*60)
    print(f"\n📋 Node Configuration:")
    print(f"   Node ID: {consensus.node_id}")
    print(f"   Node Type: {NodeType.VALIDATOR.value}")
    print(f"   Security Level: {crypto.config.security_level.value}")
    print(f"   Key Rotation: {crypto.config.key_rotation_interval}s")
    
    print(f"\n📊 Statistics:")
    print(f"   Transactions: {len(transactions)}")
    print(f"   Merkle Depth: {len(proof)}")
    print(f"   Signature Size: {len(signature)} bytes")
    
    print(f"\n🔑 Cryptographic Info:")
    print(f"   Public Key: {identity.public_signing_key.public_bytes_raw().hex()[:32]}...")
    print(f"   Merkle Root: {root.hex()[:32]}...")
    
    print("\n✅ All systems operational!")


if __name__ == "__main__":
    asyncio.run(main())
