#!/usr/bin/env python3
"""
Test script for quantum currency system with perfectly coherent signals
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate
from openagi.token_rules import validate_harmonic_tx, apply_token_effects

def test_perfect_coherence():
    """Test with perfectly coherent signals"""
    print("🧪 Testing Perfect Coherence Validation")
    print("=" * 50)
    
    # Generate time base
    t = np.linspace(0, 0.5, 2048)
    
    # Node A: coherent sine (50 Hz)
    a_vals = np.sin(2 * np.pi * 50 * t)
    
    # Node B: identical signal
    b_vals = np.sin(2 * np.pi * 50 * t)
    
    # Node C: identical signal
    c_vals = np.sin(2 * np.pi * 50 * t)
    
    # Create snapshots for each node
    print("📸 Generating snapshots...")
    snapshot_a = make_snapshot("node-A", t.tolist(), a_vals.tolist(), secret_key="keyA")
    snapshot_b = make_snapshot("node-B", t.tolist(), b_vals.tolist(), secret_key="keyB")
    snapshot_c = make_snapshot("node-C", t.tolist(), c_vals.tolist(), secret_key="keyC")
    
    # Calculate coherence
    print("🔄 Calculating coherence...")
    cs_ab = compute_coherence_score(snapshot_a, [snapshot_b])
    cs_ac = compute_coherence_score(snapshot_a, [snapshot_c])
    cs_bc = compute_coherence_score(snapshot_b, [snapshot_c])
    
    print(f"   Coherence A-B: {cs_ab:.4f}")
    print(f"   Coherence A-C: {cs_ac:.4f}")
    print(f"   Coherence B-C: {cs_bc:.4f}")
    
    # Test recursive validation
    bundle = [snapshot_a, snapshot_b, snapshot_c]
    is_valid, proof_bundle = recursive_validate(bundle, threshold=0.75)
    
    print(f"   Aggregated Coherence: {proof_bundle.aggregated_CS if proof_bundle else 0.0:.4f}")
    print(f"   Validation: {'✅ APPROVED' if is_valid else '❌ REJECTED'}")
    
    # Create transaction
    tx = {
        "id": "tx-test-" + str(int(time.time())),
        "type": "harmonic",
        "action": "mint",
        "token": "FLX",
        "sender": "node-A",
        "receiver": "node-A",
        "amount": 100,
        "aggregated_cs": proof_bundle.aggregated_CS if proof_bundle else 0.0,
        "sender_chr": 0.8,
        "timestamp": time.time()
    }
    
    config = {"mint_threshold": 0.75, "min_chr": 0.6}
    ok = validate_harmonic_tx(tx, config)
    
    print(f"   Transaction Validation: {'✅ APPROVED' if ok else '❌ REJECTED'}")
    
    if ok:
        print("💰 Minting 100 FLX tokens...")
        ledger = {"balances": {}, "chr": {}}
        ledger = apply_token_effects(ledger, tx)
        print(f"   Updated Ledger: {ledger}")
    else:
        print("🚫 Transaction rejected due to low coherence/CHR")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_perfect_coherence()