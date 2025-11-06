#!/usr/bin/env python3
"""
Test script for quantum currency system with high coherence signals
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_coherence_score
from openagi.token_rules import validate_harmonic_tx, apply_token_effects

def synth_signal(freq, phase, duration=0.5, sample_rate=2048):
    """Generate a synthetic signal"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    x = np.sin(2 * np.pi * freq * t + phase)
    return t, x

def test_high_coherence():
    """Test with high coherence signals"""
    print("🧪 Testing High Coherence Validation")
    print("=" * 50)
    
    # Generate time base
    t = np.linspace(0, 0.5, 2048)
    
    # Node A: coherent sine (50 Hz)
    a_vals = np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))
    
    # Node B: same frequency, same phase (coherent)
    b_vals = np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))
    
    # Node C: same frequency, slightly different phase (coherent)
    c_vals = np.sin(2 * np.pi * 50 * t + 0.1) + 0.01 * np.random.randn(len(t))
    
    # Create snapshots for each node
    print("📸 Generating snapshots...")
    snapshot_a = make_snapshot("node-A", t.tolist(), a_vals.tolist(), secret_key="keyA")
    snapshot_b = make_snapshot("node-B", t.tolist(), b_vals.tolist(), secret_key="keyB")
    snapshot_c = make_snapshot("node-C", t.tolist(), c_vals.tolist(), secret_key="keyC")
    
    # Calculate coherence
    print("🔄 Calculating coherence...")
    cs = compute_coherence_score(snapshot_a, [snapshot_b, snapshot_c])
    
    # Create transaction
    tx = {
        "id": "tx-test-" + str(int(time.time())),
        "type": "harmonic",
        "action": "mint",
        "token": "FLX",
        "sender": "node-A",
        "receiver": "node-A",
        "amount": 100,
        "aggregated_cs": cs,
        "sender_chr": 0.8,
        "timestamp": time.time()
    }
    
    config = {"mint_threshold": 0.75, "min_chr": 0.6}
    ok = validate_harmonic_tx(tx, config)
    
    print(f"📊 Results:")
    print(f"   Coherence Score: {cs:.4f}")
    print(f"   Validation: {'✅ APPROVED' if ok else '❌ REJECTED'}")
    
    if ok:
        print("💰 Minting 100 FLX tokens...")
        ledger = {"balances": {}, "chr": {}}
        ledger = apply_token_effects(ledger, tx)
        print(f"   Updated Ledger: {ledger}")
    else:
        print("🚫 Transaction rejected due to low coherence/CHR")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_high_coherence()