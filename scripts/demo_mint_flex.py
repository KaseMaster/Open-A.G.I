#!/usr/bin/env python3
"""
Demo: Harmonic-Validated Mint Transaction
Demonstrates minting FLX tokens based on harmonic coherence validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from openagi.harmonic_validation import make_snapshot, compute_coherence_score
from openagi.token_rules import validate_harmonic_tx, apply_token_effects


def synth(freq, phase, t):
    """Generate synthetic signal"""
    return np.sin(2*np.pi*freq*t + phase)


def main():
    """Main demo function"""
    # Generate coherent signals
    t = np.linspace(0, 0.5, 1024)
    a_vals = synth(100, 0, t)
    b_vals = synth(100, 0, t)
    c_vals = synth(110, 0.5, t)

    a = make_snapshot("node-A", t.tolist(), a_vals.tolist(), secret_key="keyA")
    b = make_snapshot("node-B", t.tolist(), b_vals.tolist(), secret_key="keyB")
    c = make_snapshot("node-C", t.tolist(), c_vals.tolist(), secret_key="keyC")

    cs = compute_coherence_score(a, [b, c])

    tx = {
        "id": "tx001",
        "type": "harmonic",
        "action": "mint",
        "token": "FLX",
        "sender": "node-A",
        "receiver": "node-A",
        "amount": 100,
        "aggregated_cs": cs,
        "sender_chr": 0.8
    }

    config = {"mint_threshold": 0.75, "min_chr": 0.6}
    ok = validate_harmonic_tx(tx, config)
    print(f"Aggregated CS: {cs:.3f}")
    if ok:
        print("✅ Transaction valid; minting 100 FLX")
        ledger = {"balances": {}, "chr": {}}
        ledger = apply_token_effects(ledger, tx)
        print("Ledger state:", ledger)
    else:
        print("❌ Rejected due to low coherence/CHR")


if __name__ == "__main__":
    main()