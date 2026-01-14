#!/usr/bin/env python3
"""
Client Node for Harmonic Currency Network
Generates harmonic snapshots and sends transactions to validators
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import numpy as np
from openagi.harmonic_validation import make_snapshot, compute_coherence_score


def synth(freq, phase, t):
    """Generate synthetic signal"""
    return np.sin(2*np.pi*freq*t + phase)


def main():
    """Main client function"""
    t = np.linspace(0, 0.5, 1024)
    a_vals = synth(100, 0, t)
    b_vals = synth(100, 0, t)
    c_vals = synth(110, 0.2, t)
    a = make_snapshot("node-A", t.tolist(), a_vals.tolist(), secret_key="keyA")
    b = make_snapshot("node-B", t.tolist(), b_vals.tolist(), secret_key="keyB")
    c = make_snapshot("node-C", t.tolist(), c_vals.tolist(), secret_key="keyC")

    cs = compute_coherence_score(a, [b, c])
    tx = {
        "id": "txDemo",
        "type": "harmonic",
        "action": "mint",
        "token": "FLX",
        "sender": "node-A",
        "receiver": "node-A",
        "amount": 50,
        "aggregated_cs": cs,
        "sender_chr": 0.9,
    }

    for port in [8001, 8002, 8003]:
        try:
            r = requests.post(f"http://localhost:{port}/validate", json=tx)
            print(f"Validator {port-8000+1} →", r.json())
        except Exception as e:
            print(f"Error contacting validator {port-8000+1}: {e}")


if __name__ == "__main__":
    main()