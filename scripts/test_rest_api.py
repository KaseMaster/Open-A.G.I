#!/usr/bin/env python3
"""
Test script for REST API
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import numpy as np

def test_api():
    # Test data
    t = np.linspace(0, 0.5, 1024).tolist()
    values = np.sin(2*np.pi*100*np.linspace(0, 0.5, 1024)).tolist()
    
    # Test snapshot endpoint
    snapshot_data = {
        "node_id": "test-node",
        "times": t,
        "values": values,
        "secret_key": "test-key"
    }
    
    try:
        response = requests.post("http://localhost:5000/snapshot", json=snapshot_data)
        print("Snapshot endpoint response:", response.status_code)
        if response.status_code == 200:
            print("Snapshot created successfully")
        else:
            print("Snapshot creation failed:", response.text)
    except Exception as e:
        print("Error contacting snapshot endpoint:", e)
    
    # Test ledger endpoint
    try:
        response = requests.get("http://localhost:5000/ledger")
        print("Ledger endpoint response:", response.status_code)
        if response.status_code == 200:
            print("Ledger retrieved successfully:", response.json())
        else:
            print("Ledger retrieval failed:", response.text)
    except Exception as e:
        print("Error contacting ledger endpoint:", e)

if __name__ == "__main__":
    test_api()