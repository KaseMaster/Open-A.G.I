#!/usr/bin/env python3
"""
Simple API server for Quantum Currency System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "Quantum Currency API Server", "status": "running", "timestamp": time.time()})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route("/metrics")
def metrics():
    # Mock metrics data
    return jsonify({
        "h_internal": 0.975,
        "caf": 1.025,
        "entropy_rate": 0.0018,
        "connected_systems": 8,
        "global_status": "stable",
        "timestamp": time.time()
    })

if __name__ == "__main__":
    print("Starting simple Quantum Currency API server...")
    print("Navigate to http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)