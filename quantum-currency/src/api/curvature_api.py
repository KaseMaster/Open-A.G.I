#!/usr/bin/env python3
"""
Curvature API for Quantum Currency Dashboard
Provides real-time metrics for the dashboard
"""

from flask import Flask, jsonify
import numpy as np
import time
import threading
import json

app = Flask(__name__)

# Global metrics storage
current_metrics = {
    "timestamp": time.time(),
    "rsi": 0.92,
    "cs": 0.98,
    "gas": 0.995,
    "R_Omega_magnitude": 2.159210e-62,
    "T_Omega": 7.236968e-38,
    "safe_mode": False,
    "stability_state": "STABLE",
    "lambda_opt": 0.75,
    "delta_lambda": 0.005
}

def update_metrics_periodically():
    """Update metrics periodically to simulate real system"""
    global current_metrics
    while True:
        time.sleep(2)  # Update every 2 seconds
        
        # Simulate small random fluctuations
        current_metrics["timestamp"] = time.time()
        current_metrics["rsi"] = max(0.85, min(0.99, current_metrics["rsi"] + np.random.normal(0, 0.01)))
        current_metrics["cs"] = max(0.90, min(1.00, current_metrics["cs"] + np.random.normal(0, 0.005)))
        current_metrics["gas"] = max(0.95, min(1.00, current_metrics["gas"] + np.random.normal(0, 0.003)))
        current_metrics["lambda_opt"] = max(0.5, min(1.0, current_metrics["lambda_opt"] + np.random.normal(0, 0.01)))
        current_metrics["delta_lambda"] = max(0.0, min(0.01, current_metrics["delta_lambda"] + np.random.normal(0, 0.001)))

# Start metrics update thread
metrics_thread = threading.Thread(target=update_metrics_periodically, daemon=True)
metrics_thread.start()

@app.route('/field/curvature_metrics', methods=['GET'])
def get_curvature_metrics():
    """Get current curvature metrics"""
    return jsonify(current_metrics)

@app.route('/field/curvature_stream_info', methods=['GET'])
def get_curvature_stream_info():
    """Get information about the curvature stream endpoint"""
    return jsonify({
        "endpoint": "/field/curvature_stream",
        "protocol": "WebSocket",
        "interval": "20ms",
        "metrics": [
            "RSI", "CS", "GAS", "RΩ", "TΩ", "safe_mode", "stability_state"
        ],
        "status": "active"
    })

@app.route('/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    return jsonify({
        "status": "operational",
        "coherence": current_metrics["cs"],
        "gas": current_metrics["gas"],
        "rsi": current_metrics["rsi"],
        "lambda_opt": current_metrics["lambda_opt"],
        "delta_lambda": current_metrics["delta_lambda"],
        "thresholds": {
            "gas_min": 0.95,
            "cs_min": 0.90,
            "rsi_min": 0.65
        }
    })

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Quantum Currency Curvature API",
        "version": "1.2.0"
    })

if __name__ == '__main__':
    print("⚛️ Quantum Currency Curvature API Server")
    print("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)