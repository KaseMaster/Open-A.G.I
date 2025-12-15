#!/usr/bin/env python3
"""
Main API Server for Quantum Currency
Combines curvature metrics and transaction processing
"""

from flask import Flask, jsonify, request
import numpy as np
import time
import threading
import uuid
import sys
import os

# Add the parent directory to the path so we can import the routes
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

# In-memory storage for transactions and wallets
transactions = []
wallets = {
    "SYSTEM_A": {"flx_balance": 1000.0, "chr_score": 0.95},
    "USER_ALPHA": {"flx_balance": 500.5, "chr_score": 0.88},
    "NODE_001": {"flx_balance": 250.75, "chr_score": 0.92}
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

def check_coherence(sender_chr, amount):
    """
    Check if transaction meets coherence requirements
    C_system >= 0.95 for approval
    """
    # Simplified coherence check
    coherence_score = sender_chr  # In a real system, this would be more complex
    
    # Additional check based on amount (larger amounts require higher coherence)
    amount_factor = min(1.0, amount / 1000.0)  # Normalize amount
    adjusted_threshold = 0.95 - (amount_factor * 0.1)  # Lower threshold for larger amounts
    
    return coherence_score >= adjusted_threshold

# Start metrics update thread
metrics_thread = threading.Thread(target=update_metrics_periodically, daemon=True)
metrics_thread.start()

# Import and register ledger blueprint
try:
    from api.routes.ledger import ledger_bp
    app.register_blueprint(ledger_bp, url_prefix='')
    print("Ledger API routes registered successfully")
except ImportError as e:
    print(f"Warning: Could not import ledger routes: {e}")

# Import and register mint blueprint
try:
    from api.routes.mint import mint_bp
    app.register_blueprint(mint_bp, url_prefix='')
    print("Mint API routes registered successfully")
except ImportError as e:
    print(f"Warning: Could not import mint routes: {e}")

# === Curvature Metrics Endpoints ===

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

# === Transaction Processing Endpoints ===

@app.route('/ledger/commit', methods=['POST'])
def commit_transaction():
    """Commit a quantum transaction with coherence validation"""
    try:
        tx_data = request.json
        if not tx_data:
            return jsonify({"error": "No transaction data provided"}), 400
            
        sender = tx_data.get('sender', 'unknown')
        receiver = tx_data.get('receiver', 'unknown')
        amount = float(tx_data.get('amount', 0))
        sender_chr = float(tx_data.get('sender_chr', 0))
        tx_id = tx_data.get('id', str(uuid.uuid4()))
        
        # Check coherence requirements
        if not check_coherence(sender_chr, amount):
            return jsonify({
                "status": "REJECTED",
                "tx_id": tx_id,
                "C_system": sender_chr,
                "message": f"TX {tx_id} rejected - Sub-critical coherence ({sender_chr:.4f} < 0.95)"
            }), 400
        
        # Check sender balance
        if sender not in wallets or wallets[sender]["flx_balance"] < amount:
            return jsonify({
                "status": "REJECTED",
                "tx_id": tx_id,
                "message": f"TX {tx_id} rejected - Insufficient balance"
            }), 400
        
        # Process transaction
        wallets[sender]["flx_balance"] -= amount
        if receiver not in wallets:
            wallets[receiver] = {"flx_balance": 0, "chr_score": 0.5}
        wallets[receiver]["flx_balance"] += amount
        
        # Record transaction
        transaction = {
            "id": tx_id,
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "sender_chr": sender_chr,
            "timestamp": time.time(),
            "status": "APPROVED"
        }
        transactions.append(transaction)
        
        return jsonify({
            "status": "APPROVED",
            "tx_id": tx_id,
            "C_system": sender_chr,
            "message": f"TX {tx_id} approved"
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "REJECTED",
            "error": f"Transaction processing error: {str(e)}"
        }), 500

@app.route('/wallets', methods=['GET'])
def get_wallets():
    """Get all wallet information"""
    return jsonify(wallets)

@app.route('/wallets/<wallet_id>', methods=['GET'])
def get_wallet(wallet_id):
    """Get specific wallet information"""
    if wallet_id in wallets:
        return jsonify(wallets[wallet_id])
    else:
        return jsonify({"error": "Wallet not found"}), 404

@app.route('/transactions', methods=['GET'])
def get_transactions():
    """Get transaction history"""
    return jsonify(transactions)

@app.route('/transactions/<tx_id>', methods=['GET'])
def get_transaction(tx_id):
    """Get specific transaction"""
    for tx in transactions:
        if tx["id"] == tx_id:
            return jsonify(tx)
    return jsonify({"error": "Transaction not found"}), 404

# === Coherence Endpoint ===

@app.route('/coherence', methods=['POST'])
def calculate_coherence():
    """Calculate coherence score from provided data"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # In a real implementation, this would perform actual coherence calculations
        # For now, we'll return a mock coherence score
        coherence_score = 0.95 + np.random.normal(0, 0.02)  # Random score around 0.95
        coherence_score = max(0.0, min(1.0, coherence_score))  # Clamp between 0 and 1
        
        return jsonify({
            "coherence_score": round(coherence_score, 4),
            "timestamp": time.time()
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Coherence calculation error: {str(e)}"
        }), 500

# === System Status Endpoints ===

@app.route('/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    total_flx = sum(wallet["flx_balance"] for wallet in wallets.values())
    avg_chr = np.mean([wallet["chr_score"] for wallet in wallets.values()])
    
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
        },
        "system_metrics": {
            "total_flx_circulating": total_flx,
            "average_chr_score": avg_chr,
            "active_wallets": len(wallets),
            "total_transactions": len(transactions)
        }
    })

@app.route('/system/metrics', methods=['GET'])
def get_system_metrics():
    """Get system metrics for dashboard"""
    total_flx = sum(wallet["flx_balance"] for wallet in wallets.values())
    avg_chr = np.mean([wallet["chr_score"] for wallet in wallets.values()])
    
    return jsonify({
        "total_flx_circulating": total_flx,
        "average_chr_score": avg_chr,
        "active_wallets": len(wallets),
        "total_transactions": len(transactions)
    })

# === Health Check Endpoint ===

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Quantum Currency Main API",
        "version": "1.2.0"
    })

if __name__ == '__main__':
    print("⚛️ Quantum Currency Main API Server")
    print("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)