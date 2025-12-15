#!/usr/bin/env python3
"""
Transaction API for Quantum Currency
Handles transaction processing with coherence validation
"""

from flask import Flask, jsonify, request
import numpy as np
import time
import uuid

app = Flask(__name__)

# In-memory storage for transactions and wallets
transactions = []
wallets = {
    "SYSTEM_A": {"flx_balance": 1000.0, "chr_score": 0.95},
    "USER_ALPHA": {"flx_balance": 500.5, "chr_score": 0.88},
    "NODE_001": {"flx_balance": 250.75, "chr_score": 0.92}
}

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

if __name__ == '__main__':
    print("⚛️ Quantum Currency Transaction API Server")
    print("Starting server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)