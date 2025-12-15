#!/usr/bin/env python3
"""
Ledger route for Quantum Currency API with coherence validation and QRA integration
"""

import logging
import numpy as np
import os
import sys
from flask import Blueprint, jsonify, request

ledger_bp = Blueprint('ledger', __name__)

# Global ledger state (in a real implementation, this would be a database)
ledger = {"balances": {}, "chr": {}, "transactions": []}

# Global module availability flag
stability_module_available = False

def initialize_modules():
    """Initialize required modules"""
    global stability_module_available
    
    try:
        # Add parent directory to path for imports
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Test if we can import the required modules
        from core.stability import enforce_governing_law
        from qra.generator import QRAGenerator
        from emission.caf import CoherenceAugmentationFunction
        
        stability_module_available = True
        logging.info("Ledger modules available")
    except ImportError as e:
        stability_module_available = False
        logging.warning(f"Modules not available, coherence validation disabled: {e}")

def load_QRA(node_id):
    """Load QRA for a specific node"""
    try:
        from qra.generator import QRAGenerator
        qra_generator = QRAGenerator()
        return qra_generator.load_qra(node_id)
    except ImportError:
        return {}

def save_to_ledger(tx_data, result, fee=0.01):
    """Save validated transaction to ledger"""
    tx_id = tx_data.get('id', 'unknown')
    
    # Add transaction to ledger
    ledger_entry = {
        "tx_data": tx_data,
        "result": result,
        "fee": fee,
        "timestamp": result.get('timestamp', 'unknown')
    }
    
    ledger["transactions"].append(ledger_entry)
    logging.info(f"Transaction {tx_id} committed to ledger with fee {fee}")

def calculate_harmonic_fee(sender_qra, recipient_qra):
    """
    Calculate transaction fee based on harmonic ratio between sender and recipient
    """
    # Coherence resonance check (Φ-harmonic ratio)
    sender_coherence = sender_qra.get('Coherence_Score', 0.95)
    recipient_coherence = recipient_qra.get('Coherence_Score', 0.95)
    
    if sender_coherence > 0:
        harmonic_ratio = recipient_coherence / sender_coherence
    else:
        harmonic_ratio = 1.0
    
    # Allowed harmonic ratios (simple, perfect fifths, etc.)
    allowed_ratios = [3/2, 4/3, 5/4]
    tolerance = 0.01

    # Check if harmonic ratio is close to allowed ratios
    is_harmonic = any(abs(harmonic_ratio - r) < tolerance for r in allowed_ratios)
    
    if is_harmonic:
        # Base rate for harmonic transactions
        tx_fee = 0.01
    else:
        # Inertial Resistance Fee for non-harmonic transactions
        inertial_cost = sender_qra.get('I_eff_Cost', 0.05)
        tx_fee = 0.01 + 0.1 * inertial_cost
        
    return tx_fee

@ledger_bp.route("/ledger", methods=["GET"])
def get_ledger():
    """Get current ledger state"""
    return jsonify(ledger)

@ledger_bp.route("/ledger/commit", methods=["POST"])
def commit_quantum_transaction():
    """Commit a quantum transaction with coherence validation and QRA checks"""
    # Initialize modules if not already done
    if not stability_module_available:
        initialize_modules()
    
    if not stability_module_available:
        return jsonify({"error": "Stability module not available"}), 503
    
    try:
        from core.stability import enforce_governing_law
        
        tx_data = request.json
        if not tx_data:
            return jsonify({"error": "No transaction data provided"}), 400
            
        sender = tx_data.get("sender")
        recipient = tx_data.get("recipient")
        
        # Load QRA for sender and recipient
        sender_qra = load_QRA(sender) if sender else {}
        recipient_qra = load_QRA(recipient) if recipient else {}
        
        # Calculate transaction fee based on harmonic ratios
        if sender_qra and recipient_qra:
            tx_fee = calculate_harmonic_fee(sender_qra, recipient_qra)
        else:
            tx_fee = 0.01  # Default fee if QRA not available
        
        state_vector = tx_data.get("state", {})
        memory_trace = tx_data.get("memory", {})
        
        # Enforce governing law
        result = enforce_governing_law(state_vector, memory_trace, tx_data)

        # Commit only if coherent
        C_system = result.get('C_system', 0)
        if C_system >= 0.95:
            save_to_ledger(tx_data, result, fee=tx_fee)
            
            # Apply CAF emission if conditions are met
            try:
                from emission.caf import CoherenceAugmentationFunction
                from haru.autoregression import HARU
                haru_model = HARU.load_or_initialize()
                GAS_target = haru_model.get_GAS_target()
                phi_inv = 1.0 / 1.618  # Approximate inverse of golden ratio
                
                # Emit tokens based on CAF policy
                caf = CoherenceAugmentationFunction()
                emission_amount = caf.emit_tokens(sender, C_system, GAS_target, phi_inv)
                
                return jsonify({
                    "status": "APPROVED",
                    "tx_id": tx_data.get('id', 'unknown'),
                    "C_system": C_system,
                    "fee": tx_fee,
                    "emission": emission_amount,
                    "message": f"TX {tx_data.get('id', 'unknown')} approved"
                }), 200
            except Exception as e:
                logging.warning(f"CAF emission failed: {e}")
                return jsonify({
                    "status": "APPROVED",
                    "tx_id": tx_data.get('id', 'unknown'),
                    "C_system": C_system,
                    "fee": tx_fee,
                    "emission": 0,
                    "message": f"TX {tx_data.get('id', 'unknown')} approved (emission failed)"
                }), 200
        else:
            return jsonify({
                "status": "REJECTED",
                "tx_id": tx_data.get('id', 'unknown'),
                "C_system": C_system,
                "message": f"TX {tx_data.get('id', 'unknown')} rejected - Sub-critical coherence"
            }), 400
            
    except Exception as e:
        logging.error(f"Transaction processing error: {str(e)}")
        return jsonify({
            "status": "REJECTED",
            "error": f"Transaction processing error: {str(e)}"
        }), 400

# Initialize modules when the module is loaded
initialize_modules()