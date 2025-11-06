#!/usr/bin/env python3
"""
Ledger route for Quantum Currency API
"""

from flask import Blueprint, jsonify

ledger_bp = Blueprint('ledger', __name__)

# Global ledger state (in a real implementation, this would be a database)
ledger = {"balances": {}, "chr": {}}

@ledger_bp.route("/ledger", methods=["GET"])
def get_ledger():
    """Get current ledger state"""
    return jsonify(ledger)