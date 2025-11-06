#!/usr/bin/env python3
"""
Validation route for Quantum Currency API
"""

from flask import Blueprint, request, jsonify
from openagi.harmonic_validation import compute_coherence_score, HarmonicSnapshot

validate_bp = Blueprint('validate', __name__)

@validate_bp.route("/coherence", methods=["POST"])
def coherence():
    """Calculate coherence score between local and remote snapshots"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Convert dict data to HarmonicSnapshot objects
    local_data = data.get("local", {})
    remote_data_list = data.get("remotes", [])
    
    # Create local snapshot
    local_snapshot = HarmonicSnapshot(
        node_id=local_data.get("node_id", "local"),
        timestamp=local_data.get("timestamp", 0.0),
        times=local_data.get("times", []),
        values=local_data.get("values", []),
        spectrum=local_data.get("spectrum", []),
        spectrum_hash=local_data.get("spectrum_hash", ""),
        CS=local_data.get("CS", 0.0),
        phi_params=local_data.get("phi_params", {}),
        signature=local_data.get("signature", None)
    )
    
    # Create remote snapshots
    remote_snapshots = []
    for remote_data in remote_data_list:
        remote_snapshot = HarmonicSnapshot(
            node_id=remote_data.get("node_id", "remote"),
            timestamp=remote_data.get("timestamp", 0.0),
            times=remote_data.get("times", []),
            values=remote_data.get("values", []),
            spectrum=remote_data.get("spectrum", []),
            spectrum_hash=remote_data.get("spectrum_hash", ""),
            CS=remote_data.get("CS", 0.0),
            phi_params=remote_data.get("phi_params", {}),
            signature=remote_data.get("signature", None)
        )
        remote_snapshots.append(remote_snapshot)
    
    cs = compute_coherence_score(local_snapshot, remote_snapshots)
    return jsonify({"coherence_score": cs})