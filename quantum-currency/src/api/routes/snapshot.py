#!/usr/bin/env python3
"""
Snapshot route for Quantum Currency API
"""

from flask import Blueprint, request, jsonify
from core.harmonic_validation import make_snapshot

snapshot_bp = Blueprint('snapshot', __name__)

# Global snapshots storage (in a real implementation, this would be a database)
snapshots = {}

@snapshot_bp.route("/snapshot", methods=["POST"])
def snapshot():
    """Generate a signed snapshot"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    snap = make_snapshot(
        node_id=data.get("node_id", ""),
        times=data.get("times", []),
        values=data.get("values", []),
        secret_key=data.get("secret_key", "")
    )
    
    # Store snapshot
    snapshots[snap.node_id] = snap.__dict__
    
    return jsonify(snap.__dict__)

@snapshot_bp.route("/snapshots", methods=["GET"])
def get_snapshots():
    """Get all snapshots"""
    return jsonify(list(snapshots.values()))