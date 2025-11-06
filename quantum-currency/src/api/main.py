#!/usr/bin/env python3
"""
REST API for Quantum Currency System
Provides endpoints for snapshot generation, coherence calculation, minting, and ledger queries
"""

import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
from core.harmonic_validation import make_snapshot, compute_coherence_score, HarmonicSnapshot
from core.token_rules import validate_harmonic_tx, apply_token_effects

app = Flask(__name__)
ledger = {"balances": {}, "chr": {}}
config = {"mint_threshold": 0.75, "min_chr": 0.6}

# Initialize database
DB_PATH = "quantum_currency.db"

def init_db():
    """Initialize the database with tables for ledger and snapshots"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create ledger table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ledger (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Create snapshots table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            data TEXT,
            timestamp REAL
        )
    ''')
    
    # Create transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            data TEXT,
            timestamp REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Load ledger from database if it exists
    load_ledger_from_db()

def save_ledger_to_db():
    """Save the current ledger state to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Save balances
    cursor.execute("INSERT OR REPLACE INTO ledger (key, value) VALUES (?, ?)", 
                  ("balances", json.dumps(ledger["balances"])))
    
    # Save CHR scores
    cursor.execute("INSERT OR REPLACE INTO ledger (key, value) VALUES (?, ?)", 
                  ("chr", json.dumps(ledger["chr"])))
    
    conn.commit()
    conn.close()

def load_ledger_from_db():
    """Load the ledger state from the database"""
    global ledger
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Load balances
    cursor.execute("SELECT value FROM ledger WHERE key = ?", ("balances",))
    row = cursor.fetchone()
    if row:
        ledger["balances"] = json.loads(row[0])
    
    # Load CHR scores
    cursor.execute("SELECT value FROM ledger WHERE key = ?", ("chr",))
    row = cursor.fetchone()
    if row:
        ledger["chr"] = json.loads(row[0])
    
    conn.close()

def save_transaction_to_db(tx):
    """Save a transaction to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("INSERT OR REPLACE INTO transactions (id, data, timestamp) VALUES (?, ?, ?)", 
                  (tx["id"], json.dumps(tx), tx.get("timestamp", 0)))
    
    conn.commit()
    conn.close()

def save_snapshot_to_db(snapshot_id, snapshot_data):
    """Save a snapshot to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("INSERT OR REPLACE INTO snapshots (id, data, timestamp) VALUES (?, ?, ?)", 
                  (snapshot_id, json.dumps(snapshot_data), snapshot_data.get("timestamp", 0)))
    
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

@app.route("/snapshot", methods=["POST"])
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
    
    # Save snapshot to database
    save_snapshot_to_db(snap.node_id, snap.__dict__)
    
    return jsonify(snap.__dict__)

@app.route("/mint", methods=["POST"])
def mint():
    """Validate and mint FLX tokens"""
    tx = request.json
    ok = validate_harmonic_tx(tx, config)
    if not ok:
        return jsonify({"status": "rejected", "reason": "coherence or CHR too low"}), 400
    
    apply_token_effects(ledger, tx)
    
    # Save transaction and updated ledger to database
    save_transaction_to_db(tx)
    save_ledger_to_db()
    
    return jsonify({"status": "accepted", "ledger": ledger})

@app.route("/ledger", methods=["GET"])
def get_ledger():
    """Get current ledger state"""
    return jsonify(ledger)

@app.route("/coherence", methods=["POST"])
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

@app.route("/transactions", methods=["GET"])
def get_transactions():
    """Get all transactions from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT data FROM transactions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    transactions = [json.loads(row[0]) for row in rows]
    
    conn.close()
    return jsonify(transactions)

@app.route("/snapshots", methods=["GET"])
def get_snapshots():
    """Get all snapshots from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT data FROM snapshots ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    snapshots = [json.loads(row[0]) for row in rows]
    
    conn.close()
    return jsonify(snapshots)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)