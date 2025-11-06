#!/usr/bin/env python3
"""
Validator Node for Harmonic Currency Network
Implements validation logic for harmonic transactions
"""

import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
import os
import logging
from openagi.harmonic_validation import compute_coherence_score
from openagi.token_rules import validate_harmonic_tx, apply_token_effects

app = Flask(__name__)
log = logging.getLogger("validator")

ledger = {"balances": {}, "chr": {}}
validator_id = os.getenv("VALIDATOR_ID", "0")
config = {"mint_threshold": 0.75, "min_chr": 0.6}

# Initialize database
DB_PATH = f"validator_{validator_id}.db"

def init_db():
    """Initialize the database with tables for ledger and transactions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create ledger table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ledger (
            key TEXT PRIMARY KEY,
            value TEXT
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
                  (tx.get("id", ""), json.dumps(tx), tx.get("timestamp", 0)))
    
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

@app.route("/validate", methods=["POST"])
def validate():
    """Validate a harmonic transaction"""
    tx = request.json
    if not tx:
        return jsonify({"error": "No transaction data provided"}), 400
    
    cs = tx.get("aggregated_cs", 0)
    ok = validate_harmonic_tx(tx, config)
    if ok:
        apply_token_effects(ledger, tx)
        log.info(f"Validator {validator_id}: accepted tx {tx.get('id', 'unknown')}")
        
        # Save transaction and updated ledger to database
        save_transaction_to_db(tx)
        save_ledger_to_db()
        
        return jsonify({"validator": validator_id, "result": "accepted", "cs": cs})
    else:
        log.info(f"Validator {validator_id}: rejected tx {tx.get('id', 'unknown')}")
        return jsonify({"validator": validator_id, "result": "rejected", "cs": cs})


@app.route("/ledger", methods=["GET"])
def get_ledger():
    """Get current ledger state"""
    return jsonify(ledger)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)