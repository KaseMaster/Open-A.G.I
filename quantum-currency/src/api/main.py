#!/usr/bin/env python3
"""
REST API for Quantum Currency System
Provides endpoints for snapshot generation, coherence calculation, minting, and ledger queries
"""

import sys
import os
import json
import sqlite3
import html
import time  # Add missing import
import hashlib
import secrets
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'openagi'))

from flask import Flask, request, jsonify, send_from_directory
from core.harmonic_validation import make_snapshot, compute_coherence_score, HarmonicSnapshot
from core.token_rules import validate_harmonic_tx, apply_token_effects
from core.cal_engine import CALEngine  # New UHES component
from models.quantum_memory import UnifiedFieldMemory, QuantumPacket  # New UHES component
from models.coherent_db import CoherentDatabase  # New UHES component
from models.entropy_monitor import EntropyMonitor  # New UHES component
from models.ai_governance import AIGovernance  # New UHES component
from models.harmonic_wallet import HarmonicWallet, WalletAccount, Transaction  # Wallet component
from models.external_network import ExternalNetworkConnector, CoherenceBridge, ResonanceData  # External network integration
from core.global_harmonic_synchronizer import GlobalHarmonicSynchronizer  # Global Harmonic Synchronizer
from models.human_feedback import HumanFeedbackSystem  # Human Feedback Integration
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the path to the UI dashboard directory
UI_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'ui-dashboard')

# If the above path doesn't work, try an absolute path
if not os.path.exists(UI_DIR):
    UI_DIR = os.path.join('D:', 'AI AGENT CODERV1', 'QUANTUM CURRENCY', 'ui-dashboard')

# Import the new inter-system councils module
from src.models.inter_system_councils import InterSystemCouncils
from src.models.ethical_coherence_governance import EthicalCoherenceGovernance, EthicalRule

# Import Quantum Coherence AI for OpenAGI integration
from models.quantum_coherence_ai import QuantumCoherenceAI, EconomicOptimization

app = Flask(__name__, static_folder=UI_DIR, static_url_path='')
ledger = {"balances": {}, "chr": {}}
config = {"mint_threshold": 0.75, "min_chr": 0.6}

# Initialize database
# Use a mounted volume for persistence in Docker environments
DB_PATH = "/data/quantum_currency.db" if os.path.exists("/data") else "quantum_currency.db"

# Initialize UHES components
cal_engine = CALEngine()
ufm = UnifiedFieldMemory()
cdb = CoherentDatabase(ufm)
entropy_monitor = EntropyMonitor(ufm)
governance = AIGovernance(cdb, ufm)

# Initialize wallet storage
wallets = {}  # In-memory wallet storage (in production, this would be in a database)

# Initialize external network connectors
external_networks = {}  # In-memory external network storage

# Create default external network connector
external_networks["default"] = ExternalNetworkConnector("quantum-currency-network")

# Initialize Global Harmonic Synchronizer
global_synchronizer = GlobalHarmonicSynchronizer("quantum-currency-global-network")

# Initialize Human Feedback System
human_feedback_system = HumanFeedbackSystem("quantum-currency-human-feedback")

# Initialize the inter-system councils system
inter_system_councils = InterSystemCouncils(governance)

# Initialize the Ethical Coherence Governance engine
ecg_engine = EthicalCoherenceGovernance(governance)

# Initialize Quantum Coherence AI for OpenAGI integration
quantum_ai = QuantumCoherenceAI("quantum-currency-openagi-network")

# Add this function to initialize wallets with entropy monitor
def initialize_wallet_with_entropy_monitor(wallet_id):
    """Initialize wallet with entropy monitor"""
    if wallet_id in wallets:
        wallet = wallets[wallet_id]
        if hasattr(wallet, 'set_entropy_monitor'):
            wallet.set_entropy_monitor(entropy_monitor)

def init_db():
    """Initialize the database with tables for ledger and snapshots"""
    try:
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
        
        # Create wallets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wallets (
                wallet_id TEXT PRIMARY KEY,
                wallet_data TEXT,
                created_at REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Load ledger from database if it exists
        load_ledger_from_db()
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Fallback to in-memory ledger if database fails
        global ledger
        ledger = {"balances": {}, "chr": {}}

def save_ledger_to_db():
    """Save the current ledger state to the database"""
    try:
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
    except Exception as e:
        print(f"Error saving ledger to database: {e}")

def load_ledger_from_db():
    """Load the ledger state from the database"""
    try:
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
    except Exception as e:
        print(f"Error loading ledger from database: {e}")
        # Initialize with empty ledger if database fails
        ledger["balances"] = {}
        ledger["chr"] = {}

def save_transaction_to_db(tx):
    """Save a transaction to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("INSERT OR REPLACE INTO transactions (id, data, timestamp) VALUES (?, ?, ?)", 
                      (tx["id"], json.dumps(tx), tx.get("timestamp", 0)))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving transaction to database: {e}")

def save_snapshot_to_db(snapshot_id, snapshot_data):
    """Save a snapshot to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("INSERT OR REPLACE INTO snapshots (id, data, timestamp) VALUES (?, ?, ?)", 
                      (snapshot_id, json.dumps(snapshot_data), snapshot_data.get("timestamp", 0)))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving snapshot to database: {e}")

def save_wallet_to_db(wallet_id, wallet_data):
    """Save a wallet to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("INSERT OR REPLACE INTO wallets (wallet_id, wallet_data, created_at) VALUES (?, ?, ?)", 
                      (wallet_id, json.dumps(wallet_data), time.time()))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving wallet to database: {e}")

def load_wallets_from_db():
    """Load wallets from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT wallet_id, wallet_data FROM wallets")
        rows = cursor.fetchall()
        
        for wallet_id, wallet_data in rows:
            wallets[wallet_id] = json.loads(wallet_data)
        
        conn.close()
    except Exception as e:
        print(f"Error loading wallets from database: {e}")

# Initialize the database when the app starts
init_db()

def sanitize_data(data):
    """Sanitize data to prevent XSS attacks"""
    if isinstance(data, dict):
        return {key: sanitize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, str):
        return html.escape(data, quote=True)
    else:
        return data

def safe_get(data, key, default=None):
    """Safely get a value from a dictionary, ensuring it's the right type"""
    if isinstance(data, dict):
        value = data.get(key, default)
        # Ensure we return the correct type
        if key in ['node_id', 'secret_key', 'spectrum_hash']:
            if value is None:
                return default if isinstance(default, str) else ""
            try:
                return str(value)
            except:
                return default if isinstance(default, str) else ""
        elif key in ['times', 'values', 'spectrum']:
            if value is None:
                return default if isinstance(default, list) else []
            try:
                return list(value) if hasattr(value, '__iter__') and not isinstance(value, str) else [float(value)] if value else []
            except:
                return default if isinstance(default, list) else []
        elif key in ['timestamp', 'CS']:
            if value is None:
                return default if isinstance(default, (int, float)) else 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return default if isinstance(default, (int, float)) else 0.0
        elif key in ['phi_params']:
            if value is None:
                return default if isinstance(default, dict) else {}
            try:
                return dict(value) if hasattr(value, '__iter__') else {}
            except:
                return default if isinstance(default, dict) else {}
        elif key in ['signature']:
            return value if value is not None else default
        return value
    return default

# Print information about the UI directory
print(f"Looking for UI files in: {UI_DIR}")
if os.path.exists(UI_DIR):
    print("UI directory found")
    ui_files = os.listdir(UI_DIR)
    print(f"UI files: {ui_files}")
else:
    print("UI directory NOT found")
    # List the parent directory to see what's available
    parent_dir = os.path.dirname(UI_DIR)
    if os.path.exists(parent_dir):
        print(f"Contents of parent directory: {os.listdir(parent_dir)}")

@app.route("/snapshot", methods=["POST"])
def snapshot():
    """Generate a signed snapshot"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Sanitize input data
    sanitized_data = sanitize_data(data)
    
    snap = make_snapshot(
        node_id=safe_get(sanitized_data, "node_id", ""),
        times=safe_get(sanitized_data, "times", []),
        values=safe_get(sanitized_data, "values", []),
        secret_key=safe_get(sanitized_data, "secret_key", "")
    )
    
    # Save snapshot to database
    save_snapshot_to_db(snap.node_id, snap.__dict__)
    
    # Sanitize output data
    sanitized_output = sanitize_data(snap.__dict__)
    return jsonify(sanitized_output)

@app.route("/mint", methods=["POST"])
def mint():
    """Validate and mint FLX tokens"""
    tx = request.json
    # Sanitize input data
    sanitized_tx = sanitize_data(tx)
    ok = validate_harmonic_tx(sanitized_tx, config)
    if not ok:
        return jsonify({"status": "rejected", "reason": "coherence or CHR too low"}), 400
    
    apply_token_effects(ledger, sanitized_tx)
    
    # Save transaction and updated ledger to database
    save_transaction_to_db(sanitized_tx)
    save_ledger_to_db()
    
    # Sanitize output data
    sanitized_ledger = sanitize_data(ledger)
    return jsonify({"status": "accepted", "ledger": sanitized_ledger})

@app.route("/ledger", methods=["GET"])
def get_ledger():
    """Get current ledger state"""
    # Sanitize output data
    sanitized_ledger = sanitize_data(ledger)
    return jsonify(sanitized_ledger)

@app.route("/coherence", methods=["POST"])
def coherence():
    """Calculate coherence score between local and remote snapshots"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Sanitize input data
    sanitized_data = sanitize_data(data)
    
    # Convert dict data to HarmonicSnapshot objects
    local_data = safe_get(sanitized_data, "local", {})
    remote_data_list = safe_get(sanitized_data, "remotes", [])
    
    # Create local snapshot
    local_snapshot = HarmonicSnapshot(
        node_id=safe_get(local_data, "node_id", "local"),
        timestamp=safe_get(local_data, "timestamp", 0.0),
        times=safe_get(local_data, "times", []),
        values=safe_get(local_data, "values", []),
        spectrum=safe_get(local_data, "spectrum", []),
        spectrum_hash=safe_get(local_data, "spectrum_hash", ""),
        CS=safe_get(local_data, "CS", 0.0),
        phi_params=safe_get(local_data, "phi_params", {}),
        signature=safe_get(local_data, "signature", None)
    )
    
    # Create remote snapshots
    remote_snapshots = []
    for remote_data in remote_data_list:
        remote_snapshot = HarmonicSnapshot(
            node_id=safe_get(remote_data, "node_id", "remote"),
            timestamp=safe_get(remote_data, "timestamp", 0.0),
            times=safe_get(remote_data, "times", []),
            values=safe_get(remote_data, "values", []),
            spectrum=safe_get(remote_data, "spectrum", []),
            spectrum_hash=safe_get(remote_data, "spectrum_hash", ""),
            CS=safe_get(remote_data, "CS", 0.0),
            phi_params=safe_get(remote_data, "phi_params", {}),
            signature=safe_get(remote_data, "signature", None)
        )
        remote_snapshots.append(remote_snapshot)
    
    cs = compute_coherence_score(local_snapshot, remote_snapshots)
    # Sanitize output data
    sanitized_result = sanitize_data({"coherence_score": cs})
    return jsonify(sanitized_result)

@app.route("/transactions", methods=["GET"])
def get_transactions():
    """Get all transactions from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT data FROM transactions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    transactions = [json.loads(row[0]) for row in rows]
    
    conn.close()
    # Sanitize output data
    sanitized_transactions = sanitize_data(transactions)
    return jsonify(sanitized_transactions)

@app.route("/snapshots", methods=["GET"])
def get_snapshots():
    """Get all snapshots from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT data FROM snapshots ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    snapshots = [json.loads(row[0]) for row in rows]
    
    conn.close()
    # Sanitize output data
    sanitized_snapshots = sanitize_data(snapshots)
    return jsonify(sanitized_snapshots)

@app.route("/uhes/status", methods=["GET"])
def get_uhes_status():
    """Get UHES system status"""
    try:
        # Get statistics from all components
        ufm_stats = ufm.get_layer_statistics()
        cdb_stats = cdb.get_database_statistics()
        gov_stats = governance.get_network_topology()
        
        status = {
            "cal_engine": "active",
            "unified_field_memory": ufm_stats,
            "coherent_database": cdb_stats,
            "entropy_monitor": {
                "high_entropy_packets": len(entropy_monitor.high_entropy_packets),
                "monitored_packets": len(entropy_monitor.entropy_history)
            },
            "ai_governance": gov_stats,
            "timestamp": time.time()
        }
        
        # Sanitize output data
        sanitized_status = sanitize_data(status)
        return jsonify(sanitized_status)
    except Exception as e:
        return jsonify({"error": f"Failed to get UHES status: {str(e)}"}), 500

@app.route("/uhes/create_packet", methods=["POST"])
def create_quantum_packet():
    """Create a new Quantum Packet in UFM"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        omega_vector = safe_get(sanitized_data, "omega_vector", [])
        psi_score = safe_get(sanitized_data, "psi_score", 0.0)
        scale_level = safe_get(sanitized_data, "scale_level", "Lϕ")
        data_payload = safe_get(sanitized_data, "data_payload", None)
        
        # Create packet
        packet = ufm.create_quantum_packet(
            omega_vector=omega_vector,
            psi_score=psi_score,
            scale_level=scale_level,
            data_payload=data_payload
        )
        
        # Store packet
        success = ufm.store_packet(packet)
        
        if success:
            # Add to CDB
            cdb.add_quantum_packet(packet)
            
            # Sanitize output data
            sanitized_packet = sanitize_data(packet.to_dict())
            return jsonify({
                "status": "success",
                "packet": sanitized_packet
            })
        else:
            return jsonify({"status": "error", "message": "Failed to store packet"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Failed to create packet: {str(e)}"}), 500

@app.route("/uhes/wave_query", methods=["POST"])
def wave_propagation_query():
    """Perform wave propagation query in CDB"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        start_omega = safe_get(sanitized_data, "start_omega", [])
        start_scale = safe_get(sanitized_data, "start_scale", "LΦ")
        max_depth = safe_get(sanitized_data, "max_depth", 5)
        
        # Perform wave propagation query
        results = cdb.wave_propagation_query(
            start_omega=start_omega,
            start_scale=start_scale,
            max_depth=max_depth
        )
        
        # Format results
        formatted_results = []
        for packet, weight in results[:10]:  # Limit to top 10
            formatted_results.append({
                "packet_id": packet.id,
                "psi_score": packet.psi_score,
                "scale_level": packet.scale_level,
                "weight": weight
            })
        
        # Sanitize output data
        sanitized_results = sanitize_data(formatted_results)
        return jsonify({
            "status": "success",
            "results": sanitized_results
        })
        
    except Exception as e:
        return jsonify({"error": f"Wave query failed: {str(e)}"}), 500

@app.route("/uhes/monitor_entropy", methods=["POST"])
def monitor_entropy():
    """Monitor entropy in UFM and trigger self-healing if needed"""
    try:
        # Monitor all packets
        actions = entropy_monitor.monitor_all_packets()
        
        # Sanitize output data
        sanitized_actions = sanitize_data(actions)
        return jsonify({
            "status": "success",
            "actions_taken": sanitized_actions
        })
        
    except Exception as e:
        return jsonify({"error": f"Entropy monitoring failed: {str(e)}"}), 500

@app.route("/uhes/governance/proposal", methods=["POST"])
def create_governance_proposal():
    """Create a new governance proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        proposer_id = safe_get(sanitized_data, "proposer_id", "")
        title = safe_get(sanitized_data, "title", "")
        description = safe_get(sanitized_data, "description", "")
        target_omega = safe_get(sanitized_data, "target_omega", {})
        
        # Create proposal
        proposal_id = governance.create_proposal(
            proposer_id=proposer_id,
            title=title,
            description=description,
            target_omega=target_omega
        )
        
        if proposal_id:
            return jsonify({
                "status": "success",
                "proposal_id": proposal_id
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create proposal"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Proposal creation failed: {str(e)}"}), 500

@app.route("/uhes/governance/vote", methods=["POST"])
def vote_on_proposal():
    """Vote on a governance proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        validator_id = safe_get(sanitized_data, "validator_id", "")
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        vote_weight = safe_get(sanitized_data, "vote_weight", 0.0)
        
        # Record vote
        success = governance.vote_on_proposal(
            validator_id=validator_id,
            proposal_id=proposal_id,
            vote_weight=vote_weight
        )
        
        if success:
            return jsonify({"status": "success", "message": "Vote recorded"})
        else:
            return jsonify({"status": "error", "message": "Failed to record vote"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Voting failed: {str(e)}"}), 500

# Wallet API endpoints
@app.route("/wallet/create", methods=["POST"])
def create_wallet():
    """Create a new harmonic wallet"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        wallet_name = data.get("wallet_name", "default-wallet")
        
        # Create wallet instance
        wallet = HarmonicWallet(wallet_name)
        wallet_id = wallet.wallet_id
        
        # Set entropy monitor
        wallet.set_entropy_monitor(entropy_monitor)
        
        # Store wallet in memory and database
        wallets[wallet_id] = wallet
        save_wallet_to_db(wallet_id, {
            "wallet_name": wallet_name,
            "wallet_id": wallet_id,
            "accounts": {},
            "transactions": []
        })
        
        return jsonify({
            "status": "success",
            "wallet_id": wallet_id,
            "wallet_name": wallet_name
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create wallet: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/generate_keypair", methods=["POST"])
def generate_keypair(wallet_id):
    """Generate a new keypair for a wallet"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        account_name = data.get("account_name", "default")
        key_type = data.get("key_type", "Ed25519")
        
        # Generate keypair
        address = wallet.generate_harmonic_keypair(account_name, key_type)
        
        if address:
            account = wallet.get_account(address)
            return jsonify({
                "status": "success",
                "address": address,
                "account": {
                    "chr_score": account.chr_score,
                    "created_at": account.created_at
                }
            })
        else:
            return jsonify({"error": "Failed to generate keypair"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to generate keypair: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/accounts", methods=["GET"])
def get_wallet_accounts(wallet_id):
    """Get all accounts in a wallet"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        accounts_data = {}
        
        for address, account in wallet.accounts.items():
            accounts_data[address] = {
                "chr_balance": account.chr_balance,
                "flx_balance": account.flx_balance,
                "psy_balance": account.psy_balance,
                "atr_balance": account.atr_balance,
                "res_balance": account.res_balance,
                "chr_score": account.chr_score,
                "created_at": account.created_at,
                "last_activity": account.last_activity
            }
        
        return jsonify({
            "status": "success",
            "accounts": accounts_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get accounts: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/balance/<address>/<token_type>", methods=["GET"])
def get_balance(wallet_id, address, token_type):
    """Get token balance for an account"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        balance = wallet.get_balance(address, token_type.upper())
        
        return jsonify({
            "status": "success",
            "balance": balance,
            "token_type": token_type.upper(),
            "address": address
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get balance: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/transaction", methods=["POST"])
def create_transaction(wallet_id):
    """Create a new transaction"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        sender = data.get("sender")
        receiver = data.get("receiver")
        token_type = data.get("token_type")
        amount = float(data.get("amount", 0))
        memo = data.get("memo")
        
        # Create transaction
        tx_id = wallet.create_transaction(sender, receiver, token_type.upper(), amount, memo)
        
        if tx_id:
            return jsonify({
                "status": "success",
                "transaction_id": tx_id,
                "sender": sender,
                "receiver": receiver,
                "token_type": token_type.upper(),
                "amount": amount
            })
        else:
            return jsonify({"error": "Failed to create transaction"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to create transaction: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/transactions/<address>", methods=["GET"])
def get_transaction_history(wallet_id, address):
    """Get transaction history for an account"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        limit = int(request.args.get("limit", 10))
        
        transactions = wallet.get_transaction_history(address, limit)
        transactions_data = []
        
        for tx in transactions:
            transactions_data.append({
                "tx_id": tx.tx_id,
                "sender": tx.sender,
                "receiver": tx.receiver,
                "token_type": tx.token_type,
                "amount": tx.amount,
                "timestamp": tx.timestamp,
                "fee": tx.fee,
                "memo": tx.memo,
                "status": tx.status
            })
        
        return jsonify({
            "status": "success",
            "transactions": transactions_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get transaction history: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/resonance", methods=["GET"])
def get_wallet_resonance(wallet_id):
    """Get real-time resonance data for a wallet"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        
        # Calculate wallet resonance metrics
        total_coherence = 0.0
        total_entropy = 0.0
        account_count = len(wallet.accounts)
        
        if account_count > 0:
            for account in wallet.accounts.values():
                total_coherence += account.chr_score
                # Calculate entropy based on balance distribution
                total_balance = (account.flx_balance + account.chr_balance + 
                               account.psy_balance + account.atr_balance + 
                               account.res_balance)
                total_entropy += total_balance * 0.1  # Simple entropy calculation
            
            avg_coherence = total_coherence / account_count
            avg_entropy = total_entropy / account_count
        else:
            avg_coherence = 0.0
            avg_entropy = 0.0
        
        # Calculate flow based on recent transactions
        recent_transactions = [tx for tx in wallet.transactions 
                             if time.time() - tx.timestamp < 3600]  # Last hour
        flow = len(recent_transactions)
        
        resonance_data = {
            "status": "success",
            "wallet_id": wallet_id,
            "coherence": avg_coherence,
            "entropy": avg_entropy,
            "flow": flow,
            "account_count": account_count,
            "timestamp": time.time()
        }
        
        return jsonify(resonance_data)
    except Exception as e:
        return jsonify({"error": f"Failed to get resonance data: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/stake", methods=["POST"])
def stake_tokens(wallet_id):
    """Stake tokens for rewards"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        address = data.get("address")
        token_type = data.get("token_type")
        amount = float(data.get("amount", 0))
        
        if not address or not token_type or amount <= 0:
            return jsonify({"error": "Invalid staking parameters"}), 400
        
        # Stake tokens
        stake_id = wallet.stake_tokens(address, token_type, amount)
        
        if stake_id:
            return jsonify({
                "status": "success",
                "stake_id": stake_id,
                "message": f"Successfully staked {amount} {token_type}"
            })
        else:
            return jsonify({"error": "Failed to stake tokens"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to stake tokens: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/unstake/<stake_id>", methods=["POST"])
def unstake_tokens(wallet_id, stake_id):
    """Unstake tokens"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        
        # Unstake tokens
        success = wallet.unstake_tokens(stake_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Successfully unstaked tokens"
            })
        else:
            return jsonify({"error": "Failed to unstake tokens"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to unstake tokens: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/staking_records/<address>", methods=["GET"])
def get_staking_records(wallet_id, address):
    """Get staking records for an account"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        records = wallet.get_staking_records(address)
        
        records_data = []
        for record in records:
            records_data.append({
                "stake_id": record.stake_id,
                "token_type": record.token_type,
                "amount": record.amount,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "reward_rate": record.reward_rate,
                "status": record.status
            })
        
        return jsonify({
            "status": "success",
            "records": records_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get staking records: {str(e)}"}), 500

@app.route("/wallet/<wallet_id>/rebalance", methods=["POST"])
def rebalance_wallet(wallet_id):
    """Rebalance wallet flow dynamically based on entropy metrics"""
    try:
        if wallet_id not in wallets:
            return jsonify({"error": "Wallet not found"}), 404
        
        wallet = wallets[wallet_id]
        
        # Rebalance flow dynamically
        result = wallet.rebalance_flow_dynamically()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to rebalance wallet: {str(e)}"}), 500

# Add AI-driven coherence tuning endpoint
@app.route("/uhes/tuning/trigger", methods=["POST"])
def trigger_coherence_tuning():
    """Trigger AI-driven coherence tuning for performance optimization"""
    try:
        # Trigger tuning
        tuning_result = cal_engine.ai_driven_coherence_tuning()
        
        return jsonify(tuning_result)
    except Exception as e:
        return jsonify({"error": f"Coherence tuning failed: {str(e)}"}), 500

@app.route("/uhes/tuning/status", methods=["GET"])
def get_tuning_status():
    """Get Auto-Balance Mode status"""
    try:
        status = cal_engine.get_auto_balance_mode_status()
        return jsonify({
            "status": "success",
            "auto_balance_mode": status
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get tuning status: {str(e)}"}), 500

@app.route("/uhes/tuning/metrics", methods=["POST"])
def record_performance_metrics():
    """Record performance metrics for AI-driven tuning"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        latency = float(data.get("latency", 0))
        memory_usage = float(data.get("memory_usage", 0))
        throughput = float(data.get("throughput", 0))
        
        # Record metrics
        cal_engine.record_performance_metrics(latency, memory_usage, throughput)
        
        return jsonify({
            "status": "success",
            "message": "Performance metrics recorded"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to record metrics: {str(e)}"}), 500

# External Network Integration APIs
@app.route("/external/connect", methods=["POST"])
def connect_external_system():
    """Connect to an external system"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        name = data.get("name", "")
        url = data.get("url", "")
        api_key = data.get("api_key", "")
        
        # Connect to external system
        system_id = external_networks["default"].connect_external_system(name, url, api_key)
        
        if system_id:
            return jsonify({
                "status": "success",
                "system_id": system_id,
                "message": f"Connected to {name}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to connect to external system"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to connect to external system: {str(e)}"}), 500

@app.route("/external/disconnect/<system_id>", methods=["POST"])
def disconnect_external_system(system_id):
    """Disconnect from an external system"""
    try:
        # Disconnect from external system
        success = external_networks["default"].disconnect_external_system(system_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Disconnected from external system"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to disconnect from external system"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to disconnect from external system: {str(e)}"}), 500

@app.route("/external/bridge/create", methods=["POST"])
def create_coherence_bridge():
    """Create a coherence bridge between two systems"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        source_system = data.get("source_system", "")
        target_system = data.get("target_system", "")
        
        # Create coherence bridge
        bridge_id = external_networks["default"].create_coherence_bridge(source_system, target_system)
        
        if bridge_id:
            return jsonify({
                "status": "success",
                "bridge_id": bridge_id,
                "message": "Coherence bridge created"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create coherence bridge"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to create coherence bridge: {str(e)}"}), 500

@app.route("/external/bridge/validate/<bridge_id>", methods=["POST"])
def validate_harmonic_parity(bridge_id):
    """Validate harmonic parity for a coherence bridge"""
    try:
        # Validate harmonic parity
        is_valid = external_networks["default"].validate_harmonic_parity(bridge_id)
        
        return jsonify({
            "status": "success",
            "valid": is_valid,
            "message": "Harmonic parity validation completed"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to validate harmonic parity: {str(e)}"}), 500

@app.route("/external/bridge/validate/realtime/<bridge_id>", methods=["POST"])
def validate_real_time_parity(bridge_id):
    """Validate harmonic parity through real-time checksums"""
    try:
        # Validate real-time parity
        validation_result = external_networks["default"].validate_real_time_parity(bridge_id)
        
        return jsonify(validation_result)
    except Exception as e:
        return jsonify({"error": f"Failed to validate real-time parity: {str(e)}"}), 500

@app.route("/external/resonance/<system_id>", methods=["GET"])
def get_external_resonance_data(system_id):
    """Get resonance data from an external system"""
    try:
        # Get resonance data
        resonance_data = external_networks["default"].get_external_resonance_data(system_id)
        
        if resonance_data:
            return jsonify({
                "status": "success",
                "resonance_data": {
                    "system_id": resonance_data.system_id,
                    "timestamp": resonance_data.timestamp,
                    "coherence": resonance_data.coherence,
                    "entropy": resonance_data.entropy,
                    "flow": resonance_data.flow,
                    "checksum": resonance_data.checksum
                }
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to get resonance data"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to get resonance data: {str(e)}"}), 500

@app.route("/external/synchronize/<system_id>", methods=["POST"])
def synchronize_with_external_system(system_id):
    """Synchronize with an external system"""
    try:
        # Synchronize with external system
        sync_result = external_networks["default"].synchronize_with_external_system(system_id)
        
        return jsonify(sync_result)
    except Exception as e:
        return jsonify({"error": f"Failed to synchronize with external system: {str(e)}"}), 500

@app.route("/external/network/topology", methods=["GET"])
def get_network_topology():
    """Get external network topology"""
    try:
        # Get network topology
        topology = external_networks["default"].get_network_topology()
        
        return jsonify({
            "status": "success",
            "topology": topology
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get network topology: {str(e)}"}), 500

@app.route("/external/synchronize/harmonic/<system_id>", methods=["POST"])
def harmonic_synchronization(system_id):
    """Perform harmonic synchronization with an external system"""
    try:
        # Perform harmonic synchronization
        sync_result = external_networks["default"].harmonic_synchronization_protocol(system_id)
        
        return jsonify(sync_result)
    except Exception as e:
        return jsonify({"error": f"Failed to perform harmonic synchronization: {str(e)}"}), 500

@app.route("/external/resonance/exchange/<system_id>", methods=["POST"])
def two_way_resonance_exchange(system_id):
    """Perform two-way resonance exchange with an external system"""
    try:
        # Perform two-way resonance exchange
        exchange_result = external_networks["default"].two_way_resonance_exchange(system_id)
        
        return jsonify(exchange_result)
    except Exception as e:
        return jsonify({"error": f"Failed to perform two-way resonance exchange: {str(e)}"}), 500

@app.route("/external/resonance/exchange/continuous", methods=["POST"])
def continuous_resonance_exchange():
    """Perform continuous two-way resonance exchange with multiple systems"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        system_ids = data.get("system_ids", [])
        
        # Perform continuous resonance exchange
        exchange_result = external_networks["default"].continuous_resonance_exchange(system_ids)
        
        return jsonify(exchange_result)
    except Exception as e:
        return jsonify({"error": f"Failed to perform continuous resonance exchange: {str(e)}"}), 500

@app.route("/external/feedback/collect/<system_id>", methods=["POST"])
def collect_real_time_feedback(system_id):
    """Collect real-time feedback from a connected system"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        # Collect feedback
        success = external_networks["default"].collect_real_time_feedback(system_id, data)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Feedback collected successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to collect feedback"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to collect feedback: {str(e)}"}), 500

@app.route("/external/feedback/recent", methods=["GET"])
def get_real_time_feedback():
    """Get recent real-time feedback from connected systems"""
    try:
        limit = int(request.args.get("limit", 100))
        
        # Get feedback
        feedback = external_networks["default"].get_real_time_feedback(limit)
        
        return jsonify({
            "status": "success",
            "feedback": feedback
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get feedback: {str(e)}"}), 500

@app.route("/external/feedback/trends", methods=["GET"])
def analyze_feedback_trends():
    """Analyze trends in real-time feedback"""
    try:
        # Analyze feedback trends
        trends = external_networks["default"].analyze_feedback_trends()
        
        return jsonify(trends)
    except Exception as e:
        return jsonify({"error": f"Failed to analyze feedback trends: {str(e)}"}), 500

@app.route("/external/entropy/listener/register", methods=["POST"])
def register_entropy_listener():
    """Register an external resonance listener"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        listener_id = data.get("listener_id", "")
        system_id = data.get("system_id", "")
        
        # Register listener
        success = external_networks["default"].register_entropy_listener(listener_id, system_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Entropy listener registered successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to register entropy listener"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to register entropy listener: {str(e)}"}), 500

@app.route("/external/entropy/listener/unregister/<listener_id>", methods=["POST"])
def unregister_entropy_listener(listener_id):
    """Unregister an external resonance listener"""
    try:
        # Unregister listener
        success = external_networks["default"].unregister_entropy_listener(listener_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Entropy listener unregistered successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to unregister entropy listener"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to unregister entropy listener: {str(e)}"}), 500

@app.route("/external/entropy/monitor/<listener_id>", methods=["GET"])
def monitor_interaction_entropy(listener_id):
    """Monitor interaction entropy for a registered listener"""
    try:
        # Monitor entropy
        result = external_networks["default"].monitor_interaction_entropy(listener_id)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to monitor interaction entropy: {str(e)}"}), 500

@app.route("/external/entropy/summary", methods=["GET"])
def get_entropy_monitoring_summary():
    """Get entropy monitoring summary"""
    try:
        # Get summary
        summary = external_networks["default"].get_entropy_monitoring_summary()
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Failed to get entropy monitoring summary: {str(e)}"}), 500

@app.route("/external/bridge/configure/<bridge_id>", methods=["POST"])
def configure_coherence_bridge(bridge_id):
    """Configure a coherence bridge"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        translation_map = data.get("translation_map", {})
        sync_frequency = data.get("sync_frequency", 300)
        
        # Configure bridge
        success = external_networks["default"].configure_coherence_bridge(bridge_id, translation_map, sync_frequency)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Coherence bridge configured successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to configure coherence bridge"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to configure coherence bridge: {str(e)}"}), 500

@app.route("/external/bridge/translate/<bridge_id>", methods=["POST"])
def translate_data_for_bridge(bridge_id):
    """Translate data for a coherence bridge"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        # Translate data
        translated_data = external_networks["default"].translate_data_for_bridge(bridge_id, data)
        
        return jsonify({
            "status": "success",
            "translated_data": translated_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to translate data: {str(e)}"}), 500

@app.route("/external/bridge/sync/<bridge_id>", methods=["POST"])
def sync_coherence_bridge(bridge_id):
    """Synchronize a coherence bridge"""
    try:
        # Synchronize bridge
        result = external_networks["default"].sync_coherence_bridge(bridge_id)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to synchronize coherence bridge: {str(e)}"}), 500

# Inter-System Councils API endpoints
@app.route("/uhes/councils/register", methods=["POST"])
def register_external_system():
    """Register an external system for participation in inter-system councils"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        system_id = safe_get(sanitized_data, "system_id", "")
        initial_coherence = safe_get(sanitized_data, "initial_coherence", 0.5)
        initial_trust = safe_get(sanitized_data, "initial_trust", 0.5)
        
        # Register system
        success = inter_system_councils.register_external_system(
            system_id=system_id,
            initial_coherence=initial_coherence,
            initial_trust=initial_trust
        )
        
        if success:
            return jsonify({"status": "success", "message": "System registered"})
        else:
            return jsonify({"status": "error", "message": "Failed to register system"}), 400
            
    except Exception as e:
        return jsonify({"error": f"System registration failed: {str(e)}"}), 500

@app.route("/uhes/councils/form", methods=["POST"])
def form_council():
    """Form a new inter-system council"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        council_name = safe_get(sanitized_data, "council_name", "")
        member_systems = safe_get(sanitized_data, "member_systems", [])
        
        # Form council
        council_id = inter_system_councils.form_council(
            council_name=council_name,
            member_systems=member_systems
        )
        
        if council_id:
            return jsonify({
                "status": "success",
                "council_id": council_id
            })
        else:
            return jsonify({"status": "error", "message": "Failed to form council"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Council formation failed: {str(e)}"}), 500

@app.route("/uhes/councils/proposal", methods=["POST"])
def create_inter_system_proposal():
    """Create a new inter-system governance proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        council_id = safe_get(sanitized_data, "council_id", "")
        proposer_system = safe_get(sanitized_data, "proposer_system", "")
        title = safe_get(sanitized_data, "title", "")
        description = safe_get(sanitized_data, "description", "")
        target_systems = safe_get(sanitized_data, "target_systems", [])
        target_changes = safe_get(sanitized_data, "target_changes", {})
        
        # Create proposal
        proposal_id = inter_system_councils.create_inter_system_proposal(
            council_id=council_id,
            proposer_system=proposer_system,
            title=title,
            description=description,
            target_systems=target_systems,
            target_changes=target_changes
        )
        
        if proposal_id:
            return jsonify({
                "status": "success",
                "proposal_id": proposal_id
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create proposal"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Proposal creation failed: {str(e)}"}), 500

@app.route("/uhes/councils/vote", methods=["POST"])
def vote_on_inter_system_proposal():
    """Vote on an inter-system governance proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        system_id = safe_get(sanitized_data, "system_id", "")
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        vote_weight = safe_get(sanitized_data, "vote_weight", 0.0)
        
        # Record vote
        success = inter_system_councils.vote_on_inter_system_proposal(
            system_id=system_id,
            proposal_id=proposal_id,
            vote_weight=vote_weight
        )
        
        if success:
            return jsonify({"status": "success", "message": "Vote recorded"})
        else:
            return jsonify({"status": "error", "message": "Failed to record vote"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Voting failed: {str(e)}"}), 500

@app.route("/uhes/councils/evaluate", methods=["POST"])
def evaluate_inter_system_proposal():
    """Evaluate an inter-system proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        
        # Evaluate proposal
        is_approved, score = inter_system_councils.evaluate_inter_system_proposal(
            proposal_id=proposal_id
        )
        
        return jsonify({
            "status": "success",
            "approved": is_approved,
            "score": score
        })
            
    except Exception as e:
        return jsonify({"error": f"Proposal evaluation failed: {str(e)}"}), 500

@app.route("/uhes/councils/implement", methods=["POST"])
def implement_inter_system_proposal():
    """Implement an approved inter-system proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        
        # Implement proposal
        success = inter_system_councils.implement_inter_system_proposal(
            proposal_id=proposal_id
        )
        
        if success:
            return jsonify({"status": "success", "message": "Proposal implemented"})
        else:
            return jsonify({"status": "error", "message": "Failed to implement proposal"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Proposal implementation failed: {str(e)}"}), 500

@app.route("/uhes/councils/info/<council_id>", methods=["GET"])
def get_council_info(council_id):
    """Get detailed information about a council"""
    try:
        council_info = inter_system_councils.get_council_info(council_id)
        
        if council_info:
            return jsonify({
                "status": "success",
                "council_info": council_info
            })
        else:
            return jsonify({"status": "error", "message": "Council not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to get council info: {str(e)}"}), 500

@app.route("/uhes/councils/reputation/<system_id>", methods=["GET"])
def get_system_reputation(system_id):
    """Get reputation information for a system"""
    try:
        reputation_info = inter_system_councils.get_system_reputation(system_id)
        
        if reputation_info:
            return jsonify({
                "status": "success",
                "reputation_info": reputation_info
            })
        else:
            return jsonify({"status": "error", "message": "System not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to get system reputation: {str(e)}"}), 500

# Global Harmonic Synchronizer APIs
@app.route("/global-synchronizer/metrics", methods=["GET"])
def get_global_coherence_metrics():
    """Get global coherence metrics"""
    try:
        # Get global coherence metrics
        metrics = global_synchronizer.aggregate_coherence_metrics()
        
        if metrics:
            return jsonify({
                "status": "success",
                "metrics": {
                    "timestamp": metrics.timestamp,
                    "global_coherence": metrics.global_coherence,
                    "regional_entropy_hotspots": metrics.regional_entropy_hotspots,
                    "systemic_equilibrium": metrics.systemic_equilibrium,
                    "connected_nodes": metrics.connected_nodes
                }
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to aggregate coherence metrics"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to get global coherence metrics: {str(e)}"}), 500

@app.route("/global-synchronizer/map", methods=["GET"])
def get_global_coherence_map():
    """Get global coherence map"""
    try:
        # Get global coherence map
        coherence_map = global_synchronizer.get_global_coherence_map()
        
        return jsonify(coherence_map)
    except Exception as e:
        return jsonify({"error": f"Failed to get global coherence map: {str(e)}"}), 500

@app.route("/global-synchronizer/analytics", methods=["GET"])
def get_systemic_equilibrium_analytics():
    """Get systemic equilibrium analytics"""
    try:
        # Get systemic equilibrium analytics
        analytics = global_synchronizer.get_systemic_equilibrium_analytics()
        
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"error": f"Failed to get systemic equilibrium analytics: {str(e)}"}), 500

@app.route("/global-synchronizer/caf", methods=["GET"])
def calculate_coherence_amplification_factor():
    """Calculate Coherence Amplification Factor"""
    try:
        # Calculate CAF
        caf_result = global_synchronizer.calculate_coherence_amplification_factor()
        
        return jsonify(caf_result)
    except Exception as e:
        return jsonify({"error": f"Failed to calculate CAF: {str(e)}"}), 500

@app.route("/global-synchronizer/distribute-feedback", methods=["POST"])
def distribute_stabilizing_feedback():
    """Distribute stabilizing feedback to connected nodes"""
    try:
        # Distribute stabilizing feedback
        feedback_result = global_synchronizer.distribute_stabilizing_feedback()
        
        return jsonify(feedback_result)
    except Exception as e:
        return jsonify({"error": f"Failed to distribute stabilizing feedback: {str(e)}"}), 500

@app.route("/global-synchronizer/patterns", methods=["GET"])
def identify_disharmony_patterns():
    """Identify disharmony patterns using adaptive learning algorithms"""
    try:
        # Identify disharmony patterns
        patterns = global_synchronizer.identify_disharmony_patterns()
        
        return jsonify({
            "status": "success",
            "patterns": patterns,
            "count": len(patterns)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to identify disharmony patterns: {str(e)}"}), 500

@app.route("/global-synchronizer/learn", methods=["POST"])
def learn_from_adjustments():
    """Learn from past field adjustments to improve future corrections"""
    try:
        # Learn from adjustments
        learning_result = global_synchronizer.learn_from_adjustments()
        
        return jsonify(learning_result)
    except Exception as e:
        return jsonify({"error": f"Failed to learn from adjustments: {str(e)}"}), 500

# Human Feedback APIs
@app.route("/human-feedback/submit", methods=["POST"])
def submit_human_feedback():
    """Submit human feedback for coherence recalibration"""
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400
        
        user_id = data.get("user_id", "anonymous")
        feedback_type = data.get("feedback_type", "general")
        message = data.get("message", "")
        rating = data.get("rating", 5)
        
        # Submit feedback
        feedback_id = human_feedback_system.submit_feedback(user_id, feedback_type, message, rating)
        
        if feedback_id:
            return jsonify({
                "status": "success",
                "feedback_id": feedback_id,
                "message": "Feedback submitted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to submit feedback"
            }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to submit feedback: {str(e)}"}), 500

@app.route("/human-feedback/process", methods=["POST"])
def process_human_feedback():
    """Process human feedback to generate coherence recalibration recommendations"""
    try:
        # Process feedback
        processing_result = human_feedback_system.process_feedback()
        
        return jsonify(processing_result)
    except Exception as e:
        return jsonify({"error": f"Failed to process feedback: {str(e)}"}), 500

@app.route("/human-feedback/history", methods=["GET"])
def get_feedback_history():
    """Get human feedback history"""
    try:
        limit = int(request.args.get("limit", 50))
        
        # Get feedback history
        history = human_feedback_system.get_feedback_history(limit)
        
        return jsonify({
            "status": "success",
            "feedback": history,
            "count": len(history)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get feedback history: {str(e)}"}), 500

@app.route("/human-feedback/summary", methods=["GET"])
def get_feedback_summary():
    """Get human feedback summary statistics"""
    try:
        # Get feedback summary
        summary = human_feedback_system.get_feedback_summary()
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Failed to get feedback summary: {str(e)}"}), 500

# Route to serve the main index.html file
# Add a route to serve index.html at the root
# Inter-System Councils API endpoints
@app.route("/uhes/councils/register", methods=["POST"])
def register_inter_system():
    """Register an external system for participation in inter-system councils"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        system_id = safe_get(sanitized_data, "system_id", "")
        initial_coherence = safe_get(sanitized_data, "initial_coherence", 0.5)
        initial_trust = safe_get(sanitized_data, "initial_trust", 0.5)
        
        # Register system
        success = inter_system_councils.register_external_system(
            system_id=system_id,
            initial_coherence=initial_coherence,
            initial_trust=initial_trust
        )
        
        if success:
            return jsonify({"status": "success", "message": "System registered"})
        else:
            return jsonify({"status": "error", "message": "Failed to register system"}), 400
            
    except Exception as e:
        return jsonify({"error": f"System registration failed: {str(e)}"}), 500

@app.route("/uhes/councils/form", methods=["POST"])
def form_inter_system_council():
    """Form a new inter-system council"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        council_name = safe_get(sanitized_data, "council_name", "")
        member_systems = safe_get(sanitized_data, "member_systems", [])
        
        # Form council
        council_id = inter_system_councils.form_council(
            council_name=council_name,
            member_systems=member_systems
        )
        
        if council_id:
            return jsonify({
                "status": "success",
                "council_id": council_id
            })
        else:
            return jsonify({"status": "error", "message": "Failed to form council"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Council formation failed: {str(e)}"}), 500

@app.route("/uhes/councils/proposal", methods=["POST"])
def create_inter_system_council_proposal():
    """Create a new inter-system governance proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        council_id = safe_get(sanitized_data, "council_id", "")
        proposer_system = safe_get(sanitized_data, "proposer_system", "")
        title = safe_get(sanitized_data, "title", "")
        description = safe_get(sanitized_data, "description", "")
        target_systems = safe_get(sanitized_data, "target_systems", [])
        target_changes = safe_get(sanitized_data, "target_changes", {})
        
        # Create proposal
        proposal_id = inter_system_councils.create_inter_system_proposal(
            council_id=council_id,
            proposer_system=proposer_system,
            title=title,
            description=description,
            target_systems=target_systems,
            target_changes=target_changes
        )
        
        if proposal_id:
            return jsonify({
                "status": "success",
                "proposal_id": proposal_id
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create proposal"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Proposal creation failed: {str(e)}"}), 500

@app.route("/uhes/councils/vote", methods=["POST"])
def vote_on_inter_system_council_proposal():
    """Vote on an inter-system governance proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        system_id = safe_get(sanitized_data, "system_id", "")
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        vote_weight = safe_get(sanitized_data, "vote_weight", 0.0)
        
        # Record vote
        success = inter_system_councils.vote_on_inter_system_proposal(
            system_id=system_id,
            proposal_id=proposal_id,
            vote_weight=vote_weight
        )
        
        if success:
            return jsonify({"status": "success", "message": "Vote recorded"})
        else:
            return jsonify({"status": "error", "message": "Failed to record vote"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Voting failed: {str(e)}"}), 500

@app.route("/uhes/councils/evaluate", methods=["POST"])
def evaluate_inter_system_council_proposal():
    """Evaluate an inter-system proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        
        # Evaluate proposal
        is_approved, score = inter_system_councils.evaluate_inter_system_proposal(
            proposal_id=proposal_id
        )
        
        return jsonify({
            "status": "success",
            "approved": is_approved,
            "score": score
        })
            
    except Exception as e:
        return jsonify({"error": f"Proposal evaluation failed: {str(e)}"}), 500

@app.route("/uhes/councils/implement", methods=["POST"])
def implement_inter_system_council_proposal():
    """Implement an approved inter-system proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Sanitize input data
        sanitized_data = sanitize_data(data)
        
        # Extract parameters
        proposal_id = safe_get(sanitized_data, "proposal_id", "")
        
        # Implement proposal
        success = inter_system_councils.implement_inter_system_proposal(
            proposal_id=proposal_id
        )
        
        if success:
            return jsonify({"status": "success", "message": "Proposal implemented"})
        else:
            return jsonify({"status": "error", "message": "Failed to implement proposal"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Proposal implementation failed: {str(e)}"}), 500

@app.route("/uhes/councils/info/<council_id>", methods=["GET"])
def get_inter_system_council_info(council_id):
    """Get detailed information about a council"""
    try:
        council_info = inter_system_councils.get_council_info(council_id)
        
        if council_info:
            return jsonify({
                "status": "success",
                "council_info": council_info
            })
        else:
            return jsonify({"status": "error", "message": "Council not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to get council info: {str(e)}"}), 500

@app.route("/uhes/councils/reputation/<system_id>", methods=["GET"])
def get_inter_system_reputation(system_id):
    """Get reputation information for a system"""
    try:
        reputation_info = inter_system_councils.get_system_reputation(system_id)
        
        if reputation_info:
            return jsonify({
                "status": "success",
                "reputation_info": reputation_info
            })
        else:
            return jsonify({"status": "error", "message": "System not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to get system reputation: {str(e)}"}), 500

# Ethical Coherence Governance (ECG) API endpoints
@app.route("/uhes/ecg/rules", methods=["GET"])
def get_ecg_rules():
    """Get all ethical rules in the ECG system"""
    try:
        rules = {}
        for rule_id, rule in ecg_engine.rules.items():
            rules[rule_id] = {
                "name": rule.name,
                "description": rule.description,
                "category": rule.category,
                "weight": rule.weight,
                "threshold": rule.threshold,
                "active": rule.active
            }
        
        return jsonify({
            "status": "success",
            "rules": rules
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get ECG rules: {str(e)}"}), 500

@app.route("/uhes/ecg/rules/add", methods=["POST"])
def add_ecg_rule():
    """Add a new ethical rule to the ECG system"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        rule_id = data.get("rule_id", f"rule_{int(time.time() * 1000000)}")
        name = data.get("name", "")
        description = data.get("description", "")
        category = data.get("category", "general")
        weight = data.get("weight", 0.5)
        threshold = data.get("threshold", 0.8)
        formula = data.get("formula", "")
        
        # Create rule
        rule = EthicalRule(
            rule_id=rule_id,
            name=name,
            description=description,
            category=category,
            weight=weight,
            threshold=threshold,
            formula=formula
        )
        
        # Add rule
        success = ecg_engine.add_ethical_rule(rule)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Rule added successfully",
                "rule_id": rule_id
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to add rule"
            }), 400
    except Exception as e:
        return jsonify({"error": f"Failed to add ECG rule: {str(e)}"}), 500

@app.route("/uhes/ecg/assess/proposal/<proposal_id>", methods=["POST"])
def assess_proposal_ethics(proposal_id):
    """Assess the ethical compliance of a governance proposal"""
    try:
        # Get proposal
        proposal = governance.proposals.get(proposal_id)
        if not proposal:
            return jsonify({
                "status": "error",
                "message": "Proposal not found"
            }), 404
        
        # Assess proposal ethics
        assessment = ecg_engine.assess_proposal_ethics(proposal)
        
        return jsonify({
            "status": "success",
            "assessment": {
                "assessment_id": assessment.assessment_id,
                "target_id": assessment.target_id,
                "timestamp": assessment.timestamp,
                "overall_score": assessment.overall_score,
                "category_scores": assessment.category_scores,
                "rule_compliance": assessment.rule_compliance,
                "violations": assessment.violations,
                "recommendations": assessment.recommendations,
                "approved": assessment.approved
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to assess proposal ethics: {str(e)}"}), 500

@app.route("/uhes/ecg/report/transparency", methods=["GET"])
def generate_transparency_report():
    """Generate a transparency report for governance activities"""
    try:
        # Get period from query parameters (default 30 days)
        period_days = int(request.args.get("period_days", 30))
        
        # Generate transparency report
        report = ecg_engine.generate_transparency_report(period_days)
        
        return jsonify({
            "status": "success",
            "report": {
                "report_id": report.report_id,
                "period_start": report.period_start,
                "period_end": report.period_end,
                "generated_at": report.generated_at,
                "activities_count": len(report.activities),
                "ethical_scores": report.ethical_scores,
                "violations_summary": report.violations_summary,
                "recommendations": report.recommendations
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate transparency report: {str(e)}"}), 500

@app.route("/uhes/ecg/compliance/summary", methods=["GET"])
def get_ethical_compliance_summary():
    """Get a summary of ethical compliance across the system"""
    try:
        # Get compliance summary
        summary = ecg_engine.get_ethical_compliance_summary()
        
        return jsonify({
            "status": "success",
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get compliance summary: {str(e)}"}), 500

@app.route("/uhes/ecg/enforce/proposal/<proposal_id>", methods=["POST"])
def enforce_ethical_governance(proposal_id):
    """Enforce ethical governance by assessing and potentially blocking proposals"""
    try:
        # Get proposal
        proposal = governance.proposals.get(proposal_id)
        if not proposal:
            return jsonify({
                "status": "error",
                "message": "Proposal not found"
            }), 404
        
        # Enforce ethical governance
        approved, message = ecg_engine.enforce_ethical_governance(proposal)
        
        return jsonify({
            "status": "success",
            "approved": approved,
            "message": message
        })
    except Exception as e:
        return jsonify({"error": f"Failed to enforce ethical governance: {str(e)}"}), 500

# OpenAGI Harmonic Outreach API endpoints
@app.route("/uhes/openagi/outreach/propose", methods=["POST"])
async def propose_harmonic_outreach():
    """Propose a harmonic outreach initiative through OpenAGI modules"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        initiative_data = data.get("initiative_data", {})
        economic_state_data = data.get("economic_state", {})
        
        # Create economic optimization object
        economic_state = EconomicOptimization(
            timestamp=economic_state_data.get("timestamp", time.time()),
            coherence_stability_index=economic_state_data.get("coherence_stability_index", 0.8),
            recommended_minting_rate=economic_state_data.get("recommended_minting_rate", 0.05),
            inflation_adjustment=economic_state_data.get("inflation_adjustment", 0.0),
            token_flow_optimization=economic_state_data.get("token_flow_optimization", {})
        )
        
        # Propose outreach initiative
        decision = await quantum_ai.propose_harmonic_outreach_initiative(
            initiative_data=initiative_data,
            economic_state=economic_state
        )
        
        return jsonify({
            "status": "success",
            "decision": {
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp,
                "decision_type": decision.decision_type,
                "description": decision.description,
                "confidence": decision.confidence,
                "impact_assessment": decision.impact_assessment,
                "explanation": decision.explanation,
                "implementation_plan": decision.implementation_plan
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to propose outreach initiative: {str(e)}"}), 500

@app.route("/uhes/openagi/outreach/opportunities", methods=["POST"])
async def evaluate_network_expansion_opportunities():
    """Evaluate network expansion opportunities for harmonic outreach"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        current_state = data.get("current_state", {})
        
        # Evaluate opportunities
        opportunities = await quantum_ai.evaluate_network_expansion_opportunities(current_state)
        
        return jsonify({
            "status": "success",
            "opportunities": opportunities
        })
    except Exception as e:
        return jsonify({"error": f"Failed to evaluate opportunities: {str(e)}"}), 500

@app.route("/uhes/openagi/outreach/generate", methods=["POST"])
async def generate_harmonic_outreach_proposal():
    """Generate a complete harmonic outreach proposal"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        opportunity = data.get("opportunity", {})
        system_state = data.get("system_state", {})
        
        # Generate proposal
        proposal = await quantum_ai.generate_harmonic_outreach_proposal(opportunity, system_state)
        
        return jsonify({
            "status": "success",
            "proposal": proposal
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate proposal: {str(e)}"}), 500

@app.route('/')
def index():
    """Serve the main index.html file"""
    return app.send_static_file('index.html')

if __name__ == "__main__":
    # Print the UI directory path for debugging
    print(f"UI Directory: {UI_DIR}")
    print(f"UI Directory exists: {os.path.exists(UI_DIR)}")
    app.run(host="0.0.0.0", port=5000)