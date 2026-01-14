#!/usr/bin/env python3
"""
Multi-node simulation test with artificial latency injection
"""

import sys
import os
import time
import random
import threading
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate
from openagi.token_rules import validate_harmonic_tx, apply_token_effects

class LatencySimulator:
    """Simulates network latency between nodes"""
    
    def __init__(self, min_latency: float = 0.01, max_latency: float = 0.1):
        self.min_latency = min_latency
        self.max_latency = max_latency
    
    def inject_latency(self):
        """Inject random latency"""
        latency = random.uniform(self.min_latency, self.max_latency)
        time.sleep(latency)
        return latency

class NodeSimulator:
    """Simulates a validator node"""
    
    def __init__(self, node_id: str, latency_simulator: LatencySimulator):
        self.node_id = node_id
        self.latency_simulator = latency_simulator
        self.transactions = []
        self.ledger = {"balances": {}, "chr": {}}
    
    def generate_signal(self, freq: float, phase: float, duration: float = 0.5) -> tuple:
        """Generate a synthetic signal"""
        t = np.linspace(0, duration, 1024)
        x = np.sin(2 * np.pi * freq * t + phase) + 0.01 * np.random.randn(len(t))
        return t, x
    
    def create_snapshot(self, freq: float, phase: float) -> Any:
        """Create a harmonic snapshot"""
        t, values = self.generate_signal(freq, phase)
        return make_snapshot(self.node_id, t.tolist(), values.tolist(), secret_key=f"key{self.node_id}")
    
    def process_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Process a transaction with simulated latency"""
        # Inject latency
        latency = self.latency_simulator.inject_latency()
        
        # Validate transaction
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        ok = validate_harmonic_tx(tx, config)
        
        if ok:
            apply_token_effects(self.ledger, tx)
            self.transactions.append(tx)
            return {
                "node_id": self.node_id,
                "result": "accepted",
                "latency": latency,
                "timestamp": time.time()
            }
        else:
            return {
                "node_id": self.node_id,
                "result": "rejected",
                "latency": latency,
                "timestamp": time.time()
            }

def run_multi_node_simulation():
    """Run multi-node simulation with latency injection"""
    print("🧪 Multi-Node Simulation Test with Latency Injection")
    print("=" * 60)
    
    # Initialize latency simulator
    latency_sim = LatencySimulator(min_latency=0.02, max_latency=0.08)
    
    # Create simulated nodes
    nodes = [
        NodeSimulator("node-A", latency_sim),
        NodeSimulator("node-B", latency_sim),
        NodeSimulator("node-C", latency_sim)
    ]
    
    print(f"📍 Created {len(nodes)} simulated nodes")
    
    # Generate coherent signals for high coherence test
    print("🔄 Generating coherent signals...")
    snapshots = []
    for i, node in enumerate(nodes):
        # Create coherent signals with slight phase differences
        freq = 50.0  # 50 Hz
        phase = i * 0.1  # Small phase difference
        snapshot = node.create_snapshot(freq, phase)
        snapshots.append(snapshot)
    
    # Calculate coherence between nodes
    print("📊 Calculating coherence scores...")
    coherence_scores = []
    for i, local_node in enumerate(nodes):
        local_snapshot = snapshots[i]
        remote_snapshots = [s for j, s in enumerate(snapshots) if i != j]
        cs = compute_coherence_score(local_snapshot, remote_snapshots)
        coherence_scores.append(cs)
        print(f"   {local_node.node_id}: {cs:.4f}")
    
    # Perform recursive validation
    print("🔍 Performing recursive validation...")
    is_valid, proof_bundle = recursive_validate(snapshots, threshold=0.75)
    
    if proof_bundle:
        print(f"   Aggregated Coherence: {proof_bundle.aggregated_CS:.4f}")
        print(f"   Validation Result: {'✅ APPROVED' if is_valid else '❌ REJECTED'}")
    
    # Simulate transactions if validation passed
    if is_valid and proof_bundle:
        print("💰 Simulating transactions...")
        transactions_results = []
        
        # Create transactions for each node
        for i, node in enumerate(nodes):
            tx = {
                "id": f"tx-{node.node_id}-{int(time.time())}",
                "type": "harmonic",
                "action": "mint",
                "token": "FLX",
                "sender": node.node_id,
                "receiver": node.node_id,
                "amount": 100 + i * 10,  # Different amounts
                "aggregated_cs": proof_bundle.aggregated_CS,
                "sender_chr": 0.8 + i * 0.05,  # Different CHR scores
                "timestamp": time.time()
            }
            
            # Process transaction with latency
            result = node.process_transaction(tx)
            transactions_results.append(result)
        
        # Display results
        print("\n📈 Transaction Results:")
        total_latency = 0
        successful_tx = 0
        for result in transactions_results:
            status = "✅" if result["result"] == "accepted" else "❌"
            print(f"   {result['node_id']}: {status} (Latency: {result['latency']:.3f}s)")
            total_latency += result["latency"]
            if result["result"] == "accepted":
                successful_tx += 1
        
        avg_latency = total_latency / len(transactions_results)
        print(f"\n📊 Summary:")
        print(f"   Successful Transactions: {successful_tx}/{len(transactions_results)}")
        print(f"   Average Latency: {avg_latency:.3f}s")
        print(f"   Total Simulation Time: {time.time() - transactions_results[0]['timestamp']:.3f}s")
    
    print("\n✅ Multi-node simulation completed successfully!")

if __name__ == "__main__":
    run_multi_node_simulation()