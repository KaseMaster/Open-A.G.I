#!/usr/bin/env python3
"""
Gravity Well Governance
Field-level security with dynamic node isolation for coherence disruption prevention
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GravityWellGovernance:
    """Implements field-level security with Gravity Well detection and node isolation"""
    
    def __init__(self, security_log_file: str = "logs/security_log.json"):
        self.security_log_file = security_log_file
        self.isolation_history = []
        self.G_crit = 0.05  # Critical gravity threshold
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(security_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Load existing isolation history if available
        self._load_isolation_history()
    
    def _load_isolation_history(self):
        """Load isolation history from file"""
        if os.path.exists(self.security_log_file):
            try:
                with open(self.security_log_file, 'r') as f:
                    self.isolation_history = json.load(f)
                logger.info(f"Loaded {len(self.isolation_history)} isolation records")
            except Exception as e:
                logger.warning(f"Failed to load isolation history: {e}")
                self.isolation_history = []
    
    def _save_isolation_history(self):
        """Save isolation history to file"""
        try:
            with open(self.security_log_file, 'w') as f:
                json.dump(self.isolation_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save isolation history: {e}")
    
    def compute_g_vector(self, transaction_cluster: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute gravity vector for a transaction cluster
        
        Args:
            transaction_cluster: List of transactions in the cluster
            
        Returns:
            Dict mapping node IDs to their gravity magnitudes
        """
        g_vector = {}
        
        # For each node in the cluster, compute gravity based on transaction patterns
        for tx in transaction_cluster:
            sender = tx.get('sender')
            recipient = tx.get('recipient')
            amount = tx.get('amount', 0)
            coherence_score = tx.get('coherence_score', 0.95)
            
            # Compute gravity contribution for sender
            if sender:
                # Gravity increases with large transactions and low coherence
                g_contribution = amount * (1.0 - coherence_score)
                if sender in g_vector:
                    g_vector[sender] += g_contribution
                else:
                    g_vector[sender] = g_contribution
            
            # Compute gravity contribution for recipient
            if recipient:
                # Similar calculation for recipient
                g_contribution = amount * (1.0 - coherence_score)
                if recipient in g_vector:
                    g_vector[recipient] += g_contribution
                else:
                    g_vector[recipient] = g_contribution
        
        return g_vector
    
    def isolate_node_if_needed(self, node_id: str, g_mag: float) -> bool:
        """
        Isolate a node if its gravity magnitude exceeds critical threshold
        
        Args:
            node_id: Node identifier
            g_mag: Gravity magnitude for the node
            
        Returns:
            bool: True if node was isolated, False otherwise
        """
        if g_mag > self.G_crit:
            # Node needs isolation
            isolation_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "node_id": node_id,
                "gravity_magnitude": float(g_mag),
                "action": "isolated",
                "reason": "Gravity Well detected"
            }
            
            self.isolation_history.append(isolation_record)
            self._save_isolation_history()
            
            logger.warning(f"[SECURITY] Node {node_id} isolated: Gravity Well detected (g={g_mag:.4f})")
            return True
        return False
    
    def process_transaction_cluster(self, transaction_cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a transaction cluster for Gravity Well detection
        
        Args:
            transaction_cluster: List of transactions to process
            
        Returns:
            Dict with processing results
        """
        # Compute gravity vector
        g_vector = self.compute_g_vector(transaction_cluster)
        
        # Check each node for isolation
        isolated_nodes = []
        for node_id, g_mag in g_vector.items():
            if self.isolate_node_if_needed(node_id, g_mag):
                isolated_nodes.append(node_id)
        
        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processed_transactions": len(transaction_cluster),
            "gravity_vector": g_vector,
            "isolated_nodes": isolated_nodes,
            "critical_threshold": self.G_crit
        }
        
        if isolated_nodes:
            logger.info(f"Isolated {len(isolated_nodes)} nodes due to Gravity Well detection")
        
        return result
    
    def get_isolation_history(self, node_id: Optional[str] = None) -> list:
        """
        Get isolation history for a specific node or all nodes
        
        Args:
            node_id: Node identifier (optional)
            
        Returns:
            list: Isolation records
        """
        if node_id:
            return [record for record in self.isolation_history if record.get("node_id") == node_id]
        return self.isolation_history
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security statistics
        
        Returns:
            Dict with security statistics
        """
        total_isolations = len(self.isolation_history)
        unique_nodes = len(set(record.get("node_id") for record in self.isolation_history))
        
        # Calculate average gravity magnitude of isolated nodes
        if self.isolation_history:
            avg_gravity = np.mean([record.get("gravity_magnitude", 0) 
                                 for record in self.isolation_history])
        else:
            avg_gravity = 0.0
        
        return {
            "total_isolations": total_isolations,
            "unique_isolated_nodes": unique_nodes,
            "average_gravity_magnitude": float(avg_gravity),
            "critical_threshold": self.G_crit
        }

# Example usage
def main():
    gravity_governance = GravityWellGovernance()
    
    # Example transaction cluster
    transaction_cluster = [
        {
            "sender": "node_001",
            "recipient": "node_002",
            "amount": 100.0,
            "coherence_score": 0.92
        },
        {
            "sender": "node_003",
            "recipient": "node_004",
            "amount": 500.0,
            "coherence_score": 0.85  # Lower coherence, higher gravity
        }
    ]
    
    # Process cluster
    result = gravity_governance.process_transaction_cluster(transaction_cluster)
    print("Gravity Well Governance Results:")
    print(json.dumps(result, indent=2))
    
    # Get stats
    stats = gravity_governance.get_security_stats()
    print("\nSecurity Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()