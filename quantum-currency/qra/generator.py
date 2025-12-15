#!/usr/bin/env python3
"""
QRA (Quantum Resonance Authentication) Generator
Generates bioresonant QRA keys for all nodes/users in the quantum currency system
"""

import argparse
import json
import logging
import numpy as np
import os
import sys
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QRAGenerator:
    """Generates and manages QRA keys for quantum currency system nodes/users"""
    
    def __init__(self, qra_dir: str = "qra/keys"):
        self.qra_dir = qra_dir
        # Create directory if it doesn't exist
        if not os.path.exists(qra_dir):
            os.makedirs(qra_dir)
        
    def generate_qra_key(self, node_id: str) -> Dict[str, Any]:
        """
        Generate a QRA key for a specific node/user
        
        Args:
            node_id: Unique identifier for the node/user
            
        Returns:
            Dictionary containing QRA key data
        """
        # Generate bioresonant parameters
        coherence_score = np.random.uniform(0.85, 1.0)  # Coherence score between 0.85-1.0
        phi_ratio = np.random.uniform(1.6, 1.62)  # Close to golden ratio
        inertial_efficiency = np.random.uniform(0.01, 0.1)  # I_eff cost
        
        # Create QRA key
        qra_key = {
            "node_id": node_id,
            "qra_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "Coherence_Score": float(coherence_score),
            "Phi_Ratio": float(phi_ratio),
            "I_eff_Cost": float(inertial_efficiency),
            "harmonic_signature": self._generate_harmonic_signature(coherence_score, phi_ratio),
            "version": "1.3"
        }
        
        return qra_key
    
    def _generate_harmonic_signature(self, coherence_score: float, phi_ratio: float) -> List[float]:
        """
        Generate a harmonic signature based on coherence and phi ratio
        
        Args:
            coherence_score: Node's coherence score
            phi_ratio: Phi ratio value
            
        Returns:
            List of harmonic signature values
        """
        # Generate a signature based on mathematical relationships
        signature = []
        for i in range(5):  # Generate 5 signature components
            # Create harmonic relationships
            component = coherence_score * (phi_ratio ** (i/2)) + np.random.normal(0, 0.01)
            signature.append(float(component))
        return signature
    
    def generate_all_nodes(self, node_list: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate QRA keys for all nodes in the system
        
        Args:
            node_list: List of node IDs (if None, generates sample nodes)
            
        Returns:
            Dictionary mapping node IDs to their QRA keys
        """
        if node_list is None:
            # Generate sample nodes if none provided
            node_list = [f"node_{i:03d}" for i in range(1, 11)]  # nodes 001-010
            node_list.extend(["SYSTEM_A", "USER_ALPHA", "NODE_001"])  # Add some specific nodes
            
        qra_keys = {}
        
        logger.info(f"Generating QRA keys for {len(node_list)} nodes")
        
        for node_id in node_list:
            qra_key = self.generate_qra_key(node_id)
            qra_keys[node_id] = qra_key
            
            # Save to file
            filename = os.path.join(self.qra_dir, f"{node_id}_qra.json")
            with open(filename, 'w') as f:
                json.dump(qra_key, f, indent=2)
                
            logger.debug(f"Generated QRA key for {node_id}")
        
        logger.info(f"✅ Generated QRA keys for all {len(node_list)} nodes")
        return qra_keys
    
    def load_qra(self, node_id: str) -> Dict[str, Any]:
        """
        Load a QRA key for a specific node
        
        Args:
            node_id: Node ID to load QRA for
            
        Returns:
            QRA key data or empty dict if not found
        """
        filename = os.path.join(self.qra_dir, f"{node_id}_qra.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"QRA key not found for node {node_id}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='QRA Generator for Quantum Currency System')
    parser.add_argument('--generate_all_nodes', action='store_true', help='Generate QRA keys for all nodes')
    parser.add_argument('--node_id', type=str, help='Generate QRA key for specific node')
    parser.add_argument('--qra_dir', type=str, default="qra/keys", help='Directory to store QRA keys')
    
    args = parser.parse_args()
    
    qra_generator = QRAGenerator(qra_dir=args.qra_dir)
    
    if args.generate_all_nodes:
        logger.info("[QRA] Generating bioresonant QRA keys for all nodes")
        qra_keys = qra_generator.generate_all_nodes()
        print(f"Generated {len(qra_keys)} QRA keys")
        
    elif args.node_id:
        logger.info(f"[QRA] Generating bioresonant QRA key for node {args.node_id}")
        qra_key = qra_generator.generate_qra_key(args.node_id)
        filename = os.path.join(args.qra_dir, f"{args.node_id}_qra.json")
        with open(filename, 'w') as f:
            json.dump(qra_key, f, indent=2)
        print(f"Generated QRA key for {args.node_id}")
        print(json.dumps(qra_key, indent=2))
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()