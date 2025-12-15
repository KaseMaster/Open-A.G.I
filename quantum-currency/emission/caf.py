#!/usr/bin/env python3
"""
Coherence Augmentation Function (CAF)
Implements quantum value emission based on system coherence
"""

import logging
import numpy as np
from typing import Dict, Any
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoherenceAugmentationFunction:
    """Implements CAF emission policy based on system coherence"""
    
    def __init__(self, emission_log_file: str = "emission/emission_log.json"):
        self.emission_log_file = emission_log_file
        self.emission_history = []
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(emission_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Load existing emission history if available
        self._load_emission_history()
    
    def _load_emission_history(self):
        """Load emission history from file"""
        if os.path.exists(self.emission_log_file):
            try:
                with open(self.emission_log_file, 'r') as f:
                    self.emission_history = json.load(f)
                logger.info(f"Loaded {len(self.emission_history)} emission records")
            except Exception as e:
                logger.warning(f"Failed to load emission history: {e}")
                self.emission_history = []
    
    def _save_emission_history(self):
        """Save emission history to file"""
        try:
            with open(self.emission_log_file, 'w') as f:
                json.dump(self.emission_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emission history: {e}")
    
    def emit_tokens(self, node_id: str, C_system: float, GAS_target: float, phi_inv: float) -> float:
        """
        Emit tokens based on CAF policy
        
        Args:
            node_id: Node identifier
            C_system: Current system coherence
            GAS_target: Target GAS value
            phi_inv: Inverse phi value from HARU model
            
        Returns:
            float: Amount of Ω-tokens emitted
        """
        # Scaling coefficient
        alpha = 1.0
        
        # Calculate emission using CAF formula
        # R_emission = max(0, alpha * ln(C_system / GAS_target) * phi_inv)
        if C_system > 0 and GAS_target > 0:
            try:
                ratio = C_system / GAS_target
                log_ratio = np.log(ratio) if ratio > 0 else 0
                R_emission = max(0, alpha * log_ratio * phi_inv)
            except Exception as e:
                logger.warning(f"Error calculating emission for {node_id}: {e}")
                R_emission = 0.0
        else:
            R_emission = 0.0
            
        # Record emission
        if R_emission > 0:
            emission_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "node_id": node_id,
                "C_system": float(C_system),
                "GAS_target": float(GAS_target),
                "phi_inv": float(phi_inv),
                "emission_amount": float(R_emission)
            }
            
            self.emission_history.append(emission_record)
            self._save_emission_history()
            
            logger.info(f"[EMISSION] Node {node_id} emitting {R_emission:.4f} Ω-tokens")
            
        return R_emission
    
    def get_node_emission_history(self, node_id: str) -> list:
        """
        Get emission history for a specific node
        
        Args:
            node_id: Node identifier
            
        Returns:
            list: Emission records for the node
        """
        return [record for record in self.emission_history if record.get("node_id") == node_id]
    
    def get_total_emission(self) -> float:
        """
        Get total emission across all nodes
        
        Returns:
            float: Total Ω-tokens emitted
        """
        return sum(record.get("emission_amount", 0) for record in self.emission_history)
    
    def get_recent_emission_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get emission statistics for recent period
        
        Args:
            hours: Time period in hours
            
        Returns:
            Dict with emission statistics
        """
        # For simplicity, we'll return basic stats
        # In a real implementation, we would filter by timestamp
        total_emission = self.get_total_emission()
        node_count = len(set(record.get("node_id") for record in self.emission_history))
        
        return {
            "total_emission": total_emission,
            "active_nodes": node_count,
            "average_per_node": total_emission / max(1, node_count) if node_count > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    caf = CoherenceAugmentationFunction()
    
    # Example emission
    node_id = "node_001"
    C_system = 0.98
    GAS_target = 0.95
    phi_inv = 0.618  # Inverse of golden ratio
    
    emission_amount = caf.emit_tokens(node_id, C_system, GAS_target, phi_inv)
    print(f"Emitted {emission_amount:.4f} Ω-tokens for {node_id}")
    
    # Get stats
    stats = caf.get_recent_emission_stats()
    print(f"Total emission: {stats['total_emission']:.4f} Ω-tokens")
    print(f"Active nodes: {stats['active_nodes']}")