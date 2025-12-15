#!/usr/bin/env python3
"""
Shard Manager for QECS
Enables parallelized shard governance with low-latency correction
"""

import json
import time
import threading
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ShardStatus:
    """Represents the status of a shard"""
    shard_id: str
    g_vector: List[float]
    coherence_score: float
    action_efficiency: float
    gravity_well_count: int
    stable: bool
    last_update: float

class ShardManager:
    """Manages parallelized shard governance for low-latency correction"""
    
    def __init__(self):
        self.shards: Dict[str, ShardStatus] = {}
        self.governance_loops: Dict[str, threading.Thread] = {}
        self.running = False
        self.q_seed_affinity_groups: Dict[str, List[str]] = {}
        
    def register_shard(self, shard_id: str, initial_g_vector: Optional[List[float]] = None):
        """
        Register a new shard with the manager
        
        Args:
            shard_id: Unique identifier for the shard
            initial_g_vector: Initial g-vector for the shard
        """
        if initial_g_vector is None:
            initial_g_vector = [0.0, 0.0, 0.0]
            
        self.shards[shard_id] = ShardStatus(
            shard_id=shard_id,
            g_vector=initial_g_vector,
            coherence_score=0.95,
            action_efficiency=0.8,
            gravity_well_count=0,
            stable=True,
            last_update=time.time()
        )
        
        logger.info(f"_registered: {shard_id}")
    
    def group_shards_by_q_seed_affinity(self):
        """
        Group nodes by resonant Q-Seed affinity
        In a real implementation, this would use actual Q-Seed values
        """
        # For demonstration, we'll group shards into 3 affinity groups
        shard_ids = list(self.shards.keys())
        group_size = max(1, len(shard_ids) // 3)
        
        for i, shard_id in enumerate(shard_ids):
            group_id = f"Q_GROUP_{i // group_size}"
            if group_id not in self.q_seed_affinity_groups:
                self.q_seed_affinity_groups[group_id] = []
            self.q_seed_affinity_groups[group_id].append(shard_id)
        
        logger.info(f"Grouped {len(shard_ids)} shards into {len(self.q_seed_affinity_groups)} affinity groups")
    
    def start_all_local_governance_loops(self):
        """Start governance loops for all registered shards"""
        self.running = True
        for shard_id in self.shards:
            self._start_governance_loop(shard_id)
        logger.info("Started all local governance loops")
    
    def _start_governance_loop(self, shard_id: str):
        """
        Start a governance loop for a specific shard
        
        Args:
            shard_id: ID of the shard to start governance for
        """
        def governance_loop():
            while self.running and shard_id in self.shards:
                try:
                    self._governance_cycle(shard_id)
                    time.sleep(1)  # 1 second interval
                except Exception as e:
                    logger.error(f"Error in governance loop for shard {shard_id}: {e}")
                    time.sleep(5)  # Wait before retrying
        
        thread = threading.Thread(target=governance_loop, daemon=True)
        thread.start()
        self.governance_loops[shard_id] = thread
        logger.info(f"Started governance loop for shard {shard_id}")
    
    def _governance_cycle(self, shard_id: str):
        """
        Execute a single governance cycle for a shard
        
        Args:
            shard_id: ID of the shard to govern
        """
        if shard_id not in self.shards:
            return
            
        shard = self.shards[shard_id]
        
        # Simulate monitoring g-vector
        # In a real implementation, this would interface with actual shard components
        new_g_vector = [
            shard.g_vector[0] + np.random.normal(0, 0.01),
            shard.g_vector[1] + np.random.normal(0, 0.01),
            shard.g_vector[2] + np.random.normal(0, 0.01)
        ]
        
        # Update shard status
        shard.g_vector = new_g_vector
        shard.last_update = time.time()
        
        # Calculate g-vector magnitude
        g_magnitude = np.linalg.norm(new_g_vector)
        
        # Apply Gravity Well Gating
        if g_magnitude > 1.5:  # G_CRIT
            shard.gravity_well_count += 1
            logger.warning(f"Gravity Well detected in shard {shard_id}: |g|={g_magnitude:.4f}")
            # In a real implementation, this would trigger node isolation
        else:
            shard.stable = True
            
        # Update coherence score based on g-vector stability
        shard.coherence_score = float(max(0.0, min(1.0, 0.95 - (g_magnitude * 0.1))))
        
        # Update action efficiency
        shard.action_efficiency = float(max(0.0, min(1.0, 0.8 + np.random.normal(0, 0.05))))
        
        logger.debug(f"Governance cycle for shard {shard_id}: |g|={g_magnitude:.4f}, coherence={shard.coherence_score:.4f}")
    
    def stop_all_governance_loops(self):
        """Stop all governance loops"""
        self.running = False
        for shard_id, thread in self.governance_loops.items():
            if thread.is_alive():
                logger.info(f"Stopping governance loop for shard {shard_id}")
        self.governance_loops.clear()
        logger.info("Stopped all governance loops")
    
    def audit_and_consolidate_shard_status(self) -> Dict[str, Any]:
        """
        Audit all shards and consolidate their status
        
        Returns:
            Dictionary with consolidated shard status
        """
        if not self.shards:
            return {"status": "no_shards_registered", "shard_count": 0}
        
        # Collect metrics from all shards
        total_coherence = 0.0
        total_efficiency = 0.0
        unstable_shards = 0
        gravity_wells = 0
        
        for shard in self.shards.values():
            total_coherence += shard.coherence_score
            total_efficiency += shard.action_efficiency
            if not shard.stable:
                unstable_shards += 1
            gravity_wells += shard.gravity_well_count
        
        avg_coherence = total_coherence / len(self.shards)
        avg_efficiency = total_efficiency / len(self.shards)
        
        # Determine overall system status
        system_stable = unstable_shards == 0 and avg_coherence > 0.90
        
        report = {
            "timestamp": time.time(),
            "shard_count": len(self.shards),
            "average_coherence": avg_coherence,
            "average_efficiency": avg_efficiency,
            "unstable_shards": unstable_shards,
            "total_gravity_wells": gravity_wells,
            "system_stable": system_stable,
            "status": "STABLE" if system_stable else "UNSTABLE"
        }
        
        logger.info(f"Shard audit completed: {report['status']} (avg_coherence={avg_coherence:.4f})")
        return report
    
    def get_shard_status(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific shard
        
        Args:
            shard_id: ID of the shard to query
            
        Returns:
            Dictionary with shard status or None if not found
        """
        if shard_id in self.shards:
            shard = self.shards[shard_id]
            return {
                "shard_id": shard.shard_id,
                "g_vector": shard.g_vector,
                "g_magnitude": np.linalg.norm(shard.g_vector),
                "coherence_score": shard.coherence_score,
                "action_efficiency": shard.action_efficiency,
                "gravity_well_count": shard.gravity_well_count,
                "stable": shard.stable,
                "last_update": shard.last_update
            }
        return None
    
    def get_all_shard_statuses(self) -> List[Dict[str, Any]]:
        """
        Get the status of all shards
        
        Returns:
            List of dictionaries with shard statuses
        """
        statuses = []
        for shard_id in self.shards:
            status = self.get_shard_status(shard_id)
            if status:
                statuses.append(status)
        return statuses

# Example usage
if __name__ == "__main__":
    # Create shard manager
    shard_manager = ShardManager()
    
    # Register some shards
    for i in range(10):
        shard_manager.register_shard(f"SHARD_{i:03d}")
    
    # Group shards by Q-Seed affinity
    shard_manager.group_shards_by_q_seed_affinity()
    
    # Start governance loops
    shard_manager.start_all_local_governance_loops()
    
    # Let it run for a bit
    time.sleep(10)
    
    # Audit shard status
    report = shard_manager.audit_and_consolidate_shard_status()
    print(f"System report: {report}")
    
    # Stop governance loops
    shard_manager.stop_all_governance_loops()