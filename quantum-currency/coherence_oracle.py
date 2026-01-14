#!/usr/bin/env python3
"""
Coherence Oracle for QECS
Enables continuous learning from governance decisions
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GovernanceCycleLog:
    """Represents a governance cycle log entry"""
    timestamp: float
    state_t: Dict[str, Any]  # Current system state
    action_t: Dict[str, Any]  # Action taken
    reward_t_plus_1: float  # Reward received
    next_state_t_plus_1: Dict[str, Any]  # Next system state
    protocol_version: str  # Protocol version at time of action
    cycle_id: str  # Unique identifier for this cycle

class CoherenceOracle:
    """Coherence Oracle for continuous learning from governance decisions"""
    
    def __init__(self):
        self.governance_log: List[GovernanceCycleLog] = []
        self.lambda_history: List[float] = []
        self.I_eff_history: List[float] = []
        self.coherence_history: List[float] = []
        self.protocol_versions: List[str] = []
        
    def log_governance_cycle(self, system_report: Dict[str, Any]):
        """
        Log a governance cycle entry
        
        Args:
            system_report: System report containing state, action, and reward information
        """
        timestamp = time.time()
        
        # Extract state information
        state_t = {
            "coherence_score": system_report.get("average_coherence", 0.95),
            "action_efficiency": system_report.get("average_efficiency", 0.8),
            "unstable_shards": system_report.get("unstable_shards", 0),
            "gravity_wells": system_report.get("total_gravity_wells", 0),
            "system_stable": system_report.get("system_stable", True)
        }
        
        # Extract action information (what was done in this cycle)
        action_t = {
            "shard_governance_actions": system_report.get("shard_governance_actions", 0),
            "protocol_adjustments": system_report.get("protocol_adjustments", 0),
            "anomaly_corrections": system_report.get("anomaly_corrections", 0),
            "optimization_vector_applied": system_report.get("optimization_vector_applied", False)
        }
        
        # Calculate reward (higher coherence and efficiency = higher reward)
        coherence_score = state_t["coherence_score"]
        action_efficiency = state_t["action_efficiency"]
        reward_t_plus_1 = coherence_score * 0.7 + action_efficiency * 0.3
        
        # Next state is the same as current state for this log entry
        next_state_t_plus_1 = state_t.copy()
        
        # Get protocol version
        protocol_version = system_report.get("protocol_version", "HSMF_v2.0")
        
        # Generate cycle ID
        cycle_id = f"cycle_{int(timestamp * 1000000) % 1000000}"
        
        # Create log entry
        log_entry = GovernanceCycleLog(
            timestamp=timestamp,
            state_t=state_t,
            action_t=action_t,
            reward_t_plus_1=reward_t_plus_1,
            next_state_t_plus_1=next_state_t_plus_1,
            protocol_version=protocol_version,
            cycle_id=cycle_id
        )
        
        # Add to log
        self.governance_log.append(log_entry)
        
        # Update histories
        self.coherence_history.append(coherence_score)
        self.I_eff_history.append(1.0 - action_efficiency)  # Convert efficiency to cost
        self.protocol_versions.append(protocol_version)
        
        # Keep only recent history (last 1000 entries)
        if len(self.governance_log) > 1000:
            self.governance_log = self.governance_log[-1000:]
            self.coherence_history = self.coherence_history[-1000:]
            self.I_eff_history = self.I_eff_history[-1000:]
            self.protocol_versions = self.protocol_versions[-1000:]
        
        logger.info(f"Governance cycle logged: {cycle_id} (reward={reward_t_plus_1:.4f})")
    
    def get_long_term_lambda_history(self) -> List[float]:
        """
        Get long-term λ history for protocol evolution analysis
        
        Returns:
            List of λ values over time
        """
        # In a real implementation, this would track actual λ values
        # For now, we'll simulate based on coherence history
        if not self.coherence_history:
            return [0.5] * 100  # Default initialization
            
        # Convert coherence history to simulated λ values
        # Higher coherence should correlate with more stable λ
        lambda_history = []
        for coherence in self.coherence_history:
            # Simulate λ values that stabilize as coherence improves
            lambda_val = 0.3 + 0.4 * coherence + np.random.normal(0, 0.05)
            lambda_history.append(max(0.1, min(1.0, lambda_val)))
            
        return lambda_history
    
    def get_recent_governance_cycles(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent governance cycle logs
        
        Args:
            count: Number of recent cycles to retrieve
            
        Returns:
            List of recent governance cycle logs
        """
        recent_logs = self.governance_log[-count:] if self.governance_log else []
        return [asdict(log) for log in recent_logs]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics based on governance history
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.governance_log:
            return {
                "total_cycles": 0,
                "average_reward": 0.0,
                "coherence_trend": "unknown",
                "stability_score": 0.0
            }
        
        # Calculate average reward
        rewards = [log.reward_t_plus_1 for log in self.governance_log]
        avg_reward = sum(rewards) / len(rewards)
        
        # Analyze coherence trend
        if len(self.coherence_history) >= 10:
            recent_coherence = self.coherence_history[-10:]
            old_coherence = self.coherence_history[-20:-10] if len(self.coherence_history) >= 20 else [0.9]
            recent_avg = sum(recent_coherence) / len(recent_coherence)
            old_avg = sum(old_coherence) / len(old_coherence)
            
            if recent_avg > old_avg * 1.05:
                coherence_trend = "improving"
            elif recent_avg < old_avg * 0.95:
                coherence_trend = "declining"
            else:
                coherence_trend = "stable"
        else:
            coherence_trend = "insufficient_data"
        
        # Calculate stability score (percentage of stable cycles)
        stable_cycles = sum(1 for log in self.governance_log if log.state_t["system_stable"])
        stability_score = stable_cycles / len(self.governance_log)
        
        return {
            "total_cycles": len(self.governance_log),
            "average_reward": avg_reward,
            "coherence_trend": coherence_trend,
            "stability_score": stability_score,
            "latest_coherence": self.coherence_history[-1] if self.coherence_history else 0.95
        }
    
    def export_governance_data(self, filepath: str):
        """
        Export governance data to a JSON file
        
        Args:
            filepath: Path to export file
        """
        export_data = {
            "export_timestamp": time.time(),
            "governance_log": [asdict(log) for log in self.governance_log],
            "lambda_history": self.lambda_history,
            "I_eff_history": self.I_eff_history,
            "coherence_history": self.coherence_history,
            "protocol_versions": self.protocol_versions
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Governance data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export governance data: {e}")
    
    def get_training_data_for_paf(self) -> List[Dict[str, Any]]:
        """
        Get training data formatted for the PAF engine
        
        Returns:
            List of training data records
        """
        training_data = []
        
        for log in self.governance_log:
            record = {
                "timestamp": log.timestamp,
                "I_eff": 1.0 - log.state_t["action_efficiency"],  # Convert efficiency to cost
                "g_vector_magnitude": log.state_t["gravity_wells"] * 0.5,  # Simulate g-vector magnitude
                "coherence_score": log.state_t["coherence_score"],
                "action_efficiency": log.state_t["action_efficiency"]
            }
            training_data.append(record)
        
        return training_data

# Example usage
if __name__ == "__main__":
    # Create coherence oracle
    oracle = CoherenceOracle()
    
    # Simulate some governance cycles
    for i in range(50):
        system_report = {
            "average_coherence": 0.90 + 0.05 * np.sin(i * 0.2),
            "average_efficiency": 0.75 + 0.1 * np.cos(i * 0.15),
            "unstable_shards": max(0, 3 - i // 10),
            "total_gravity_wells": max(0, 5 - i // 5),
            "system_stable": i > 5,
            "protocol_version": "HSMF_v2.0"
        }
        
        oracle.log_governance_cycle(system_report)
        time.sleep(0.01)  # Small delay to simulate time passing
    
    # Get performance metrics
    metrics = oracle.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Get long-term lambda history
    lambda_history = oracle.get_long_term_lambda_history()
    print(f"Lambda history length: {len(lambda_history)}")
    
    # Export data
    oracle.export_governance_data("coherence_oracle_data.json")