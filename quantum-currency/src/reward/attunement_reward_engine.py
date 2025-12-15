#!/usr/bin/env python3
"""
Attunement Reward Engine for Quantum Currency Coherence System
Implements dynamic reward distribution for T2 and T5 tokens
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RewardEpoch:
    """Represents a reward epoch"""
    epoch_id: str
    start_time: float
    end_time: float
    t2_rewards_distributed: float = 0.0
    t5_rewards_distributed: float = 0.0
    validators_rewarded: List[str] = field(default_factory=list)
    memory_nodes_contributed: List[str] = field(default_factory=list)


@dataclass
class MemoryNodeContribution:
    """Represents a memory node contribution"""
    node_id: str
    rphiv: float  # RΦV score
    lambda_node: float  # λ_node value
    contribution_time: float
    reward_amount: float = 0.0


class AttunementRewardEngine:
    """
    Attunement Reward Engine to distribute T2 and T5 tokens dynamically
    """
    
    def __init__(self, network_id: str = "quantum-currency-rewards-001"):
        self.network_id = network_id
        self.reward_epochs: List[RewardEpoch] = []
        self.memory_contributions: List[MemoryNodeContribution] = []
        self.reward_log: List[Dict[str, Any]] = []
        
        # Reward configuration
        self.base_t2_reward_pool = 10000.0  # Base T2 tokens per epoch
        self.deficit_multiplier_threshold = 0.85  # Threshold for deficit mode
        self.t5_reward_factor = 0.1  # Factor for T5 rewards based on memory contributions
        
    def distribute_t2_rewards(self, validators: Dict[str, Any], network_coherence: float, 
                             deficit_multiplier: float = 1.0) -> Dict[str, float]:
        """
        Distribute T2 rewards dynamically based on Ψ and network coherence
        
        Args:
            validators: Dictionary of validators with their metrics
            network_coherence: Current network coherence score
            deficit_multiplier: Multiplier for deficit mode (default: 1.0)
            
        Returns:
            Dict mapping validator IDs to reward amounts
        """
        # Calculate total T2 rewards for this epoch
        base_reward = self.base_t2_reward_pool
        if network_coherence < self.deficit_multiplier_threshold:
            # Apply deficit multiplier
            adjusted_reward = base_reward * (1 + deficit_multiplier)
        else:
            adjusted_reward = base_reward
            
        # Distribute rewards based on validator performance
        total_psi = sum(v.get('psi_score', 0.0) for v in validators.values())
        rewards = {}
        
        for validator_id, validator in validators.items():
            psi_score = validator.get('psi_score', 0.0)
            if total_psi > 0:
                # Proportional to psi score
                reward_share = (psi_score / total_psi) * adjusted_reward
                rewards[validator_id] = reward_share
            else:
                # Equal distribution if no psi scores
                rewards[validator_id] = adjusted_reward / len(validators) if validators else 0.0
                
        # Log reward distribution
        self._log_reward_distribution("T2", rewards, network_coherence, deficit_multiplier)
        
        return rewards
    
    def calculate_t5_rewards(self, memory_contributions: List[MemoryNodeContribution]) -> Dict[str, float]:
        """
        Calculate T5 rewards based on memory node contributions
        
        Args:
            memory_contributions: List of memory node contributions
            
        Returns:
            Dict mapping node IDs to reward amounts
        """
        rewards = {}
        total_t5_rewards = 0.0
        
        # Calculate rewards for each memory node
        for contribution in memory_contributions:
            # T5_reward = sum(node.RphiV * lambda_node for node in contributed_nodes)
            reward_amount = contribution.rphiv * contribution.lambda_node * self.t5_reward_factor
            rewards[contribution.node_id] = reward_amount
            contribution.reward_amount = reward_amount
            total_t5_rewards += reward_amount
            
        # Log reward distribution
        self._log_reward_distribution("T5", rewards, total_t5_rewards)
        
        return rewards
    
    def _log_reward_distribution(self, token_type: str, rewards: Dict[str, float], 
                                *args) -> None:
        """
        Log reward distribution for audit purposes
        
        Args:
            token_type: Type of token (T2 or T5)
            rewards: Dict mapping addresses to reward amounts
            *args: Additional arguments for logging
        """
        log_entry = {
            "timestamp": time.time(),
            "token_type": token_type,
            "rewards": rewards,
            "total_distributed": sum(rewards.values()),
            "additional_info": args
        }
        self.reward_log.append(log_entry)
        
        # Print summary
        print(f"Distributed {sum(rewards.values()):.2f} {token_type} tokens to {len(rewards)} recipients")
    
    def start_new_epoch(self) -> str:
        """
        Start a new reward epoch
        
        Returns:
            str: Epoch ID
        """
        epoch_id = f"epoch-{int(time.time())}"
        epoch = RewardEpoch(
            epoch_id=epoch_id,
            start_time=time.time(),
            end_time=time.time() + 3600  # 1 hour epochs
        )
        self.reward_epochs.append(epoch)
        return epoch_id
    
    def end_epoch(self, epoch_id: str, t2_rewards: float, t5_rewards: float, 
                  validators_rewarded: List[str], memory_nodes_contributed: List[str]) -> bool:
        """
        End a reward epoch and record results
        
        Args:
            epoch_id: ID of epoch to end
            t2_rewards: Total T2 rewards distributed
            t5_rewards: Total T5 rewards distributed
            validators_rewarded: List of validator IDs that received rewards
            memory_nodes_contributed: List of memory node IDs that contributed
            
        Returns:
            bool: True if epoch ended successfully, False otherwise
        """
        # Find epoch
        epoch = None
        for e in self.reward_epochs:
            if e.epoch_id == epoch_id:
                epoch = e
                break
                
        if not epoch:
            print(f"Epoch {epoch_id} not found")
            return False
            
        # Update epoch data
        epoch.end_time = time.time()
        epoch.t2_rewards_distributed = t2_rewards
        epoch.t5_rewards_distributed = t5_rewards
        epoch.validators_rewarded = validators_rewarded
        epoch.memory_nodes_contributed = memory_nodes_contributed
        
        print(f"Epoch {epoch_id} ended. T2: {t2_rewards:.2f}, T5: {t5_rewards:.2f}")
        return True
    
    def get_reward_log(self) -> List[Dict[str, Any]]:
        """Get reward distribution log"""
        return self.reward_log.copy()
    
    def get_current_epoch(self) -> Optional[RewardEpoch]:
        """Get current (latest) epoch"""
        if self.reward_epochs:
            return self.reward_epochs[-1]
        return None


# Example usage and testing
if __name__ == "__main__":
    # Create reward engine
    reward_engine = AttunementRewardEngine()
    
    # Mock validators data
    validators = {
        "validator-001": {"psi_score": 0.92},
        "validator-002": {"psi_score": 0.87},
        "validator-003": {"psi_score": 0.95},
        "validator-004": {"psi_score": 0.78},
        "validator-005": {"psi_score": 0.89}
    }
    
    # Test T2 reward distribution
    print("Testing T2 reward distribution...")
    t2_rewards = reward_engine.distribute_t2_rewards(validators, 0.85, 1.2)
    print(f"T2 rewards: {t2_rewards}")
    
    # Test with deficit mode
    t2_rewards_deficit = reward_engine.distribute_t2_rewards(validators, 0.80, 1.5)
    print(f"T2 rewards (deficit): {t2_rewards_deficit}")
    
    # Test T5 reward calculation
    print("\nTesting T5 reward calculation...")
    contributions = [
        MemoryNodeContribution("node-001", 0.95, 0.88, time.time()),
        MemoryNodeContribution("node-002", 0.87, 0.92, time.time()),
        MemoryNodeContribution("node-003", 0.91, 0.85, time.time())
    ]
    
    t5_rewards = reward_engine.calculate_t5_rewards(contributions)
    print(f"T5 rewards: {t5_rewards}")
    
    # Test epoch management
    print("\nTesting epoch management...")
    epoch_id = reward_engine.start_new_epoch()
    print(f"Started epoch: {epoch_id}")
    
    current_epoch = reward_engine.get_current_epoch()
    print(f"Current epoch: {current_epoch}")
    
    # End epoch
    reward_engine.end_epoch(
        epoch_id=epoch_id,
        t2_rewards=sum(t2_rewards.values()),
        t5_rewards=sum(t5_rewards.values()),
        validators_rewarded=list(t2_rewards.keys()),
        memory_nodes_contributed=list(t5_rewards.keys())
    )
    
    # Check reward log
    print(f"\nReward log entries: {len(reward_engine.get_reward_log())}")
    
    print("\n✅ Attunement reward engine demo completed!")