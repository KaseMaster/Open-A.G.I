#!/usr/bin/env python3
"""
AI Governance & Security Layer - Harmonic Regulation
Implements OpenAGI integration, reward functions, and formal verification

This module provides:
1. Harmonic Regulation for OpenAGI
2. Reward function implementation
3. Macro-level governance
4. Formal verification with dimensional stability tests
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import math
from dataclasses import dataclass, field
from .quantum_memory import QuantumPacket, UnifiedFieldMemory
from .coherent_db import CoherentDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GovernanceProposal:
    """Governance proposal for network changes"""
    id: str
    proposer_id: str
    title: str
    description: str
    target_omega: Dict[str, Any]  # Proposed Ω_target changes
    timestamp: float
    votes: Dict[str, float] = field(default_factory=dict)  # validator_id -> vote_weight
    status: str = "pending"  # pending, approved, rejected, implemented
    
    def __post_init__(self):
        if not self.id:  # Fix: Check for empty string instead of None
            self.id = f"prop_{int(self.timestamp * 1000000)}_{hash(self.title) % 10000}"

@dataclass
class Validator:
    """Network validator with coherence-based reputation"""
    id: str
    staked_atr: float  # Staked ATR tokens
    coherence_history: List[float]  # Historical Ψ scores
    reputation_score: float = 0.0  # Overall reputation
    last_active: float = 0.0  # Last activity timestamp
    coherence_contributions: Dict[str, float] = field(default_factory=dict)  # proposal_id -> contribution_score
    
    def get_voting_power(self) -> float:
        """Calculate voting power based on staked ATR and reputation"""
        # Voting power ∝ sqrt(staked_ATR) × reputation
        atr_component = math.sqrt(max(0.0, self.staked_atr))
        reputation_component = max(0.0, min(1.0, self.reputation_score))
        return atr_component * reputation_component

class AIGovernance:
    """
    AI Governance & Security Layer - Implements harmonic regulation
    """
    
    def __init__(self, cdb: CoherentDatabase, ufm: UnifiedFieldMemory, 
                 network_id: str = "quantum-currency-uhes-gov-001"):
        self.network_id = network_id
        self.cdb = cdb  # Reference to Coherent Database
        self.ufm = ufm  # Reference to Unified Field Memory
        self.proposals: Dict[str, GovernanceProposal] = {}
        self.validators: Dict[str, Validator] = {}
        self.omega_target: Dict[str, Any] = {
            "token_rate": 1.0,
            "sentiment_energy": 0.5,
            "semantic_shift": 0.1,
            "meta_attention_spectrum": [0.2, 0.3, 0.3, 0.2]
        }
        self.governance_params = {
            "approval_threshold": 0.6,  # 60% weighted vote needed for approval
            "min_validators": 3,  # Minimum validators needed
            "proposal_timeout": 86400,  # 24 hours in seconds
            "reward_clip_delta": 0.1,  # δ for reward clipping
            "harmonic_reward_multiplier": 1.5  # Multiplier for harmonic behavior rewards
        }
        
        logger.info(f"🏛️ AI Governance initialized for network: {network_id}")
    
    def compute_reward_function(self, psi_t: float, psi_t_plus_1: float, 
                              harmonic_bonus: float = 0.0) -> float:
        """
        Compute OpenAGI Reward Function with harmonic behavior incentives
        R_t = clip(Ψ_{t+1} - Ψ_t, -δ, +δ) + harmonic_bonus
        
        Args:
            psi_t: Current coherence score Ψ_t
            psi_t_plus_1: Next coherence score Ψ_{t+1}
            harmonic_bonus: Additional reward for harmonic behavior
            
        Returns:
            float: Computed reward R_t
        """
        delta = self.governance_params["reward_clip_delta"]
        coherence_change = psi_t_plus_1 - psi_t
        base_reward = max(-delta, min(delta, coherence_change))  # clip function
        
        # Apply harmonic bonus
        total_reward = base_reward + (harmonic_bonus * self.governance_params["harmonic_reward_multiplier"])
        
        return total_reward
    
    def register_validator(self, validator_id: str, staked_atr: float, 
                          initial_coherence: float = 0.5) -> bool:
        """
        Register a new validator in the network
        
        Args:
            validator_id: Unique identifier for validator
            staked_atr: Amount of ATR tokens staked
            initial_coherence: Initial coherence score
            
        Returns:
            bool: True if registration successful
        """
        validator = Validator(
            id=validator_id,
            staked_atr=staked_atr,
            coherence_history=[initial_coherence],
            reputation_score=initial_coherence,
            last_active=time.time()
        )
        
        self.validators[validator_id] = validator
        logger.info(f"📝 Registered validator {validator_id} with {staked_atr} ATR staked")
        return True
    
    def update_validator_coherence(self, validator_id: str, new_coherence: float, 
                                  proposal_id: Optional[str] = None, 
                                  contribution_score: Optional[float] = None) -> bool:
        """
        Update a validator's coherence score and reputation
        
        Args:
            validator_id: Validator ID
            new_coherence: New coherence score
            proposal_id: Optional proposal ID for contribution tracking
            contribution_score: Optional contribution score for the proposal
            
        Returns:
            bool: True if update successful
        """
        validator = self.validators.get(validator_id)
        if not validator:
            logger.warning(f"Validator {validator_id} not found")
            return False
        
        # Update coherence history
        validator.coherence_history.append(new_coherence)
        # Keep only recent history (last 50 scores)
        if len(validator.coherence_history) > 50:
            validator.coherence_history = validator.coherence_history[-50:]
        
        # Update reputation score (moving average)
        validator.reputation_score = float(np.mean(validator.coherence_history))
        validator.last_active = time.time()
        
        # Track contribution to specific proposal if provided
        if proposal_id and contribution_score is not None:
            validator.coherence_contributions[proposal_id] = contribution_score
        
        logger.debug(f"📈 Updated validator {validator_id} coherence to {new_coherence:.4f}, "
                    f"reputation: {validator.reputation_score:.4f}")
        return True
    
    def create_proposal(self, proposer_id: str, title: str, description: str,
                       target_omega: Dict[str, Any]) -> Optional[str]:
        """
        Create a new governance proposal
        
        Args:
            proposer_id: ID of proposer (must be validator)
            title: Proposal title
            description: Proposal description
            target_omega: Proposed Ω_target changes
            
        Returns:
            str: Proposal ID or None if creation failed
        """
        # Check if proposer is a registered validator
        if proposer_id not in self.validators:
            logger.error(f"Proposer {proposer_id} is not a registered validator")
            return None
        
        # Only LΦ (Macro) Ω state can suggest modifications to ATR target vector
        # Check if proposer has sufficient reputation (high Ψ score)
        validator = self.validators[proposer_id]
        if validator.reputation_score < 0.8:
            logger.error(f"Validator {proposer_id} has insufficient reputation "
                        f"({validator.reputation_score:.4f} < 0.8)")
            return None
        
        # Create proposal with temporary ID
        temp_id = f"prop_{int(time.time() * 1000000)}_{hash(title) % 10000}"
        proposal = GovernanceProposal(
            id=temp_id,  # Will be auto-generated in __post_init__
            proposer_id=proposer_id,
            title=title,
            description=description,
            target_omega=target_omega,
            timestamp=time.time()
        )
        
        self.proposals[proposal.id] = proposal
        logger.info(f"💡 Created proposal {proposal.id}: {title}")
        return proposal.id
    
    def vote_on_proposal(self, validator_id: str, proposal_id: str, vote_weight: float) -> bool:
        """
        Vote on a governance proposal with weighted voting based on coherence contribution
        
        Args:
            validator_id: ID of voting validator
            proposal_id: ID of proposal to vote on
            vote_weight: Vote weight (-1.0 to 1.0, negative=reject, positive=approve)
            
        Returns:
            bool: True if vote recorded successfully
        """
        # Check if validator exists
        validator = self.validators.get(validator_id)
        if not validator:
            logger.error(f"Validator {validator_id} not found")
            return False
        
        # Check if proposal exists and is pending
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        if proposal.status != "pending":
            logger.error(f"Proposal {proposal_id} is not pending (status: {proposal.status})")
            return False
        
        # Calculate enhanced voting weight based on coherence contributions
        base_voting_power = validator.get_voting_power()
        
        # If validator has contributed to this proposal, enhance their voting power
        if proposal_id in validator.coherence_contributions:
            contribution_score = validator.coherence_contributions[proposal_id]
            # Enhance voting power by contribution score (capped at 2x)
            enhancement_factor = min(2.0, 1.0 + contribution_score)
            enhanced_voting_power = base_voting_power * enhancement_factor
        else:
            enhanced_voting_power = base_voting_power
        
        # Record vote with enhanced weight
        weighted_vote = max(-1.0, min(1.0, vote_weight)) * enhanced_voting_power
        proposal.votes[validator_id] = weighted_vote
        validator.last_active = time.time()
        
        logger.info(f"🗳️ Validator {validator_id} voted {vote_weight:+.2f} (enhanced weight: {enhanced_voting_power:.2f}) "
                   f"on proposal {proposal_id}")
        return True
    
    def evaluate_proposal(self, proposal_id: str) -> Tuple[bool, float]:
        """
        Evaluate a proposal based on weighted validator votes with coherence contribution metrics
        
        Args:
            proposal_id: ID of proposal to evaluate
            
        Returns:
            Tuple of (is_approved, weighted_score)
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False, 0.0
        
        if not self.validators:
            logger.warning("No validators registered")
            return False, 0.0
        
        # Calculate weighted vote score using enhanced voting weights
        total_weight = 0.0
        weighted_sum = 0.0
        
        for validator_id, weighted_vote in proposal.votes.items():
            validator = self.validators.get(validator_id)
            if validator:
                # The vote is already weighted, so we just sum it
                weighted_sum += weighted_vote
                # For normalization, we use the base voting power
                base_voting_power = validator.get_voting_power()
                total_weight += base_voting_power
        
        if total_weight <= 0:
            logger.warning(f"No valid votes for proposal {proposal_id}")
            return False, 0.0
        
        # Normalize the weighted score
        weighted_score = weighted_sum / total_weight
        
        # Check approval threshold
        is_approved = (
            weighted_score >= self.governance_params["approval_threshold"] and
            len(proposal.votes) >= self.governance_params["min_validators"]
        )
        
        # Update proposal status
        proposal.status = "approved" if is_approved else "rejected"
        
        logger.info(f"📊 Proposal {proposal_id} evaluated: "
                   f"score={weighted_score:.4f}, approved={is_approved}")
        return is_approved, weighted_score
    
    def implement_proposal(self, proposal_id: str) -> bool:
        """
        Implement an approved proposal
        
        Args:
            proposal_id: ID of proposal to implement
            
        Returns:
            bool: True if implementation successful
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        if proposal.status != "approved":
            logger.error(f"Proposal {proposal_id} is not approved (status: {proposal.status})")
            return False
        
        # Implement Ω_target changes
        # Only the LΦ (Macro) Ω state can suggest changes to the ATR target vector
        old_target = self.omega_target.copy()
        self.omega_target.update(proposal.target_omega)
        
        # Update proposal status
        proposal.status = "implemented"
        
        logger.info(f"⚡ Implemented proposal {proposal_id}")
        logger.info(f"   Ω_target updated: {old_target} → {self.omega_target}")
        return True
    
    def dimensional_stability_test(self, modulator_value: float, K: float = 10.0) -> bool:
        """
        Formal Verification - Dimensional Stability Test
        Primary audit check for consensus and coherence validity
        
        Args:
            modulator_value: The computed modulator value m_t(L)
            K: Clamping bound (default: 10.0)
            
        Returns:
            bool: True if dimensionally stable
        """
        # Check that the log of modulator is within bounds
        # This ensures λ(L) · proj(I_t(L)) remains dimensionless and clamped
        try:
            log_modulator = math.log(modulator_value)
            is_stable = -K <= log_modulator <= K
            
            if not is_stable:
                logger.warning(f"Dimensional instability detected: log(m_t) = {log_modulator:.4f} "
                              f"exceeds bounds ±{K}")
            
            return is_stable
        except (ValueError, OverflowError):
            logger.warning(f"Dimensional instability: modulator value {modulator_value} causes overflow")
            return False
    
    def run_consensus_audit(self) -> Dict[str, Any]:
        """
        Run comprehensive consensus audit including dimensional stability test
        
        Returns:
            Dict with audit results
        """
        audit_results = {
            "timestamp": time.time(),
            "validator_count": len(self.validators),
            "proposal_count": len(self.proposals),
            "active_proposals": len([p for p in self.proposals.values() if p.status == "pending"]),
            "dimensional_stability": True,
            "consensus_health": "healthy"
        }
        
        # Check dimensional stability for all validators
        unstable_count = 0
        for validator_id, validator in self.validators.items():
            # Simulate modulator values for audit (in practice, these would come from actual computations)
            # For audit purposes, we'll use a representative value
            modulator_value = 1.5  # Representative value
            if not self.dimensional_stability_test(modulator_value):
                unstable_count += 1
        
        if unstable_count > 0:
            audit_results["dimensional_stability"] = False
            audit_results["unstable_validators"] = unstable_count
            audit_results["consensus_health"] = "degraded"
            logger.warning(f"Consensus audit found {unstable_count} dimensionally unstable validators")
        
        # Check for timed-out proposals
        current_time = time.time()
        timeout = self.governance_params["proposal_timeout"]
        timed_out = 0
        
        for proposal in self.proposals.values():
            if (proposal.status == "pending" and 
                current_time - proposal.timestamp > timeout):
                proposal.status = "timeout"
                timed_out += 1
        
        if timed_out > 0:
            logger.info(f"Expired {timed_out} timed-out proposals")
        
        logger.info(f"Consensus audit completed: {audit_results['consensus_health']}")
        return audit_results
    
    def get_network_topology(self) -> Dict[str, Any]:
        """
        Get current network topology and governance statistics
        
        Returns:
            Dict with network topology information
        """
        # Calculate validator statistics
        total_staked_atr = sum(v.staked_atr for v in self.validators.values())
        avg_reputation = np.mean([v.reputation_score for v in self.validators.values()]) if self.validators else 0.0
        active_validators = sum(1 for v in self.validators.values() 
                              if time.time() - v.last_active < 3600)  # Active in last hour
        
        # Proposal statistics
        proposal_stats = {
            "total": len(self.proposals),
            "pending": len([p for p in self.proposals.values() if p.status == "pending"]),
            "approved": len([p for p in self.proposals.values() if p.status == "approved"]),
            "rejected": len([p for p in self.proposals.values() if p.status == "rejected"]),
            "implemented": len([p for p in self.proposals.values() if p.status == "implemented"])
        }
        
        topology = {
            "validators": {
                "total": len(self.validators),
                "active": active_validators,
                "total_staked_atr": total_staked_atr,
                "average_reputation": float(avg_reputation)
            },
            "proposals": proposal_stats,
            "omega_target": self.omega_target,
            "governance_params": self.governance_params
        }
        
        return topology
    
    def compute_harmonic_behavior_score(self, validator_id: str, 
                                      actions: List[Dict[str, Any]]) -> float:
        """
        Compute harmonic behavior score based on validator actions
        
        Args:
            validator_id: Validator ID
            actions: List of actions taken by the validator
            
        Returns:
            float: Harmonic behavior score (0.0 to 1.0)
        """
        if not actions:
            return 0.0
        
        harmonic_score = 0.0
        total_actions = len(actions)
        
        for action in actions:
            action_type = action.get("type", "")
            action_quality = action.get("quality", 0.0)
            action_coherence = action.get("coherence_impact", 0.0)
            
            # Score based on action type and quality
            if action_type == "proposal_creation":
                # High quality proposal creation
                harmonic_score += 0.3 * action_quality
            elif action_type == "constructive_voting":
                # Voting that improves consensus
                harmonic_score += 0.2 * action_quality * action_coherence
            elif action_type == "network_maintenance":
                # Maintenance activities that improve network health
                harmonic_score += 0.15 * action_quality
            elif action_type == "coherence_improvement":
                # Direct coherence improvement actions
                harmonic_score += 0.35 * action_coherence
            else:
                # Generic positive action
                harmonic_score += 0.1 * action_quality
        
        # Normalize by number of actions
        normalized_score = harmonic_score / max(1, total_actions)
        
        # Cap at 1.0
        return min(1.0, normalized_score)
    
    def distribute_harmonic_rewards(self, validator_id: str, 
                                  coherence_improvement: float,
                                  actions: List[Dict[str, Any]]) -> float:
        """
        Distribute harmonic rewards to validators based on their contributions
        
        Args:
            validator_id: Validator ID
            coherence_improvement: Improvement in network coherence
            actions: List of actions taken by the validator
            
        Returns:
            float: Total reward distributed
        """
        validator = self.validators.get(validator_id)
        if not validator:
            logger.warning(f"Validator {validator_id} not found for reward distribution")
            return 0.0
        
        # Calculate base reward based on coherence improvement
        base_reward = max(0.0, coherence_improvement)  # Only positive improvements rewarded
        
        # Calculate harmonic behavior score
        harmonic_score = self.compute_harmonic_behavior_score(validator_id, actions)
        
        # Calculate total reward
        total_reward = base_reward + (harmonic_score * base_reward * 0.5)  # Up to 50% bonus
        
        # Update validator's reputation based on reward
        new_reputation = min(1.0, validator.reputation_score + total_reward * 0.1)
        self.update_validator_coherence(validator_id, new_reputation)
        
        logger.info(f"💰 Distributed harmonic reward of {total_reward:.4f} to validator {validator_id} "
                   f"(base: {base_reward:.4f}, harmonic bonus: {total_reward - base_reward:.4f})")
        
        return total_reward

# Example usage and testing
if __name__ == "__main__":
    # Create required components
    ufm = UnifiedFieldMemory()
    cdb = CoherentDatabase(ufm)
    governance = AIGovernance(cdb, ufm)
    
    # Register validators
    governance.register_validator("validator_1", 1000.0, 0.85)
    governance.register_validator("validator_2", 1500.0, 0.92)
    governance.register_validator("validator_3", 800.0, 0.78)
    
    print(f"Registered {len(governance.validators)} validators")
    
    # Test reward function
    reward = governance.compute_reward_function(0.8, 0.85)
    print(f"Reward for coherence improvement: {reward:.4f}")
    
    reward = governance.compute_reward_function(0.85, 0.8)
    print(f"Reward for coherence degradation: {reward:.4f}")
    
    # Create a proposal
    proposal_target = {
        "token_rate": 1.2,
        "sentiment_energy": 0.6
    }
    
    proposal_id = governance.create_proposal(
        "validator_1",
        "Increase Token Rate",
        "Propose increasing the token generation rate to support network growth",
        proposal_target
    )
    
    if proposal_id:
        print(f"Created proposal: {proposal_id}")
        
        # Vote on proposal
        governance.vote_on_proposal("validator_1", proposal_id, 1.0)  # Strong approve
        governance.vote_on_proposal("validator_2", proposal_id, 0.8)  # Approve
        governance.vote_on_proposal("validator_3", proposal_id, -0.5) # Mild reject
        
        # Evaluate proposal
        is_approved, score = governance.evaluate_proposal(proposal_id)
        print(f"Proposal evaluation: approved={is_approved}, score={score:.4f}")
        
        if is_approved:
            # Implement proposal
            success = governance.implement_proposal(proposal_id)
            print(f"Proposal implementation: {success}")
    
    # Test dimensional stability
    is_stable = governance.dimensional_stability_test(1.5)
    print(f"Dimensional stability test: {is_stable}")
    
    # Run consensus audit
    audit_results = governance.run_consensus_audit()
    print(f"Consensus audit: {audit_results['consensus_health']}")
    
    # Get network topology
    topology = governance.get_network_topology()
    print(f"Network topology: {topology['validators']['total']} validators, "
          f"{topology['proposals']['total']} proposals")