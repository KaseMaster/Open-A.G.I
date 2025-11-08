#!/usr/bin/env python3
"""
Inter-System Councils Module
Extends DAO protocols to enable cross-system governance and coordination

This module implements:
1. Inter-system council formation and management
2. Cross-chain governance proposals
3. Multi-system voting mechanisms
4. Council reputation and trust scoring
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import time
import math
from dataclasses import dataclass, field
from .ai_governance import AIGovernance, GovernanceProposal, Validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InterSystemCouncil:
    """Represents an inter-system council for cross-system governance"""
    council_id: str
    member_systems: List[str]  # List of system IDs that are members
    council_name: str
    formation_timestamp: float
    council_reputation: float = 0.0  # Overall council reputation score
    active_proposals: Dict[str, 'InterSystemProposal'] = field(default_factory=dict)
    council_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.council_params:
            self.council_params = {
                "approval_threshold": 0.65,  # 65% consensus needed
                "min_systems": 2,  # Minimum systems needed
                "proposal_timeout": 172800,  # 48 hours in seconds
                "reputation_weight": 0.3  # Weight of reputation in voting
            }

@dataclass
class InterSystemProposal:
    """Represents a cross-system governance proposal"""
    id: str
    council_id: str
    proposer_system: str
    title: str
    description: str
    target_systems: List[str]  # Systems this proposal affects
    target_changes: Dict[str, Any]  # Proposed changes
    timestamp: float
    votes: Dict[str, Dict[str, float]] = field(default_factory=dict)  # system_id -> {vote_weight, timestamp}
    status: str = "pending"  # pending, approved, rejected, implemented
    implementation_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemReputation:
    """Tracks reputation metrics for external systems"""
    system_id: str
    coherence_score: float  # Historical coherence with our system
    trust_score: float  # Trustworthiness based on past interactions
    participation_score: float  # Active participation in councils
    last_interaction: float  # Timestamp of last interaction
    successful_proposals: int = 0  # Number of successfully implemented proposals
    total_proposals: int = 0  # Total proposals submitted

class InterSystemCouncils:
    """
    Inter-System Councils - Extends DAO protocols for cross-system governance
    """
    
    def __init__(self, ai_governance: AIGovernance):
        self.ai_governance = ai_governance  # Reference to existing AI governance system
        self.councils: Dict[str, InterSystemCouncil] = {}
        self.system_reputations: Dict[str, SystemReputation] = {}
        self.inter_system_proposals: Dict[str, InterSystemProposal] = {}
        
        logger.info("🏛️ Inter-System Councils system initialized")
    
    def register_external_system(self, system_id: str, 
                                initial_coherence: float = 0.5,
                                initial_trust: float = 0.5) -> bool:
        """
        Register an external system for participation in inter-system councils
        
        Args:
            system_id: Unique identifier for external system
            initial_coherence: Initial coherence score with our system
            initial_trust: Initial trust score
            
        Returns:
            bool: True if registration successful
        """
        if system_id in self.system_reputations:
            logger.warning(f"System {system_id} already registered")
            return False
        
        reputation = SystemReputation(
            system_id=system_id,
            coherence_score=initial_coherence,
            trust_score=initial_trust,
            participation_score=0.0,
            last_interaction=time.time()
        )
        
        self.system_reputations[system_id] = reputation
        logger.info(f"📝 Registered external system {system_id}")
        return True
    
    def update_system_reputation(self, system_id: str, 
                                coherence_delta: float = 0.0,
                                trust_delta: float = 0.0,
                                participated: bool = False) -> bool:
        """
        Update reputation metrics for an external system
        
        Args:
            system_id: System ID
            coherence_delta: Change in coherence score
            trust_delta: Change in trust score
            participated: Whether system participated in recent council activity
            
        Returns:
            bool: True if update successful
        """
        reputation = self.system_reputations.get(system_id)
        if not reputation:
            logger.warning(f"System {system_id} not found")
            return False
        
        # Update coherence score (clamped between 0 and 1)
        reputation.coherence_score = max(0.0, min(1.0, reputation.coherence_score + coherence_delta))
        
        # Update trust score (clamped between 0 and 1)
        reputation.trust_score = max(0.0, min(1.0, reputation.trust_score + trust_delta))
        
        # Update participation score
        if participated:
            reputation.participation_score = min(1.0, reputation.participation_score + 0.1)
        else:
            reputation.participation_score = max(0.0, reputation.participation_score - 0.05)
        
        reputation.last_interaction = time.time()
        
        logger.debug(f"📈 Updated system {system_id} reputation - "
                    f"coherence: {reputation.coherence_score:.4f}, "
                    f"trust: {reputation.trust_score:.4f}")
        return True
    
    def form_council(self, council_name: str, 
                    member_systems: List[str]) -> Optional[str]:
        """
        Form a new inter-system council
        
        Args:
            council_name: Name of the council
            member_systems: List of system IDs to include
            
        Returns:
            str: Council ID or None if formation failed
        """
        # Validate member systems
        valid_systems = [system for system in member_systems if system in self.system_reputations]
        if len(valid_systems) < 2:
            logger.error("Need at least 2 valid systems to form council")
            return None
        
        # Create council ID
        council_id = f"council_{int(time.time() * 1000000)}_{hash(council_name) % 10000}"
        
        # Create council
        council = InterSystemCouncil(
            council_id=council_id,
            member_systems=valid_systems,
            council_name=council_name,
            formation_timestamp=time.time()
        )
        
        self.councils[council_id] = council
        logger.info(f"🏛️ Formed inter-system council {council_id}: {council_name} "
                   f"with {len(valid_systems)} members")
        return council_id
    
    def create_inter_system_proposal(self, council_id: str,
                                   proposer_system: str,
                                   title: str,
                                   description: str,
                                   target_systems: List[str],
                                   target_changes: Dict[str, Any]) -> Optional[str]:
        """
        Create a new inter-system governance proposal
        
        Args:
            council_id: ID of council this proposal belongs to
            proposer_system: ID of system proposing the change
            title: Proposal title
            description: Proposal description
            target_systems: Systems this proposal affects
            target_changes: Proposed changes to implement
            
        Returns:
            str: Proposal ID or None if creation failed
        """
        # Check if council exists
        council = self.councils.get(council_id)
        if not council:
            logger.error(f"Council {council_id} not found")
            return None
        
        # Check if proposer is a member of the council
        if proposer_system not in council.member_systems:
            logger.error(f"System {proposer_system} is not a member of council {council_id}")
            return None
        
        # Check if target systems are valid
        valid_targets = [system for system in target_systems if system in self.system_reputations]
        if not valid_targets:
            logger.error("No valid target systems specified")
            return None
        
        # Create proposal ID
        proposal_id = f"inter_prop_{int(time.time() * 1000000)}_{hash(title) % 10000}"
        
        # Create proposal
        proposal = InterSystemProposal(
            id=proposal_id,
            council_id=council_id,
            proposer_system=proposer_system,
            title=title,
            description=description,
            target_systems=valid_targets,
            target_changes=target_changes,
            timestamp=time.time()
        )
        
        self.inter_system_proposals[proposal_id] = proposal
        council.active_proposals[proposal_id] = proposal
        
        logger.info(f"💡 Created inter-system proposal {proposal_id}: {title}")
        return proposal_id
    
    def vote_on_inter_system_proposal(self, system_id: str,
                                     proposal_id: str,
                                     vote_weight: float) -> bool:
        """
        Vote on an inter-system governance proposal
        
        Args:
            system_id: ID of voting system
            proposal_id: ID of proposal to vote on
            vote_weight: Vote weight (-1.0 to 1.0, negative=reject, positive=approve)
            
        Returns:
            bool: True if vote recorded successfully
        """
        # Check if system exists
        if system_id not in self.system_reputations:
            logger.error(f"System {system_id} not registered")
            return False
        
        # Check if proposal exists and is pending
        proposal = self.inter_system_proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        if proposal.status != "pending":
            logger.error(f"Proposal {proposal_id} is not pending (status: {proposal.status})")
            return False
        
        # Check if system is member of the council
        council = self.councils.get(proposal.council_id)
        if not council or system_id not in council.member_systems:
            logger.error(f"System {system_id} is not a member of council {proposal.council_id}")
            return False
        
        # Get system reputation for enhanced voting weight
        reputation = self.system_reputations[system_id]
        
        # Calculate enhanced voting weight based on reputation
        # Base weight is clamped between -1 and 1
        base_vote_weight = max(-1.0, min(1.0, vote_weight))
        
        # Enhancement factor based on system reputation
        # Higher reputation increases voting influence
        reputation_factor = (
            reputation.coherence_score * 0.4 +
            reputation.trust_score * 0.4 +
            reputation.participation_score * 0.2
        )
        
        # Enhanced weight (up to 2x base weight for highly reputable systems)
        enhanced_weight = base_vote_weight * (1.0 + reputation_factor)
        
        # Record vote
        proposal.votes[system_id] = {
            "vote_weight": enhanced_weight,
            "timestamp": time.time()
        }
        
        # Update system participation
        self.update_system_reputation(system_id, participated=True)
        
        logger.info(f"🗳️ System {system_id} voted {base_vote_weight:+.2f} "
                   f"(enhanced weight: {enhanced_weight:.2f}) on proposal {proposal_id}")
        return True
    
    def evaluate_inter_system_proposal(self, proposal_id: str) -> Tuple[bool, float]:
        """
        Evaluate an inter-system proposal based on weighted votes
        
        Args:
            proposal_id: ID of proposal to evaluate
            
        Returns:
            Tuple of (is_approved, weighted_score)
        """
        proposal = self.inter_system_proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False, 0.0
        
        council = self.councils.get(proposal.council_id)
        if not council:
            logger.error(f"Council {proposal.council_id} not found")
            return False, 0.0
        
        if not council.member_systems:
            logger.warning("No member systems in council")
            return False, 0.0
        
        # Calculate weighted vote score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for system_id, vote_data in proposal.votes.items():
            # Check if system is still a member of the council
            if system_id in council.member_systems:
                vote_weight = vote_data["vote_weight"]
                weighted_sum += vote_weight
                # For normalization, we use the base reputation score
                reputation = self.system_reputations[system_id]
                base_reputation = (
                    reputation.coherence_score * 0.4 +
                    reputation.trust_score * 0.4 +
                    reputation.participation_score * 0.2
                )
                total_weight += max(0.1, base_reputation)  # Minimum weight to avoid division by zero
        
        if total_weight <= 0:
            logger.warning(f"No valid votes for proposal {proposal_id}")
            return False, 0.0
        
        # Normalize the weighted score
        weighted_score = weighted_sum / total_weight
        
        # Check approval threshold
        is_approved = (
            weighted_score >= council.council_params["approval_threshold"] and
            len(proposal.votes) >= council.council_params["min_systems"]
        )
        
        # Update proposal status
        proposal.status = "approved" if is_approved else "rejected"
        
        logger.info(f"📊 Inter-system proposal {proposal_id} evaluated: "
                   f"score={weighted_score:.4f}, approved={is_approved}")
        return is_approved, weighted_score
    
    def implement_inter_system_proposal(self, proposal_id: str) -> bool:
        """
        Implement an approved inter-system proposal
        
        Args:
            proposal_id: ID of proposal to implement
            
        Returns:
            bool: True if implementation successful
        """
        proposal = self.inter_system_proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        if proposal.status != "approved":
            logger.error(f"Proposal {proposal_id} is not approved (status: {proposal.status})")
            return False
        
        # Simulate implementation of changes
        # In a real system, this would involve actual cross-system communication
        implementation_log = {
            "timestamp": time.time(),
            "implemented_by": "inter_system_councils",
            "changes": proposal.target_changes,
            "target_systems": proposal.target_systems,
            "status": "simulated_success"
        }
        
        proposal.implementation_results = implementation_log
        proposal.status = "implemented"
        
        # Update system reputations based on successful implementation
        for system_id in proposal.target_systems:
            if system_id in self.system_reputations:
                # Increase coherence for systems that successfully implement changes
                self.update_system_reputation(system_id, coherence_delta=0.05, trust_delta=0.03)
                # Update proposal success count
                self.system_reputations[system_id].successful_proposals += 1
                self.system_reputations[system_id].total_proposals += 1
        
        logger.info(f"⚡ Implemented inter-system proposal {proposal_id}")
        return True
    
    def get_council_info(self, council_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a council
        
        Args:
            council_id: ID of council
            
        Returns:
            Dict with council information or None if not found
        """
        council = self.councils.get(council_id)
        if not council:
            return None
        
        # Get member system reputations
        member_reputations = {}
        for system_id in council.member_systems:
            if system_id in self.system_reputations:
                reputation = self.system_reputations[system_id]
                member_reputations[system_id] = {
                    "coherence_score": reputation.coherence_score,
                    "trust_score": reputation.trust_score,
                    "participation_score": reputation.participation_score
                }
        
        return {
            "council_id": council.council_id,
            "council_name": council.council_name,
            "member_systems": council.member_systems,
            "member_count": len(council.member_systems),
            "member_reputations": member_reputations,
            "formation_timestamp": council.formation_timestamp,
            "council_reputation": council.council_reputation,
            "active_proposals": len(council.active_proposals),
            "council_params": council.council_params
        }
    
    def get_system_reputation(self, system_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reputation information for a system
        
        Args:
            system_id: ID of system
            
        Returns:
            Dict with reputation information or None if not found
        """
        reputation = self.system_reputations.get(system_id)
        if not reputation:
            return None
        
        return {
            "system_id": reputation.system_id,
            "coherence_score": reputation.coherence_score,
            "trust_score": reputation.trust_score,
            "participation_score": reputation.participation_score,
            "last_interaction": reputation.last_interaction,
            "success_rate": reputation.successful_proposals / max(1, reputation.total_proposals),
            "total_proposals": reputation.total_proposals
        }

# Example usage and testing
if __name__ == "__main__":
    # This would normally integrate with the existing AI governance system
    # For demonstration, we'll create a mock that matches the expected interface
    from unittest.mock import Mock
    
    mock_governance = Mock(spec=AIGovernance)
    councils = InterSystemCouncils(mock_governance)
    
    # Register external systems
    councils.register_external_system("system_alpha", 0.85, 0.92)
    councils.register_external_system("system_beta", 0.78, 0.85)
    councils.register_external_system("system_gamma", 0.92, 0.88)
    
    print(f"Registered {len(councils.system_reputations)} external systems")
    
    # Form a council
    council_id = councils.form_council(
        "Harmonic Economic Council",
        ["system_alpha", "system_beta", "system_gamma"]
    )
    
    if council_id:
        print(f"Formed council: {council_id}")
        
        # Get council info
        council_info = councils.get_council_info(council_id)
        if council_info:
            print(f"Council has {council_info['member_count']} members")
        
        # Create an inter-system proposal
        proposal_changes = {
            "token_rate": 1.15,
            "cross_system_bridge": "enabled",
            "harmonic_frequency": 440.0
        }
        
        proposal_id = councils.create_inter_system_proposal(
            council_id=council_id,
            proposer_system="system_alpha",
            title="Cross-System Token Bridge",
            description="Enable token bridging between all council members",
            target_systems=["system_alpha", "system_beta", "system_gamma"],
            target_changes=proposal_changes
        )
        
        if proposal_id:
            print(f"Created inter-system proposal: {proposal_id}")
            
            # Vote on proposal
            councils.vote_on_inter_system_proposal("system_alpha", proposal_id, 1.0)  # Strong approve
            councils.vote_on_inter_system_proposal("system_beta", proposal_id, 0.8)   # Approve
            councils.vote_on_inter_system_proposal("system_gamma", proposal_id, 0.9)  # Approve
            
            # Evaluate proposal
            is_approved, score = councils.evaluate_inter_system_proposal(proposal_id)
            print(f"Proposal evaluation: approved={is_approved}, score={score:.4f}")
            
            if is_approved:
                # Implement proposal
                success = councils.implement_inter_system_proposal(proposal_id)
                print(f"Proposal implementation: {success}")
            
            # Check system reputations
            for system_id in ["system_alpha", "system_beta", "system_gamma"]:
                rep_info = councils.get_system_reputation(system_id)
                if rep_info:
                    print(f"System {system_id} reputation: "
                          f"coherence={rep_info['coherence_score']:.4f}, "
                          f"trust={rep_info['trust_score']:.4f}")
