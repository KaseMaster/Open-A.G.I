#!/usr/bin/env python3
"""
Governance Voting System for Quantum Currency
Implements Ψ-gated governance with CHR-weighted quadratic voting

This module provides:
1. Proposal creation and management
2. Quadratic vote handling with Ψ-gating
3. Validator eligibility checks based on coherence scores
4. Integration with CAL feedback for harmonic adjustments
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from core.validator_staking import Validator

@dataclass
class Proposal:
    """Represents a governance proposal"""
    id: str
    title: str
    description: str
    creator: str  # Validator address
    created_at: float
    voting_start: float
    voting_end: float
    status: str = "active"  # active, passed, rejected, cancelled
    votes_for: int = 0
    votes_against: int = 0
    total_weighted_votes: float = 0.0
    vote_power_distribution: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Vote:
    """Represents a single vote"""
    proposal_id: str
    validator_address: str
    vote: bool  # True for yes, False for no
    vote_power: float  # Weighted vote power
    timestamp: float
    signature: str

class GovernanceVotingSystem:
    """
    Governance Voting System with Ψ-gated eligibility and CHR-weighted quadratic voting
    """
    
    def __init__(self, network_id: str = "quantum-currency-governance-001"):
        self.network_id = network_id
        self.proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, List[Vote]] = {}  # proposal_id -> list of votes
        self.minimum_psi_threshold = 0.85  # Only validators with Ψ ≥ 0.85 can participate
        self.proposal_fee = 100.0  # FLX tokens required to create a proposal
        
    def create_proposal(self, title: str, description: str, creator: str, 
                       voting_duration_hours: int = 168) -> str:
        """
        Create a new governance proposal
        
        Args:
            title: Proposal title
            description: Proposal description
            creator: Validator address creating the proposal
            voting_duration_hours: Duration of voting period in hours (default: 168 = 1 week)
            
        Returns:
            str: Proposal ID
        """
        # Generate unique proposal ID
        proposal_data = f"{title}{description}{creator}{time.time()}"
        proposal_id = hashlib.sha256(proposal_data.encode()).hexdigest()[:32]
        
        # Create proposal
        proposal = Proposal(
            id=proposal_id,
            title=title,
            description=description,
            creator=creator,
            created_at=time.time(),
            voting_start=time.time() + 3600,  # Voting starts in 1 hour
            voting_end=time.time() + 3600 + (voting_duration_hours * 3600)
        )
        
        self.proposals[proposal_id] = proposal
        self.votes[proposal_id] = []
        
        return proposal_id
    
    def is_validator_eligible(self, validator: Validator) -> bool:
        """
        Check if a validator is eligible to participate in governance
        
        Args:
            validator: Validator to check
            
        Returns:
            bool: True if eligible, False otherwise
        """
        # Check if validator has sufficient coherence score
        if hasattr(validator, 'psi_score_history') and validator.psi_score_history:
            # Calculate rolling 90-day average Ψ score
            recent_psi_scores = validator.psi_score_history[-90:] if len(validator.psi_score_history) >= 90 else validator.psi_score_history
            if recent_psi_scores:
                avg_psi = sum(recent_psi_scores) / len(recent_psi_scores)
                return avg_psi >= self.minimum_psi_threshold
        
        # If no history, check current psi_score if available
        if hasattr(validator, 'psi_score'):
            return validator.psi_score >= self.minimum_psi_threshold
            
        return False
    
    def calculate_vote_power(self, validator: Validator) -> float:
        """
        Calculate vote power using CHR-weighted quadratic voting
        vote_power = √(CHR_balance) × Ψ
        
        Args:
            validator: Validator to calculate vote power for
            
        Returns:
            float: Vote power
        """
        # Get CHR balance
        chr_balance = getattr(validator, 'chr_balance', 0.0)
        
        # Get current Ψ score
        psi_score = getattr(validator, 'psi_score', 0.0)
        
        # Calculate vote power
        vote_power = (chr_balance ** 0.5) * psi_score
        
        return max(0.0, vote_power)
    
    def cast_vote(self, proposal_id: str, validator: Validator, vote: bool, signature: str) -> bool:
        """
        Cast a vote on a proposal
        
        Args:
            proposal_id: ID of proposal to vote on
            validator: Validator casting the vote
            vote: True for yes, False for no
            signature: Signature to verify vote authenticity
            
        Returns:
            bool: True if vote was cast successfully, False otherwise
        """
        # Check if proposal exists and is active
        if proposal_id not in self.proposals:
            print(f"Proposal {proposal_id} not found")
            return False
            
        proposal = self.proposals[proposal_id]
        current_time = time.time()
        
        # Check if voting is open
        if current_time < proposal.voting_start:
            print(f"Voting not started yet. Current: {current_time}, Start: {proposal.voting_start}")
            return False
        if current_time > proposal.voting_end:
            print(f"Voting already ended. Current: {current_time}, End: {proposal.voting_end}")
            return False
            
        # Check if proposal is still active
        if proposal.status != "active":
            print(f"Proposal not active. Status: {proposal.status}")
            return False
            
        # Check validator eligibility
        if not self.is_validator_eligible(validator):
            print(f"Validator {validator.validator_id} not eligible")
            return False
            
        # Check if validator has already voted
        for existing_vote in self.votes[proposal_id]:
            if existing_vote.validator_address == validator.validator_id:
                print(f"Validator {validator.validator_id} already voted")
                return False  # Validator already voted
                
        # Calculate vote power
        vote_power = self.calculate_vote_power(validator)
        print(f"Calculated vote power: {vote_power}")
        
        # Create vote
        new_vote = Vote(
            proposal_id=proposal_id,
            validator_address=validator.validator_id,
            vote=vote,
            vote_power=vote_power,
            timestamp=current_time,
            signature=signature
        )
        
        # Add vote to proposal
        self.votes[proposal_id].append(new_vote)
        
        # Update proposal vote counts
        if vote:
            proposal.votes_for += 1
        else:
            proposal.votes_against += 1
            
        proposal.total_weighted_votes += vote_power
        proposal.vote_power_distribution[validator.validator_id] = vote_power
        
        return True
    
    def tally_votes(self, proposal_id: str) -> Dict[str, Any]:
        """
        Tally votes for a proposal
        
        Args:
            proposal_id: ID of proposal to tally
            
        Returns:
            Dict with tally results
        """
        if proposal_id not in self.proposals:
            return {"error": "Proposal not found"}
            
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        # Calculate weighted vote totals
        total_for = sum(vote.vote_power for vote in votes if vote.vote)
        total_against = sum(vote.vote_power for vote in votes if not vote.vote)
        
        # Determine outcome
        passed = total_for > total_against
        participation_rate = len(votes) / 100 if 100 > 0 else 0  # Assuming 100 eligible validators
        
        return {
            "proposal_id": proposal_id,
            "title": proposal.title,
            "total_for": total_for,
            "total_against": total_against,
            "passed": passed,
            "participation_rate": participation_rate,
            "total_votes": len(votes),
            "vote_power_distribution": proposal.vote_power_distribution
        }
    
    def get_active_proposals(self) -> List[Proposal]:
        """
        Get all active proposals
        
        Returns:
            List of active proposals
        """
        current_time = time.time()
        active_proposals = []
        
        for proposal in self.proposals.values():
            if (proposal.status == "active" and 
                current_time >= proposal.voting_start and 
                current_time <= proposal.voting_end):
                active_proposals.append(proposal)
                
        return active_proposals
    
    def get_proposal_results(self, proposal_id: str) -> Dict[str, Any]:
        """
        Get detailed results for a proposal
        
        Args:
            proposal_id: ID of proposal
            
        Returns:
            Dict with proposal results
        """
        if proposal_id not in self.proposals:
            return {"error": "Proposal not found"}
            
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        return {
            "proposal": proposal,
            "votes": votes,
            "tally": self.tally_votes(proposal_id)
        }

# Example usage and testing
if __name__ == "__main__":
    # Create governance system
    gov_system = GovernanceVotingSystem()
    
    # Create a test proposal
    proposal_id = gov_system.create_proposal(
        title="Increase Block Size",
        description="Proposal to increase the maximum block size to 2MB",
        creator="validator-001"
    )
    
    print(f"Created proposal: {proposal_id}")
    print(f"Active proposals: {len(gov_system.get_active_proposals())}")
    
    # Test with actual Validator instance
    from core.validator_staking import Validator
    
    # Create a test validator
    validator = Validator(
        validator_id="validator-001",
        operator_address="valoper1xyz...",
        chr_score=0.95,
        total_staked={"FLX": 10000.0, "ATR": 5000.0},
        total_delegated={"FLX": 2000.0, "ATR": 1000.0},
        psi_score=0.9,
        psi_score_history=[0.85, 0.88, 0.9, 0.87, 0.89],
        chr_balance=1000.0
    )
    
    # Test eligibility
    is_eligible = gov_system.is_validator_eligible(validator)
    print(f"Validator eligible: {is_eligible}")
    
    # Test vote power calculation
    vote_power = gov_system.calculate_vote_power(validator)
    print(f"Vote power: {vote_power:.4f}")
    
    # Test vote casting
    vote_cast = gov_system.cast_vote(proposal_id, validator, True, "mock_signature")
    print(f"Vote cast successfully: {vote_cast}")
    
    # Test tally
    results = gov_system.get_proposal_results(proposal_id)
    print(f"Proposal results: {results['tally']}")