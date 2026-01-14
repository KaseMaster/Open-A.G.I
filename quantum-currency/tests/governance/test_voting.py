#!/usr/bin/env python3
"""
Test suite for Governance Voting System
"""

import sys
import os
import pytest
import time
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import with direct path manipulation for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'governance'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'core'))

from voting import GovernanceVotingSystem, Proposal, Vote
from validator_staking import Validator

class TestGovernanceVotingSystem:
    """Test cases for GovernanceVotingSystem"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.gov_system = GovernanceVotingSystem()
        
    def test_create_proposal(self):
        """Test creating a new proposal"""
        proposal_id = self.gov_system.create_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            creator="validator-001"
        )
        
        assert proposal_id is not None
        assert len(proposal_id) == 32  # SHA256 hex digest truncated to 32 chars
        assert proposal_id in self.gov_system.proposals
        assert len(self.gov_system.proposals) == 1
        
    def test_validator_eligibility(self):
        """Test validator eligibility checks"""
        # Create eligible validator
        eligible_validator = Validator(
            validator_id="validator-001",
            operator_address="valoper1xyz...",
            chr_score=0.95,
            total_staked={"FLX": 10000.0, "ATR": 5000.0},
            total_delegated={"FLX": 2000.0, "ATR": 1000.0},
            psi_score=0.9,
            psi_score_history=[0.85, 0.88, 0.9, 0.87, 0.89],
            chr_balance=1000.0
        )
        
        # Create ineligible validator
        ineligible_validator = Validator(
            validator_id="validator-002",
            operator_address="valoper2abc...",
            chr_score=0.75,
            total_staked={"FLX": 5000.0, "ATR": 2500.0},
            total_delegated={"FLX": 1000.0, "ATR": 500.0},
            psi_score=0.8,  # Below threshold
            psi_score_history=[0.75, 0.78, 0.8, 0.77, 0.79],
            chr_balance=500.0
        )
        
        # Test eligibility
        assert self.gov_system.is_validator_eligible(eligible_validator) == True
        assert self.gov_system.is_validator_eligible(ineligible_validator) == False
        
    def test_calculate_vote_power(self):
        """Test vote power calculation"""
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
        
        vote_power = self.gov_system.calculate_vote_power(validator)
        
        # vote_power = √(CHR_balance) × Ψ
        # vote_power = √(1000) × 0.9 = 31.62 × 0.9 = 28.46
        expected_vote_power = (1000.0 ** 0.5) * 0.9
        assert abs(vote_power - expected_vote_power) < 0.01
        
    def test_cast_vote(self):
        """Test casting a vote"""
        # Create proposal
        proposal_id = self.gov_system.create_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            creator="validator-001"
        )
        
        # Modify the proposal to start voting immediately
        proposal = self.gov_system.proposals[proposal_id]
        proposal.voting_start = time.time() - 1  # Start 1 second ago
        proposal.voting_end = time.time() + 3600  # End in 1 hour
        
        # Create validator
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
        
        # Cast vote
        vote_success = self.gov_system.cast_vote(
            proposal_id=proposal_id,
            validator=validator,
            vote=True,
            signature="test_signature"
        )
        
        assert vote_success == True
        assert len(self.gov_system.votes[proposal_id]) == 1
        
        # Try to vote again (should fail)
        vote_success2 = self.gov_system.cast_vote(
            proposal_id=proposal_id,
            validator=validator,
            vote=False,
            signature="test_signature2"
        )
        
        assert vote_success2 == False
        assert len(self.gov_system.votes[proposal_id]) == 1
        
    def test_tally_votes(self):
        """Test vote tallying"""
        # Create proposal
        proposal_id = self.gov_system.create_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            creator="validator-001"
        )
        
        # Modify the proposal to start voting immediately
        proposal = self.gov_system.proposals[proposal_id]
        proposal.voting_start = time.time() - 1  # Start 1 second ago
        proposal.voting_end = time.time() + 3600  # End in 1 hour
        
        # Create validators
        validator1 = Validator(
            validator_id="validator-001",
            operator_address="valoper1xyz...",
            chr_score=0.95,
            total_staked={"FLX": 10000.0, "ATR": 5000.0},
            total_delegated={"FLX": 2000.0, "ATR": 1000.0},
            psi_score=0.9,
            psi_score_history=[0.85, 0.88, 0.9, 0.87, 0.89],
            chr_balance=1000.0
        )
        
        validator2 = Validator(
            validator_id="validator-002",
            operator_address="valoper2abc...",
            chr_score=0.85,
            total_staked={"FLX": 5000.0, "ATR": 2500.0},
            total_delegated={"FLX": 1000.0, "ATR": 500.0},
            psi_score=0.85,
            psi_score_history=[0.85, 0.85, 0.85, 0.85, 0.85],
            chr_balance=500.0
        )
        
        # Cast votes
        vote1_success = self.gov_system.cast_vote(proposal_id, validator1, True, "sig1")
        vote2_success = self.gov_system.cast_vote(proposal_id, validator2, False, "sig2")
        
        assert vote1_success == True
        assert vote2_success == True
        
        # Tally votes
        results = self.gov_system.tally_votes(proposal_id)
        
        assert results["total_for"] > 0
        assert results["total_against"] > 0
        assert results["total_votes"] == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])