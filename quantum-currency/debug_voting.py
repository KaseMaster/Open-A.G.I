#!/usr/bin/env python3
"""
Debug script for Governance Voting System
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from governance.voting import GovernanceVotingSystem
from core.validator_staking import Validator

# Create governance system
gov_system = GovernanceVotingSystem()

# Create a test proposal with immediate voting
proposal_id = gov_system.create_proposal(
    title="Test Proposal",
    description="This is a test proposal",
    creator="validator-001",
    voting_duration_hours=1  # Short duration for testing
)

print(f"Created proposal: {proposal_id}")

# Modify the proposal to start voting immediately
proposal = gov_system.proposals[proposal_id]
proposal.voting_start = time.time() - 1  # Start 1 second ago
proposal.voting_end = time.time() + 3600  # End in 1 hour

print(f"Voting start: {proposal.voting_start}, current time: {time.time()}")

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

print(f"Validator PSI score: {validator.psi_score}")
print(f"Validator PSI history: {validator.psi_score_history}")

# Test eligibility
is_eligible = gov_system.is_validator_eligible(validator)
print(f"Validator eligible: {is_eligible}")

# Test vote power calculation
vote_power = gov_system.calculate_vote_power(validator)
print(f"Vote power: {vote_power:.4f}")

# Test vote casting
vote_success = gov_system.cast_vote(
    proposal_id=proposal_id,
    validator=validator,
    vote=True,
    signature="test_signature"
)

print(f"Vote cast successfully: {vote_success}")

if vote_success:
    # Test tally
    results = gov_system.tally_votes(proposal_id)
    print(f"Proposal results: {results}")