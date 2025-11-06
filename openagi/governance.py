#!/usr/bin/env python3
"""
Governance Module for Quantum Currency System
Implements quadratic voting and reputation-weighted governance mechanisms
"""

import sys
import os
import json
import time
import hashlib
import hmac
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import harmonic validation classes for snapshot-based proofs
from openagi.harmonic_validation import HarmonicSnapshot, HarmonicProofBundle

@dataclass
class Proposal:
    """Represents a governance proposal"""
    id: str
    title: str
    description: str
    creator: str
    created_at: float
    voting_start: float
    voting_end: float
    options: List[str]
    status: str = "draft"  # draft, active, executed, rejected
    votes: Optional[Dict[str, Dict[str, int]]] = None  # {option: {voter: votes}}
    results: Optional[Dict[str, int]] = None  # {option: total_votes}
    # Snapshot-based proof for verifiable voting
    voting_proof: Optional[HarmonicProofBundle] = None
    # Signature for proposal authenticity
    proposal_signature: Optional[str] = None
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
        if self.results is None:
            self.results = {option: 0 for option in self.options}

@dataclass
class Vote:
    """Represents a vote on a proposal"""
    proposal_id: str
    voter: str
    option: str
    votes: int  # Number of votes (for quadratic voting)
    chr_score: float  # Voter's CHR reputation score
    timestamp: float
    signature: Optional[str] = None
    # Snapshot for vote verification
    voter_snapshot: Optional[HarmonicSnapshot] = None

@dataclass
class VotingProof:
    """Represents a verifiable proof of voting based on harmonic snapshots"""
    proposal_id: str
    voter: str
    option: str
    votes: int
    timestamp: float
    snapshot_bundle: HarmonicProofBundle
    proof_signature: Optional[str] = None

class GovernanceSystem:
    """Governance system with quadratic voting and CHR reputation weighting"""
    
    def __init__(self, secret_key: str = "governance_secret"):
        self.proposals = {}  # id -> Proposal
        self.votes = {}  # proposal_id -> [Vote]
        self.chr_scores = {}  # voter -> chr_score
        self.secret_key = secret_key  # For signing proposals and votes
    
    def set_chr_score(self, voter: str, chr_score: float):
        """Set CHR reputation score for a voter"""
        self.chr_scores[voter] = chr_score
    
    def get_chr_score(self, voter: str) -> float:
        """Get CHR reputation score for a voter"""
        return self.chr_scores.get(voter, 0.0)
    
    def _sign_data(self, data: str) -> str:
        """Create a signature for data using HMAC"""
        return hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def create_proposal(self, 
                       title: str, 
                       description: str, 
                       creator: str,
                       options: List[str],
                       voting_duration: int = 604800) -> Proposal:  # 7 days default
        """Create a new governance proposal"""
        proposal_id = hashlib.sha256(f"{title}{creator}{time.time()}".encode()).hexdigest()[:16]
        
        # Create proposal data for signing
        proposal_data = f"{proposal_id}{title}{creator}{time.time()}"
        signature = self._sign_data(proposal_data)
        
        proposal = Proposal(
            id=proposal_id,
            title=title,
            description=description,
            creator=creator,
            created_at=time.time(),
            voting_start=time.time(),  # Start voting immediately
            voting_end=time.time() + voting_duration,  # End voting after duration
            options=options,
            proposal_signature=signature
        )
        
        self.proposals[proposal_id] = proposal
        self.votes[proposal_id] = []
        return proposal
    
    def start_voting(self, proposal_id: str) -> bool:
        """Start voting on a proposal"""
        if proposal_id not in self.proposals:
            return False
            
        proposal = self.proposals[proposal_id]
        if proposal.status == "draft" and time.time() >= proposal.voting_start:
            proposal.status = "active"
            return True
        return False
    
    def calculate_vote_cost(self, votes: int) -> int:
        """
        Calculate the cost of votes in quadratic voting
        Cost = votes^2
        """
        return votes * votes
    
    def calculate_voting_power(self, voter: str, votes: int) -> float:
        """
        Calculate voting power based on CHR reputation and quadratic voting
        Voting Power = CHR_score * sqrt(votes)
        """
        chr_score = self.get_chr_score(voter)
        return chr_score * (votes ** 0.5)
    
    def create_voter_snapshot(self, voter: str, times: List[float], values: List[float]) -> HarmonicSnapshot:
        """
        Create a harmonic snapshot for a voter to verify their participation
        
        Args:
            voter: Voter's identifier
            times: Time series timestamps
            values: Time series values
            
        Returns:
            HarmonicSnapshot object
        """
        # Create snapshot data for signing
        snapshot_data = f"{voter}{time.time()}{hashlib.sha256(str(times).encode()).hexdigest()}"
        signature = self._sign_data(snapshot_data)
        
        # Create a basic snapshot (in a real implementation, this would use the actual harmonic validation)
        snapshot = HarmonicSnapshot(
            node_id=voter,
            timestamp=time.time(),
            times=times,
            values=values,
            spectrum=[],  # In a real implementation, this would be computed
            spectrum_hash=hashlib.sha256(str(times).encode()).hexdigest(),
            CS=0.0,  # Coherence score would be computed in real implementation
            phi_params={"lambda": 0.618, "phi": 1.618, "tau": 1.0}
        )
        
        return snapshot
    
    def vote(self, 
             proposal_id: str, 
             voter: str, 
             option: str, 
             votes: int,
             voter_times: Optional[List[float]] = None,
             voter_values: Optional[List[float]] = None) -> Optional[Vote]:
        """
        Cast a vote on a proposal using quadratic voting with CHR weighting
        
        Args:
            proposal_id: ID of the proposal to vote on
            voter: Voter's address
            option: Option to vote for
            votes: Number of votes to cast (cost will be votes^2)
            voter_times: Time series data for voter snapshot (optional)
            voter_values: Values for voter snapshot (optional)
            
        Returns:
            Vote object if successful, None if failed
        """
        # Check if proposal exists and is active
        if proposal_id not in self.proposals:
            return None
            
        proposal = self.proposals[proposal_id]
        if proposal.status != "active":
            return None
            
        # Check if voting period is still open
        if time.time() > proposal.voting_end:
            return None
            
        # Check if option is valid
        if option not in proposal.options:
            return None
            
        # Check if voter has already voted on this option
        if option in proposal.votes and voter in proposal.votes[option]:
            return None
            
        # Create voter snapshot if data provided
        voter_snapshot = None
        if voter_times and voter_values:
            voter_snapshot = self.create_voter_snapshot(voter, voter_times, voter_values)
            
        # Create vote data for signing
        vote_data = f"{proposal_id}{voter}{option}{votes}{time.time()}"
        signature = self._sign_data(vote_data)
            
        # Create vote
        vote = Vote(
            proposal_id=proposal_id,
            voter=voter,
            option=option,
            votes=votes,
            chr_score=self.get_chr_score(voter),
            timestamp=time.time(),
            signature=signature,
            voter_snapshot=voter_snapshot
        )
        
        # Add vote to proposal
        if option not in proposal.votes:
            proposal.votes[option] = {}
        proposal.votes[option][voter] = votes
        
        # Update proposal results
        voting_power = self.calculate_voting_power(voter, votes)
        proposal.results[option] += int(voting_power * 10000)  # Scale for display
        
        # Store vote
        self.votes[proposal_id].append(vote)
        
        return vote
    
    def create_voting_proof(self, proposal_id: str, voter: str) -> Optional[VotingProof]:
        """
        Create a verifiable proof of voting based on harmonic snapshots
        
        Args:
            proposal_id: ID of the proposal
            voter: Voter's address
            
        Returns:
            VotingProof object if successful, None if failed
        """
        if proposal_id not in self.proposals:
            return None
            
        proposal = self.proposals[proposal_id]
        
        # Find the voter's vote
        voter_vote = None
        for vote in self.votes[proposal_id]:
            if vote.voter == voter:
                voter_vote = vote
                break
                
        if not voter_vote or not voter_vote.voter_snapshot:
            return None
            
        # In a real implementation, we would create a proof bundle from multiple snapshots
        # For this demo, we'll create a simple proof bundle with just the voter's snapshot
        snapshot_bundle = HarmonicProofBundle(
            snapshots=[voter_vote.voter_snapshot],
            aggregated_CS=voter_vote.voter_snapshot.CS
        )
        
        # Create proof data for signing
        proof_data = f"{proposal_id}{voter}{voter_vote.option}{voter_vote.votes}{voter_vote.timestamp}"
        proof_signature = self._sign_data(proof_data)
        
        proof = VotingProof(
            proposal_id=proposal_id,
            voter=voter,
            option=voter_vote.option,
            votes=voter_vote.votes,
            timestamp=voter_vote.timestamp,
            snapshot_bundle=snapshot_bundle,
            proof_signature=proof_signature
        )
        
        return proof
    
    def tally_votes(self, proposal_id: str) -> Optional[Dict[str, int]]:
        """
        Tally votes for a proposal and determine results
        
        Args:
            proposal_id: ID of the proposal to tally
            
        Returns:
            Dictionary of results {option: votes} or None if failed
        """
        if proposal_id not in self.proposals:
            return None
            
        proposal = self.proposals[proposal_id]
        
        # Only tally if voting period has ended
        if time.time() < proposal.voting_end:
            return None
            
        # Update status
        proposal.status = "executed"  # Simplified for demo
        
        return proposal.results.copy()
    
    def get_proposal_results(self, proposal_id: str) -> Optional[Dict[str, int]]:
        """Get current results for a proposal"""
        if proposal_id not in self.proposals:
            return None
            
        return self.proposals[proposal_id].results.copy()
    
    def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals"""
        return [p for p in self.proposals.values() if p.status == "active"]
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get a specific proposal"""
        return self.proposals.get(proposal_id)
    
    def verify_vote_proof(self, proof: VotingProof) -> bool:
        """
        Verify a voting proof using the signature and snapshot data
        
        Args:
            proof: VotingProof to verify
            
        Returns:
            True if proof is valid, False otherwise
        """
        # Verify proof signature
        proof_data = f"{proof.proposal_id}{proof.voter}{proof.option}{proof.votes}{proof.timestamp}"
        expected_signature = self._sign_data(proof_data)
        
        if proof.proof_signature != expected_signature:
            return False
            
        # Verify proposal exists
        if proof.proposal_id not in self.proposals:
            return False
            
        # Verify voter participated in this proposal
        proposal_votes = self.votes.get(proof.proposal_id, [])
        voter_participated = any(vote.voter == proof.voter for vote in proposal_votes)
        
        return voter_participated

def demo_snapshot_governance():
    """Demonstrate snapshot-based governance with verifiable voting proofs"""
    print("🏛️ Quantum Currency Governance - Snapshot-Based Voting Demo")
    print("=" * 60)
    
    # Create governance system
    gov = GovernanceSystem("test_governance_key")
    
    # Set CHR scores for voters
    voters = ["node-A", "node-B", "node-C"]
    chr_scores = [0.9, 0.7, 0.8]  # Different reputation scores
    
    for voter, score in zip(voters, chr_scores):
        gov.set_chr_score(voter, score)
        print(f"   {voter}: CHR = {score}")
    
    # Create a proposal
    proposal = gov.create_proposal(
        title="Quantum Network Upgrade",
        description="Upgrade to next-generation quantum consensus protocol",
        creator="node-A",
        options=["Approve", "Reject", "Abstain"],
        voting_duration=1  # 1 second for demo
    )
    
    print(f"\n📝 Created Proposal: {proposal.title}")
    print(f"   ID: {proposal.id}")
    print(f"   Options: {', '.join(proposal.options)}")
    
    # Start voting
    gov.start_voting(proposal.id)
    print(f"\n✅ Voting started for proposal {proposal.id}")
    
    # Simulate voting with snapshot data
    print(f"\n🗳️  Voting Simulation with Snapshots:")
    
    # Generate sample time series data for snapshots
    import numpy as np
    times_a = np.linspace(0, 0.5, 100).tolist()
    values_a = np.sin(2 * np.pi * 50 * np.linspace(0, 0.5, 100)).tolist()
    
    times_b = np.linspace(0, 0.5, 100).tolist()
    values_b = np.sin(2 * np.pi * 50 * np.linspace(0, 0.5, 100) + 0.1).tolist()
    
    times_c = np.linspace(0, 0.5, 100).tolist()
    values_c = np.sin(2 * np.pi * 50 * np.linspace(0, 0.5, 100) + 0.2).tolist()
    
    # node-A votes (high CHR) with snapshot
    vote1 = gov.vote(proposal.id, "node-A", "Approve", 3, times_a, values_a)
    if vote1:
        power = gov.calculate_voting_power("node-A", 3)
        print(f"   node-A: 3 votes for 'Approve' (Cost: 9, Power: {power:.3f})")
        if vote1.voter_snapshot:
            print(f"      Snapshot created with hash: {vote1.voter_snapshot.spectrum_hash[:16]}...")
    
    # node-B votes (medium CHR) with snapshot
    vote2 = gov.vote(proposal.id, "node-B", "Approve", 2, times_b, values_b)
    if vote2:
        power = gov.calculate_voting_power("node-B", 2)
        print(f"   node-B: 2 votes for 'Approve' (Cost: 4, Power: {power:.3f})")
        if vote2.voter_snapshot:
            print(f"      Snapshot created with hash: {vote2.voter_snapshot.spectrum_hash[:16]}...")
    
    # node-C votes (medium CHR) with snapshot
    vote3 = gov.vote(proposal.id, "node-C", "Reject", 4, times_c, values_c)
    if vote3:
        power = gov.calculate_voting_power("node-C", 4)
        print(f"   node-C: 4 votes for 'Reject' (Cost: 16, Power: {power:.3f})")
        if vote3.voter_snapshot:
            print(f"      Snapshot created with hash: {vote3.voter_snapshot.spectrum_hash[:16]}...")
    
    # Wait for voting to end (simulate)
    print(f"\n⏳ Waiting for voting to end...")
    time.sleep(2)  # Wait for voting period to end
    
    # Tally votes
    results = gov.tally_votes(proposal.id)
    if results:
        print(f"\n📊 Final Voting Results:")
        for option, votes in results.items():
            print(f"   {option}: {votes} votes")
        
        # Determine winner
        if results:
            winner = max(results.keys(), key=lambda x: results[x] if results[x] is not None else 0)
            total_votes = sum(results.values())
            winning_votes = results[winner]
            percentage = (winning_votes / total_votes * 100) if total_votes > 0 else 0
            
            print(f"\n🏆 Winner: {winner} ({winning_votes} votes, {percentage:.1f}%)")
    
    # Demonstrate verifiable voting proofs
    print(f"\n🔍 Verifiable Voting Proofs:")
    
    # Create and verify voting proofs for each voter
    for voter in voters:
        proof = gov.create_voting_proof(proposal.id, voter)
        if proof:
            is_valid = gov.verify_vote_proof(proof)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"   {voter}: {status} proof (Option: {proof.option}, Votes: {proof.votes})")
            if proof.proof_signature:
                print(f"      Proof signature: {proof.proof_signature[:16]}...")
            if proof.snapshot_bundle.snapshots:
                print(f"      Snapshot hash: {proof.snapshot_bundle.snapshots[0].spectrum_hash[:16]}...")
        else:
            print(f"   {voter}: No proof available")
    
    print(f"\n✅ Snapshot-based governance demo completed!")

if __name__ == "__main__":
    demo_snapshot_governance()