#!/usr/bin/env python3
"""
On-Chain Governance System for Quantum Currency
Implements on-chain governance proposals and voting mechanisms
"""

import sys
import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing governance module
from openagi.governance import GovernanceSystem, Proposal

@dataclass
class GovernanceProposal:
    """Represents an on-chain governance proposal"""
    proposal_id: str
    title: str
    description: str
    proposer: str
    proposed_at: float
    voting_start: float
    voting_end: float
    status: str  # "draft", "active", "passed", "rejected", "executed"
    proposal_type: str  # "parameter_change", "upgrade", "grant", "constitution"
    parameters: Dict  # Parameters to be changed or actions to be taken
    required_quorum: float  # Minimum percentage of votes needed (0.0 to 1.0)
    required_threshold: float  # Minimum percentage of yes votes (0.0 to 1.0)
    total_votes: Dict[str, float]  # {"yes": amount, "no": amount, "abstain": amount}
    voter_participation: Dict[str, str]  # {voter_address: vote_choice}
    execution_timestamp: Optional[float] = None
    execution_result: Optional[str] = None

@dataclass
class GovernanceVote:
    """Represents a vote on a governance proposal"""
    vote_id: str
    proposal_id: str
    voter_address: str
    vote_choice: str  # "yes", "no", "abstain"
    voting_power: float  # Weighted voting power
    voted_at: float
    signature: Optional[str] = None  # For verification

@dataclass
class GovernanceParameter:
    """Represents a governable parameter"""
    param_name: str
    current_value: float
    min_value: float
    max_value: float
    change_step: float  # Minimum change allowed
    governance_proposals: List[str]  # List of proposal IDs that affected this parameter

class OnChainGovernanceSystem:
    """
    Implements on-chain governance proposals and voting mechanisms
    """
    
    def __init__(self, governance_key: str = "default-governance-key"):
        self.governance_system = GovernanceSystem(governance_key)
        self.proposals: Dict[str, GovernanceProposal] = {}
        self.votes: Dict[str, GovernanceVote] = {}
        self.parameters: Dict[str, GovernanceParameter] = {}
        self.governance_config = {
            "default_quorum": 0.4,  # 40% of total voting power
            "default_threshold": 0.51,  # 51% majority
            "min_proposal_deposit": 1000.0,  # Minimum FLX deposit to propose
            "voting_period": 7 * 24 * 60 * 60,  # 7 days in seconds
            "proposal_cooldown": 24 * 60 * 60,  # 1 day cooldown between proposals
        }
        self._initialize_parameters()
        self.last_proposal_time = 0.0
    
    def _initialize_parameters(self):
        """Initialize governable parameters"""
        parameters_data = [
            ("mint_threshold", 0.75, 0.5, 0.95, 0.01),
            ("min_chr", 0.6, 0.1, 0.9, 0.05),
            ("block_time", 13.8, 5.0, 30.0, 0.1),
            ("validator_count", 100, 50, 200, 1),
            ("transaction_fee", 0.001, 0.0001, 0.01, 0.0001)
        ]
        
        for name, current, min_val, max_val, step in parameters_data:
            param = GovernanceParameter(
                param_name=name,
                current_value=current,
                min_value=min_val,
                max_value=max_val,
                change_step=step,
                governance_proposals=[]
            )
            self.parameters[name] = param
    
    def create_proposal(self, title: str, description: str, proposer: str,
                       proposal_type: str, parameters: Dict,
                       deposit_amount: float = 0.0) -> Optional[str]:
        """
        Create a new governance proposal
        
        Args:
            title: Title of the proposal
            description: Detailed description
            proposer: Address of the proposer
            proposal_type: Type of proposal
            parameters: Parameters to be changed or actions to be taken
            deposit_amount: Amount of FLX deposited to propose
            
        Returns:
            Proposal ID if successful, None otherwise
        """
        # Check if enough time has passed since last proposal
        current_time = time.time()
        if current_time - self.last_proposal_time < self.governance_config["proposal_cooldown"]:
            print("Must wait before creating another proposal")
            return None
        
        # Check deposit requirement
        if deposit_amount < self.governance_config["min_proposal_deposit"]:
            print(f"Deposit {deposit_amount} is below minimum {self.governance_config['min_proposal_deposit']}")
            return None
        
        # Create proposal ID
        proposal_id = f"prop-{int(current_time)}-{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        # Set voting timeline
        voting_start = current_time + 24 * 60 * 60  # Start voting in 1 day
        voting_end = voting_start + self.governance_config["voting_period"]
        
        # Create proposal
        proposal = GovernanceProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposer=proposer,
            proposed_at=current_time,
            voting_start=voting_start,
            voting_end=voting_end,
            status="draft",
            proposal_type=proposal_type,
            parameters=parameters,
            required_quorum=self.governance_config["default_quorum"],
            required_threshold=self.governance_config["default_threshold"],
            total_votes={"yes": 0.0, "no": 0.0, "abstain": 0.0},
            voter_participation={}
        )
        
        self.proposals[proposal_id] = proposal
        self.last_proposal_time = current_time
        
        return proposal_id
    
    def start_voting(self, proposal_id: str) -> bool:
        """
        Start voting on a proposal
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            True if successful, False otherwise
        """
        if proposal_id not in self.proposals:
            print(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.proposals[proposal_id]
        current_time = time.time()
        
        # Check if it's time to start voting
        if current_time < proposal.voting_start:
            print("Voting has not started yet")
            return False
        
        # Check if voting has already ended
        if current_time > proposal.voting_end:
            print("Voting has already ended")
            return False
        
        # Start voting
        proposal.status = "active"
        return True
    
    def cast_vote(self, proposal_id: str, voter_address: str, vote_choice: str,
                 voting_power: float, signature: Optional[str] = None) -> Optional[str]:
        """
        Cast a vote on a proposal
        
        Args:
            proposal_id: ID of the proposal
            voter_address: Address of the voter
            vote_choice: Vote choice ("yes", "no", "abstain")
            voting_power: Weighted voting power
            signature: Optional signature for verification
            
        Returns:
            Vote ID if successful, None otherwise
        """
        # Validate inputs
        if proposal_id not in self.proposals:
            print(f"Proposal {proposal_id} not found")
            return None
        
        if vote_choice not in ["yes", "no", "abstain"]:
            print("Invalid vote choice")
            return None
        
        proposal = self.proposals[proposal_id]
        
        # Check if voting is active
        if proposal.status != "active":
            print("Voting is not active for this proposal")
            return None
        
        # Check if voting period has ended
        current_time = time.time()
        if current_time > proposal.voting_end:
            print("Voting period has ended")
            return None
        
        # Check if voter has already voted
        if voter_address in proposal.voter_participation:
            print("Voter has already participated")
            return None
        
        # Create vote ID
        vote_id = f"vote-{int(current_time)}-{hashlib.md5(f'{proposal_id}{voter_address}'.encode()).hexdigest()[:8]}"
        
        # Create vote
        vote = GovernanceVote(
            vote_id=vote_id,
            proposal_id=proposal_id,
            voter_address=voter_address,
            vote_choice=vote_choice,
            voting_power=voting_power,
            voted_at=current_time,
            signature=signature
        )
        
        self.votes[vote_id] = vote
        
        # Update proposal
        proposal.total_votes[vote_choice] += voting_power
        proposal.voter_participation[voter_address] = vote_choice
        
        return vote_id
    
    def tally_votes(self, proposal_id: str) -> Optional[Dict]:
        """
        Tally votes for a proposal and determine outcome
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Dictionary with results or None if proposal not found
        """
        if proposal_id not in self.proposals:
            print(f"Proposal {proposal_id} not found")
            return None
        
        proposal = self.proposals[proposal_id]
        current_time = time.time()
        
        # Check if voting has ended
        if current_time < proposal.voting_end:
            print("Voting has not ended yet")
            return None
        
        # Calculate total voting power
        total_voting_power = sum(proposal.total_votes.values())
        
        # Check quorum
        # In a real implementation, we would compare to total network voting power
        # For this simulation, we'll use a fixed value
        network_voting_power = 100000.0  # Simulated total network voting power
        quorum = total_voting_power / network_voting_power
        
        if quorum < proposal.required_quorum:
            proposal.status = "rejected"
            return {
                "status": "rejected",
                "reason": "quorum_not_met",
                "quorum": quorum,
                "required_quorum": proposal.required_quorum
            }
        
        # Calculate results
        yes_votes = proposal.total_votes["yes"]
        no_votes = proposal.total_votes["no"]
        total_valid_votes = yes_votes + no_votes  # Abstains don't count toward threshold
        
        if total_valid_votes == 0:
            proposal.status = "rejected"
            return {
                "status": "rejected",
                "reason": "no_valid_votes",
                "yes_votes": yes_votes,
                "no_votes": no_votes
            }
        
        # Calculate threshold
        threshold = yes_votes / total_valid_votes
        
        # Determine outcome
        if threshold >= proposal.required_threshold:
            proposal.status = "passed"
            outcome = "passed"
        else:
            proposal.status = "rejected"
            outcome = "rejected"
        
        return {
            "status": outcome,
            "quorum": quorum,
            "threshold": threshold,
            "required_threshold": proposal.required_threshold,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "abstain_votes": proposal.total_votes["abstain"],
            "total_voting_power": total_voting_power,
            "voter_count": len(proposal.voter_participation)
        }
    
    def execute_proposal(self, proposal_id: str) -> bool:
        """
        Execute a passed proposal
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            True if successful, False otherwise
        """
        if proposal_id not in self.proposals:
            print(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.proposals[proposal_id]
        
        # Check if proposal has passed
        if proposal.status != "passed":
            print("Proposal has not passed")
            return False
        
        # Execute based on proposal type
        try:
            if proposal.proposal_type == "parameter_change":
                self._execute_parameter_change(proposal)
            elif proposal.proposal_type == "upgrade":
                self._execute_upgrade(proposal)
            elif proposal.proposal_type == "grant":
                self._execute_grant(proposal)
            elif proposal.proposal_type == "constitution":
                self._execute_constitution_change(proposal)
            else:
                print(f"Unknown proposal type: {proposal.proposal_type}")
                return False
            
            # Mark as executed
            proposal.status = "executed"
            proposal.execution_timestamp = time.time()
            return True
            
        except Exception as e:
            proposal.execution_result = f"Error: {str(e)}"
            return False
    
    def _execute_parameter_change(self, proposal: GovernanceProposal):
        """Execute a parameter change proposal"""
        for param_name, new_value in proposal.parameters.items():
            if param_name in self.parameters:
                param = self.parameters[param_name]
                
                # Validate new value is within bounds
                if param.min_value <= new_value <= param.max_value:
                    # Check that change is at least the minimum step
                    if abs(new_value - param.current_value) >= param.change_step:
                        old_value = param.current_value
                        param.current_value = new_value
                        param.governance_proposals.append(proposal.proposal_id)
                        proposal.execution_result = f"Changed {param_name} from {old_value} to {new_value}"
                    else:
                        raise ValueError(f"Change to {param_name} is smaller than minimum step {param.change_step}")
                else:
                    raise ValueError(f"New value {new_value} for {param_name} is outside bounds [{param.min_value}, {param.max_value}]")
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
    
    def _execute_upgrade(self, proposal: GovernanceProposal):
        """Execute a network upgrade proposal"""
        version = proposal.parameters.get("version", "unknown")
        features = proposal.parameters.get("features", [])
        proposal.execution_result = f"Upgraded to version {version} with features: {', '.join(features)}"
        # In a real implementation, this would trigger actual upgrade procedures
    
    def _execute_grant(self, proposal: GovernanceProposal):
        """Execute a grant proposal"""
        recipient = proposal.parameters.get("recipient", "unknown")
        amount = proposal.parameters.get("amount", 0.0)
        purpose = proposal.parameters.get("purpose", "general")
        proposal.execution_result = f"Granted {amount} FLX to {recipient} for {purpose}"
        # In a real implementation, this would transfer tokens
    
    def _execute_constitution_change(self, proposal: GovernanceProposal):
        """Execute a constitution change proposal"""
        changes = proposal.parameters.get("changes", {})
        proposal.execution_result = f"Constitution updated with {len(changes)} changes"
        # In a real implementation, this would update governance rules
    
    def get_proposal_info(self, proposal_id: str) -> Optional[Dict]:
        """
        Get detailed information about a proposal
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Dictionary with proposal information or None if not found
        """
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        
        info = {
            "proposal_id": proposal.proposal_id,
            "title": proposal.title,
            "description": proposal.description,
            "proposer": proposal.proposer,
            "proposed_at": proposal.proposed_at,
            "voting_start": proposal.voting_start,
            "voting_end": proposal.voting_end,
            "status": proposal.status,
            "proposal_type": proposal.proposal_type,
            "parameters": proposal.parameters,
            "required_quorum": proposal.required_quorum,
            "required_threshold": proposal.required_threshold,
            "total_votes": proposal.total_votes,
            "voter_count": len(proposal.voter_participation),
            "execution_result": proposal.execution_result
        }
        
        # Add time information
        current_time = time.time()
        if proposal.status == "draft":
            info["time_until_voting"] = max(0, proposal.voting_start - current_time)
        elif proposal.status == "active":
            info["time_until_voting_ends"] = max(0, proposal.voting_end - current_time)
        elif proposal.status in ["passed", "rejected", "executed"]:
            info["voting_ended_ago"] = current_time - proposal.voting_end
        
        return info
    
    def get_active_proposals(self) -> List[Dict]:
        """
        Get list of active proposals
        
        Returns:
            List of active proposal information
        """
        active_proposals = []
        current_time = time.time()
        
        for proposal_id, proposal in self.proposals.items():
            # Include draft proposals that will become active, active proposals, 
            # and proposals that have ended but haven't been tallied yet
            if (proposal.status == "draft" and proposal.voting_start <= current_time + 24*3600) or \
               (proposal.status == "active") or \
               (proposal.status in ["active", "draft"] and proposal.voting_end <= current_time):
                active_proposals.append(self.get_proposal_info(proposal_id))
        
        return active_proposals
    
    def get_governance_stats(self) -> Dict:
        """
        Get governance system statistics
        
        Returns:
            Dictionary with governance statistics
        """
        total_proposals = len(self.proposals)
        passed_proposals = len([p for p in self.proposals.values() if p.status == "passed"])
        rejected_proposals = len([p for p in self.proposals.values() if p.status == "rejected"])
        executed_proposals = len([p for p in self.proposals.values() if p.status == "executed"])
        active_proposals = len([p for p in self.proposals.values() if p.status == "active"])
        
        # Calculate total votes
        total_votes = 0
        total_voting_power = 0.0
        for proposal in self.proposals.values():
            total_votes += len(proposal.voter_participation)
            total_voting_power += sum(proposal.total_votes.values())
        
        return {
            "total_proposals": total_proposals,
            "passed_proposals": passed_proposals,
            "rejected_proposals": rejected_proposals,
            "executed_proposals": executed_proposals,
            "active_proposals": active_proposals,
            "total_votes": total_votes,
            "total_voting_power": total_voting_power,
            "pass_rate": passed_proposals / max(total_proposals, 1),
            "parameters": {name: param.current_value for name, param in self.parameters.items()}
        }

def demo_onchain_governance():
    """Demonstrate on-chain governance system capabilities"""
    print("🏛️  On-Chain Governance System Demo")
    print("=" * 35)
    
    # Create governance system instance
    governance = OnChainGovernanceSystem("quantum-governance-key")
    
    # Show initial governance stats
    print("\n📊 Initial Governance Stats:")
    stats = governance.get_governance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Show governable parameters
    print("\n⚙️  Governable Parameters:")
    for name, param in governance.parameters.items():
        print(f"   {name}: {param.current_value} (range: {param.min_value}-{param.max_value})")
    
    # Create proposals
    print("\n📝 Creating Proposals:")
    
    # Parameter change proposal
    param_proposal_id = governance.create_proposal(
        title="Adjust Mint Threshold",
        description="Lower the mint threshold to increase token supply growth",
        proposer="validator-001",
        proposal_type="parameter_change",
        parameters={"mint_threshold": 0.70},
        deposit_amount=1500.0
    )
    
    if param_proposal_id:
        print(f"   Parameter change proposal created: {param_proposal_id}")
    else:
        print("   Failed to create parameter change proposal")
    
    # Reset cooldown for next proposal (for demo purposes)
    governance.last_proposal_time = time.time() - governance.governance_config["proposal_cooldown"] - 1
    
    # Network upgrade proposal
    upgrade_proposal_id = governance.create_proposal(
        title="Quantum Network Upgrade v2.0",
        description="Upgrade to next-generation quantum consensus protocol",
        proposer="developer-team",
        proposal_type="upgrade",
        parameters={
            "version": "2.0.0",
            "features": ["quantum_consensus", "enhanced_security", "improved_throughput"]
        },
        deposit_amount=2000.0
    )
    
    if upgrade_proposal_id:
        print(f"   Upgrade proposal created: {upgrade_proposal_id}")
    else:
        print("   Failed to create upgrade proposal")
    
    # Reset cooldown for next proposal (for demo purposes)
    governance.last_proposal_time = time.time() - governance.governance_config["proposal_cooldown"] - 1
    
    # Grant proposal
    grant_proposal_id = governance.create_proposal(
        title="Research Grant for Quantum Algorithms",
        description="Fund research into quantum-resistant cryptographic algorithms",
        proposer="research-committee",
        proposal_type="grant",
        parameters={
            "recipient": "quantum-research-lab",
            "amount": 50000.0,
            "purpose": "cryptographic_research"
        },
        deposit_amount=1000.0
    )
    
    if grant_proposal_id:
        print(f"   Grant proposal created: {grant_proposal_id}")
    else:
        print("   Failed to create grant proposal")
    
    # Show created proposals
    print("\n📋 Created Proposals:")
    for proposal_id in [param_proposal_id, upgrade_proposal_id, grant_proposal_id]:
        if proposal_id:
            info = governance.get_proposal_info(proposal_id)
            if info:
                print(f"   {info['title']}:")
                print(f"      ID: {info['proposal_id']}")
                print(f"      Type: {info['proposal_type']}")
                print(f"      Status: {info['status']}")
                print(f"      Proposed by: {info['proposer']}")
    
    # Simulate voting (fast-forward time to start voting)
    print("\n🗳️  Simulating Voting:")
    
    # Start voting on proposals
    for proposal_id in [param_proposal_id, upgrade_proposal_id, grant_proposal_id]:
        if proposal_id:
            # Fast-forward proposal to active status for demo
            if proposal_id in governance.proposals:
                governance.proposals[proposal_id].status = "active"
                governance.proposals[proposal_id].voting_start = time.time() - 3600  # Started 1 hour ago
                governance.proposals[proposal_id].voting_end = time.time() + 3600  # Ends in 1 hour
            
            print(f"   Voting started for {proposal_id}")
    
    # Cast votes with majority "yes" to ensure passage
    voters = [
        ("validator-001", 10000.0),
        ("validator-002", 8000.0),
        ("validator-003", 12000.0),
        ("large-holder-001", 50000.0),
        ("community-member-001", 1000.0),
        ("community-member-002", 1500.0)
    ]
    
    # For the parameter change proposal, ensure it passes by having more "yes" votes
    if param_proposal_id and param_proposal_id in governance.proposals:
        print(f"   Voting on {param_proposal_id} (ensuring passage):")
        governance.proposals[param_proposal_id].total_votes = {"yes": 0.0, "no": 0.0, "abstain": 0.0}
        governance.proposals[param_proposal_id].voter_participation = {}
        
        # Ensure majority yes votes for parameter change proposal
        for i, (voter_address, voting_power) in enumerate(voters):
            vote_choice = "yes" if i < 4 else "no"  # First 4 voters vote yes, last 2 vote no
            vote_id = governance.cast_vote(
                proposal_id=param_proposal_id,
                voter_address=voter_address,
                vote_choice=vote_choice,
                voting_power=voting_power
            )
            if vote_id:
                print(f"      {voter_address} voted {vote_choice} with power {voting_power}")
    
    # For other proposals, use random voting
    vote_choices = ["yes", "no", "abstain"]
    
    for proposal_id in [upgrade_proposal_id, grant_proposal_id]:
        if proposal_id:
            print(f"   Voting on {proposal_id}:")
            # Reset votes for clean voting
            if proposal_id in governance.proposals:
                governance.proposals[proposal_id].total_votes = {"yes": 0.0, "no": 0.0, "abstain": 0.0}
                governance.proposals[proposal_id].voter_participation = {}
            
            for voter_address, voting_power in voters:
                vote_choice = np.random.choice(vote_choices, p=[0.6, 0.3, 0.1])  # 60% yes, 30% no, 10% abstain
                vote_id = governance.cast_vote(
                    proposal_id=proposal_id,
                    voter_address=voter_address,
                    vote_choice=vote_choice,
                    voting_power=voting_power
                )
                if vote_id:
                    print(f"      {voter_address} voted {vote_choice} with power {voting_power}")
    
    # Fast-forward time to end voting
    for proposal_id in [param_proposal_id, upgrade_proposal_id, grant_proposal_id]:
        if proposal_id and proposal_id in governance.proposals:
            governance.proposals[proposal_id].voting_end = time.time() - 3600  # Ended 1 hour ago
    
    # Tally votes
    print("\n📊 Tallying Votes:")
    for proposal_id in [param_proposal_id, upgrade_proposal_id, grant_proposal_id]:
        if proposal_id:
            results = governance.tally_votes(proposal_id)
            if results:
                info = governance.get_proposal_info(proposal_id)
                if info:
                    print(f"   {info['title']}:")
                    print(f"      Status: {results['status']}")
                    print(f"      Quorum: {results['quorum']:.1%} (required: {info['required_quorum']:.1%})")
                    print(f"      Threshold: {results['threshold']:.1%} (required: {info['required_threshold']:.1%})")
                    print(f"      Yes: {results['yes_votes']:.0f}, No: {results['no_votes']:.0f}, Abstain: {results['abstain_votes']:.0f}")
    
    # Execute passed proposals
    print("\n⚡ Executing Proposals:")
    for proposal_id in [param_proposal_id, upgrade_proposal_id, grant_proposal_id]:
        if proposal_id:
            info = governance.get_proposal_info(proposal_id)
            if info and info['status'] == 'passed':
                success = governance.execute_proposal(proposal_id)
                if success:
                    # Refresh info after execution
                    updated_info = governance.get_proposal_info(proposal_id)
                    if updated_info:
                        print(f"   Executed {updated_info['title']}")
                        print(f"      Result: {updated_info.get('execution_result', 'No result')}")
                    else:
                        print(f"   Executed proposal {proposal_id}")
                else:
                    if info:
                        print(f"   Failed to execute {info['title']}")
    
    # Show final governance stats
    print("\n📊 Final Governance Stats:")
    stats = governance.get_governance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Show updated parameters
    print("\n⚙️  Updated Parameters:")
    for name, param in governance.parameters.items():
        print(f"   {name}: {param.current_value} (changed in {len(param.governance_proposals)} proposals)")
    
    print("\n✅ On-chain governance system demo completed!")

if __name__ == "__main__":
    demo_onchain_governance()