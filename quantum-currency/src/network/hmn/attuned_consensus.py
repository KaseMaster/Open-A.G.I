#!/usr/bin/env python3
"""
λ(t)-Attuned BFT Consensus for Harmonic Mesh Network
Implements adaptive consensus with self-healing mechanisms based on coherence metrics
"""

import asyncio
import time
import json
import hashlib
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsensusMode(Enum):
    """Consensus operation modes"""
    IDLE = 0
    REGULAR = 1
    EMERGENCY = 2

class ConsensusActionType(Enum):
    """Types of consensus actions"""
    SLASHING = "slashing"
    BOOSTING = "boosting"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STATE_RECOVERY = "state_recovery"
    SHARD_COORDINATION = "shard_coordination"

@dataclass
class ConsensusAction:
    """Represents a consensus action to be executed"""
    action_type: ConsensusActionType
    target: str  # validator_id or parameter_name
    value: Any  # amount, new_value, etc.
    reason: str
    priority: int = 1  # 1=highest, 5=lowest
    weight: float = 1.0  # Weight based on validator stake and psi score

@dataclass
class ConsensusRound:
    """Represents a consensus round"""
    round_id: str
    mode: ConsensusMode
    actions: List[ConsensusAction]
    participants: List[str]
    timestamp: float
    duration: float = 0.0
    shard_id: Optional[str] = None  # For multi-shard consensus
    votes: Dict[str, bool] = field(default_factory=dict)  # validator_id -> vote
    finality_proof: Optional[str] = None  # Cryptographic proof of finality

@dataclass
class ValidatorState:
    """Represents the state of a validator"""
    validator_id: str
    psi_score: float
    coherence_contribution: float
    uptime: float
    stake_amount: float
    active: bool = True
    shard_id: Optional[str] = None  # For multi-shard support
    last_vote: Optional[bool] = None
    vote_timestamp: Optional[float] = None

@dataclass
class ShardState:
    """Represents the state of a shard"""
    shard_id: str
    validators: List[str]
    coherence_score: float
    last_consensus_time: float
    pending_actions: List[ConsensusAction]

@dataclass
class ConsensusMetrics:
    """Metrics for consensus observability"""
    rounds_completed: int = 0
    emergency_rounds: int = 0
    avg_round_duration: float = 0.0
    validator_participation_rate: float = 1.0
    slashing_events: int = 0
    boosting_events: int = 0
    failed_rounds: int = 0
    rollback_events: int = 0

class AttunedConsensus:
    """
    Implements λ(t)-Attuned BFT Consensus for the Harmonic Mesh Network
    Provides adaptive consensus scheduling and self-healing mechanisms
    """
    
    def __init__(self, node_id: str, network_config: Dict[str, Any]):
        self.node_id = node_id
        self.network_config = network_config
        self.validators: Dict[str, ValidatorState] = {}
        self.shards: Dict[str, ShardState] = {}  # For multi-shard consensus
        self.consensus_history: List[ConsensusRound] = []
        self.current_mode: ConsensusMode = ConsensusMode.IDLE
        self.last_consensus_time: float = 0.0
        self.emergency_threshold: float = 0.7  # Ĉ(t) < 0.7 triggers emergency
        self.rollback_history: List[Dict[str, Any]] = []  # For automatic recovery
        self.metrics = ConsensusMetrics()
        
        # Parallel consensus support
        self.parallel_consensus_lock = threading.Lock()
        self.active_consensus_rounds: Dict[str, ConsensusRound] = {}
        
        # Configuration
        self.config = {
            "base_epoch_time": 60.0,  # seconds
            "min_epoch_duration": 10.0,  # seconds
            "emergency_consensus_timeout": 5.0,  # seconds
            "consensus_quorum": 0.67,  # 2/3 of validators needed
            "slashing_threshold": 0.6,  # Ψ < 0.6 triggers slashing
            "boosting_threshold": 0.9,  # Ψ > 0.9 eligible for boosting
            "max_parallel_rounds": 5,  # Maximum parallel consensus rounds
            "rollback_depth": 10,  # Maximum rollback depth for recovery
            "vote_timeout": 10.0,  # Seconds to wait for votes
        }
        
        logger.info(f"λ(t)-Attuned Consensus initialized for node: {node_id}")
    
    def calculate_dynamic_epoch(self, λ_t: float) -> float:
        """
        Calculate dynamic epoch duration based on λ(t)
        Epoch duration = Base Time × λ(t)
        """
        dynamic_epoch = self.config["base_epoch_time"] * λ_t
        return max(self.config["min_epoch_duration"], dynamic_epoch)
    
    def should_trigger_consensus(self, network_state: Dict[str, Any]) -> ConsensusMode:
        """
        Determine if consensus should be triggered and what type
        """
        Ĉ_t = network_state.get("coherence_density", 0.8)
        λ_t = network_state.get("lambda_t", 0.7)
        current_time = time.time()
        
        # Self-Healing Consensus for emergency state
        if Ĉ_t < self.emergency_threshold:
            logger.warning(f"Emergency consensus triggered: Ĉ(t)={Ĉ_t:.3f} < {self.emergency_threshold}")
            return ConsensusMode.EMERGENCY
        
        # Regular consensus based on normal conditions
        time_since_last_consensus = current_time - self.last_consensus_time
        dynamic_epoch = self.calculate_dynamic_epoch(λ_t)
        
        if time_since_last_consensus >= dynamic_epoch:
            return ConsensusMode.REGULAR
        
        return ConsensusMode.IDLE
    
    def self_healing_consensus(self, network_state: Dict[str, Any]) -> List[ConsensusAction]:
        """
        Execute self-healing consensus when Ĉ(t) < 0.7
        """
        Ĉ_t = network_state.get("coherence_density", 0.8)
        if Ĉ_t >= self.emergency_threshold:
            return []
        
        logger.info("Executing self-healing consensus")
        emergency_actions = []
        
        # Mass slashing for validators with low Ψ scores
        low_psi_validators = self._get_low_psi_validators()
        for validator in low_psi_validators:
            slashing_amount = self._calculate_slashing_amount(validator.psi_score)
            emergency_actions.append(ConsensusAction(
                action_type=ConsensusActionType.SLASHING,
                target=validator.validator_id,
                value=slashing_amount,
                reason=f"Low Ψ score: {validator.psi_score:.3f}",
                priority=1,
                weight=self._calculate_validator_weight(validator)
            ))
        
        # T4 boosts for high-performing validators
        high_psi_validators = self._get_high_psi_validators()
        for validator in high_psi_validators:
            boost_amount = self._calculate_boost_amount(validator.psi_score)
            emergency_actions.append(ConsensusAction(
                action_type=ConsensusActionType.BOOSTING,
                target=validator.validator_id,
                value=boost_amount,
                reason=f"High Ψ score: {validator.psi_score:.3f}",
                priority=2,
                weight=self._calculate_validator_weight(validator)
            ))
        
        return emergency_actions
    
    def regular_consensus(self, network_state: Dict[str, Any]) -> List[ConsensusAction]:
        """
        Execute regular consensus operations
        """
        logger.info("Executing regular consensus")
        regular_actions = []
        
        # Parameter adjustments based on network state
        λ_t = network_state.get("lambda_t", 0.7)
        Ĉ_t = network_state.get("coherence_density", 0.8)
        
        # Adjust parameters if needed
        if λ_t < 0.5:  # High instability
            regular_actions.append(ConsensusAction(
                action_type=ConsensusActionType.PARAMETER_ADJUSTMENT,
                target="consensus_sensitivity",
                value="increase",
                reason=f"High instability: λ(t)={λ_t:.3f}",
                priority=3,
                weight=1.0
            ))
        elif λ_t > 0.9:  # High stability
            regular_actions.append(ConsensusAction(
                action_type=ConsensusActionType.PARAMETER_ADJUSTMENT,
                target="consensus_sensitivity",
                value="decrease",
                reason=f"High stability: λ(t)={λ_t:.3f}",
                priority=4,
                weight=1.0
            ))
        
        # Validator performance adjustments
        low_contributors = self._get_low_contribution_validators()
        for validator in low_contributors:
            regular_actions.append(ConsensusAction(
                action_type=ConsensusActionType.PARAMETER_ADJUSTMENT,
                target=f"validator_{validator.validator_id}_weight",
                value="reduce",
                reason=f"Low contribution: {validator.coherence_contribution:.3f}",
                priority=4,
                weight=self._calculate_validator_weight(validator)
            ))
        
        return regular_actions
    
    def execute_consensus_round(self, network_state: Dict[str, Any]) -> Optional[ConsensusRound]:
        """
        Execute a consensus round based on current network state
        """
        consensus_mode = self.should_trigger_consensus(network_state)
        
        if consensus_mode == ConsensusMode.IDLE:
            return None
        
        start_time = time.time()
        logger.info(f"Starting {consensus_mode.name} consensus round")
        
        # Determine participants (in a real implementation, this would use actual validator set)
        participants = list(self.validators.keys()) if self.validators else ["validator-1", "validator-2", "validator-3"]
        
        # Generate consensus actions based on mode
        if consensus_mode == ConsensusMode.EMERGENCY:
            actions = self.self_healing_consensus(network_state)
            self.metrics.emergency_rounds += 1
        else:
            actions = self.regular_consensus(network_state)
        
        # Sort actions by priority and weight
        actions.sort(key=lambda x: (x.priority, -x.weight))
        
        # Create consensus round record
        round_id = f"consensus-{int(start_time)}"
        consensus_round = ConsensusRound(
            round_id=round_id,
            mode=consensus_mode,
            actions=actions,
            participants=participants,
            timestamp=start_time
        )
        
        # Execute consensus with voting
        try:
            votes = self._conduct_voting(consensus_round)
            consensus_round.votes = votes
            
            # Check if consensus was reached
            if self._check_consensus_reached(votes):
                # Execute actions (in a real implementation, this would coordinate with other nodes)
                self._execute_consensus_actions(actions)
                
                # Generate finality proof
                consensus_round.finality_proof = self._generate_finality_proof(consensus_round)
                
                # Update metrics
                self.metrics.rounds_completed += 1
            else:
                logger.warning(f"Consensus not reached in round {round_id}")
                self.metrics.failed_rounds += 1
                return None
                
        except Exception as e:
            logger.error(f"Error during consensus execution: {e}")
            self.metrics.failed_rounds += 1
            
            # Attempt rollback if configured
            if self.config.get("auto_rollback_on_failure", True):
                self._attempt_rollback(consensus_round)
            
            return None
        
        # Record completion
        end_time = time.time()
        consensus_round.duration = end_time - start_time
        self.consensus_history.append(consensus_round)
        self.last_consensus_time = end_time
        self.current_mode = ConsensusMode.IDLE
        
        # Keep only recent history
        if len(self.consensus_history) > 100:
            self.consensus_history = self.consensus_history[-100:]
        
        # Update average round duration
        total_duration = sum(round.duration for round in self.consensus_history)
        self.metrics.avg_round_duration = total_duration / max(1, len(self.consensus_history))
        
        logger.info(f"Completed {consensus_mode.name} consensus round in {consensus_round.duration:.3f}s")
        logger.info(f"Executed {len(actions)} consensus actions")
        
        return consensus_round
    
    def _conduct_voting(self, consensus_round: ConsensusRound) -> Dict[str, bool]:
        """Conduct voting for a consensus round"""
        votes = {}
        
        # In a real implementation, this would communicate with validators
        # For now, we'll simulate voting based on validator properties
        for validator_id in consensus_round.participants:
            # Simulate vote based on validator state
            validator = self.validators.get(validator_id)
            if validator and validator.active:
                # Higher psi score and uptime lead to higher chance of positive vote
                vote_probability = min(1.0, (validator.psi_score * 0.7 + validator.uptime * 0.3))
                vote = True if vote_probability > 0.5 else False
                votes[validator_id] = vote
                
                # Update validator's last vote
                validator.last_vote = vote
                validator.vote_timestamp = time.time()
            else:
                # Inactive validators vote negatively
                votes[validator_id] = False
        
        return votes
    
    def _check_consensus_reached(self, votes: Dict[str, bool]) -> bool:
        """Check if consensus was reached based on votes"""
        if not votes:
            return False
        
        positive_votes = sum(1 for vote in votes.values() if vote)
        total_votes = len(votes)
        quorum = self.config["consensus_quorum"]
        
        return (positive_votes / total_votes) >= quorum
    
    def _calculate_validator_weight(self, validator: ValidatorState) -> float:
        """Calculate validator weight based on stake and psi score"""
        # Weight is a combination of stake amount and psi score
        # Normalize stake amount (assuming max stake is 100000)
        normalized_stake = min(1.0, validator.stake_amount / 100000.0)
        
        # Weight formula: 0.6 * normalized_stake + 0.4 * psi_score
        weight = 0.6 * normalized_stake + 0.4 * validator.psi_score
        return max(0.1, weight)  # Minimum weight of 0.1
    
    def _get_low_psi_validators(self) -> List[ValidatorState]:
        """Get validators with low Ψ scores"""
        return [
            validator for validator in self.validators.values()
            if validator.psi_score < self.config["slashing_threshold"]
        ]
    
    def _get_high_psi_validators(self) -> List[ValidatorState]:
        """Get validators with high Ψ scores"""
        return [
            validator for validator in self.validators.values()
            if validator.psi_score > self.config["boosting_threshold"]
        ]
    
    def _get_low_contribution_validators(self) -> List[ValidatorState]:
        """Get validators with low coherence contributions"""
        avg_contribution = sum(v.coherence_contribution for v in self.validators.values()) / max(1, len(self.validators))
        return [
            validator for validator in self.validators.values()
            if validator.coherence_contribution < avg_contribution * 0.5
        ]
    
    def _calculate_slashing_amount(self, psi_score: float) -> float:
        """Calculate slashing amount based on Ψ score"""
        # Higher penalty for lower Ψ scores
        penalty_factor = max(0.0, (self.config["slashing_threshold"] - psi_score) / self.config["slashing_threshold"])
        return min(0.1, penalty_factor * 0.5)  # Max 10% slash
    
    def _calculate_boost_amount(self, psi_score: float) -> float:
        """Calculate boost amount based on Ψ score"""
        # Higher reward for higher Ψ scores
        reward_factor = max(0.0, (psi_score - self.config["boosting_threshold"]) / (1.0 - self.config["boosting_threshold"]))
        return min(0.05, reward_factor * 0.1)  # Max 5% boost
    
    def _execute_consensus_actions(self, actions: List[ConsensusAction]):
        """Execute consensus actions with detailed logging"""
        for action in actions:
            try:
                logger.info(f"Executing {action.action_type.value} on {action.target}: {action.value} ({action.reason}) [Weight: {action.weight:.2f}]")
                
                # Update metrics based on action type
                if action.action_type == ConsensusActionType.SLASHING:
                    self.metrics.slashing_events += 1
                elif action.action_type == ConsensusActionType.BOOSTING:
                    self.metrics.boosting_events += 1
                
                # In a real implementation, this would coordinate with the actual components
                # to execute slashing, boosting, parameter adjustments, etc.
                
            except Exception as e:
                logger.error(f"Error executing action {action.action_type.value} on {action.target}: {e}")
    
    def _generate_finality_proof(self, consensus_round: ConsensusRound) -> str:
        """Generate cryptographic proof of consensus finality"""
        # In a real implementation, this would use actual cryptographic signing
        # For now, we'll create a mock proof
        proof_data = f"{consensus_round.round_id}:{consensus_round.timestamp}:{len(consensus_round.votes)}"
        proof = hashlib.sha256(proof_data.encode()).hexdigest()
        return proof
    
    def add_validator(self, validator_id: str, psi_score: float, stake_amount: float, shard_id: Optional[str] = None):
        """Add a validator to the consensus set"""
        self.validators[validator_id] = ValidatorState(
            validator_id=validator_id,
            psi_score=psi_score,
            coherence_contribution=0.0,
            uptime=1.0,
            stake_amount=stake_amount,
            shard_id=shard_id
        )
        
        # Add to shard if specified
        if shard_id:
            if shard_id not in self.shards:
                self.shards[shard_id] = ShardState(
                    shard_id=shard_id,
                    validators=[],
                    coherence_score=0.8,
                    last_consensus_time=0.0,
                    pending_actions=[]
                )
            self.shards[shard_id].validators.append(validator_id)
        
        logger.info(f"Added validator {validator_id} to consensus set")
    
    def update_validator_state(self, validator_id: str, psi_score: Optional[float] = None, 
                              coherence_contribution: Optional[float] = None, uptime: Optional[float] = None):
        """Update validator state"""
        if validator_id in self.validators:
            validator = self.validators[validator_id]
            if psi_score is not None:
                validator.psi_score = psi_score
            if coherence_contribution is not None:
                validator.coherence_contribution = coherence_contribution
            if uptime is not None:
                validator.uptime = uptime
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        return {
            "current_mode": self.current_mode.name,
            "validators_count": len(self.validators),
            "consensus_history_count": len(self.consensus_history),
            "last_consensus_time": self.last_consensus_time,
            "emergency_threshold": self.emergency_threshold,
            "metrics": {
                "rounds_completed": self.metrics.rounds_completed,
                "emergency_rounds": self.metrics.emergency_rounds,
                "avg_round_duration": self.metrics.avg_round_duration,
                "validator_participation_rate": self.metrics.validator_participation_rate,
                "slashing_events": self.metrics.slashing_events,
                "boosting_events": self.metrics.boosting_events,
                "failed_rounds": self.metrics.failed_rounds,
                "rollback_events": self.metrics.rollback_events
            }
        }
    
    def _attempt_rollback(self, failed_round: ConsensusRound):
        """Attempt to rollback a failed consensus round"""
        try:
            # Record rollback information
            rollback_info = {
                "round_id": failed_round.round_id,
                "timestamp": time.time(),
                "actions": [action.__dict__ for action in failed_round.actions],
                "reason": "Consensus failure"
            }
            
            self.rollback_history.append(rollback_info)
            
            # Keep only recent rollback history
            if len(self.rollback_history) > self.config["rollback_depth"]:
                self.rollback_history = self.rollback_history[-self.config["rollback_depth"]:]
            
            self.metrics.rollback_events += 1
            logger.info(f"Rollback attempted for failed consensus round {failed_round.round_id}")
            
        except Exception as e:
            logger.error(f"Error during rollback attempt: {e}")
    
    def start_parallel_consensus_round(self, shard_id: str, network_state: Dict[str, Any]) -> Optional[str]:
        """Start a parallel consensus round for a specific shard"""
        with self.parallel_consensus_lock:
            # Check if we can start another parallel round
            if len(self.active_consensus_rounds) >= self.config["max_parallel_rounds"]:
                logger.warning("Maximum parallel consensus rounds reached")
                return None
            
            # Create a consensus round for this shard
            start_time = time.time()
            round_id = f"consensus-{shard_id}-{int(start_time)}"
            
            # Get shard validators
            shard = self.shards.get(shard_id)
            if not shard:
                logger.error(f"Shard {shard_id} not found")
                return None
            
            participants = shard.validators
            
            # Create consensus round
            consensus_round = ConsensusRound(
                round_id=round_id,
                mode=ConsensusMode.REGULAR,
                actions=shard.pending_actions,
                participants=participants,
                timestamp=start_time,
                shard_id=shard_id
            )
            
            # Add to active rounds
            self.active_consensus_rounds[round_id] = consensus_round
            
            logger.info(f"Started parallel consensus round {round_id} for shard {shard_id}")
            return round_id
    
    def complete_parallel_consensus_round(self, round_id: str, success: bool):
        """Complete a parallel consensus round"""
        with self.parallel_consensus_lock:
            if round_id in self.active_consensus_rounds:
                consensus_round = self.active_consensus_rounds[round_id]
                
                # Record completion
                end_time = time.time()
                consensus_round.duration = end_time - consensus_round.timestamp
                
                if success:
                    # Add to history
                    self.consensus_history.append(consensus_round)
                    self.metrics.rounds_completed += 1
                    
                    # Clear pending actions for this shard
                    if consensus_round.shard_id:
                        shard = self.shards.get(consensus_round.shard_id)
                        if shard:
                            shard.pending_actions = []
                            shard.last_consensus_time = end_time
                
                # Remove from active rounds
                del self.active_consensus_rounds[round_id]
                
                logger.info(f"Completed parallel consensus round {round_id} ({'success' if success else 'failed'})")

# Example usage and testing
async def demo_attuned_consensus():
    """Demonstrate the enhanced λ(t)-Attuned Consensus"""
    print("⚖️ Enhanced λ(t)-Attuned Consensus Demo")
    print("=" * 45)
    
    # Create consensus instance
    network_config = {
        "validator_count": 5,
        "consensus_threshold": 0.67
    }
    
    consensus = AttunedConsensus("node-001", network_config)
    
    # Add sample validators with shards
    validators_data = [
        ("validator-1", 0.95, 10000.0, "shard-1"),
        ("validator-2", 0.87, 8000.0, "shard-1"),
        ("validator-3", 0.65, 12000.0, "shard-2"),  # Low Ψ score
        ("validator-4", 0.92, 9000.0, "shard-2"),
        ("validator-5", 0.55, 7000.0, "shard-3"),   # Very low Ψ score
    ]
    
    for validator_id, psi_score, stake, shard_id in validators_data:
        consensus.add_validator(validator_id, psi_score, stake, shard_id)
    
    print(f"✅ Added {len(validators_data)} validators across shards")
    
    # Show consensus stats
    stats = consensus.get_consensus_stats()
    print(f"📊 Consensus Stats: {stats['validators_count']} validators")
    
    # Test emergency consensus trigger
    print("\n🚨 Testing Emergency Consensus (Ĉ(t) < 0.7)")
    emergency_state = {
        "coherence_density": 0.65,  # Below emergency threshold
        "lambda_t": 0.4
    }
    
    consensus_round = consensus.execute_consensus_round(emergency_state)
    if consensus_round:
        print(f"🔄 Executed {consensus_round.mode.name} consensus round")
        print(f"⚡ Actions executed: {len(consensus_round.actions)}")
        for action in consensus_round.actions[:3]:  # Show first 3 actions
            print(f"   • {action.action_type.value} on {action.target}: {action.value} (Weight: {action.weight:.2f})")
    
    # Test regular consensus
    print("\n🔄 Testing Regular Consensus")
    regular_state = {
        "coherence_density": 0.85,
        "lambda_t": 0.7
    }
    
    consensus_round = consensus.execute_consensus_round(regular_state)
    if consensus_round:
        print(f"🔄 Executed {consensus_round.mode.name} consensus round")
        print(f"⚡ Actions executed: {len(consensus_round.actions)}")
        for action in consensus_round.actions[:3]:  # Show first 3 actions
            print(f"   • {action.action_type.value} on {action.target}: {action.value} (Weight: {action.weight:.2f})")
    
    # Test parallel consensus
    print("\nParallelGroup Consensus Testing")
    shard_round_id = consensus.start_parallel_consensus_round("shard-1", regular_state)
    if shard_round_id:
        print(f"ParallelGroup consensus started for shard-1: {shard_round_id}")
        consensus.complete_parallel_consensus_round(shard_round_id, True)
        print("ParallelGroup consensus completed")
    
    # Show final stats
    final_stats = consensus.get_consensus_stats()
    print(f"\n📊 Final Consensus Stats: {final_stats['validators_count']} validators")
    print(f"📈 Metrics: {final_stats['metrics']}")
    
    print("\n✅ Enhanced λ(t)-Attuned Consensus demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_attuned_consensus())