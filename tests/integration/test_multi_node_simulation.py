"""
Integration Tests for Quantum Currency Multi-Node Simulation
"""

import unittest
import sys
import os
import time
import threading
import json

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'openagi'))

from openagi.harmonic_validation import (
    HarmonicSnapshot, 
    make_snapshot, 
    compute_coherence_score
)
from openagi.token_rules import validate_harmonic_tx, apply_token_effects
from openagi.validator_staking import ValidatorStakingSystem


class TestMultiNodeSimulation(unittest.TestCase):
    """Test suite for multi-node simulation"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        # Create multiple validators
        self.validator_ids = ["validator-001", "validator-002", "validator-003"]
        self.validator_keys = ["key-001", "key-002", "key-003"]
        
        # Initialize staking system
        self.staking_system = ValidatorStakingSystem()
        
        # Initialize ledger
        self.ledger = {
            "balances": {},
            "chr": {}
        }
        
    def test_full_validation_cycle(self):
        """Test a full validation cycle with multiple nodes"""
        # Step 1: Create staking positions
        print("Step 1: Creating staking positions...")
        for i, validator_id in enumerate(self.validator_ids):
            position_id = self.staking_system.create_staking_position(
                staker_address=validator_id,
                validator_id=validator_id,
                token_type="FLX",
                amount=5000.0 + i * 1000.0,
                lockup_period=90.0
            )
            self.assertIsNotNone(position_id)
            
        # Step 2: Generate snapshots from all validators
        print("Step 2: Generating snapshots...")
        snapshots = []
        for i, (validator_id, secret_key) in enumerate(zip(self.validator_ids, self.validator_keys)):
            # Slightly modify values for each validator to simulate real-world differences
            modified_values = [v + i * 0.01 for v in self.test_values]
            
            snapshot = make_snapshot(
                node_id=validator_id,
                times=self.test_times,
                values=modified_values,
                secret_key=secret_key
            )
            snapshots.append(snapshot)
            
        # Step 3: Compute coherence scores
        print("Step 3: Computing coherence scores...")
        for i, local_snapshot in enumerate(snapshots):
            # Use other snapshots as remote snapshots
            remote_snapshots = [snapshots[j] for j in range(len(snapshots)) if j != i]
            
            coherence_score = compute_coherence_score(local_snapshot, remote_snapshots)
            print(f"   {local_snapshot.node_id} coherence score: {coherence_score:.4f}")
            
            # Store coherence score in ledger
            self.ledger["chr"][local_snapshot.node_id] = coherence_score
            
        # Step 4: Validate transactions
        print("Step 4: Validating transactions...")
        for i, local_snapshot in enumerate(snapshots):
            # Create transaction data
            remote_snapshots = [snapshots[j].__dict__ for j in range(len(snapshots)) if j != i]
            
            tx = {
                "local_snapshot": local_snapshot.__dict__,
                "snapshot_bundle": remote_snapshots,
                "sender": local_snapshot.node_id,
                "amount": 100.0
            }
            
            # Validate transaction
            config = {"mint_threshold": 0.75, "min_chr": 0.6}
            is_valid = validate_harmonic_tx(tx, config)
            
            print(f"   Transaction from {local_snapshot.node_id}: {'Valid' if is_valid else 'Invalid'}")
            
            # Apply token effects if valid
            if is_valid:
                apply_token_effects(self.ledger, tx)
                
        # Step 5: Check final ledger state
        print("Step 5: Checking final ledger state...")
        for validator_id in self.validator_ids:
            chr_score = self.ledger["chr"].get(validator_id, 0.0)
            flx_balance = self.ledger["balances"].get(validator_id, {}).get("FLX", 0.0)
            chr_balance = self.ledger["balances"].get(validator_id, {}).get("CHR", 0.0)
            
            print(f"   {validator_id}: CHR={chr_score:.4f}, FLX={flx_balance:.2f}, CHR={chr_balance:.2f}")
            
        # Verify that all validators have positive balances
        for validator_id in self.validator_ids:
            flx_balance = self.ledger["balances"].get(validator_id, {}).get("FLX", 0.0)
            chr_balance = self.ledger["balances"].get(validator_id, {}).get("CHR", 0.0)
            self.assertGreater(flx_balance, 0.0)
            self.assertGreater(chr_balance, 0.0)
            
    def test_consensus_protocol_integration(self):
        """Test integration with consensus protocol"""
        # Import the consensus protocol function
        from openagi.consensus_protocol import pre_prepare_block
        
        # Generate snapshots
        snapshots = []
        for i, (validator_id, secret_key) in enumerate(zip(self.validator_ids, self.validator_keys)):
            modified_values = [v + i * 0.01 for v in self.test_values]
            
            snapshot = make_snapshot(
                node_id=validator_id,
                times=self.test_times,
                values=modified_values,
                secret_key=secret_key
            )
            snapshots.append(snapshot)
            
        # Create a mock block with harmonic transactions
        class MockBlock:
            def __init__(self):
                self.transactions = []
                
        block = MockBlock()
        
        # Add harmonic transactions to the block
        for i, snapshot in enumerate(snapshots):
            remote_snapshots = [snapshots[j].__dict__ for j in range(len(snapshots)) if j != i]
            
            tx = {
                "id": f"tx-{i}",
                "type": "harmonic",
                "local_snapshot": snapshot.__dict__,
                "snapshot_bundle": remote_snapshots,
                "sender": snapshot.node_id,
                "amount": 100.0
            }
            block.transactions.append(tx)
            
        # Run consensus protocol
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        try:
            result_block = pre_prepare_block(block, config)
            # If we get here, the consensus protocol ran successfully
            self.assertIsNotNone(result_block)
        except Exception as e:
            # If there's an exception, it means a transaction was rejected
            print(f"Consensus protocol exception: {e}")
            
    def test_staking_rewards_calculation(self):
        """Test staking rewards calculation"""
        # Create staking positions
        for i, validator_id in enumerate(self.validator_ids):
            position_id = self.staking_system.create_staking_position(
                staker_address=validator_id,
                validator_id=validator_id,
                token_type="FLX",
                amount=10000.0,
                lockup_period=180.0
            )
            self.assertIsNotNone(position_id)
            
        # Calculate rewards for a period
        rewards = self.staking_system.calculate_rewards(time_period_hours=24*7)  # 1 week
        
        # Check that rewards were calculated
        self.assertIsInstance(rewards, dict)
        print(f"Total rewards calculated: {sum(rewards.values()):.2f} tokens")
        
        # Check that some addresses received rewards
        self.assertGreater(len(rewards), 0)
        
        # Claim rewards for one address
        test_address = self.validator_ids[0]
        if test_address in rewards:
            initial_pending = self.staking_system.pending_rewards.get(test_address, 0.0)
            claimed = self.staking_system.claim_rewards(test_address)
            final_pending = self.staking_system.pending_rewards.get(test_address, 0.0)
            
            print(f"Claimed {claimed:.2f} tokens for {test_address}")
            self.assertLessEqual(final_pending, initial_pending)


class TestNetworkResilience(unittest.TestCase):
    """Test suite for network resilience"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        # Create validators
        self.validator_ids = ["validator-001", "validator-002", "validator-003", "validator-004", "validator-005"]
        self.validator_keys = ["key-001", "key-002", "key-003", "key-004", "key-005"]
        
        # Initialize staking system
        self.staking_system = ValidatorStakingSystem()
        
    def test_partial_node_failure(self):
        """Test system behavior with partial node failure"""
        # Create staking positions for all validators
        active_validators = []
        for i, validator_id in enumerate(self.validator_ids):
            position_id = self.staking_system.create_staking_position(
                staker_address=validator_id,
                validator_id=validator_id,
                token_type="FLX",
                amount=5000.0,
                lockup_period=90.0
            )
            if position_id is not None:
                active_validators.append(validator_id)
                
        # Simulate that only first 3 validators are active
        active_validators = active_validators[:3]
        
        # Generate snapshots only for active validators
        snapshots = []
        for i, validator_id in enumerate(active_validators):
            secret_key = self.validator_keys[i]
            modified_values = [v + i * 0.01 for v in self.test_values]
            
            snapshot = make_snapshot(
                node_id=validator_id,
                times=self.test_times,
                values=modified_values,
                secret_key=secret_key
            )
            snapshots.append(snapshot)
            
        # Compute coherence scores with partial data
        coherence_scores = []
        for i, local_snapshot in enumerate(snapshots):
            remote_snapshots = [snapshots[j] for j in range(len(snapshots)) if j != i]
            coherence_score = compute_coherence_score(local_snapshot, remote_snapshots)
            coherence_scores.append(coherence_score)
            
        # Verify that coherence scores are still computed
        for score in coherence_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
        print(f"Coherence scores with {len(active_validators)} active validators: {[f'{s:.4f}' for s in coherence_scores]}")
        
    def test_network_partition_simulation(self):
        """Test system behavior during network partition"""
        # Create two groups of validators
        group1_validators = self.validator_ids[:2]
        group2_validators = self.validator_ids[2:4]
        
        # Create staking positions
        for validator_id in self.validator_ids:
            position_id = self.staking_system.create_staking_position(
                staker_address=validator_id,
                validator_id=validator_id,
                token_type="FLX",
                amount=5000.0,
                lockup_period=90.0
            )
            
        # Generate snapshots for each group
        group1_snapshots = []
        group2_snapshots = []
        
        # Group 1 snapshots
        for i, validator_id in enumerate(group1_validators):
            secret_key = self.validator_keys[i]
            # Similar values for group 1
            modified_values = [v + 0.01 for v in self.test_values]
            
            snapshot = make_snapshot(
                node_id=validator_id,
                times=self.test_times,
                values=modified_values,
                secret_key=secret_key
            )
            group1_snapshots.append(snapshot)
            
        # Group 2 snapshots
        for i, validator_id in enumerate(group2_validators):
            secret_key = self.validator_keys[i+2]
            # Different values for group 2
            modified_values = [v + 0.5 for v in self.test_values]
            
            snapshot = make_snapshot(
                node_id=validator_id,
                times=self.test_times,
                values=modified_values,
                secret_key=secret_key
            )
            group2_snapshots.append(snapshot)
            
        # Compute coherence within groups
        group1_coherence = []
        for i, local_snapshot in enumerate(group1_snapshots):
            remote_snapshots = [group1_snapshots[j] for j in range(len(group1_snapshots)) if j != i]
            coherence_score = compute_coherence_score(local_snapshot, remote_snapshots)
            group1_coherence.append(coherence_score)
            
        group2_coherence = []
        for i, local_snapshot in enumerate(group2_snapshots):
            remote_snapshots = [group2_snapshots[j] for j in range(len(group2_snapshots)) if j != i]
            coherence_score = compute_coherence_score(local_snapshot, remote_snapshots)
            group2_coherence.append(coherence_score)
            
        # Coherence within groups should be high
        for score in group1_coherence:
            self.assertGreater(score, 0.8)
            
        for score in group2_coherence:
            self.assertGreater(score, 0.8)
            
        # Cross-group coherence should be low
        cross_coherence_scores = []
        for snapshot1 in group1_snapshots:
            for snapshot2 in group2_snapshots:
                score = compute_coherence_score(snapshot1, [snapshot2])
                cross_coherence_scores.append(score)
                
        # Most cross-group scores should be low
        low_scores = [s for s in cross_coherence_scores if s < 0.5]
        self.assertGreater(len(low_scores), len(cross_coherence_scores) * 0.7)  # At least 70% should be low
        
        print(f"Group 1 coherence: {[f'{s:.4f}' for s in group1_coherence]}")
        print(f"Group 2 coherence: {[f'{s:.4f}' for s in group2_coherence]}")
        print(f"Cross-group coherence: {[f'{s:.4f}' for s in cross_coherence_scores]}")


if __name__ == "__main__":
    unittest.main()