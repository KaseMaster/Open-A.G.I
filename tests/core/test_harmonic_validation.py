"""
Core Tests for Quantum Currency Harmonic Validation and Token Logic
"""

import unittest
import sys
import os
import numpy as np

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'openagi'))

from openagi.harmonic_validation import (
    HarmonicSnapshot, 
    make_snapshot, 
    compute_coherence_score
)
from openagi.token_rules import validate_harmonic_tx, apply_token_effects
from openagi.validator_staking import ValidatorStakingSystem


class TestHarmonicValidation(unittest.TestCase):
    """Test suite for harmonic validation logic"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
    def test_make_snapshot(self):
        """Test snapshot creation"""
        snapshot = make_snapshot(
            node_id="test-validator",
            times=self.test_times,
            values=self.test_values,
            secret_key="test-secret"
        )
        
        self.assertIsInstance(snapshot, HarmonicSnapshot)
        self.assertEqual(snapshot.node_id, "test-validator")
        self.assertEqual(len(snapshot.times), len(self.test_times))
        self.assertEqual(len(snapshot.values), len(self.test_values))
        self.assertIsNotNone(snapshot.spectrum)
        self.assertIsNotNone(snapshot.spectrum_hash)
        self.assertIsNotNone(snapshot.CS)
        self.assertIsNotNone(snapshot.phi_params)
        self.assertIsNotNone(snapshot.signature)
        
    def test_compute_coherence_score_high_similarity(self):
        """Test coherence score computation with high similarity"""
        # Create two very similar snapshots
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times,
            values=self.test_values,
            secret_key="secret-1"
        )
        
        # Slightly modified values for the second snapshot
        modified_values = [v + 0.01 for v in self.test_values]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times,
            values=modified_values,
            secret_key="secret-2"
        )
        
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Should be high (close to 1.0) due to similarity
        self.assertGreater(coherence_score, 0.8)
        self.assertLessEqual(coherence_score, 1.0)
        
    def test_compute_coherence_score_low_similarity(self):
        """Test coherence score computation with low similarity"""
        # Create two very different snapshots
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times,
            values=self.test_values,
            secret_key="secret-1"
        )
        
        # Very different values for the second snapshot
        different_values = [10.0, 5.0, 2.0, 8.0, 3.0, 1.0]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times,
            values=different_values,
            secret_key="secret-2"
        )
        
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Should be low (close to 0.0) due to dissimilarity
        self.assertGreaterEqual(coherence_score, 0.0)
        self.assertLess(coherence_score, 0.5)


class TestTokenRules(unittest.TestCase):
    """Test suite for token rules and validation"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = {
            "mint_threshold": 0.75,
            "min_chr": 0.6
        }
        
        self.test_ledger = {
            "balances": {
                "validator-1": {"FLX": 1000.0, "CHR": 500.0},
                "validator-2": {"FLX": 1500.0, "CHR": 750.0}
            },
            "chr": {
                "validator-1": 0.85,
                "validator-2": 0.92
            }
        }
        
    def test_validate_harmonic_tx_valid(self):
        """Test validation of a valid harmonic transaction"""
        # Create test snapshots with high coherence
        test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=test_times,
            values=test_values,
            secret_key="secret-1"
        )
        
        # Slightly modified values for the second snapshot
        modified_values = [v + 0.01 for v in test_values]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=test_times,
            values=modified_values,
            secret_key="secret-2"
        )
        
        tx = {
            "local_snapshot": snapshot1.__dict__,
            "snapshot_bundle": [snapshot2.__dict__],
            "sender": "validator-1",
            "amount": 100.0
        }
        
        is_valid = validate_harmonic_tx(tx, self.test_config)
        self.assertTrue(is_valid)
        
    def test_validate_harmonic_tx_low_coherence(self):
        """Test validation of a transaction with low coherence"""
        # Create test snapshots with low coherence
        test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=test_times,
            values=test_values,
            secret_key="secret-1"
        )
        
        # Very different values for the second snapshot
        different_values = [10.0, 5.0, 2.0, 8.0, 3.0, 1.0]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=test_times,
            values=different_values,
            secret_key="secret-2"
        )
        
        tx = {
            "local_snapshot": snapshot1.__dict__,
            "snapshot_bundle": [snapshot2.__dict__],
            "sender": "validator-1",
            "amount": 100.0
        }
        
        is_valid = validate_harmonic_tx(tx, self.test_config)
        self.assertFalse(is_valid)
        
    def test_apply_token_effects(self):
        """Test applying token effects to the ledger"""
        # Create test snapshots with high coherence
        test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        test_values = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=test_times,
            values=test_values,
            secret_key="secret-1"
        )
        
        # Slightly modified values for the second snapshot
        modified_values = [v + 0.01 for v in test_values]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=test_times,
            values=modified_values,
            secret_key="secret-2"
        )
        
        tx = {
            "local_snapshot": snapshot1.__dict__,
            "snapshot_bundle": [snapshot2.__dict__],
            "sender": "validator-1",
            "amount": 100.0
        }
        
        # Store initial balances
        initial_flx = self.test_ledger["balances"]["validator-1"]["FLX"]
        initial_chr = self.test_ledger["balances"]["validator-1"]["CHR"]
        
        # Apply token effects
        apply_token_effects(self.test_ledger, tx)
        
        # Check that balances increased
        self.assertGreater(self.test_ledger["balances"]["validator-1"]["FLX"], initial_flx)
        self.assertGreater(self.test_ledger["balances"]["validator-1"]["CHR"], initial_chr)


class TestValidatorStaking(unittest.TestCase):
    """Test suite for validator staking system"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.staking_system = ValidatorStakingSystem()
        
    def test_create_staking_position(self):
        """Test creating a staking position"""
        validator_id = "validator-001"
        staker_address = "test-staker"
        amount = 5000.0
        lockup_period = 90.0
        
        # Create staking position
        position_id = self.staking_system.create_staking_position(
            staker_address=staker_address,
            validator_id=validator_id,
            token_type="FLX",
            amount=amount,
            lockup_period=lockup_period
        )
        
        # Check that the position was created
        self.assertIsNotNone(position_id)
        
        # Check that the validator's total staked increased
        validator_info = self.staking_system.get_validator_info(validator_id)
        if validator_info is not None:
            self.assertEqual(validator_info["total_staked"], amount)
        
    def test_unstake_tokens(self):
        """Test unstaking tokens"""
        validator_id = "validator-001"
        staker_address = "test-staker"
        amount = 5000.0
        lockup_period = 90.0
        
        # Create staking position
        position_id = self.staking_system.create_staking_position(
            staker_address=staker_address,
            validator_id=validator_id,
            token_type="FLX",
            amount=amount,
            lockup_period=lockup_period
        )
        
        # Check that position was created
        self.assertIsNotNone(position_id)
        
        # Try to unstake (should fail due to lockup period)
        if position_id is not None:
            success = self.staking_system.unstake_tokens(position_id)
            self.assertFalse(success)
        
    def test_get_staking_apr(self):
        """Test getting staking APR"""
        validator_id = "validator-001"
        
        # Get APR for validator
        apr = self.staking_system.get_staking_apr(validator_id)
        
        # Should be greater than 0
        if apr is not None:
            self.assertGreater(apr, 0.0)
        
    def test_get_validator_info(self):
        """Test getting validator information"""
        validator_id = "validator-001"
        
        # Get validator info
        info = self.staking_system.get_validator_info(validator_id)
        
        # Check that we got valid info
        self.assertIsNotNone(info)
        if info is not None:
            self.assertEqual(info["validator_id"], validator_id)
            self.assertIn("total_staked", info)
            self.assertIn("total_delegated", info)
            self.assertIn("chr_score", info)


if __name__ == "__main__":
    unittest.main()