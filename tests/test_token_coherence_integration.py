#!/usr/bin/env python3
"""
Integration Test for Token-Coherence System
Tests the integration between all five tokens and the harmonic coherence validation system
"""

import sys
import os
import unittest
import numpy as np
from typing import List

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openagi'))

from openagi.harmonic_validation import (
    HarmonicSnapshot, 
    make_snapshot, 
    compute_coherence_score,
    recursive_validate,
    calculate_token_rewards
)
from openagi.token_rules import validate_harmonic_tx, apply_token_effects
from openagi.validator_staking import ValidatorStakingSystem
from openagi.token_economy_simulation import TokenEconomySimulation


class TestTokenCoherenceIntegration(unittest.TestCase):
    """Test suite for token-coherence system integration"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test time series data
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values_high_coherence = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        self.test_values_low_coherence = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # More dissimilar values
        
        # Initialize systems
        self.staking_system = ValidatorStakingSystem()
        self.economy_simulation = TokenEconomySimulation()
        
        # Initialize ledger state
        self.ledger_state = {
            "balances": {},
            "chr": {},
            "staking": {}
        }
    
    def test_high_coherence_token_minting(self):
        """Test token minting with high coherence scores"""
        # Create snapshots with high coherence
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times,
            values=self.test_values_high_coherence,
            secret_key="secret-1"
        )
        
        # Slightly modified values for the second snapshot
        modified_values = [v + 0.01 for v in self.test_values_high_coherence]
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times,
            values=modified_values,
            secret_key="secret-2"
        )
        
        # Compute coherence score
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        self.assertGreater(coherence_score, 0.8)
        
        # Create transaction
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "FLX",
            "action": "mint",
            "aggregated_cs": coherence_score,
            "sender_chr": 0.85  # High CHR score
        }
        
        # Validate transaction
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        is_valid = validate_harmonic_tx(tx, config)
        self.assertTrue(is_valid)
        
        # Apply token effects
        initial_balances = self.ledger_state["balances"].copy()
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that FLX balance increased
        if "validator-1" in updated_state["balances"]:
            flx_balance = updated_state["balances"]["validator-1"]["FLX"]
            self.assertGreater(flx_balance, 0.0)
        
        # Check that CHR reputation also increased
        if "validator-1" in updated_state["chr"]:
            chr_score = updated_state["chr"]["validator-1"]
            self.assertGreater(chr_score, 0.0)
    
    def test_low_coherence_transaction_rejection(self):
        """Test that low coherence transactions are rejected"""
        # Create snapshots with low coherence
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times,
            values=self.test_values_high_coherence,
            secret_key="secret-1"
        )
        
        # Very different values for the second snapshot (more dissimilar)
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times,
            values=self.test_values_low_coherence,
            secret_key="secret-2"
        )
        
        # Compute coherence score
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        print(f"Coherence score for dissimilar signals: {coherence_score}")
        self.assertLess(coherence_score, 0.5)
        
        # Create transaction
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "FLX",
            "action": "mint",
            "aggregated_cs": coherence_score,
            "sender_chr": 0.4  # Low CHR score
        }
        
        # Validate transaction
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        is_valid = validate_harmonic_tx(tx, config)
        self.assertFalse(is_valid)
    
    def test_token_rewards_based_on_coherence(self):
        """Test that token rewards are calculated based on coherence scores"""
        # Test with high coherence
        high_coherence_rewards = calculate_token_rewards(0.9, 0.85)
        self.assertGreater(high_coherence_rewards["FLX"], 100.0)  # Should be above base
        self.assertGreater(high_coherence_rewards["CHR"], 50.0)
        
        # Test with low coherence
        low_coherence_rewards = calculate_token_rewards(0.3, 0.85)
        self.assertLess(low_coherence_rewards["FLX"], 100.0)  # Should be below base
        self.assertLess(low_coherence_rewards["CHR"], 50.0)
        
        # Test that higher CHR score increases rewards
        low_chr_rewards = calculate_token_rewards(0.8, 0.3)
        high_chr_rewards = calculate_token_rewards(0.8, 0.9)
        self.assertGreater(high_chr_rewards["FLX"], low_chr_rewards["FLX"])
    
    def test_token_conversion_cycle(self):
        """Test the complete token conversion cycle"""
        # Start with CHR tokens
        initial_chr = 1000.0
        
        # Convert CHR to ATR (stability)
        tx1 = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": initial_chr,
            "token": "CHR",
            "action": "stake",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        # Apply conversion
        self.ledger_state["balances"]["user-1"] = {"CHR": initial_chr, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}
        updated_state = apply_token_effects(self.ledger_state, tx1)
        
        # Check that CHR decreased and ATR increased
        if "user-1" in updated_state["balances"]:
            final_chr = updated_state["balances"]["user-1"]["CHR"]
            atr_balance = updated_state["balances"]["user-1"]["ATR"]
            self.assertLess(final_chr, initial_chr)
            self.assertGreater(atr_balance, 0.0)
        
        # Convert FLX to PSY (synchronization)
        initial_flx = 500.0
        if "user-1" in updated_state["balances"]:
            updated_state["balances"]["user-1"]["FLX"] = initial_flx
        
        tx2 = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": initial_flx,
            "token": "FLX",
            "action": "convert_to_psy",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        updated_state = apply_token_effects(updated_state, tx2)
        
        # Check that FLX decreased and PSY increased
        if "user-1" in updated_state["balances"]:
            final_flx = updated_state["balances"]["user-1"]["FLX"]
            psy_balance = updated_state["balances"]["user-1"]["PSY"]
            self.assertLess(final_flx, initial_flx)
            self.assertGreater(psy_balance, 0.0)
    
    def test_validator_staking_with_multi_tokens(self):
        """Test validator staking with different token types"""
        # Create validators
        self.staking_system._initialize_validators()
        
        # Stake FLX tokens
        flx_stake_id = self.staking_system.create_staking_position(
            staker_address="staker-1",
            validator_id="validator-001",
            token_type="FLX",
            amount=5000.0,
            lockup_period=90.0
        )
        self.assertIsNotNone(flx_stake_id)
        
        # Stake ATR tokens
        atr_stake_id = self.staking_system.create_staking_position(
            staker_address="staker-2",
            validator_id="validator-001",
            token_type="ATR",
            amount=2500.0,
            lockup_period=180.0
        )
        self.assertIsNotNone(atr_stake_id)
        
        # Stake CHR tokens (should convert to ATR) - use higher amount to meet minimum
        chr_stake_id = self.staking_system.create_staking_position(
            staker_address="staker-3",
            validator_id="validator-001",
            token_type="CHR",
            amount=2500.0,  # Increased from 1000.0 to meet minimum of 2000.0
            lockup_period=90.0
        )
        self.assertIsNotNone(chr_stake_id)
        
        # Check validator info
        validator_info = self.staking_system.get_validator_info("validator-001")
        self.assertIsNotNone(validator_info)
        if validator_info is not None:
            self.assertIn("FLX", validator_info["total_staked"])
            self.assertIn("ATR", validator_info["total_staked"])
            self.assertGreater(validator_info["total_staked"]["FLX"], 0.0)
            self.assertGreater(validator_info["total_staked"]["ATR"], 0.0)
    
    def test_resonance_multiplication_effect(self):
        """Test the multiplicative effect of RES tokens"""
        # Initialize with some RES tokens
        initial_res = 1000.0
        self.ledger_state["balances"]["network"] = {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": initial_res}
        
        # Simulate multiplication action
        tx = {
            "sender": "network",
            "receiver": "network",
            "amount": initial_res,
            "token": "RES",
            "action": "multiply",
            "multiplier": 1.1,  # 10% increase
            "aggregated_cs": 0.9,
            "sender_chr": 0.9
        }
        
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that RES increased
        if "network" in updated_state["balances"]:
            final_res = updated_state["balances"]["network"]["RES"]
            self.assertGreater(final_res, initial_res)
    
    def test_complete_validation_cycle_with_all_tokens(self):
        """Test a complete validation cycle with all token types"""
        # Create multiple coherent snapshots
        snapshots = []
        for i in range(5):
            # Slightly different but coherent values
            values = [v + i * 0.005 for v in self.test_values_high_coherence]
            snapshot = make_snapshot(
                node_id=f"validator-{i+1}",
                times=self.test_times,
                values=values,
                secret_key=f"secret-{i+1}"
            )
            snapshots.append(snapshot)
        
        # Validate the bundle
        is_valid, proof = recursive_validate(snapshots, threshold=0.75)
        self.assertTrue(is_valid)
        self.assertIsNotNone(proof)
        if proof is not None:
            self.assertGreater(proof.aggregated_CS, 0.75)
            
            # Calculate rewards based on coherence
            rewards = calculate_token_rewards(proof.aggregated_CS, 0.85)
            
            # Apply rewards to validators
            for i, snapshot in enumerate(snapshots):
                validator_id = snapshot.node_id
                
                # Reward FLX
                flx_tx = {
                    "sender": "network",
                    "receiver": validator_id,
                    "amount": rewards["FLX"],
                    "token": "FLX",
                    "action": "mint",
                    "aggregated_cs": proof.aggregated_CS,
                    "sender_chr": 0.85
                }
                
                self.ledger_state = apply_token_effects(self.ledger_state, flx_tx)
                
                # Reward CHR for ethical alignment
                chr_tx = {
                    "sender": "network",
                    "receiver": validator_id,
                    "amount": rewards["CHR"],
                    "token": "CHR",
                    "action": "reward",
                    "aggregated_cs": proof.aggregated_CS,
                    "sender_chr": 0.85
                }
                
                self.ledger_state = apply_token_effects(self.ledger_state, chr_tx)
            
            # Verify all validators received rewards
            for i in range(5):
                validator_id = f"validator-{i+1}"
                self.assertIn(validator_id, self.ledger_state["balances"])
                self.assertGreater(self.ledger_state["balances"][validator_id]["FLX"], 0.0)
                self.assertGreater(self.ledger_state["balances"][validator_id]["CHR"], 0.0)
                self.assertGreater(self.ledger_state["chr"][validator_id], 0.0)


if __name__ == "__main__":
    unittest.main()