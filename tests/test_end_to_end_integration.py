#!/usr/bin/env python3
"""
End-to-end integration tests for the complete quantum currency system
"""

import sys
import os
import unittest
import numpy as np

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openagi'))

from openagi.harmonic_validation import (
    make_snapshot, 
    compute_coherence_score, 
    recursive_validate,
    calculate_token_rewards
)
from openagi.token_rules import validate_harmonic_tx, apply_token_effects
from openagi.validator_staking import ValidatorStakingSystem
from openagi.consensus_protocol import pre_prepare_block


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration test suite"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test time series data
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values_high_coherence = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        
        # Initialize systems
        self.staking_system = ValidatorStakingSystem()
        self.ledger_state = {
            "balances": {},
            "chr": {},
            "staking": {}
        }

    def test_complete_validation_and_reward_cycle(self):
        """Test complete validation and reward cycle"""
        # Create multiple coherent snapshots
        snapshots = []
        validators = []
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
            validators.append(f"validator-{i+1}")
        
        # Validate the bundle
        is_valid, proof = recursive_validate(snapshots, threshold=0.75)
        self.assertTrue(is_valid)
        self.assertIsNotNone(proof)
        
        if proof is not None:
            # Calculate rewards based on coherence
            rewards = calculate_token_rewards(proof.aggregated_CS, 0.85)
            
            # Apply rewards to validators
            for validator_id in validators:
                # Reward FLX
                flx_tx = {
                    "id": f"flx-reward-{validator_id}",
                    "sender": "network",
                    "receiver": validator_id,
                    "amount": rewards["FLX"],
                    "token": "FLX",
                    "action": "mint",
                    "aggregated_cs": proof.aggregated_CS,
                    "sender_chr": 0.85,
                    "type": "harmonic"
                }
                
                self.ledger_state = apply_token_effects(self.ledger_state, flx_tx)
                
                # Reward CHR for ethical alignment
                chr_tx = {
                    "id": f"chr-reward-{validator_id}",
                    "sender": "network",
                    "receiver": validator_id,
                    "amount": rewards["CHR"],
                    "token": "CHR",
                    "action": "reward",
                    "aggregated_cs": proof.aggregated_CS,
                    "sender_chr": 0.85,
                    "type": "harmonic"
                }
                
                self.ledger_state = apply_token_effects(self.ledger_state, chr_tx)
            
            # Verify all validators received rewards
            for validator_id in validators:
                self.assertIn(validator_id, self.ledger_state["balances"])
                self.assertGreater(self.ledger_state["balances"][validator_id]["FLX"], 0.0)
                self.assertGreater(self.ledger_state["balances"][validator_id]["CHR"], 0.0)
                self.assertGreater(self.ledger_state["chr"][validator_id], 0.0)

    def test_multi_token_staking_system(self):
        """Test multi-token staking system"""
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
        
        # Stake CHR tokens (should convert to ATR)
        chr_stake_id = self.staking_system.create_staking_position(
            staker_address="staker-3",
            validator_id="validator-001",
            token_type="CHR",
            amount=2500.0,  # Increased to meet minimum
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

    def test_consensus_protocol_with_harmonic_transactions(self):
        """Test consensus protocol with harmonic transactions"""
        # Create a block with multiple harmonic transactions
        block = {
            "transactions": [
                {
                    "id": "tx-001",
                    "sender": "validator-1",
                    "receiver": "validator-1",
                    "amount": 100.0,
                    "token": "FLX",
                    "action": "mint",
                    "aggregated_cs": 0.85,
                    "sender_chr": 0.85,
                    "type": "harmonic"
                },
                {
                    "id": "tx-002",
                    "sender": "validator-2",
                    "receiver": "validator-2",
                    "amount": 50.0,
                    "token": "CHR",
                    "action": "reward",
                    "aggregated_cs": 0.85,
                    "sender_chr": 0.85,
                    "type": "harmonic"
                }
            ]
        }
        
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        # Process block through consensus protocol
        processed_block = pre_prepare_block(block, config)
        self.assertIsNotNone(processed_block)
        self.assertEqual(len(processed_block["transactions"]), 2)

    def test_token_conversion_chain(self):
        """Test the complete token conversion chain"""
        # Initialize ledger with tokens and network account
        self.ledger_state["balances"] = {
            "user-1": {"CHR": 1000.0, "FLX": 500.0, "PSY": 200.0, "ATR": 300.0, "RES": 50.0},
            "network": {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}
        }
        
        # 1. Convert CHR to ATR (stake)
        chr_to_atr_tx = {
            "id": "tx-003",
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "CHR",
            "action": "stake",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8,
            "type": "harmonic"
        }
        
        initial_chr = self.ledger_state["balances"]["user-1"]["CHR"]
        initial_atr = self.ledger_state["balances"]["user-1"]["ATR"]
        
        self.ledger_state = apply_token_effects(self.ledger_state, chr_to_atr_tx)
        
        # Check conversion
        final_chr = self.ledger_state["balances"]["user-1"]["CHR"]
        final_atr = self.ledger_state["balances"]["user-1"]["ATR"]
        self.assertLess(final_chr, initial_chr)
        self.assertGreater(final_atr, initial_atr)
        
        # 2. Convert FLX to PSY
        flx_to_psy_tx = {
            "id": "tx-004",
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 200.0,
            "token": "FLX",
            "action": "convert_to_psy",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8,
            "type": "harmonic"
        }
        
        initial_flx = self.ledger_state["balances"]["user-1"]["FLX"]
        initial_psy = self.ledger_state["balances"]["user-1"]["PSY"]
        
        self.ledger_state = apply_token_effects(self.ledger_state, flx_to_psy_tx)
        
        # Check conversion
        final_flx = self.ledger_state["balances"]["user-1"]["FLX"]
        final_psy = self.ledger_state["balances"]["user-1"]["PSY"]
        self.assertLess(final_flx, initial_flx)
        self.assertGreater(final_psy, initial_psy)
        
        # 3. Convert PSY to ATR
        psy_to_atr_tx = {
            "id": "tx-005",
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 100.0,
            "token": "PSY",
            "action": "convert_to_atr",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8,
            "type": "harmonic"
        }
        
        initial_psy = self.ledger_state["balances"]["user-1"]["PSY"]
        initial_atr = self.ledger_state["balances"]["user-1"]["ATR"]
        
        self.ledger_state = apply_token_effects(self.ledger_state, psy_to_atr_tx)
        
        # Check conversion
        final_psy = self.ledger_state["balances"]["user-1"]["PSY"]
        final_atr = self.ledger_state["balances"]["user-1"]["ATR"]
        self.assertLess(final_psy, initial_psy)
        self.assertGreater(final_atr, initial_atr)
        
        # 4. Convert ATR to RES
        atr_to_res_tx = {
            "id": "tx-006",
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 150.0,
            "token": "ATR",
            "action": "convert_to_res",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8,
            "type": "harmonic"
        }
        
        initial_atr = self.ledger_state["balances"]["user-1"]["ATR"]
        initial_res = self.ledger_state["balances"]["user-1"]["RES"]
        
        self.ledger_state = apply_token_effects(self.ledger_state, atr_to_res_tx)
        
        # Check conversion
        final_atr = self.ledger_state["balances"]["user-1"]["ATR"]
        final_res = self.ledger_state["balances"]["user-1"]["RES"]
        self.assertLess(final_atr, initial_atr)
        self.assertGreater(final_res, initial_res)
        
        # 5. RES multiplication effect
        res_multiply_tx = {
            "id": "tx-007",
            "sender": "user-1",
            "receiver": "user-1",
            "amount": self.ledger_state["balances"]["user-1"]["RES"],
            "token": "RES",
            "action": "multiply",
            "multiplier": 1.1,  # 10% increase
            "aggregated_cs": 0.9,
            "sender_chr": 0.9,
            "type": "harmonic"
        }
        
        initial_res = self.ledger_state["balances"]["user-1"]["RES"]
        self.ledger_state = apply_token_effects(self.ledger_state, res_multiply_tx)
        final_res = self.ledger_state["balances"]["user-1"]["RES"]
        self.assertGreater(final_res, initial_res)


if __name__ == "__main__":
    unittest.main()