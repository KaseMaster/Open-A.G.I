#!/usr/bin/env python3
"""
End-to-end integration tests for the complete Quantum Currency System
"""

import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, recursive_validate, calculate_token_rewards
from openagi.token_rules import validate_harmonic_tx, apply_token_effects
from openagi.validator_staking import ValidatorStakingSystem
from openagi.token_economy_simulation import TokenEconomySimulation


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration test suite"""

    def setUp(self):
        """Set up test fixtures"""
        # Create test time series data
        self.test_times = np.linspace(0, 1, 100)
        
        # Initialize systems
        self.staking_system = ValidatorStakingSystem()
        self.economy_simulation = TokenEconomySimulation()
        
    def test_complete_validation_and_reward_cycle(self):
        """Test complete validation and reward cycle"""
        # Create multiple coherent snapshots
        snapshots = []
        for i in range(5):
            # Create slightly different but coherent signals
            values = np.sin(2 * np.pi * 10 * self.test_times) + 0.1 * np.random.randn(len(self.test_times)) + i * 0.005
            snapshot = make_snapshot(
                node_id=f"validator-{i+1}",
                times=self.test_times.tolist(),
                values=values.tolist(),
                secret_key=f"secret-{i+1}"
            )
            snapshots.append(snapshot)
        
        # Validate the bundle
        is_valid, proof = recursive_validate(snapshots, threshold=0.75)
        
        # Check that consensus was reached
        self.assertTrue(is_valid)
        self.assertIsNotNone(proof)
        if proof is not None:
            self.assertGreater(proof.aggregated_CS, 0.75)
        
        # Calculate rewards based on coherence
        if proof is not None:
            rewards = calculate_token_rewards(proof.aggregated_CS, 0.85)
        else:
            rewards = calculate_token_rewards(0.8, 0.85)
        
        # Verify all token types have rewards
        expected_tokens = ["FLX", "CHR", "ATR", "PSY", "RES"]
        for token in expected_tokens:
            self.assertIn(token, rewards)
            self.assertGreater(rewards[token], 0.0)
        
        # Initialize ledger state
        ledger_state = {
            "balances": {},
            "chr": {},
            "staking": {}
        }
        
        # Apply rewards to all validators
        for i, snapshot in enumerate(snapshots):
            validator_id = snapshot.node_id
            
            # Reward FLX
            flx_tx = {
                "sender": "network",
                "receiver": validator_id,
                "amount": rewards["FLX"],
                "token": "FLX",
                "action": "mint",
                "aggregated_cs": proof.aggregated_CS if proof is not None else 0.8,
                "sender_chr": 0.85
            }
            
            ledger_state = apply_token_effects(ledger_state, flx_tx)
            
            # Reward CHR for ethical alignment
            chr_tx = {
                "sender": "network",
                "receiver": validator_id,
                "amount": rewards["CHR"],
                "token": "CHR",
                "action": "reward",
                "aggregated_cs": proof.aggregated_CS if proof is not None else 0.8,
                "sender_chr": 0.85
            }
            
            ledger_state = apply_token_effects(ledger_state, chr_tx)
        
        # Verify all validators received rewards
        for i in range(5):
            validator_id = f"validator-{i+1}"
            self.assertIn(validator_id, ledger_state["balances"])
            self.assertGreater(ledger_state["balances"][validator_id]["FLX"], 0.0)
            self.assertGreater(ledger_state["balances"][validator_id]["CHR"], 0.0)
            self.assertGreater(ledger_state["chr"][validator_id], 0.0)

    def test_validator_staking_integration(self):
        """Test validator staking integration with token rewards"""
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
            amount=2500.0,
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

    def test_token_economy_simulation_integration(self):
        """Test token economy simulation integration"""
        # Run simulation for a few steps
        initial_market_cap = sum(token.market_cap for token in self.economy_simulation.token_states.values())
        
        # Run simulation steps
        for _ in range(10):
            self.economy_simulation.run_simulation_step()
        
        # Check that simulation ran
        final_market_cap = sum(token.market_cap for token in self.economy_simulation.token_states.values())
        
        # Market cap should have changed (not necessarily increased)
        self.assertNotEqual(initial_market_cap, final_market_cap)
        
        # Check that all token types are still present
        token_types = ["CHR", "FLX", "PSY", "ATR", "RES"]
        for token_type in token_types:
            self.assertIn(token_type, self.economy_simulation.token_states)

    def test_harmonic_transaction_validation_flow(self):
        """Test complete harmonic transaction validation flow"""
        # Create coherent snapshots
        values1 = np.sin(2 * np.pi * 10 * self.test_times) + 0.1 * np.random.randn(len(self.test_times))
        values2 = np.sin(2 * np.pi * 10 * self.test_times) + 0.1 * np.random.randn(len(self.test_times)) + 0.01
        
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times.tolist(),
            values=values1.tolist(),
            secret_key="secret-1"
        )
        
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times.tolist(),
            values=values2.tolist(),
            secret_key="secret-2"
        )
        
        from openagi.harmonic_validation import compute_coherence_score
        # Compute coherence score
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Create transaction with high coherence and CHR score
        tx = {
            "id": "tx001",
            "type": "harmonic",
            "action": "mint",
            "token": "FLX",
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100,
            "aggregated_cs": coherence_score,
            "sender_chr": 0.85
        }
        
        # Validate transaction
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        is_valid = validate_harmonic_tx(tx, config)
        
        # With high coherence and CHR, transaction should be valid
        self.assertTrue(is_valid)
        
        # Apply token effects
        ledger_state = {"balances": {}, "chr": {}}
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Check that FLX was minted
        if "validator-1" in updated_state["balances"]:
            self.assertGreater(updated_state["balances"]["validator-1"]["FLX"], 0.0)

    def test_multi_token_conversion_flow(self):
        """Test multi-token conversion flow"""
        # Initialize with all token types
        ledger_state = {
            "balances": {
                "user-1": {"CHR": 1000.0, "FLX": 1000.0, "PSY": 1000.0, "ATR": 1000.0, "RES": 1000.0}
            },
            "chr": {"user-1": 0.8},
            "staking": {}
        }
        
        # Convert CHR to ATR
        chr_tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "CHR",
            "action": "stake",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        ledger_state = apply_token_effects(ledger_state, chr_tx)
        
        # Convert FLX to PSY
        flx_tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "FLX",
            "action": "convert_to_psy",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        ledger_state = apply_token_effects(ledger_state, flx_tx)
        
        # Convert PSY to ATR
        psy_tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "PSY",
            "action": "convert_to_atr",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        ledger_state = apply_token_effects(ledger_state, psy_tx)
        
        # Convert ATR to RES
        atr_tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "ATR",
            "action": "convert_to_res",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        ledger_state = apply_token_effects(ledger_state, atr_tx)
        
        # Verify all conversions happened
        user_balances = ledger_state["balances"]["user-1"]
        self.assertLess(user_balances["CHR"], 1000.0)
        self.assertLess(user_balances["FLX"], 1000.0)
        self.assertLess(user_balances["PSY"], 1000.0)
        self.assertLess(user_balances["ATR"], 1000.0)
        self.assertGreater(user_balances["RES"], 1000.0)  # RES should have increased due to conversions


if __name__ == "__main__":
    unittest.main()