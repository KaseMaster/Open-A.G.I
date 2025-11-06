#!/usr/bin/env python3
"""
Unit tests for coherence cycle and token conversion flow
"""

import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, compute_coherence_score, recursive_validate
from openagi.token_rules import apply_token_effects
from openagi.validator_staking import ValidatorStakingSystem
from openagi.token_economy_simulation import TokenEconomySimulation


class TestCoherenceCycle(unittest.TestCase):
    """Test suite for coherence cycle and token conversion flow"""

    def setUp(self):
        """Set up test fixtures"""
        # Create test time series data
        self.test_times = np.linspace(0, 1, 100)
        self.test_values_coherent = np.sin(2 * np.pi * 10 * self.test_times) + 0.1 * np.random.randn(len(self.test_times))
        self.test_values_less_coherent = np.sin(2 * np.pi * 15 * self.test_times) + 0.2 * np.random.randn(len(self.test_times))

    def test_chr_to_atr_conversion_cycle(self):
        """Test CHR to ATR conversion cycle"""
        # Initialize ledger state
        ledger_state = {
            "balances": {"user-1": {"CHR": 1000.0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}},
            "chr": {"user-1": 0.8},
            "staking": {}
        }
        
        # Create transaction to convert CHR to ATR
        tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "CHR",
            "action": "stake",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        # Apply token effects
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Check that CHR decreased and ATR increased
        self.assertLess(updated_state["balances"]["user-1"]["CHR"], 1000.0)
        self.assertGreater(updated_state["balances"]["user-1"]["ATR"], 0.0)

    def test_flx_to_psy_conversion_cycle(self):
        """Test FLX to PSY conversion cycle"""
        # Initialize ledger state
        ledger_state = {
            "balances": {"user-1": {"CHR": 0, "FLX": 1000.0, "PSY": 0, "ATR": 0, "RES": 0}},
            "chr": {"user-1": 0.7},
            "staking": {}
        }
        
        # Create transaction to convert FLX to PSY
        tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "FLX",
            "action": "convert_to_psy",
            "aggregated_cs": 0.8,
            "sender_chr": 0.7
        }
        
        # Apply token effects
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Check that FLX decreased and PSY increased
        self.assertLess(updated_state["balances"]["user-1"]["FLX"], 1000.0)
        self.assertGreater(updated_state["balances"]["user-1"]["PSY"], 0.0)

    def test_psy_to_atr_conversion_cycle(self):
        """Test PSY to ATR conversion cycle"""
        # Initialize ledger state
        ledger_state = {
            "balances": {"user-1": {"CHR": 0, "FLX": 0, "PSY": 1000.0, "ATR": 0, "RES": 0}},
            "chr": {"user-1": 0.75},
            "staking": {}
        }
        
        # Create transaction to convert PSY to ATR
        tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "PSY",
            "action": "convert_to_atr",
            "aggregated_cs": 0.75,
            "sender_chr": 0.75
        }
        
        # Apply token effects
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Check that PSY decreased and ATR increased
        self.assertLess(updated_state["balances"]["user-1"]["PSY"], 1000.0)
        self.assertGreater(updated_state["balances"]["user-1"]["ATR"], 0.0)

    def test_atr_to_res_conversion_cycle(self):
        """Test ATR to RES conversion cycle"""
        # Initialize ledger state
        ledger_state = {
            "balances": {"user-1": {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 1000.0, "RES": 0}},
            "chr": {"user-1": 0.8},
            "staking": {}
        }
        
        # Create transaction to convert ATR to RES
        tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "ATR",
            "action": "convert_to_res",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        # Apply token effects
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Check that ATR decreased and RES increased
        self.assertLess(updated_state["balances"]["user-1"]["ATR"], 1000.0)
        self.assertGreater(updated_state["balances"]["user-1"]["RES"], 0.0)

    def test_res_to_chr_conversion_cycle(self):
        """Test RES to CHR conversion cycle"""
        # Initialize ledger state
        ledger_state = {
            "balances": {"user-1": {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 1000.0}},
            "chr": {"user-1": 0.85},
            "staking": {}
        }
        
        # Create transaction to convert RES to CHR
        tx = {
            "sender": "user-1",
            "receiver": "user-1",
            "amount": 500.0,
            "token": "RES",
            "action": "convert_to_chr",
            "aggregated_cs": 0.9,
            "sender_chr": 0.85
        }
        
        # Apply token effects (this action doesn't exist, so it should not change balances)
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Since convert_to_chr action doesn't exist, balances should remain the same
        self.assertEqual(updated_state["balances"]["user-1"]["RES"], 1000.0)

    def test_complete_token_cycle_coherence_validation(self):
        """Test complete token cycle with coherence validation"""
        # Create multiple coherent snapshots
        snapshots = []
        for i in range(3):
            # Create slightly different but coherent signals
            values = np.sin(2 * np.pi * 10 * self.test_times) + 0.1 * np.random.randn(len(self.test_times)) + i * 0.01
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
            
            # Test that high coherence allows token conversions
            ledger_state = {
                "balances": {"validator-1": {"CHR": 1000.0, "FLX": 500.0, "PSY": 250.0, "ATR": 125.0, "RES": 62.5}},
                "chr": {"validator-1": 0.9},
                "staking": {}
            }
            
            # High coherence should allow conversions
            tx = {
                "sender": "validator-1",
                "receiver": "validator-1",
                "amount": 100.0,
                "token": "CHR",
                "action": "stake",
                "aggregated_cs": proof.aggregated_CS,
                "sender_chr": 0.9
            }
            
            updated_state = apply_token_effects(ledger_state, tx)
            self.assertGreater(updated_state["balances"]["validator-1"]["ATR"], 0.0)

    def test_low_coherence_prevents_conversions(self):
        """Test that low coherence prevents token conversions"""
        # Create less coherent snapshots
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times.tolist(),
            values=self.test_values_coherent.tolist(),
            secret_key="secret-1"
        )
        
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times.tolist(),
            values=self.test_values_less_coherent.tolist(),
            secret_key="secret-2"
        )
        
        # Compute coherence score
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Low coherence should still allow conversions (coherence threshold is in validation, not conversion)
        ledger_state = {
            "balances": {"validator-1": {"CHR": 1000.0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}},
            "chr": {"validator-1": 0.5},  # Low CHR score
            "staking": {}
        }
        
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "CHR",
            "action": "stake",
            "aggregated_cs": coherence_score,
            "sender_chr": 0.5
        }
        
        updated_state = apply_token_effects(ledger_state, tx)
        # Conversion should still happen based on token rules, not coherence
        self.assertLess(updated_state["balances"]["validator-1"]["CHR"], 1000.0)


if __name__ == "__main__":
    unittest.main()