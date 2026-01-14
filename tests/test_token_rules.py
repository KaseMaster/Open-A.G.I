#!/usr/bin/env python3
"""
Unit tests for token rules engine
"""

import sys
import os
import unittest

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openagi'))

from openagi.token_rules import validate_harmonic_tx, apply_token_effects, get_token_properties


class TestTokenRules(unittest.TestCase):
    """Test suite for token rules engine"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.ledger_state = {
            "balances": {
                "validator-1": {"CHR": 1000.0, "FLX": 500.0, "PSY": 200.0, "ATR": 300.0, "RES": 50.0},
                "validator-2": {"CHR": 800.0, "FLX": 600.0, "PSY": 150.0, "ATR": 400.0, "RES": 75.0}
            },
            "chr": {
                "validator-1": 0.85,
                "validator-2": 0.75
            },
            "staking": {}
        }

    def test_validate_harmonic_tx_valid(self):
        """Test validation of valid harmonic transaction"""
        tx = {
            "aggregated_cs": 0.85,
            "sender_chr": 0.8,
            "type": "harmonic"
        }
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        is_valid = validate_harmonic_tx(tx, config)
        self.assertTrue(is_valid)

    def test_validate_harmonic_tx_low_coherence(self):
        """Test validation rejects transaction with low coherence"""
        tx = {
            "aggregated_cs": 0.30,  # Low coherence
            "sender_chr": 0.8,
            "type": "harmonic"
        }
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        is_valid = validate_harmonic_tx(tx, config)
        self.assertFalse(is_valid)

    def test_validate_harmonic_tx_low_chr(self):
        """Test validation rejects transaction with low CHR score"""
        tx = {
            "aggregated_cs": 0.85,
            "sender_chr": 0.4,  # Low CHR
            "type": "harmonic"
        }
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        is_valid = validate_harmonic_tx(tx, config)
        self.assertFalse(is_valid)

    def test_apply_token_effects_chr_reward(self):
        """Test applying CHR reward effects"""
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "CHR",
            "action": "reward",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        initial_chr_balance = self.ledger_state["balances"]["validator-1"]["CHR"]
        initial_chr_score = self.ledger_state["chr"]["validator-1"]
        
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that CHR balance increased
        self.assertGreater(updated_state["balances"]["validator-1"]["CHR"], initial_chr_balance)
        
        # Check that CHR reputation score increased
        self.assertGreater(updated_state["chr"]["validator-1"], initial_chr_score)

    def test_apply_token_effects_flx_mint(self):
        """Test applying FLX mint effects"""
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "FLX",
            "action": "mint",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        initial_flx_balance = self.ledger_state["balances"]["validator-1"]["FLX"]
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that FLX balance increased
        self.assertGreater(updated_state["balances"]["validator-1"]["FLX"], initial_flx_balance)

    def test_apply_token_effects_chr_stake(self):
        """Test applying CHR stake effects (converts to ATR)"""
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 500.0,
            "token": "CHR",
            "action": "stake",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        initial_chr_balance = self.ledger_state["balances"]["validator-1"]["CHR"]
        initial_atr_balance = self.ledger_state["balances"]["validator-1"]["ATR"]
        
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that CHR balance decreased
        self.assertLess(updated_state["balances"]["validator-1"]["CHR"], initial_chr_balance)
        
        # Check that ATR balance increased
        self.assertGreater(updated_state["balances"]["validator-1"]["ATR"], initial_atr_balance)

    def test_apply_token_effects_flx_convert_to_psy(self):
        """Test converting FLX to PSY"""
        tx = {
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 200.0,
            "token": "FLX",
            "action": "convert_to_psy",
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        initial_flx_balance = self.ledger_state["balances"]["validator-1"]["FLX"]
        initial_psy_balance = self.ledger_state["balances"]["validator-1"]["PSY"]
        
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that FLX balance decreased
        self.assertLess(updated_state["balances"]["validator-1"]["FLX"], initial_flx_balance)
        
        # Check that PSY balance increased
        self.assertGreater(updated_state["balances"]["validator-1"]["PSY"], initial_psy_balance)

    def test_get_token_properties(self):
        """Test getting token properties"""
        # Test CHR properties
        chr_props = get_token_properties("CHR")
        self.assertEqual(chr_props["name"], "Coheron")
        self.assertFalse(chr_props["transferable"])
        
        # Test FLX properties
        flx_props = get_token_properties("FLX")
        self.assertEqual(flx_props["name"], "Φlux")
        self.assertTrue(flx_props["transferable"])
        
        # Test PSY properties
        psy_props = get_token_properties("PSY")
        self.assertEqual(psy_props["name"], "ΨSync")
        self.assertEqual(psy_props["transferable"], "semi")
        
        # Test ATR properties
        atr_props = get_token_properties("ATR")
        self.assertEqual(atr_props["name"], "Attractor")
        self.assertTrue(atr_props["stakable"])
        
        # Test RES properties
        res_props = get_token_properties("RES")
        self.assertEqual(res_props["name"], "Resonance")
        self.assertTrue(res_props["multiplicative"])

    def test_apply_token_effects_res_multiply(self):
        """Test RES multiplication effect"""
        # Initialize network with RES tokens
        self.ledger_state["balances"]["network"] = {"CHR": 0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 1000.0}
        
        tx = {
            "sender": "network",
            "receiver": "network",
            "amount": 1000.0,
            "token": "RES",
            "action": "multiply",
            "multiplier": 1.1,  # 10% increase
            "aggregated_cs": 0.9,
            "sender_chr": 0.9
        }
        
        initial_res_balance = self.ledger_state["balances"]["network"]["RES"]
        updated_state = apply_token_effects(self.ledger_state, tx)
        
        # Check that RES balance increased by multiplier
        self.assertGreater(updated_state["balances"]["network"]["RES"], initial_res_balance)


if __name__ == "__main__":
    unittest.main()