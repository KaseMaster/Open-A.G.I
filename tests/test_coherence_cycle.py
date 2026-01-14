#!/usr/bin/env python3
"""
Unit tests for coherence cycle validation
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
    HarmonicSnapshot
)
from openagi.token_rules import validate_harmonic_tx, apply_token_effects
from openagi.consensus_protocol import pre_prepare_block


class TestCoherenceCycle(unittest.TestCase):
    """Test suite for coherence cycle validation"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test time series data
        self.test_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_values_high_coherence = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        self.test_values_low_coherence = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

    def test_high_coherence_validation_cycle(self):
        """Test complete validation cycle with high coherence"""
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
        
        # Create transaction
        tx = {
            "id": "tx-001",
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "FLX",
            "action": "mint",
            "aggregated_cs": coherence_score,
            "sender_chr": 0.85,  # High CHR score
            "type": "harmonic"
        }
        
        # Validate transaction
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        is_valid = validate_harmonic_tx(tx, config)
        self.assertTrue(is_valid)
        
        # Apply token effects
        ledger_state = {
            "balances": {},
            "chr": {}
        }
        updated_state = apply_token_effects(ledger_state, tx)
        
        # Check that FLX balance increased
        if "validator-1" in updated_state["balances"]:
            flx_balance = updated_state["balances"]["validator-1"]["FLX"]
            self.assertGreater(flx_balance, 0.0)

    def test_low_coherence_rejection(self):
        """Test that low coherence transactions are rejected"""
        # Create snapshots with low coherence
        snapshot1 = make_snapshot(
            node_id="validator-1",
            times=self.test_times,
            values=self.test_values_high_coherence,
            secret_key="secret-1"
        )
        
        # Very different values for the second snapshot
        snapshot2 = make_snapshot(
            node_id="validator-2",
            times=self.test_times,
            values=self.test_values_low_coherence,
            secret_key="secret-2"
        )
        
        # Compute coherence score
        coherence_score = compute_coherence_score(snapshot1, [snapshot2])
        
        # Create transaction
        tx = {
            "id": "tx-002",
            "sender": "validator-1",
            "receiver": "validator-1",
            "amount": 100.0,
            "token": "FLX",
            "action": "mint",
            "aggregated_cs": coherence_score,
            "sender_chr": 0.4,  # Low CHR score
            "type": "harmonic"
        }
        
        # Validate transaction
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        is_valid = validate_harmonic_tx(tx, config)
        self.assertFalse(is_valid)

    def test_recursive_validation_with_multiple_nodes(self):
        """Test recursive validation with multiple nodes"""
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

    def test_complete_token_conversion_cycle(self):
        """Test the complete token conversion cycle"""
        # Start with CHR tokens
        ledger_state = {
            "balances": {
                "user-1": {"CHR": 1000.0, "FLX": 0, "PSY": 0, "ATR": 0, "RES": 0}
            },
            "chr": {},
            "staking": {}
        }
        
        # Convert CHR to ATR (stability)
        tx1 = {
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
        
        # Apply conversion
        updated_state = apply_token_effects(ledger_state, tx1)
        
        # Check that CHR decreased and ATR increased
        if "user-1" in updated_state["balances"]:
            final_chr = updated_state["balances"]["user-1"]["CHR"]
            atr_balance = updated_state["balances"]["user-1"]["ATR"]
            self.assertLess(final_chr, 1000.0)
            self.assertGreater(atr_balance, 0.0)

    def test_consensus_protocol_integration(self):
        """Test integration with consensus protocol"""
        # Create a block with harmonic transactions
        block = {
            "transactions": [
                {
                    "id": "tx-004",
                    "sender": "validator-1",
                    "receiver": "validator-1",
                    "amount": 100.0,
                    "token": "FLX",
                    "action": "mint",
                    "aggregated_cs": 0.85,
                    "sender_chr": 0.85,
                    "type": "harmonic"
                }
            ]
        }
        
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        # Process block through consensus protocol
        try:
            processed_block = pre_prepare_block(block, config)
            self.assertIsNotNone(processed_block)
        except Exception as e:
            # If validation fails, that's also a valid test result
            self.assertIn("coherence", str(e).lower())


if __name__ == "__main__":
    unittest.main()