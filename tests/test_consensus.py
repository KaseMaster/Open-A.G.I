#!/usr/bin/env python3
"""
Unit tests for consensus protocol integration
"""

import sys
import os
import unittest
import numpy as np

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openagi'))

from openagi.consensus_protocol import pre_prepare_block


class TestConsensusProtocol(unittest.TestCase):
    """Test suite for consensus protocol integration"""

    def test_pre_prepare_block_with_harmonic_transactions(self):
        """Test pre-prepare block with harmonic transactions"""
        # Create a block with harmonic transactions
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
                }
            ]
        }
        
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        # Process block through consensus protocol
        processed_block = pre_prepare_block(block, config)
        self.assertIsNotNone(processed_block)
        self.assertEqual(len(processed_block["transactions"]), 1)

    def test_pre_prepare_block_without_transactions(self):
        """Test pre-prepare block without transactions"""
        # Create a block without transactions
        block = {
            "data": "some other data"
        }
        
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        # Process block through consensus protocol
        processed_block = pre_prepare_block(block, config)
        self.assertIsNotNone(processed_block)
        self.assertEqual(processed_block, block)

    def test_pre_prepare_block_without_harmonic_transactions(self):
        """Test pre-prepare block without harmonic transactions"""
        # Create a block with regular transactions
        block = {
            "transactions": [
                {
                    "id": "tx-001",
                    "sender": "user-1",
                    "receiver": "user-2",
                    "amount": 100.0,
                    "type": "transfer"
                }
            ]
        }
        
        config = {"mint_threshold": 0.75, "min_chr": 0.6}
        
        # Process block through consensus protocol
        processed_block = pre_prepare_block(block, config)
        self.assertIsNotNone(processed_block)
        self.assertEqual(len(processed_block["transactions"]), 1)


if __name__ == "__main__":
    unittest.main()