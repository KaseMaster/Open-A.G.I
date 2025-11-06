#!/usr/bin/env python3
"""
Unit tests for consensus protocol integration with harmonic validation
"""

import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from consensus_algorithm import ConsensusEngine
from openagi.harmonic_validation import make_snapshot, recursive_validate


class TestConsensusProtocol(unittest.TestCase):
    """Test suite for consensus protocol integration"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.consensus = ConsensusEngine("test-node", ["test-node", "node-1", "node-2"])
        
    def test_consensus_initialization(self):
        """Test that consensus protocol initializes correctly"""
        self.assertIsNotNone(self.consensus)
        self.assertEqual(self.consensus.node_id, "test-node")
        self.assertEqual(self.consensus.view_number, 0)
        
    def test_node_state_initialization(self):
        """Test that node states are initialized correctly"""
        self.assertIn("test-node", self.consensus.node_states)
        self.assertIn("node-1", self.consensus.node_states)
        self.assertIn("node-2", self.consensus.node_states)
        
    def test_harmonic_validation_integration(self):
        """Test integration with harmonic validation"""
        # This test would verify that the consensus engine works with harmonic validation
        # For now, we'll just verify the basic structure
        self.assertTrue(hasattr(self.consensus, "active_proposals"))
        self.assertTrue(hasattr(self.consensus, "committed_proposals"))


if __name__ == "__main__":
    unittest.main()