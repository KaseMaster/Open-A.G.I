#!/usr/bin/env python3
"""
Unit tests for ledger API endpoints
"""

import sys
import os
import unittest
import json
from unittest.mock import patch, Mock

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
from main import app


class TestLedgerAPI(unittest.TestCase):
    """Test suite for ledger API endpoints"""

    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()

    def test_get_ledger(self):
        """Test GET /ledger endpoint"""
        response = self.app.get('/ledger')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('balances', data)
        self.assertIn('chr', data)

    def test_mint_valid_transaction(self):
        """Test POST /mint with valid transaction"""
        tx_data = {
            "id": "tx001",
            "type": "harmonic",
            "action": "mint",
            "token": "FLX",
            "sender": "node-A",
            "receiver": "node-A",
            "amount": 100,
            "aggregated_cs": 0.85,
            "sender_chr": 0.8
        }
        
        response = self.app.post('/mint', 
                                data=json.dumps(tx_data),
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'accepted')

    def test_mint_invalid_transaction(self):
        """Test POST /mint with invalid transaction"""
        tx_data = {
            "id": "tx002",
            "type": "harmonic",
            "action": "mint",
            "token": "FLX",
            "sender": "node-B",
            "receiver": "node-B",
            "amount": 100,
            "aggregated_cs": 0.30,  # Low coherence
            "sender_chr": 0.4       # Low CHR
        }
        
        response = self.app.post('/mint',
                                data=json.dumps(tx_data),
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'rejected')

    @patch('main.make_snapshot')
    def test_snapshot_endpoint(self, mock_make_snapshot):
        """Test POST /snapshot endpoint"""
        # Mock the snapshot creation
        from openagi.harmonic_validation import HarmonicSnapshot
        mock_snapshot = HarmonicSnapshot(
            node_id="test-node",
            timestamp=1234567890,
            times=[0.0, 0.1, 0.2],
            values=[1.0, 1.1, 1.2],
            spectrum=[(1.0, 0.5), (2.0, 0.3)],
            spectrum_hash="test_hash",
            CS=0.85,
            phi_params={"phi": 1.618, "lambda": 0.618},
            signature="test_signature"
        )
        mock_make_snapshot.return_value = mock_snapshot
        
        snapshot_data = {
            "node_id": "test-node",
            "times": [0.0, 0.1, 0.2],
            "values": [1.0, 1.1, 1.2],
            "secret_key": "test_secret"
        }
        
        response = self.app.post('/snapshot',
                                data=json.dumps(snapshot_data),
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        mock_make_snapshot.assert_called_once()

    def test_coherence_endpoint(self):
        """Test POST /coherence endpoint"""
        coherence_data = {
            "local": {
                "node_id": "validator-1",
                "timestamp": 1234567890,
                "times": [0.0, 0.1, 0.2],
                "values": [1.0, 1.1, 1.2],
                "spectrum": [(1.0, 0.5), (2.0, 0.3)],
                "spectrum_hash": "test_hash_1",
                "CS": 0.85,
                "phi_params": {"phi": 1.618, "lambda": 0.618}
            },
            "remotes": [{
                "node_id": "validator-2",
                "timestamp": 1234567890,
                "times": [0.0, 0.1, 0.2],
                "values": [1.0, 1.1, 1.2],
                "spectrum": [(1.0, 0.5), (2.0, 0.3)],
                "spectrum_hash": "test_hash_2",
                "CS": 0.82,
                "phi_params": {"phi": 1.618, "lambda": 0.618}
            }]
        }
        
        response = self.app.post('/coherence',
                                data=json.dumps(coherence_data),
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('coherence_score', data)


if __name__ == "__main__":
    unittest.main()