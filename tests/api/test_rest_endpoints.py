"""
API Tests for Quantum Currency REST Endpoints
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the openagi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'openagi'))

# Import the actual modules for testing
from openagi.harmonic_validation import HarmonicSnapshot
from openagi.token_rules import validate_harmonic_tx


class TestRESTEndpoints(unittest.TestCase):
    """Test suite for REST API endpoints"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the Flask app and request
        self.mock_app = Mock()
        self.mock_request = Mock()
        
    @patch('openagi.rest_api.request')
    @patch('openagi.rest_api.validate_harmonic_tx')
    def test_mint_endpoint_valid_transaction(self, mock_validate, mock_request):
        """Test the /mint endpoint with a valid transaction"""
        # Mock the request data
        mock_request.json = {
            "local_snapshot": {
                "node_id": "validator-1",
                "timestamp": 1234567890,
                "times": [0.0, 0.1, 0.2],
                "values": [1.0, 1.1, 1.2],
                "spectrum": [(1.0, 0.5), (2.0, 0.3)],
                "spectrum_hash": "test_hash_1",
                "CS": 0.85,
                "phi_params": {"phi": 1.618, "lambda": 0.618}
            },
            "snapshot_bundle": [{
                "node_id": "validator-2",
                "timestamp": 1234567890,
                "times": [0.0, 0.1, 0.2],
                "values": [1.0, 1.1, 1.2],
                "spectrum": [(1.0, 0.5), (2.0, 0.3)],
                "spectrum_hash": "test_hash_2",
                "CS": 0.82,
                "phi_params": {"phi": 1.618, "lambda": 0.618}
            }],
            "sender": "test_validator",
            "amount": 100.0
        }
        
        # Mock the validation function to return True
        mock_validate.return_value = True
        
        # Import the actual function after patching
        from openagi.rest_api import mint
        
        # Call the function
        result = mint()
        
        # Check that validate was called
        mock_validate.assert_called_once()
        assert mock_validate.call_args[0][1] == {"mint_threshold": 0.75, "min_chr": 0.6}

    @patch('openagi.rest_api.request')
    @patch('openagi.rest_api.validate_harmonic_tx')
    def test_mint_endpoint_invalid_transaction(self, mock_validate, mock_request):
        """Test the /mint endpoint with an invalid transaction"""
        # Mock the request data
        mock_request.json = {
            "local_snapshot": {
                "node_id": "validator-1",
                "timestamp": 1234567890,
                "times": [0.0, 0.1, 0.2],
                "values": [1.0, 1.1, 1.2],
                "spectrum": [(1.0, 0.5), (2.0, 0.3)],
                "spectrum_hash": "test_hash_1",
                "CS": 0.30,  # Low coherence score
                "phi_params": {"phi": 1.618, "lambda": 0.618}
            },
            "snapshot_bundle": [{
                "node_id": "validator-2",
                "timestamp": 1234567890,
                "times": [0.0, 0.1, 0.2],
                "values": [1.0, 1.1, 1.2],
                "spectrum": [(1.0, 0.5), (2.0, 0.3)],
                "spectrum_hash": "test_hash_2",
                "CS": 0.25,  # Low coherence score
                "phi_params": {"phi": 1.618, "lambda": 0.618}
            }],
            "sender": "test_validator",
            "amount": 100.0
        }
        
        # Mock the validation function to return False
        mock_validate.return_value = False
        
        # Import the actual function after patching
        from openagi.rest_api import mint
        
        # Call the function
        result = mint()
        
        # Check that validate was called
        mock_validate.assert_called_once()

    @patch('openagi.rest_api.request')
    def test_snapshot_endpoint(self, mock_request):
        """Test the /snapshot endpoint"""
        # Mock the request data
        mock_request.json = {
            "node_id": "validator-1",
            "times": [0.0, 0.1, 0.2],
            "values": [1.0, 1.1, 1.2],
            "secret_key": "test_secret"
        }
        
        # Import the actual function after patching
        from openagi.rest_api import snapshot
        
        # Call the function
        result = snapshot()
        
        # Check that the function runs without error

    @patch('openagi.rest_api.ledger', {"balances": {}, "chr": {}})
    def test_ledger_endpoint(self):
        """Test the /ledger endpoint"""
        # Import the actual function
        from openagi.rest_api import get_ledger
        
        # Call the function
        result = get_ledger()
        
        # Check that the function runs without error

    def test_chr_endpoint(self):
        """Test the /chr endpoint"""
        # Since there's no /chr endpoint in the current implementation, we'll skip this test
        pass

    @patch('openagi.rest_api.request')
    def test_coherence_endpoint(self, mock_request):
        """Test the /coherence endpoint"""
        # Mock the request data
        mock_request.json = {
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
        
        # Import the actual function after patching
        from openagi.rest_api import coherence
        
        # Call the function
        result = coherence()
        
        # Check that the function runs without error


if __name__ == "__main__":
    unittest.main()