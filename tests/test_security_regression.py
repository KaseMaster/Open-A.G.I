
import pytest
import asyncio
import json
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import MagicMock, AsyncMock, patch
from p2p_network import ConnectionManager, P2PNetworkManager, NodeType
from crypto_framework import CryptoEngine

class TestSecurityRegression:
    """Prevent security regressions from PR #12"""
    
    @pytest.mark.asyncio
    async def test_crypto_engine_mandatory(self):
        """Verify ConnectionManager requires crypto_engine"""
        with pytest.raises(ValueError, match="crypto_engine is mandatory"):
            ConnectionManager(node_id="test", port=8080, crypto_engine=None)
    
    @pytest.mark.asyncio
    async def test_rogue_peer_rejection_low_reputation(self):
        """Verify peers with low reputation are rejected in handshake"""
        # Mock CryptoEngine
        mock_crypto = MagicMock(spec=CryptoEngine)
        
        # Mock ReputationManager
        mock_reputation = MagicMock()
        mock_reputation.should_accept_connection.return_value = False # REJECT
        
        manager = ConnectionManager(node_id="test_node", port=8080, crypto_engine=mock_crypto, reputation_manager=mock_reputation)
        
        # Mock Writer
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.get_extra_info.return_value = ("127.0.0.1", 9999)
        mock_writer.wait_closed = AsyncMock()
        
        # Mock Reader
        mock_reader = AsyncMock()

        # Handshake message
        message = {"type": "handshake", "node_id": "rogue_peer"}
        
        await manager._handle_handshake_request(mock_reader, mock_writer, message)
        
        # Verify rejection
        args, _ = mock_writer.write.call_args
        response = json.loads(args[0].decode().strip())
        assert response["type"] == "handshake_error"
        assert response["error"] == "reputation_too_low"
        assert "rogue_peer" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_message_encryption_enforced_fail_closed(self):
        """Verify fallback to plaintext is disabled and raises error"""
        # Mock CryptoEngine
        mock_crypto = MagicMock(spec=CryptoEngine)
        # Mock ratchet state to simulate NO secure channel established
        mock_crypto.ratchet_states = {} 
        
        manager = ConnectionManager(node_id="test_node", port=8080, crypto_engine=mock_crypto)
        
        # Mock active connection
        mock_writer = MagicMock()
        manager.active_connections["peer_1"] = {"writer": mock_writer}
        
        # Attempt send message
        with pytest.raises(RuntimeError, match="Insecure channel rejected"):
             await manager.send_message("peer_1", {"data": "sensitive"})

        # Now simulate Established channel but encryption failure
        mock_crypto.ratchet_states = {"peer_1": "state"}
        mock_crypto.encrypt_message.return_value = None # Fail encryption
        
        with pytest.raises(RuntimeError, match="Encryption failed"):
             await manager.send_message("peer_1", {"data": "sensitive"})

