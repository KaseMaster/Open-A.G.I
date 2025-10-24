"""
Integration tests for AEGIS Framework components
Tests interactions between optimizer, security, and coordinator modules
"""

import pytest
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.blockchain.optimizer import BlockchainOptimizer, BlockCache, TransactionPoolManager
from src.aegis.security.middleware import SecurityMiddleware, RateLimiter, InputValidator
from src.aegis.ml.distributed_coordinator import DistributedTrainingCoordinator, NodeRole, TrainingPhase


# Async fixture for coordinator
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestIntegration:
    
    @pytest.fixture
    def blockchain_optimizer(self):
        return BlockchainOptimizer(
            cache_size=100,
            cache_memory_mb=10,
            tx_pool_size=1000
        )
    
    @pytest.fixture
    def security_middleware(self):
        return SecurityMiddleware()
    
    @pytest.fixture
    async def mock_p2p_network(self):
        # Create a mock P2P network manager
        mock_network = Mock()
        mock_network.broadcast_message = Mock()
        mock_network.send_direct_message = Mock()
        return mock_network
    
    @pytest.fixture
    async def distributed_coordinator(self, mock_p2p_network, event_loop):
        coordinator = DistributedTrainingCoordinator(
            node_id="test_coordinator",
            p2p_network=mock_p2p_network,
            role=NodeRole.COORDINATOR
        )
        return coordinator
    
    def test_blockchain_optimizer_with_security(self, blockchain_optimizer, security_middleware):
        """Test integration between blockchain optimizer and security middleware"""
        # Add blocks to cache
        for i in range(50):
            block_data = {
                "block_id": f"block_{i}",
                "transactions": [f"tx_{j}" for j in range(10)],
                "timestamp": time.time()
            }
            blockchain_optimizer.block_cache.put(f"hash_{i}", block_data)
        
        # Simulate security checks on block requests
        client_id = "192.168.1.100"
        
        for i in range(100):
            # Security check
            allowed, message = security_middleware.check_request_security(
                client_id=client_id,
                endpoint="/blockchain/block",
                params={"block_hash": f"hash_{i % 50}"}
            )
            
            assert allowed, f"Request blocked: {message}"
            
            # Block retrieval
            if i % 2 == 0:
                block = blockchain_optimizer.block_cache.get(f"hash_{i % 50}")
                assert block is not None
            
            # Add transaction to pool
            tx_data = {
                "tx_id": f"tx_{i}",
                "sender": client_id,
                "data": f"transaction_data_{i}",
                "timestamp": time.time()
            }
            blockchain_optimizer.tx_pool.add_transaction(
                tx_hash=f"tx_{i}",
                tx_data=tx_data,
                priority="medium" if i % 3 != 0 else "high"
            )
        
        # Verify cache statistics
        cache_stats = blockchain_optimizer.block_cache.get_stats()
        assert cache_stats["size"] == 50  # All 50 blocks should be in cache
        
        # Verify transaction pool statistics
        pool_stats = blockchain_optimizer.tx_pool.get_stats()
        assert pool_stats["pending"] == 100
        assert pool_stats["priority_counts"]["high"] >= 33
        
        # Verify security statistics
        security_stats = security_middleware.get_security_stats()
        assert security_stats["rate_limiter_stats"]["total_requests"] == 100
    
    @pytest.mark.asyncio
    async def test_security_rate_limiting_with_coordinator(self, security_middleware):
        """Test rate limiting integration with distributed coordinator"""
        # Create mock network
        mock_network = Mock()
        mock_network.broadcast_message = Mock()
        mock_network.send_direct_message = Mock()
        
        # Create coordinator
        coordinator = DistributedTrainingCoordinator(
            node_id="test_coordinator",
            p2p_network=mock_network,
            role=NodeRole.COORDINATOR
        )
        
        client_id = "aggressive_client"
        endpoint = "/training/start"
        
        # Test rate limiting
        allowed_count = 0
        blocked_count = 0
        
        for i in range(150):  # Exceed default limit of 100
            allowed, message = security_middleware.check_request_security(
                client_id=client_id,
                endpoint=endpoint,
                params={"job_id": f"job_{i}", "model_name": f"model_{i}"}
            )
            
            if allowed:
                allowed_count += 1
                # Simulate coordinator handling request
                if i < 5:  # Only process first few requests
                    message = {
                        "type": "training_start",
                        "job_id": f"job_{i}",
                        "model_name": f"model_{i}",
                        "config": {},
                        "sender": client_id,
                        "timestamp": time.time()
                    }
                    await coordinator._handle_training_start(message)
            else:
                blocked_count += 1
        
        # Verify rate limiting worked
        assert allowed_count <= 120  # Allow some burst
        assert blocked_count > 0  # Some should be blocked
        
        # Verify jobs were created
        assert len(coordinator.jobs) > 0
    
    def test_input_validation_with_optimizer(self, security_middleware, blockchain_optimizer):
        """Test input validation integration with blockchain optimizer"""
        validator = security_middleware.input_validator
        
        # Test valid inputs
        valid_inputs = [
            ("user_123", "alphanumeric"),
            ("test@example.com", "email"),
            ("abc123def456", "hex"),
            ("Hello, World!", "safe_string")
        ]
        
        for value, field_type in valid_inputs:
            is_valid, error = validator.validate_string(value, field_type)
            assert is_valid, f"Valid input rejected: {value} ({field_type}): {error}"
            
            # Add to transaction pool
            tx_data = {
                "tx_id": f"tx_{hash(value)}",
                "sender": "validated_user",
                "data": value,
                "timestamp": time.time()
            }
            success = blockchain_optimizer.tx_pool.add_transaction(
                tx_hash=f"tx_{hash(value)}",
                tx_data=tx_data
            )
            assert success
    
    def test_coordinator_security_integration(self, distributed_coordinator, security_middleware):
        """Test distributed coordinator with security middleware"""
        # Test malicious client detection
        malicious_client = "192.168.1.200"
        
        # Simulate suspicious activity
        for i in range(15):  # Exceed threshold of 10
            is_suspicious = security_middleware.detect_suspicious_activity(
                client_id=malicious_client,
                threshold=10,
                window_seconds=60
            )
            
            if is_suspicious:
                # Block the client
                security_middleware.block_client(malicious_client)
                break
        
        # Verify client is blocked
        allowed, message = security_middleware.check_request_security(
            client_id=malicious_client,
            endpoint="/training/join",
            params={"job_id": "test_job"}
        )
        
        assert not allowed
        assert "blocked" in message.lower()
        
        # Test with coordinator
        with patch.object(distributed_coordinator.p2p_network, 'send_direct_message') as mock_send:
            success = asyncio.run(distributed_coordinator.join_training_job("test_job"))
            # Should not attempt to send message to blocked client
            mock_send.assert_not_called()
    
    def test_cache_performance_with_security_load(self, blockchain_optimizer, security_middleware):
        """Test cache performance under security load"""
        # Pre-populate cache
        for i in range(100):
            block_data = {
                "id": f"block_{i}",
                "data": "x" * 1000,  # 1KB blocks
                "timestamp": time.time()
            }
            blockchain_optimizer.block_cache.put(f"hash_{i}", block_data)
        
        start_time = time.time()
        cache_hits = 0
        security_checks = 0
        
        # Simulate high load scenario
        for i in range(1000):
            # Security check
            client_id = f"client_{i % 100}"  # 100 different clients
            allowed, message = security_middleware.check_request_security(
                client_id=client_id,
                endpoint="/blockchain/query",
                params={"block_hash": f"hash_{i % 100}"}
            )
            
            if allowed:
                security_checks += 1
                # Cache access
                block = blockchain_optimizer.block_cache.get(f"hash_{i % 100}")
                if block:
                    cache_hits += 1
        
        end_time = time.time()
        
        # Verify performance
        total_time = end_time - start_time
        requests_per_second = 1000 / total_time
        
        assert requests_per_second > 100  # Should handle >100 req/sec
        assert cache_hits >= 900  # Cache hit rate should be high (90%+)
        assert security_checks == 1000  # All security checks should pass
    
    def test_transaction_pool_priority_with_security(self, blockchain_optimizer, security_middleware):
        """Test transaction pool priority with security integration"""
        # Add transactions with different priorities and security checks
        priorities = ["high", "medium", "low"]
        
        for i in range(300):
            priority = priorities[i % 3]
            client_id = f"client_{i % 50}"
            
            # Security check
            allowed, message = security_middleware.check_request_security(
                client_id=client_id,
                endpoint="/blockchain/transaction",
                params={"tx_data": f"data_{i}"}
            )
            
            if allowed:
                tx_data = {
                    "tx_id": f"tx_{i}",
                    "sender": client_id,
                    "data": f"transaction_data_{i}",
                    "timestamp": time.time(),
                    "priority": priority
                }
                
                success = blockchain_optimizer.tx_pool.add_transaction(
                    tx_hash=f"tx_{i}",
                    tx_data=tx_data,
                    priority=priority
                )
                assert success
        
        # Verify priority distribution
        stats = blockchain_optimizer.tx_pool.get_stats()
        priority_counts = stats["priority_counts"]
        
        # Each priority should have roughly equal distribution
        assert priority_counts["high"] > 50
        assert priority_counts["medium"] > 50
        assert priority_counts["low"] > 50
    
    def test_coordinator_node_management_with_security(self, distributed_coordinator, security_middleware):
        """Test coordinator node management with security integration"""
        # Simulate multiple nodes joining with security checks
        node_ids = [f"node_{i:03d}" for i in range(20)]
        
        for node_id in node_ids:
            # Security check for node
            allowed, message = security_middleware.check_request_security(
                client_id=node_id,
                endpoint="/network/join",
                params={"node_info": "test_node"}
            )
            
            if allowed:
                # Simulate node heartbeat
                message = {
                    "type": "node_heartbeat",
                    "sender": node_id,
                    "timestamp": time.time(),
                    "status": "online"
                }
                asyncio.run(distributed_coordinator._handle_node_heartbeat(message))
        
        # Verify nodes are registered
        assert len(distributed_coordinator.nodes) >= 20  # Coordinator + 20 nodes
        
        # Test node capabilities exchange
        for node_id in node_ids[:5]:  # Test first 5 nodes
            message = {
                "type": "node_capabilities",
                "sender": node_id,
                "capabilities": {
                    "federated_learning": True,
                    "gpu_available": i % 2 == 0,
                    "max_clients": 10
                },
                "timestamp": time.time()
            }
            asyncio.run(distributed_coordinator._handle_node_capabilities(message))
        
        # Verify capabilities are stored
        for node_id in node_ids[:5]:
            node_status = distributed_coordinator.get_node_status(node_id)
            assert node_status is not None
            assert "federated_learning" in node_status["capabilities"]
    
    def test_end_to_end_training_with_security(self, distributed_coordinator, security_middleware):
        """Test end-to-end training workflow with security"""
        # Start training job with security check
        client_id = "training_initiator"
        
        allowed, message = security_middleware.check_request_security(
            client_id=client_id,
            endpoint="/training/start",
            params={
                "model_name": "integration_test_model",
                "strategy": "fedavg",
                "rounds": 5
            }
        )
        
        if allowed:
            # Mock federated config
            from src.aegis.ml.federated_learning import FederatedConfig, AggregationStrategy
            
            config = FederatedConfig(
                aggregation_strategy=AggregationStrategy.FED_AVG,
                num_rounds=5,
                clients_per_round=3
            )
            
            # Start training
            job_id = asyncio.run(distributed_coordinator.start_training_job(
                model_name="integration_test_model",
                config=config,
                total_rounds=5,
                clients_per_round=3
            ))
            
            assert job_id is not None
            assert job_id in distributed_coordinator.jobs
            
            # Verify job status
            job_status = distributed_coordinator.get_job_status(job_id)
            assert job_status is not None
            assert job_status["phase"] == "idle"  # Initial phase
