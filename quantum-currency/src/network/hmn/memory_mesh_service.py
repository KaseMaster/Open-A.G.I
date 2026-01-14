#!/usr/bin/env python3
"""
Memory Mesh Service for Harmonic Mesh Network
Implements the Global Harmonic Memory Mesh with λ(t)-Attuned Gossip Protocol
"""

import asyncio
import time
import json
import hashlib
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from collections import deque
import threading
from queue import Queue, Empty
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryUpdate:
    """Represents a memory update in the harmonic mesh"""
    id: str
    content: Dict[str, Any]
    timestamp: float
    rphiv_score: float  # RΦV score
    node_id: str
    shard_id: str
    signature: Optional[str] = None
    version: int = 1

@dataclass
class MemoryProof:
    """Represents a proof for critical memory updates"""
    update_id: str
    content_hash: str
    node_signature: str
    timestamp: float
    rphiv_score: float

@dataclass
class GossipMessage:
    """Represents a gossip message between nodes"""
    sender_id: str
    updates: List[MemoryUpdate]
    λ_t: float
    timestamp: float
    message_type: str = "full"  # "full", "delta", "heartbeat"

@dataclass
class DeltaUpdate:
    """Represents a delta update for bandwidth optimization"""
    update_id: str
    field: str
    old_value: Any
    new_value: Any
    timestamp: float

class MemoryMeshService:
    """
    Implements the Memory Mesh Service for the Harmonic Mesh Network
    Manages distributed memory storage, indexing, and λ(t)-attuned gossip protocol
    """
    
    def __init__(self, node_id: str, network_config: Dict[str, Any]):
        self.node_id = node_id
        self.network_config = network_config
        self.local_memory: Dict[str, MemoryUpdate] = {}
        self.memory_index: Dict[str, Set[str]] = {}  # content_hash -> set of update_ids
        self.shard_map: Dict[str, List[str]] = {}  # shard_id -> list of node_ids
        self.peer_connections: Dict[str, Any] = {}  # node_id -> connection
        self.gossip_history: deque = deque(maxlen=1000)
        self.last_gossip_time: float = 0.0
        self.memory_archive: Dict[str, MemoryUpdate] = {}  # Archived memory for long-term storage
        self.peer_latency_map: Dict[str, float] = {}  # node_id -> latency
        self.peer_coherence_map: Dict[str, float] = {}  # node_id -> coherence_score
        
        # Peer discovery
        self.known_peers: Set[str] = set()  # All known peers in the network
        self.discovered_peers: Dict[str, Dict[str, Any]] = {}  # peer_id -> peer_info
        self.last_discovery_time: float = 0.0
        self.discovery_interval: float = 30.0  # Discover new peers every 30 seconds
        
        # Asynchronous message queue for better concurrency
        self.message_queue = Queue()
        self.message_processor_thread = None
        self.running = False
        
        # Configuration
        self.config = {
            "gossip_intervals": {
                "high": 0.5,    # seconds for high λ(t)
                "medium": 2.0,  # seconds for medium λ(t)
                "low": 10.0     # seconds for low λ(t)
            },
            "rphiv_thresholds": {
                "critical": 0.9,
                "important": 0.7,
                "normal": 0.5
            },
            "memory_limits": {
                "max_local_updates": 10000,
                "compression_threshold": 0.3,
                "archive_threshold": 0.1  # RΦV score threshold for archiving
            },
            "network": {
                "max_peers": 20,
                "peer_selection_strategy": "coherence_proximity",  # latency, coherence_proximity, hybrid
                "bandwidth_limit_kbps": 1024,  # KB/s
                "congestion_window": 10,
                "enable_tls": network_config.get("enable_tls", False),  # TLS/SSL support
                "discovery_enabled": network_config.get("discovery_enabled", True)
            }
        }
        
        # Metrics for observability
        self.metrics = {
            "gossip_messages_sent": 0,
            "gossip_messages_received": 0,
            "updates_integrated": 0,
            "memory_proofs_generated": 0,
            "bandwidth_used_kb": 0,
            "peer_selection_success_rate": 1.0,
            "peers_discovered": 0,
            "tls_connections": 0
        }
        
        # Initialize known peers from network config
        if "network_peers" in network_config:
            for peer in network_config["network_peers"]:
                self.known_peers.add(peer)
        
        logger.info(f"Memory Mesh Service initialized for node: {node_id}")
    
    def start(self):
        """Start the memory mesh service"""
        self.running = True
        self.message_processor_thread = threading.Thread(target=self._process_message_queue, daemon=True)
        self.message_processor_thread.start()
        logger.info("Memory Mesh Service started")
    
    def stop(self):
        """Stop the memory mesh service"""
        self.running = False
        if self.message_processor_thread:
            self.message_processor_thread.join(timeout=5.0)
        logger.info("Memory Mesh Service stopped")
    
    def _process_message_queue(self):
        """Process messages from the queue asynchronously"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1.0)
                self._handle_queued_message(message)
                self.message_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
    
    def _handle_queued_message(self, message: Any):
        """Handle a queued message"""
        if isinstance(message, GossipMessage):
            self.process_gossip_message(message)
        # Add other message types as needed
    
    def get_local_updates(self) -> List[MemoryUpdate]:
        """Get all local memory updates"""
        return list(self.local_memory.values())
    
    def index_updates(self, updates: List[MemoryUpdate]):
        """Index memory updates for efficient retrieval"""
        for update in updates:
            # Add to local memory
            self.local_memory[update.id] = update
            
            # Index by content hash
            content_hash = hashlib.sha256(
                json.dumps(update.content, sort_keys=True).encode()
            ).hexdigest()
            
            if content_hash not in self.memory_index:
                self.memory_index[content_hash] = set()
            self.memory_index[content_hash].add(update.id)
            
            logger.debug(f"Indexed update {update.id} with hash {content_hash}")
        
        # Check memory limits and perform maintenance
        if len(self.local_memory) > self.config["memory_limits"]["max_local_updates"]:
            self._perform_memory_maintenance()
    
    def _perform_memory_maintenance(self):
        """Perform memory maintenance including pruning and archiving"""
        # Archive low RΦV memory
        self._archive_low_rphiv_memory()
        
        # Compress memory if needed
        self._perform_compression()
        
        # Prune old memory if still over limit
        if len(self.local_memory) > self.config["memory_limits"]["max_local_updates"]:
            self._prune_oldest_memory()
    
    def _archive_low_rphiv_memory(self):
        """Archive memory with low RΦV scores"""
        archive_threshold = self.config["memory_limits"]["archive_threshold"]
        archived_count = 0
        
        # Find low RΦV updates
        low_rphiv_updates = [
            update for update in self.local_memory.values()
            if update.rphiv_score < archive_threshold
        ]
        
        # Move to archive
        for update in low_rphiv_updates:
            self.memory_archive[update.id] = update
            del self.local_memory[update.id]
            archived_count += 1
        
        if archived_count > 0:
            logger.info(f"Archived {archived_count} low RΦV memory updates")
    
    def _prune_oldest_memory(self):
        """Prune oldest memory updates to stay within limits"""
        # Sort by timestamp and remove oldest
        sorted_updates = sorted(self.local_memory.values(), key=lambda x: x.timestamp)
        prune_count = len(self.local_memory) - self.config["memory_limits"]["max_local_updates"] + 100  # Keep some buffer
        
        for update in sorted_updates[:prune_count]:
            # Remove from local memory
            del self.local_memory[update.id]
            
            # Remove from index
            content_hash = hashlib.sha256(
                json.dumps(update.content, sort_keys=True).encode()
            ).hexdigest()
            
            if content_hash in self.memory_index:
                self.memory_index[content_hash].discard(update.id)
                if not self.memory_index[content_hash]:
                    del self.memory_index[content_hash]
        
        logger.info(f"Pruned {prune_count} oldest memory updates")
    
    def discover_peers(self, network_state: Dict[str, Any]):
        """Discover new peers in the network"""
        current_time = time.time()
        if not self.config["network"]["discovery_enabled"] or \
           current_time - self.last_discovery_time < self.discovery_interval:
            return
        
        self.last_discovery_time = current_time
        
        try:
            # In a real implementation, this would use actual network discovery protocols
            # For now, we'll simulate discovery based on network configuration
            discovered_count = 0
            
            # Add any new peers from network config
            if "network_peers" in self.network_config:
                for peer in self.network_config["network_peers"]:
                    if peer not in self.known_peers:
                        self.known_peers.add(peer)
                        self.discovered_peers[peer] = {
                            "address": peer,
                            "discovery_time": current_time,
                            "coherence_score": network_state.get("coherence_density", 0.8),
                            "last_seen": current_time
                        }
                        discovered_count += 1
            
            # Simulate discovering peers from gossip messages
            for gossip_entry in list(self.gossip_history)[-10:]:  # Check last 10 gossip entries
                # In a real implementation, gossip messages would contain peer information
                pass
            
            if discovered_count > 0:
                self.metrics["peers_discovered"] += discovered_count
                logger.info(f"Discovered {discovered_count} new peers")
                
        except Exception as e:
            logger.error(f"Error during peer discovery: {e}")

    def establish_secure_connection(self, peer_id: str) -> bool:
        """Establish a secure TLS/SSL connection to a peer"""
        try:
            if not self.config["network"]["enable_tls"]:
                # TLS not enabled, use regular connection
                self.peer_connections[peer_id] = {"secure": False, "connected": True}
                return True
            
            # In a real implementation, this would establish an actual TLS/SSL connection
            # For now, we'll simulate a secure connection
            self.peer_connections[peer_id] = {"secure": True, "connected": True, "tls_version": "TLSv1.3"}
            self.metrics["tls_connections"] += 1
            logger.debug(f"Established secure TLS connection to {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error establishing secure connection to {peer_id}: {e}")
            return False

    def participate_in_gossip(self, network_state: Dict[str, Any]):
        """Participate in the λ(t)-attuned gossip protocol"""
        current_time = time.time()
        λ_t = network_state.get("lambda_t", 0.5)
        
        # Perform peer discovery
        self.discover_peers(network_state)
        
        # Determine gossip interval based on λ(t)
        gossip_interval = self._calculate_gossip_interval(λ_t)
        
        # Check if it's time to gossip
        if current_time - self.last_gossip_time < gossip_interval:
            return
        
        self.last_gossip_time = current_time
        
        try:
            # Select peers based on network conditions and coherence proximity
            peers = self._select_optimal_peers(network_state)
            
            # Establish secure connections to peers if not already connected
            for peer_id in peers:
                if peer_id not in self.peer_connections:
                    self.establish_secure_connection(peer_id)
            
            # Get priority updates to share
            updates_to_share = self._get_priority_updates(λ_t)
            
            # Determine message type based on bandwidth and update size
            message_type = self._determine_message_type(updates_to_share, λ_t)
            
            # Create gossip message
            gossip_message = GossipMessage(
                sender_id=self.node_id,
                updates=updates_to_share,
                λ_t=λ_t,
                timestamp=current_time,
                message_type=message_type
            )
            
            # Send to selected peers
            successful_sends = 0
            for peer_id in peers:
                if peer_id in self.peer_connections:
                    if self._send_gossip_message(peer_id, gossip_message):
                        successful_sends += 1
            
            # Update metrics
            self.metrics["gossip_messages_sent"] += len(peers)
            if len(peers) > 0:
                self.metrics["peer_selection_success_rate"] = successful_sends / len(peers)
            
            # Record gossip event
            self.gossip_history.append({
                "timestamp": current_time,
                "peers": len(peers),
                "updates_shared": len(updates_to_share),
                "lambda_t": λ_t,
                "message_type": message_type,
                "bandwidth_used": len(json.dumps([u.__dict__ for u in updates_to_share]).encode())
            })
            
            logger.info(f"Gossiped {len(updates_to_share)} updates ({message_type}) to {successful_sends}/{len(peers)} peers at λ(t)={λ_t:.3f}")
            
        except Exception as e:
            logger.error(f"Error during gossip participation: {e}")
    
    def _select_optimal_peers(self, network_state: Dict[str, Any]) -> List[str]:
        """Select optimal peers based on latency and coherence proximity"""
        strategy = self.config["network"]["peer_selection_strategy"]
        max_peers = self.config["network"]["max_peers"]
        
        if strategy == "latency":
            return self._select_peers_by_latency(max_peers)
        elif strategy == "coherence_proximity":
            return self._select_peers_by_coherence(network_state, max_peers)
        else:  # hybrid
            return self._select_peers_hybrid(network_state, max_peers)
    
    def _select_peers_by_latency(self, max_peers: int) -> List[str]:
        """Select peers based on network latency"""
        # Sort peers by latency
        sorted_peers = sorted(self.peer_latency_map.items(), key=lambda x: x[1])
        return [peer_id for peer_id, _ in sorted_peers[:max_peers]]
    
    def _select_peers_by_coherence(self, network_state: Dict[str, Any], max_peers: int) -> List[str]:
        """Select peers based on coherence proximity"""
        local_coherence = network_state.get("coherence_density", 0.8)
        
        # Calculate coherence distance for each peer
        peer_distances = {}
        for peer_id, peer_coherence in self.peer_coherence_map.items():
            distance = abs(peer_coherence - local_coherence)
            peer_distances[peer_id] = distance
        
        # Sort by distance (closest first)
        sorted_peers = sorted(peer_distances.items(), key=lambda x: x[1])
        return [peer_id for peer_id, _ in sorted_peers[:max_peers]]
    
    def _select_peers_hybrid(self, network_state: Dict[str, Any], max_peers: int) -> List[str]:
        """Select peers using a hybrid approach of latency and coherence"""
        # Get top peers by each metric
        latency_peers = self._select_peers_by_latency(max_peers // 2)
        coherence_peers = self._select_peers_by_coherence(network_state, max_peers // 2)
        
        # Combine and deduplicate
        combined_peers = list(set(latency_peers + coherence_peers))
        return combined_peers[:max_peers]
    
    def _determine_message_type(self, updates: List[MemoryUpdate], λ_t: float) -> str:
        """Determine optimal message type based on bandwidth and update characteristics"""
        # Calculate total message size
        message_size = len(json.dumps([u.__dict__ for u in updates]).encode())
        bandwidth_limit = self.config["network"]["bandwidth_limit_kbps"] * 1024  # Convert to bytes
        
        # At high λ(t), prefer delta updates for efficiency
        if λ_t > 0.8 and message_size > bandwidth_limit * 0.1:
            return "delta"
        # At low λ(t), prefer full updates for reliability
        elif λ_t < 0.3:
            return "full"
        # For medium λ(t), use delta if large, full if small
        else:
            return "delta" if message_size > 5000 else "full"
    
    def _calculate_gossip_interval(self, λ_t: float) -> float:
        """Calculate gossip interval based on λ(t) with adaptive adjustments"""
        if λ_t < 0.5:
            base_interval = self.config["gossip_intervals"]["high"]
        elif λ_t < 0.8:
            base_interval = self.config["gossip_intervals"]["medium"]
        else:
            base_interval = self.config["gossip_intervals"]["low"]
        
        # Adjust based on network conditions
        congestion_factor = self._calculate_congestion_factor()
        latency_factor = self._calculate_latency_factor()
        
        return base_interval * congestion_factor * latency_factor
    
    def _calculate_congestion_factor(self) -> float:
        """Calculate network congestion factor"""
        # Mock implementation - in real system, this would check actual network conditions
        queue_size = self.message_queue.qsize()
        if queue_size > self.config["network"]["congestion_window"]:
            return 1.5  # Slow down due to congestion
        elif queue_size < self.config["network"]["congestion_window"] // 2:
            return 0.8  # Speed up due to low congestion
        return 1.0  # Normal
    
    def _calculate_latency_factor(self) -> float:
        """Calculate latency factor based on peer latencies"""
        if not self.peer_latency_map:
            return 1.0
        
        avg_latency = statistics.mean(self.peer_latency_map.values())
        if avg_latency > 100:  # High latency
            return 1.3
        elif avg_latency < 20:  # Low latency
            return 0.7
        return 1.0  # Normal

    def _get_priority_updates(self, λ_t: float) -> List[MemoryUpdate]:
        """Get priority updates based on RΦV scores and λ(t)"""
        # Filter updates by RΦV threshold
        critical_threshold = self.config["rphiv_thresholds"]["critical"]
        important_threshold = self.config["rphiv_thresholds"]["important"]
        
        # Get recent updates (last 5 minutes)
        current_time = time.time()
        recent_threshold = current_time - 300  # 5 minutes
        
        priority_updates = []
        for update in self.local_memory.values():
            # Prioritize by RΦV score and recency
            is_critical = update.rphiv_score >= critical_threshold
            is_important = update.rphiv_score >= important_threshold
            is_recent = update.timestamp >= recent_threshold
            
            # At high λ(t), gossip more aggressively
            if λ_t < 0.5 or is_critical or (is_important and is_recent):
                priority_updates.append(update)
        
        # Limit to most recent 100 updates
        priority_updates.sort(key=lambda x: x.timestamp, reverse=True)
        return priority_updates[:100]
    
    def _send_gossip_message(self, peer_id: str, message: GossipMessage) -> bool:
        """Send gossip message to a peer"""
        try:
            # In a real implementation, this would use actual network communication
            # For now, we'll just log the action and simulate success
            message_size = len(json.dumps(message.__dict__, default=str).encode())
            self.metrics["bandwidth_used_kb"] += message_size / 1024
            
            logger.debug(f"Sending {message.message_type} gossip message ({message_size} bytes) to {peer_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending gossip message to {peer_id}: {e}")
            return False

    def update_peer_metrics(self, peer_id: str, latency: float, coherence_score: float):
        """Update metrics for a peer"""
        self.peer_latency_map[peer_id] = latency
        self.peer_coherence_map[peer_id] = coherence_score
    
    def process_gossip_message(self, message: GossipMessage):
        """Process incoming gossip message"""
        try:
            logger.debug(f"Processing gossip message from {message.sender_id}")
            
            # Update metrics
            self.metrics["gossip_messages_received"] += 1
            
            # Validate message
            if not self._validate_gossip_message(message):
                logger.warning(f"Invalid gossip message from {message.sender_id}")
                return
            
            # Process based on message type
            if message.message_type == "delta":
                self._process_delta_message(message)
            else:
                self._process_full_message(message)
            
        except Exception as e:
            logger.error(f"Error processing gossip message: {e}")
    
    def _process_full_message(self, message: GossipMessage):
        """Process a full gossip message"""
        # Integrate received updates
        new_updates = []
        updated_count = 0
        
        for update in message.updates:
            # Check if we already have this update
            if update.id not in self.local_memory:
                # Add to local memory
                self.local_memory[update.id] = update
                new_updates.append(update)
                updated_count += 1
                
                # Index the update
                content_hash = hashlib.sha256(
                    json.dumps(update.content, sort_keys=True).encode()
                ).hexdigest()
                
                if content_hash not in self.memory_index:
                    self.memory_index[content_hash] = set()
                self.memory_index[content_hash].add(update.id)
        
        self.metrics["updates_integrated"] += updated_count
        logger.info(f"Integrated {updated_count} new updates from {message.sender_id}")
        
        # Generate memory proofs for critical updates
        critical_updates = [
            update for update in new_updates 
            if update.rphiv_score >= self.config["rphiv_thresholds"]["critical"]
        ]
        
        for update in critical_updates:
            proof = self._generate_memory_proof(update)
            if proof:
                # Commit to Layer 1 (in a real implementation, this would submit to the ledger)
                self._commit_memory_proof_to_layer1(proof)
                self.metrics["memory_proofs_generated"] += 1
    
    def _process_delta_message(self, message: GossipMessage):
        """Process a delta gossip message"""
        # For delta messages, we need to apply changes to existing updates
        applied_deltas = 0
        
        for update in message.updates:
            if update.id in self.local_memory:
                # Apply delta changes
                local_update = self.local_memory[update.id]
                
                # For simplicity, we're treating delta as full update in this mock
                # In a real implementation, this would apply field-level changes
                self.local_memory[update.id] = update
                applied_deltas += 1
        
        logger.info(f"Applied {applied_deltas} delta updates from {message.sender_id}")
    
    def _validate_gossip_message(self, message: GossipMessage) -> bool:
        """Validate incoming gossip message"""
        # Basic validation checks
        if not message.sender_id or not message.updates:
            return False
        
        if message.timestamp > time.time() + 60:  # Allow 1 minute future tolerance
            return False
        
        # Validate each update
        for update in message.updates:
            if not update.id or not update.content or update.timestamp > time.time() + 60:
                return False
        
        return True
    
    def _generate_memory_proof(self, update: MemoryUpdate) -> Optional[MemoryProof]:
        """Generate memory proof for critical updates"""
        try:
            # In a real implementation, this would use actual cryptographic signing
            # For now, we'll create a mock signature
            content_hash = hashlib.sha256(
                json.dumps(update.content, sort_keys=True).encode()
            ).hexdigest()
            
            proof = MemoryProof(
                update_id=update.id,
                content_hash=content_hash,
                node_signature=f"mock_signature_{self.node_id}_{update.id}",
                timestamp=time.time(),
                rphiv_score=update.rphiv_score
            )
            
            logger.debug(f"Generated memory proof for update {update.id}")
            return proof
            
        except Exception as e:
            logger.error(f"Error generating memory proof: {e}")
            return None
    
    def _commit_memory_proof_to_layer1(self, proof: MemoryProof):
        """Commit memory proof to Layer 1 ledger"""
        # In a real implementation, this would submit a transaction to the blockchain
        # For now, we'll just log the action
        logger.info(f"Committing memory proof {proof.update_id} to Layer 1")
    
    def _perform_compression(self):
        """Perform adaptive compression of low-value memory"""
        # Identify low RΦV memory for compression
        compression_threshold = self.config["memory_limits"]["compression_threshold"]
        low_value_updates = [
            update for update in self.local_memory.values()
            if update.rphiv_score < compression_threshold
        ]
        
        # In a real implementation, this would actually compress the memory
        # For now, we'll just log the action
        if low_value_updates:
            logger.info(f"Identified {len(low_value_updates)} low-value updates for compression")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory mesh statistics"""
        return {
            "local_updates_count": len(self.local_memory),
            "archived_updates_count": len(self.memory_archive),
            "indexed_content_hashes": len(self.memory_index),
            "connected_peers": len(self.peer_connections),
            "recent_gossip_events": len(self.gossip_history),
            "last_gossip_time": self.last_gossip_time,
            "metrics": self.metrics.copy()
        }

# Example usage and testing
async def demo_memory_mesh_service():
    """Demonstrate the enhanced Memory Mesh Service"""
    print("🌐 Enhanced Memory Mesh Service Demo")
    print("=" * 40)
    
    # Create service instance
    network_config = {
        "shard_count": 10,
        "replication_factor": 3
    }
    
    service = MemoryMeshService("node-001", network_config)
    service.start()
    
    # Add some peer metrics
    service.update_peer_metrics("node-002", 25.5, 0.85)
    service.update_peer_metrics("node-003", 32.1, 0.78)
    service.update_peer_metrics("node-004", 18.7, 0.92)
    
    # Create sample memory updates
    sample_updates = [
        MemoryUpdate(
            id=f"update-{i}",
            content={"data": f"sample_data_{i}", "type": "harmonic_metric"},
            timestamp=time.time() - i * 60,  # Different timestamps
            rphiv_score=0.5 + (i * 0.1),  # Different RΦV scores
            node_id="node-001",
            shard_id=f"shard-{i % 3}"
        )
        for i in range(5)
    ]
    
    # Index updates
    service.index_updates(sample_updates)
    print(f"✅ Indexed {len(sample_updates)} sample updates")
    
    # Show memory stats
    stats = service.get_memory_stats()
    print(f"📊 Memory Stats: {stats['local_updates_count']} local updates")
    
    # Simulate gossip participation
    network_state = {"lambda_t": 0.6, "coherence_density": 0.85}
    service.participate_in_gossip(network_state)
    print("🔄 Participated in gossip protocol")
    
    # Process a mock gossip message
    mock_message = GossipMessage(
        sender_id="node-002",
        updates=[
            MemoryUpdate(
                id="remote-update-1",
                content={"data": "remote_data_1", "type": "coherence_metric"},
                timestamp=time.time(),
                rphiv_score=0.8,
                node_id="node-002",
                shard_id="shard-1"
            )
        ],
        λ_t=0.6,
        timestamp=time.time()
    )
    
    service.process_gossip_message(mock_message)
    print("📥 Processed mock gossip message")
    
    # Show final stats
    final_stats = service.get_memory_stats()
    print(f"📊 Final Memory Stats: {final_stats['local_updates_count']} local updates")
    print(f"📈 Metrics: {final_stats['metrics']}")
    
    service.stop()
    print("\n✅ Enhanced Memory Mesh Service demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_memory_mesh_service())