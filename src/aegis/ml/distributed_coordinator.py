"""
AEGIS Distributed Training Coordinator
Coordinates federated learning across multiple nodes in a P2P network
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
from collections import defaultdict, deque

from ..ml.federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedConfig,
    AggregationStrategy,
    ClientState,
    ServerState
)
from ..networking.p2p_network import P2PNetworkManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Phases of distributed training"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    CLIENT_SELECTION = "client_selection"
    MODEL_DISTRIBUTION = "model_distribution"
    LOCAL_TRAINING = "local_training"
    MODEL_COLLECTION = "model_collection"
    AGGREGATION = "aggregation"
    MODEL_UPDATE = "model_update"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeRole(Enum):
    """Roles in distributed training"""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


@dataclass
class TrainingJob:
    """Distributed training job configuration"""
    job_id: str
    model_name: str
    config: FederatedConfig
    total_rounds: int
    clients_per_round: int
    participants: List[str] = field(default_factory=list)
    current_round: int = 0
    phase: TrainingPhase = TrainingPhase.IDLE
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {
        "loss": [], "accuracy": [], "training_time": []
    })
    selected_clients: List[str] = field(default_factory=list)
    client_states: Dict[str, ClientState] = field(default_factory=dict)
    coordinator_node: Optional[str] = None


@dataclass
class NodeStatus:
    """Status of a training node"""
    node_id: str
    role: NodeRole
    status: str  # "online", "offline", "busy", "error"
    last_seen: float
    capabilities: Dict[str, Any] = field(default_factory=dict)
    training_jobs: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class DistributedTrainingCoordinator:
    """Coordinates federated learning across multiple nodes"""
    
    def __init__(
        self,
        node_id: str,
        p2p_network: P2PNetworkManager,
        role: NodeRole = NodeRole.COORDINATOR
    ):
        self.node_id = node_id
        self.p2p_network = p2p_network
        self.role = role
        
        self.jobs: Dict[str, TrainingJob] = {}
        self.nodes: Dict[str, NodeStatus] = {}
        self.model_registry: Dict[str, Any] = {}
        
        self.training_locks: Dict[str, asyncio.Lock] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        self._setup_message_handlers()
        self._register_self()
        
        # Start background tasks
        self.background_tasks = set()
        # Only start background tasks if we're in an async context
        try:
            asyncio.get_running_loop()
            self._start_background_tasks()
        except RuntimeError:
            # No running loop, defer background tasks start
            pass
    
    def _setup_message_handlers(self):
        """Setup message handlers for P2P communication"""
        self.message_handlers = {
            "training_start": self._handle_training_start,
            "training_join": self._handle_training_join,
            "model_request": self._handle_model_request,
            "model_update": self._handle_model_update,
            "client_selection": self._handle_client_selection,
            "training_complete": self._handle_training_complete,
            "node_heartbeat": self._handle_node_heartbeat,
            "node_capabilities": self._handle_node_capabilities,
            "error_report": self._handle_error_report
        }
    
    def _register_self(self):
        """Register this node in the network"""
        self.nodes[self.node_id] = NodeStatus(
            node_id=self.node_id,
            role=self.role,
            status="online",
            last_seen=time.time(),
            capabilities={
                "federated_learning": True,
                "max_clients": 10,
                "supported_strategies": [s.value for s in AggregationStrategy],
                "gpu_available": False
            }
        )
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.background_tasks.add(heartbeat_task)
        heartbeat_task.add_done_callback(self.background_tasks.discard)
        
        # Node discovery task
        discovery_task = asyncio.create_task(self._node_discovery_loop())
        self.background_tasks.add(discovery_task)
        discovery_task.add_done_callback(self.background_tasks.discard)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self.background_tasks.discard)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _node_discovery_loop(self):
        """Discover and update node information"""
        while True:
            try:
                await self._discover_nodes()
                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                logger.error(f"Error in node discovery loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Cleanup completed jobs and stale data"""
        while True:
            try:
                self._cleanup_stale_data()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _send_heartbeat(self):
        """Send heartbeat to all nodes"""
        message = {
            "type": "node_heartbeat",
            "sender": self.node_id,
            "timestamp": time.time(),
            "status": "online",
            "jobs": list(self.jobs.keys())
        }
        
        await self.p2p_network.broadcast_message(message)
    
    async def _discover_nodes(self):
        """Discover active nodes in the network"""
        # Request capabilities from other nodes
        message = {
            "type": "node_capabilities",
            "sender": self.node_id,
            "request": True,
            "timestamp": time.time()
        }
        
        await self.p2p_network.broadcast_message(message)
    
    def _cleanup_stale_data(self):
        """Cleanup stale jobs and node data"""
        current_time = time.time()
        
        # Remove completed jobs older than 1 hour
        stale_jobs = [
            job_id for job_id, job in self.jobs.items()
            if job.phase in [TrainingPhase.COMPLETED, TrainingPhase.FAILED]
            and current_time - job.last_update > 3600
        ]
        
        for job_id in stale_jobs:
            del self.jobs[job_id]
            if job_id in self.training_locks:
                del self.training_locks[job_id]
        
        # Remove offline nodes older than 10 minutes
        stale_nodes = [
            node_id for node_id, node in self.nodes.items()
            if current_time - node.last_seen > 600
        ]
        
        for node_id in stale_nodes:
            del self.nodes[node_id]
    
    async def start_training_job(
        self,
        model_name: str,
        config: FederatedConfig,
        total_rounds: int = 10,
        clients_per_round: int = 5
    ) -> str:
        """Start a new distributed training job"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            config=config,
            total_rounds=total_rounds,
            clients_per_round=clients_per_round,
            coordinator_node=self.node_id
        )
        
        self.jobs[job_id] = job
        self.training_locks[job_id] = asyncio.Lock()
        
        # Broadcast training start
        message = {
            "type": "training_start",
            "job_id": job_id,
            "model_name": model_name,
            "config": {
                "aggregation_strategy": config.aggregation_strategy.value,
                "num_rounds": total_rounds,
                "clients_per_round": clients_per_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate
            },
            "sender": self.node_id,
            "timestamp": time.time()
        }
        
        await self.p2p_network.broadcast_message(message)
        
        # Start first round
        asyncio.create_task(self._run_training_round(job_id))
        
        logger.info(f"Started distributed training job: {job_id}")
        return job_id
    
    async def _run_training_round(self, job_id: str):
        """Run a single training round"""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.current_round += 1
        job.last_update = time.time()
        
        if job.current_round > job.total_rounds:
            await self._complete_training_job(job_id)
            return
        
        logger.info(f"Starting round {job.current_round}/{job.total_rounds} for job {job_id}")
        
        try:
            # Select clients for this round
            await self._select_clients(job_id)
            
            # Distribute model to selected clients
            await self._distribute_model(job_id)
            
            # Collect model updates
            await self._collect_model_updates(job_id)
            
            # Aggregate updates
            await self._aggregate_updates(job_id)
            
            # Update global model
            await self._update_global_model(job_id)
            
            # Schedule next round
            asyncio.create_task(self._run_training_round(job_id))
            
        except Exception as e:
            logger.error(f"Error in training round {job.current_round}: {e}")
            await self._fail_training_job(job_id, str(e))
    
    async def _select_clients(self, job_id: str):
        """Select clients for current training round"""
        job = self.jobs[job_id]
        job.phase = TrainingPhase.CLIENT_SELECTION
        
        # Filter available nodes
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.status == "online" and node_id != self.node_id
        ]
        
        # Select clients (simple random selection for now)
        import random
        selected_clients = random.sample(
            available_nodes, 
            min(job.clients_per_round, len(available_nodes))
        )
        
        job.selected_clients = selected_clients
        
        # Notify selected clients
        message = {
            "type": "client_selection",
            "job_id": job_id,
            "selected_clients": selected_clients,
            "round": job.current_round,
            "sender": self.node_id,
            "timestamp": time.time()
        }
        
        for client_id in selected_clients:
            await self.p2p_network.send_direct_message(client_id, message)
    
    async def _distribute_model(self, job_id: str):
        """Distribute global model to selected clients"""
        job = self.jobs[job_id]
        job.phase = TrainingPhase.MODEL_DISTRIBUTION
        
        # Get current global model (simplified)
        global_model = {
            "round": job.current_round,
            "timestamp": time.time()
        }
        
        # Send to all selected clients
        message = {
            "type": "model_request",
            "job_id": job_id,
            "model": global_model,
            "round": job.current_round,
            "sender": self.node_id,
            "timestamp": time.time()
        }
        
        for client_id in job.selected_clients:
            await self.p2p_network.send_direct_message(client_id, message)
    
    async def _collect_model_updates(self, job_id: str):
        """Collect model updates from clients"""
        job = self.jobs[job_id]
        job.phase = TrainingPhase.MODEL_COLLECTION
        
        # Wait for updates from clients (with timeout)
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if we have updates from all selected clients
            received_clients = set(job.client_states.keys())
            expected_clients = set(job.selected_clients)
            
            if received_clients >= expected_clients:
                break
            
            await asyncio.sleep(1)
        
        logger.info(f"Collected updates from {len(job.client_states)} clients")
    
    async def _aggregate_updates(self, job_id: str):
        """Aggregate model updates from clients"""
        job = self.jobs[job_id]
        job.phase = TrainingPhase.AGGREGATION
        
        if not job.client_states:
            logger.warning("No client updates to aggregate")
            return
        
        # Simple averaging for now (would integrate with FL module)
        total_samples = sum(state.num_samples for state in job.client_states.values())
        
        logger.info(f"Aggregated updates from {len(job.client_states)} clients")
        job.last_update = time.time()
    
    async def _update_global_model(self, job_id: str):
        """Update global model with aggregated results"""
        job = self.jobs[job_id]
        job.phase = TrainingPhase.MODEL_UPDATE
        
        # Update metrics
        if job.client_states:
            avg_loss = sum(state.loss for state in job.client_states.values()) / len(job.client_states)
            avg_accuracy = sum(state.accuracy for state in job.client_states.values()) / len(job.client_states)
            
            job.metrics["loss"].append(avg_loss)
            job.metrics["accuracy"].append(avg_accuracy)
        
        job.last_update = time.time()
        logger.info(f"Updated global model for round {job.current_round}")
    
    async def _complete_training_job(self, job_id: str):
        """Complete training job"""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.phase = TrainingPhase.COMPLETED
        job.last_update = time.time()
        
        # Notify completion
        message = {
            "type": "training_complete",
            "job_id": job_id,
            "status": "completed",
            "final_metrics": job.metrics,
            "sender": self.node_id,
            "timestamp": time.time()
        }
        
        await self.p2p_network.broadcast_message(message)
        
        logger.info(f"Training job {job_id} completed successfully")
    
    async def _fail_training_job(self, job_id: str, error: str):
        """Fail training job"""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.phase = TrainingPhase.FAILED
        job.last_update = time.time()
        
        # Notify failure
        message = {
            "type": "training_complete",
            "job_id": job_id,
            "status": "failed",
            "error": error,
            "sender": self.node_id,
            "timestamp": time.time()
        }
        
        await self.p2p_network.broadcast_message(message)
        
        logger.error(f"Training job {job_id} failed: {error}")
    
    async def join_training_job(self, job_id: str) -> bool:
        """Join an existing training job as participant"""
        if job_id not in self.jobs:
            logger.warning(f"Job {job_id} not found")
            return False
        
        if self.role != NodeRole.PARTICIPANT:
            logger.warning("Node is not configured as participant")
            return False
        
        job = self.jobs[job_id]
        
        # Send join request
        message = {
            "type": "training_join",
            "job_id": job_id,
            "node_id": self.node_id,
            "capabilities": self.nodes[self.node_id].capabilities,
            "sender": self.node_id,
            "timestamp": time.time()
        }
        
        if job.coordinator_node:
            await self.p2p_network.send_direct_message(job.coordinator_node, message)
            return True
        
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        return {
            "job_id": job.job_id,
            "model_name": job.model_name,
            "phase": job.phase.value,
            "current_round": job.current_round,
            "total_rounds": job.total_rounds,
            "selected_clients": job.selected_clients,
            "metrics": job.metrics,
            "start_time": job.start_time,
            "last_update": job.last_update
        }
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a node"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        return {
            "node_id": node.node_id,
            "role": node.role.value,
            "status": node.status,
            "last_seen": node.last_seen,
            "capabilities": node.capabilities,
            "training_jobs": node.training_jobs,
            "performance_metrics": node.performance_metrics
        }
    
    # Message handlers
    async def _handle_training_start(self, message: Dict[str, Any]):
        """Handle training start message"""
        job_id = message.get("job_id")
        if not job_id:
            return
        
        # Create job entry
        config_data = message.get("config", {})
        strategy = AggregationStrategy(config_data.get("aggregation_strategy", "fedavg"))
        
        config = FederatedConfig(
            aggregation_strategy=strategy,
            num_rounds=config_data.get("num_rounds", 10),
            clients_per_round=config_data.get("clients_per_round", 5),
            local_epochs=config_data.get("local_epochs", 5),
            learning_rate=config_data.get("learning_rate", 0.01)
        )
        
        job = TrainingJob(
            job_id=job_id,
            model_name=message.get("model_name", "distributed_model"),
            config=config,
            total_rounds=config_data.get("num_rounds", 10),
            clients_per_round=config_data.get("clients_per_round", 5),
            coordinator_node=message.get("sender")
        )
        
        self.jobs[job_id] = job
        self.training_locks[job_id] = asyncio.Lock()
        
        logger.info(f"Joined training job: {job_id}")
    
    async def _handle_training_join(self, message: Dict[str, Any]):
        """Handle training join request"""
        job_id = message.get("job_id")
        node_id = message.get("node_id")
        
        if job_id not in self.jobs:
            return
        
        # Add node to participants if space available
        job = self.jobs[job_id]
        if len(job.participants) < job.clients_per_round * 2:  # Allow some buffer
            if node_id not in job.participants:
                job.participants.append(node_id)
                logger.info(f"Node {node_id} joined job {job_id}")
    
    async def _handle_model_request(self, message: Dict[str, Any]):
        """Handle model request from client"""
        job_id = message.get("job_id")
        sender = message.get("sender")
        
        if job_id not in self.jobs or sender not in self.nodes:
            return
        
        # In a real implementation, this would send the actual model
        logger.info(f"Model requested by {sender} for job {job_id}")
    
    async def _handle_model_update(self, message: Dict[str, Any]):
        """Handle model update from client"""
        job_id = message.get("job_id")
        sender = message.get("sender")
        client_state = message.get("client_state")
        
        if job_id not in self.jobs or not client_state:
            return
        
        job = self.jobs[job_id]
        
        # Store client state
        try:
            state = ClientState(**client_state)
            job.client_states[sender] = state
            logger.info(f"Received model update from {sender} for job {job_id}")
        except Exception as e:
            logger.error(f"Error processing client state: {e}")
    
    async def _handle_client_selection(self, message: Dict[str, Any]):
        """Handle client selection message"""
        job_id = message.get("job_id")
        selected_clients = message.get("selected_clients", [])
        
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        if self.node_id in selected_clients:
            logger.info(f"Selected for training round {message.get('round')} in job {job_id}")
    
    async def _handle_training_complete(self, message: Dict[str, Any]):
        """Handle training completion message"""
        job_id = message.get("job_id")
        status = message.get("status")
        
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.phase = TrainingPhase.COMPLETED if status == "completed" else TrainingPhase.FAILED
        job.last_update = time.time()
        
        logger.info(f"Job {job_id} marked as {status}")
    
    async def _handle_node_heartbeat(self, message: Dict[str, Any]):
        """Handle node heartbeat"""
        node_id = message.get("sender")
        if not node_id:
            return
        
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeStatus(
                node_id=node_id,
                role=NodeRole.PARTICIPANT,
                status="online",
                last_seen=time.time()
            )
        else:
            self.nodes[node_id].last_seen = time.time()
            self.nodes[node_id].status = message.get("status", "online")
    
    async def _handle_node_capabilities(self, message: Dict[str, Any]):
        """Handle node capabilities message"""
        node_id = message.get("sender")
        if not node_id:
            return
        
        if message.get("request"):
            # Send our capabilities
            response = {
                "type": "node_capabilities",
                "sender": self.node_id,
                "capabilities": self.nodes[self.node_id].capabilities,
                "timestamp": time.time()
            }
            await self.p2p_network.send_direct_message(node_id, response)
        else:
            # Update node capabilities
            capabilities = message.get("capabilities", {})
            if node_id in self.nodes:
                self.nodes[node_id].capabilities = capabilities
                self.nodes[node_id].last_seen = time.time()
    
    async def _handle_error_report(self, message: Dict[str, Any]):
        """Handle error report from node"""
        node_id = message.get("sender")
        error = message.get("error", "")
        job_id = message.get("job_id")
        
        logger.error(f"Error report from {node_id}: {error}")
        
        if job_id and job_id in self.jobs:
            # Could trigger client replacement or job adjustment
            pass
