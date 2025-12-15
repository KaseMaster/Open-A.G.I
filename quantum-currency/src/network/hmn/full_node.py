#!/usr/bin/env python3
"""
Full Node Implementation for Harmonic Mesh Network
Implements all node services: Layer 1 Ledger, CAL Engine, Mining Agent, Memory Mesh Service
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from queue import Queue
import threading
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Import HMN components
from .memory_mesh_service import MemoryMeshService
from .attuned_consensus import AttunedConsensus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
NODE_SERVICE_CALLS = Counter('hmn_node_service_calls_total', 'Total calls to node services', ['service'])
NODE_SERVICE_DURATION = Histogram('hmn_node_service_duration_seconds', 'Duration of node service calls', ['service'])
NODE_HEALTH_STATUS = Gauge('hmn_node_health_status', 'Node health status (1=healthy, 0=unhealthy)')
NODE_LAMBDA_T = Gauge('hmn_node_lambda_t', 'Current lambda(t) value')
NODE_COHERENCE_DENSITY = Gauge('hmn_node_coherence_density', 'Current coherence density')
NODE_PSI_SCORE = Gauge('hmn_node_psi_score', 'Current psi score')

@dataclass
class ServiceHealth:
    """Represents the health status of a service"""
    service_name: str
    is_healthy: bool
    last_check: float
    error_message: Optional[str] = None
    restart_count: int = 0

@dataclass
class EpochResult:
    """Represents the result of a mining epoch"""
    should_mint: bool
    mint_amount: float

class MockLayer1Ledger:
    """Mock implementation of Layer 1 Ledger with cryptographic validation"""
    
    def __init__(self):
        self.transactions = []
        self.pending_transactions = []
        self.transaction_hashes = set()
    
    def get_pending_transactions(self):
        return self.pending_transactions.copy()
    
    def validate_transaction_signature(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction cryptographic signature"""
        # In a real implementation, this would use actual cryptographic validation
        # For now, we'll check if it has a signature field
        return 'signature' in transaction
    
    def validate_transactions(self, transactions):
        # In a real implementation, this would validate transactions
        validated = []
        for tx in transactions:
            if self.validate_transaction_signature(tx):
                validated.append(tx)
            else:
                logger.warning(f"Invalid transaction signature: {tx}")
        return validated
    
    def commit_transactions(self, transactions):
        # Batch commit with optimized ordering
        # Sort transactions by priority (RΦV score if available)
        sorted_transactions = sorted(
            transactions, 
            key=lambda tx: tx.get('rphiv_score', 0), 
            reverse=True
        )
        
        for tx in sorted_transactions:
            # Create transaction hash for immutability check
            tx_hash = hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest()
            if tx_hash not in self.transaction_hashes:
                self.transactions.append(tx)
                self.transaction_hashes.add(tx_hash)
        
        self.pending_transactions = [tx for tx in self.pending_transactions if tx not in transactions]
        logger.info(f"Committed {len(sorted_transactions)} transactions to Layer 1")
    
    def submit_transaction(self, transaction):
        self.pending_transactions.append(transaction)
        logger.info(f"Submitted transaction {transaction.get('id', 'unknown')} to Layer 1")
    
    def get_transaction_count(self) -> int:
        return len(self.transactions)

class MockCALEngine:
    """Mock implementation of CAL Engine with time-series forecasting"""
    
    def __init__(self):
        self.λ_t = 0.7
        self.Ĉ_t = 0.85
        self.ψ_score = 0.9
        self.history = []
        self.forecast_history = []
    
    def compute_coherence_metrics(self):
        # Simulate changing metrics with time-series behavior
        self.Ĉ_t = max(0.5, min(1.0, self.Ĉ_t + (0.01 * (0.5 - abs(0.75 - self.Ĉ_t)))))
        # Store history for forecasting
        self.history.append({
            'timestamp': time.time(),
            'coherence_density': self.Ĉ_t,
            'lambda_t': self.λ_t,
            'psi_score': self.ψ_score
        })
        # Keep only last 1000 entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        return {"coherence_density": self.Ĉ_t}
    
    def forecast_coherence_trend(self, steps: int = 5) -> list:
        """Simple time-series forecasting based on historical data"""
        if len(self.history) < 10:
            return [self.Ĉ_t] * steps
        
        # Simple linear regression forecast
        recent_data = self.history[-10:]
        timestamps = [entry['timestamp'] for entry in recent_data]
        values = [entry['coherence_density'] for entry in recent_data]
        
        # Calculate trend
        if len(timestamps) > 1:
            time_diff = timestamps[-1] - timestamps[0]
            value_diff = values[-1] - values[0]
            trend = value_diff / time_diff if time_diff != 0 else 0
        else:
            trend = 0
        
        # Forecast next values
        forecasts = []
        for i in range(1, steps + 1):
            forecast = self.Ĉ_t + (trend * i * 10)  # 10 second intervals
            forecasts.append(max(0.5, min(1.0, forecast)))
        
        # Store forecast history
        self.forecast_history.append({
            'timestamp': time.time(),
            'forecasts': forecasts
        })
        
        return forecasts
    
    def analyze_coherence_trends(self) -> Dict[str, Any]:
        """Analyze historical coherence trends"""
        if len(self.history) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0.0,
                'volatility': 0.0,
                'stability': 'unknown'
            }
        
        # Calculate trend using linear regression
        timestamps = [entry['timestamp'] for entry in self.history[-50:]]  # Last 50 points
        values = [entry['coherence_density'] for entry in self.history[-50:]]
        
        # Normalize timestamps
        start_time = timestamps[0]
        normalized_timestamps = [t - start_time for t in timestamps]
        
        # Calculate slope (simple linear regression)
        n = len(normalized_timestamps)
        if n < 2:
            slope = 0.0
        else:
            sum_x = sum(normalized_timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(normalized_timestamps, values))
            sum_xx = sum(x * x for x in normalized_timestamps)
            
            denominator = (n * sum_xx - sum_x * sum_x)
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            else:
                slope = 0.0
        
        # Calculate volatility (standard deviation)
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        volatility = variance ** 0.5
        
        # Determine stability
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'degrading'
        
        if volatility < 0.05:
            stability = 'high'
        elif volatility < 0.1:
            stability = 'medium'
        else:
            stability = 'low'
        
        return {
            'trend': trend,
            'slope': slope,
            'volatility': volatility,
            'stability': stability,
            'data_points': len(self.history)
        }
    
    def calculate_psi_score(self):
        # Simulate changing psi score
        self.ψ_score = max(0.5, min(1.0, self.ψ_score + (0.005 * (0.75 - self.ψ_score))))
        return self.ψ_score
    
    def propose_lambda_t(self):
        # Simulate changing lambda based on coherence and psi
        target_lambda = 0.5 + (0.4 * self.Ĉ_t * self.ψ_score)
        self.λ_t = max(0.3, min(1.0, self.λ_t + (0.002 * (target_lambda - self.λ_t))))
        return self.λ_t
    
    def get_current_state(self):
        return {
            "lambda_t": self.λ_t,
            "coherence_density": self.Ĉ_t,
            "psi_score": self.ψ_score
        }

class MockMiningAgent:
    """Mock implementation of Mining Agent with adaptive minting"""
    
    def __init__(self):
        self.epoch_count = 0
        self.total_minted = 0.0
        self.transaction_priority_queue = []
        self.minting_history = []
    
    def prioritize_transactions(self, transactions: list, rphiv_scores: dict) -> list:
        """Prioritize transactions based on RΦV and Ψ scores"""
        # Sort by RΦV score (higher is better)
        return sorted(transactions, key=lambda tx: rphiv_scores.get(tx.get('id', ''), 0), reverse=True)
    
    def calculate_adaptive_minting_amount(self, network_state: Dict[str, Any]) -> float:
        """Calculate adaptive minting amount based on network state"""
        Ĉ_t = network_state.get("coherence_density", 0.8)
        λ_t = network_state.get("lambda_t", 0.7)
        ψ_score = network_state.get("psi_score", 0.9)
        
        # Base minting amount
        base_amount = 100.0
        
        # Adjust based on network health
        # Higher coherence and stability = more minting
        coherence_factor = Ĉ_t / 0.8  # Normalize around typical value
        stability_factor = λ_t / 0.7   # Normalize around typical value
        psi_factor = ψ_score / 0.9     # Normalize around typical value
        
        # Combine factors with weights
        adjustment_factor = (0.4 * coherence_factor + 0.3 * stability_factor + 0.3 * psi_factor)
        adjustment_factor = max(0.5, min(2.0, adjustment_factor))  # Clamp between 0.5 and 2.0
        
        mint_amount = base_amount * adjustment_factor
        
        # Store minting history
        self.minting_history.append({
            'timestamp': time.time(),
            'amount': mint_amount,
            'coherence': Ĉ_t,
            'lambda': λ_t,
            'psi': ψ_score,
            'factor': adjustment_factor
        })
        
        # Keep only last 100 entries
        if len(self.minting_history) > 100:
            self.minting_history = self.minting_history[-100:]
        
        return mint_amount
    
    def should_mint_in_epoch(self, network_state: Dict[str, Any]) -> bool:
        """Determine if minting should occur in this epoch based on network state"""
        Ĉ_t = network_state.get("coherence_density", 0.8)
        λ_t = network_state.get("lambda_t", 0.7)
        
        # More frequent minting when network is healthy
        if Ĉ_t > 0.9 and λ_t > 0.8:
            # Mint every 2 epochs when network is very healthy
            return self.epoch_count % 2 == 0
        elif Ĉ_t > 0.8 and λ_t > 0.6:
            # Mint every 3 epochs when network is healthy
            return self.epoch_count % 3 == 0
        elif Ĉ_t > 0.7:
            # Mint every 5 epochs when network is moderately healthy
            return self.epoch_count % 5 == 0
        else:
            # Mint every 10 epochs when network is unstable
            return self.epoch_count % 10 == 0
    
    def run_epoch(self, network_state: Optional[Dict[str, Any]] = None):
        self.epoch_count += 1
        
        # Adaptive minting based on network state
        should_mint = False
        mint_amount = 0.0
        
        if network_state:
            should_mint = self.should_mint_in_epoch(network_state)
            if should_mint:
                mint_amount = self.calculate_adaptive_minting_amount(network_state)
        else:
            # Default behavior
            should_mint = self.epoch_count % 5 == 0
            mint_amount = 50.0
        
        return EpochResult(should_mint=should_mint, mint_amount=mint_amount)
    
    def create_mint_transaction(self, epoch_result: EpochResult):
        return {
            "id": f"mint_transaction_{self.epoch_count}",
            "type": "mint",
            "amount": epoch_result.mint_amount,
            "timestamp": time.time(),
            "signature": f"mint_signature_{self.epoch_count}",
            "epoch": self.epoch_count
        }
    
    def get_minting_stats(self) -> Dict[str, Any]:
        """Get minting statistics"""
        if not self.minting_history:
            return {
                "total_minted": self.total_minted,
                "epochs_run": self.epoch_count,
                "average_mint_amount": 0.0,
                "last_mint": None
            }
        
        total_minted = sum(entry['amount'] for entry in self.minting_history)
        average_mint = total_minted / len(self.minting_history)
        last_mint = self.minting_history[-1] if self.minting_history else None
        
        return {
            "total_minted": total_minted,
            "epochs_run": self.epoch_count,
            "average_mint_amount": average_mint,
            "last_mint": last_mint,
            "history_count": len(self.minting_history)
        }

class FullNode:
    """
    Implements a Full Node for the Harmonic Mesh Network
    Runs all required services: Layer 1 Ledger, CAL Engine, Mining Agent, Memory Mesh Service
    """
    
    def __init__(self, node_id: str, network_config: Dict[str, Any]):
        self.node_id = node_id
        self.network_config = network_config
        self.running = False
        self.health_status = True
        self.service_health: Dict[str, ServiceHealth] = {}
        
        # Initialize all services
        self.ledger = MockLayer1Ledger()
        self.cal_engine = MockCALEngine()
        self.mining_agent = MockMiningAgent()
        self.memory_mesh_service = MemoryMeshService(node_id, network_config)
        self.consensus_engine = AttunedConsensus(node_id, network_config)
        
        # Asynchronous message queues for better concurrency
        self.service_queues: Dict[str, Queue] = {
            "ledger": Queue(),
            "cal_engine": Queue(),
            "mining_agent": Queue(),
            "memory_mesh": Queue(),
            "consensus": Queue()
        }
        
        # Service worker threads
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.workers_running = False
        
        # Service intervals (in seconds) - will be dynamically adjusted
        self.intervals = {
            "ledger": 1.0,
            "cal_engine": 2.0,
            "mining_agent": 5.0,
            "memory_mesh": 3.0,
            "consensus": 10.0
        }
        
        # Dynamic interval adjustment factors
        self.interval_adjustment_factors = {
            "ledger": 1.0,
            "cal_engine": 1.0,
            "mining_agent": 1.0,
            "memory_mesh": 1.0,
            "consensus": 1.0
        }
        
        # Last run times
        self.last_run = {
            "ledger": 0.0,
            "cal_engine": 0.0,
            "mining_agent": 0.0,
            "memory_mesh": 0.0,
            "consensus": 0.0
        }
        
        # Metrics server
        self.metrics_port = network_config.get("metrics_port", 8000)
        
        # Initialize service health status
        for service in ["ledger", "cal_engine", "mining_agent", "memory_mesh", "consensus"]:
            self.service_health[service] = ServiceHealth(
                service_name=service,
                is_healthy=True,
                last_check=time.time()
            )
        
        logger.info(f"Full Node initialized: {node_id}")
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def start_worker_threads(self):
        """Start worker threads for asynchronous service processing"""
        self.workers_running = True
        
        for service_name in self.service_queues.keys():
            thread = threading.Thread(
                target=self._service_worker,
                args=(service_name,),
                daemon=True
            )
            self.worker_threads[service_name] = thread
            thread.start()
        
        logger.info("Service worker threads started")
    
    def stop_worker_threads(self):
        """Stop worker threads"""
        self.workers_running = False
        # Add sentinel values to queues to wake up workers
        for queue in self.service_queues.values():
            queue.put(None)
        
        # Wait for threads to finish
        for thread in self.worker_threads.values():
            thread.join(timeout=5.0)
        
        logger.info("Service worker threads stopped")
    
    def _service_worker(self, service_name: str):
        """Worker thread function for processing service tasks"""
        queue = self.service_queues[service_name]
        
        while self.workers_running:
            try:
                task = queue.get(timeout=1.0)
                if task is None:  # Sentinel value to stop worker
                    break
                
                # Process the task
                if service_name == "ledger":
                    self._process_ledger_task(task)
                elif service_name == "cal_engine":
                    self._process_cal_engine_task(task)
                elif service_name == "mining_agent":
                    self._process_mining_agent_task(task)
                elif service_name == "memory_mesh":
                    self._process_memory_mesh_task(task)
                elif service_name == "consensus":
                    self._process_consensus_task(task)
                
                queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in {service_name} worker: {e}")
                self._update_service_health(service_name, False, str(e))
    
    def _process_ledger_task(self, task):
        """Process a ledger task"""
        # In a real implementation, this would process specific ledger operations
        pass
    
    def _process_cal_engine_task(self, task):
        """Process a CAL engine task"""
        pass
    
    def _process_mining_agent_task(self, task):
        """Process a mining agent task"""
        pass
    
    def _process_memory_mesh_task(self, task):
        """Process a memory mesh task"""
        pass
    
    def _process_consensus_task(self, task):
        """Process a consensus task"""
        pass
    
    def _update_service_health(self, service_name: str, is_healthy: bool, error_message: Optional[str] = None):
        """Update the health status of a service"""
        if service_name in self.service_health:
            health = self.service_health[service_name]
            health.is_healthy = is_healthy
            health.last_check = time.time()
            health.error_message = error_message
            
            if not is_healthy:
                health.restart_count += 1
                logger.warning(f"Service {service_name} marked as unhealthy: {error_message}")
            else:
                logger.info(f"Service {service_name} marked as healthy")
    
    def check_service_health(self) -> bool:
        """Check overall node health based on service health"""
        for service_health in self.service_health.values():
            if not service_health.is_healthy:
                return False
        return True
    
    def auto_restart_failed_services(self):
        """Automatically restart failed services"""
        for service_name, health in self.service_health.items():
            if not health.is_healthy and time.time() - health.last_check > 30:  # 30 second cooldown
                logger.info(f"Attempting to restart failed service: {service_name}")
                try:
                    # In a real implementation, this would restart the actual service
                    health.is_healthy = True
                    health.error_message = None
                    health.last_check = time.time()
                    logger.info(f"Service {service_name} restarted successfully")
                except Exception as e:
                    logger.error(f"Failed to restart service {service_name}: {e}")
    
    def adjust_service_intervals(self, network_state: Dict[str, Any]):
        """Dynamically adjust service intervals based on λ(t) and node load"""
        λ_t = network_state.get("lambda_t", 0.7)
        Ĉ_t = network_state.get("coherence_density", 0.8)
        
        # Adjust intervals based on lambda and coherence
        # Higher lambda (more instability) = shorter intervals (more frequent checks)
        # Lower coherence = shorter intervals (more frequent checks)
        
        lambda_factor = 1.0 / max(0.3, λ_t)  # Higher λ_t means lower factor
        coherence_factor = 1.0 / max(0.5, Ĉ_t)  # Lower Ĉ_t means higher factor
        
        adjustment_factor = (lambda_factor + coherence_factor) / 2.0
        adjustment_factor = max(0.5, min(2.0, adjustment_factor))  # Clamp between 0.5 and 2.0
        
        # Update adjustment factors for each service
        for service in self.interval_adjustment_factors.keys():
            self.interval_adjustment_factors[service] = adjustment_factor
        
        # Apply adjustments to intervals
        for service, base_interval in self.intervals.items():
            adjusted_interval = base_interval / adjustment_factor
            self.intervals[service] = max(0.1, adjusted_interval)  # Minimum 0.1 seconds
    
    async def run_layer1_ledger(self):
        """Run Layer 1 Ledger service - Transaction immutability and finality"""
        with NODE_SERVICE_DURATION.labels(service='ledger').time():
            NODE_SERVICE_CALLS.labels(service='ledger').inc()
            
            current_time = time.time()
            if current_time - self.last_run["ledger"] >= self.intervals["ledger"]:
                self.last_run["ledger"] = current_time
                
                try:
                    transactions = self.ledger.get_pending_transactions()
                    validated_transactions = self.ledger.validate_transactions(transactions)
                    self.ledger.commit_transactions(validated_transactions)
                    self._update_service_health("ledger", True)
                except Exception as e:
                    self._update_service_health("ledger", False, str(e))
                    logger.error(f"Error in Layer 1 Ledger service: {e}")
    
    async def run_cal_engine(self):
        """Run CAL Engine service - Computes local Ĉ(t), Ψ, proposes λ(t)"""
        with NODE_SERVICE_DURATION.labels(service='cal_engine').time():
            NODE_SERVICE_CALLS.labels(service='cal_engine').inc()
            
            current_time = time.time()
            if current_time - self.last_run["cal_engine"] >= self.intervals["cal_engine"]:
                self.last_run["cal_engine"] = current_time
                
                try:
                    coherence_metrics = self.cal_engine.compute_coherence_metrics()
                    psi_score = self.cal_engine.calculate_psi_score()
                    lambda_proposal = self.cal_engine.propose_lambda_t()
                    
                    # Update Prometheus metrics
                    NODE_LAMBDA_T.set(lambda_proposal)
                    NODE_COHERENCE_DENSITY.set(coherence_metrics['coherence_density'])
                    NODE_PSI_SCORE.set(psi_score)
                    
                    logger.debug(f"CAL Engine - Ĉ(t): {coherence_metrics['coherence_density']:.3f}, "
                               f"Ψ: {psi_score:.3f}, λ(t): {lambda_proposal:.3f}")
                    self._update_service_health("cal_engine", True)
                except Exception as e:
                    self._update_service_health("cal_engine", False, str(e))
                    logger.error(f"Error in CAL Engine service: {e}")
    
    async def run_mining_agent(self):
        """Run Mining Agent service - Executes CMF, submits Mint T0 transactions to Layer 1"""
        with NODE_SERVICE_DURATION.labels(service='mining_agent').time():
            NODE_SERVICE_CALLS.labels(service='mining_agent').inc()
            
            current_time = time.time()
            if current_time - self.last_run["mining_agent"] >= self.intervals["mining_agent"]:
                self.last_run["mining_agent"] = current_time
                
                try:
                    network_state = self.cal_engine.get_current_state()
                    epoch_result = self.mining_agent.run_epoch(network_state)
                    if epoch_result.should_mint:
                        mint_transaction = self.mining_agent.create_mint_transaction(epoch_result)
                        self.ledger.submit_transaction(mint_transaction)
                        logger.info(f"Mining Agent - Submitted mint transaction: {mint_transaction['id']} "
                                  f"with amount {epoch_result.mint_amount:.2f}")
                    self._update_service_health("mining_agent", True)
                except Exception as e:
                    self._update_service_health("mining_agent", False, str(e))
                    logger.error(f"Error in Mining Agent service: {e}")
    
    async def run_memory_mesh_service(self):
        """Run Memory Mesh Service - Stores, indexes, and gossips memory updates"""
        with NODE_SERVICE_DURATION.labels(service='memory_mesh').time():
            NODE_SERVICE_CALLS.labels(service='memory_mesh').inc()
            
            current_time = time.time()
            if current_time - self.last_run["memory_mesh"] >= self.intervals["memory_mesh"]:
                self.last_run["memory_mesh"] = current_time
                
                try:
                    # Process local memory updates
                    local_updates = self.memory_mesh_service.get_local_updates()
                    self.memory_mesh_service.index_updates(local_updates)
                    
                    # Participate in gossip protocol
                    network_state = self.cal_engine.get_current_state()
                    self.memory_mesh_service.participate_in_gossip(network_state)
                    
                    # Handle incoming gossip messages (mock)
                    # In a real implementation, this would receive actual messages from peers
                    self._update_service_health("memory_mesh", True)
                except Exception as e:
                    self._update_service_health("memory_mesh", False, str(e))
                    logger.error(f"Error in Memory Mesh Service: {e}")
    
    async def run_consensus_engine(self):
        """Run Consensus Engine - Executes λ(t)-attuned consensus"""
        with NODE_SERVICE_DURATION.labels(service='consensus').time():
            NODE_SERVICE_CALLS.labels(service='consensus').inc()
            
            current_time = time.time()
            if current_time - self.last_run["consensus"] >= self.intervals["consensus"]:
                self.last_run["consensus"] = current_time
                
                try:
                    # Get current network state
                    network_state = self.cal_engine.get_current_state()
                    
                    # Execute consensus round
                    consensus_round = self.consensus_engine.execute_consensus_round(network_state)
                    if consensus_round:
                        logger.info(f"Consensus Engine - Executed {consensus_round.mode.name} round "
                                  f"with {len(consensus_round.actions)} actions")
                    self._update_service_health("consensus", True)
                except Exception as e:
                    self._update_service_health("consensus", False, str(e))
                    logger.error(f"Error in Consensus Engine: {e}")
    
    async def run_node_services(self):
        """Run all node services in a continuous loop"""
        logger.info(f"Starting Full Node services for {self.node_id}")
        self.running = True
        
        # Start metrics server
        self.start_metrics_server()
        
        # Start worker threads
        self.start_worker_threads()
        
        try:
            while self.running:
                # Run all services
                await self.run_layer1_ledger()
                await self.run_cal_engine()
                await self.run_mining_agent()
                await self.run_memory_mesh_service()
                await self.run_consensus_engine()
                
                # Check overall health
                self.health_status = self.check_service_health()
                NODE_HEALTH_STATUS.set(1 if self.health_status else 0)
                
                # Auto-restart failed services
                self.auto_restart_failed_services()
                
                # Adjust service intervals based on network state
                network_state = self.cal_engine.get_current_state()
                self.adjust_service_intervals(network_state)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Full Node services interrupted by user")
        except Exception as e:
            logger.error(f"Error in Full Node services: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all node services"""
        logger.info(f"Stopping Full Node services for {self.node_id}")
        self.running = False
        self.stop_worker_threads()
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        return {
            "node_id": self.node_id,
            "running": self.running,
            "health_status": self.health_status,
            "cal_state": self.cal_engine.get_current_state(),
            "memory_stats": self.memory_mesh_service.get_memory_stats(),
            "consensus_stats": self.consensus_engine.get_consensus_stats(),
            "mining_epoch_count": self.mining_agent.epoch_count,
            "ledger_transaction_count": self.ledger.get_transaction_count(),
            "service_health": {
                service: {
                    "healthy": health.is_healthy,
                    "restarts": health.restart_count,
                    "last_error": health.error_message
                }
                for service, health in self.service_health.items()
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status"""
        return {
            "node_id": self.node_id,
            "overall_health": self.health_status,
            "timestamp": time.time(),
            "services": {
                service: {
                    "healthy": health.is_healthy,
                    "last_check": health.last_check,
                    "restart_count": health.restart_count,
                    "error": health.error_message
                }
                for service, health in self.service_health.items()
            }
        }

# Example usage and testing
async def demo_full_node():
    """Demonstrate the Full Node implementation"""
    print("🖥️ Full Node Demo")
    print("=" * 20)
    
    # Create node instance
    network_config = {
        "shard_count": 10,
        "replication_factor": 3,
        "validator_count": 5,
        "metrics_port": 8000
    }
    
    node = FullNode("node-001", network_config)
    
    # Add sample validators to consensus engine
    validators_data = [
        ("validator-1", 0.95, 10000.0),
        ("validator-2", 0.87, 8000.0),
        ("validator-3", 0.75, 12000.0),
        ("validator-4", 0.92, 9000.0),
        ("validator-5", 0.85, 7000.0),
    ]
    
    for validator_id, psi_score, stake in validators_data:
        node.consensus_engine.add_validator(validator_id, psi_score, stake)
    
    print(f"✅ Initialized Full Node with {len(validators_data)} validators")
    
    # Show initial stats
    stats = node.get_node_stats()
    print(f"📊 Initial Node Stats:")
    print(f"   • CAL State: Ĉ(t)={stats['cal_state']['coherence_density']:.3f}, "
          f"λ(t)={stats['cal_state']['lambda_t']:.3f}")
    print(f"   • Memory Updates: {stats['memory_stats']['local_updates_count']}")
    print(f"   • Validators: {stats['consensus_stats']['validators_count']}")
    
    # Run node services for a short time
    print("\n🔄 Running node services for 10 seconds...")
    
    # Create a task for the node services
    node_task = asyncio.create_task(node.run_node_services())
    
    # Let it run for 10 seconds
    await asyncio.sleep(10)
    
    # Stop the node
    node.stop()
    await node_task
    
    # Show final stats
    final_stats = node.get_node_stats()
    print(f"\n📊 Final Node Stats:")
    print(f"   • CAL State: Ĉ(t)={final_stats['cal_state']['coherence_density']:.3f}, "
          f"λ(t)={final_stats['cal_state']['lambda_t']:.3f}")
    print(f"   • Memory Updates: {final_stats['memory_stats']['local_updates_count']}")
    print(f"   • Mining Epochs: {final_stats['mining_epoch_count']}")
    print(f"   • Consensus Rounds: {final_stats['consensus_stats']['consensus_history_count']}")
    print(f"   • Ledger Transactions: {final_stats['ledger_transaction_count']}")
    
    # Show health status
    health = node.get_health_status()
    print(f"\n🩺 Health Status:")
    print(f"   • Overall Health: {'Healthy' if health['overall_health'] else 'Unhealthy'}")
    for service, status in health['services'].items():
        print(f"   • {service}: {'Healthy' if status['healthy'] else 'Unhealthy'}")
    
    print("\n✅ Full Node demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_full_node())