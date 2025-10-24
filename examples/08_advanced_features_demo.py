"""
AEGIS Advanced Features Demonstration
Example application showcasing dynamic validator selection, adaptive timeouts, and error handling
"""

import asyncio
import time
import random
from typing import List, Dict, Any
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.blockchain.advanced_consensus import AdvancedConsensusFeatures
from src.aegis.blockchain.consensus_protocol import HybridConsensus
from src.aegis.core.error_handling import ErrorRecoveryManager, ErrorSeverity
from src.aegis.security.middleware import SecurityMiddleware
from src.aegis.monitoring.consensus_visualizer import ConsensusVisualizer, ConsensusMetrics
from cryptography.hazmat.primitives.asymmetric import ed25519

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AEGISDemoNode:
    """Demo node showcasing advanced AEGIS features"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Create components
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.consensus = HybridConsensus(
            node_id=node_id,
            private_key=self.private_key
        )
        self.advanced_consensus = AdvancedConsensusFeatures(self.consensus)
        self.error_manager = ErrorRecoveryManager()
        self.security = SecurityMiddleware()
        self.visualizer = ConsensusVisualizer(update_interval=1.0)
        
        # Register components with error manager
        self.error_manager.register_component("consensus")
        self.error_manager.register_component("network")
        self.error_manager.register_component("storage")
        
        # Node state
        self.is_running = False
        self.consensus_round = 0
        self.performance_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "errors_handled": 0,
            "security_checks": 0
        }
    
    async def start(self):
        """Start the demo node"""
        logger.info(f"Starting AEGIS Demo Node: {self.node_id}")
        self.is_running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._consensus_simulation()),
            asyncio.create_task(self._performance_monitoring()),
            asyncio.create_task(self._error_simulation())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Demo node shutting down...")
        finally:
            self.is_running = False
    
    async def _consensus_simulation(self):
        """Simulate consensus operations"""
        while self.is_running:
            try:
                self.consensus_round += 1
                logger.info(f"Starting consensus round {self.consensus_round}")
                
                # Simulate dynamic validator selection
                await self._simulate_validator_selection()
                
                # Simulate consensus phases with adaptive timeouts
                await self._simulate_consensus_phases()
                
                # Update performance stats
                self.performance_stats["total_validations"] += 10
                self.performance_stats["successful_validations"] += 9  # 90% success rate
                
                # Wait before next round
                await asyncio.sleep(2)
                
            except Exception as e:
                await self._handle_error("consensus", e, ErrorSeverity.MEDIUM)
    
    async def _simulate_validator_selection(self):
        """Simulate dynamic validator selection"""
        logger.info("Performing dynamic validator selection...")
        
        # Create mock validators
        mock_validators = [f"validator_{i:03d}" for i in range(50)]
        
        # Record performance metrics for validators
        for validator in mock_validators:
            # Security check
            allowed, message = self.security.check_request_security(
                client_id=validator,
                endpoint="/consensus/validate",
                params={"round": self.consensus_round}
            )
            
            self.performance_stats["security_checks"] += 1
            
            if allowed:
                # Record validation results
                success = random.choice([True, True, True, False])  # 75% success rate
                response_time = random.uniform(0.1, 2.0)
                
                self.advanced_consensus.record_validation_result(
                    node_id=validator,
                    success=success,
                    response_time=response_time,
                    message_type="PREPARE"
                )
        
        # Select optimal validators
        selected_validators = await self.advanced_consensus.select_optimal_validators(
            num_validators=15,
            exclude_nodes={self.node_id}
        )
        
        logger.info(f"Selected {len(selected_validators)} validators: {selected_validators[:5]}...")
        
        # Verify selection quality
        performance_report = self.advanced_consensus.get_performance_report()
        validator_perf = performance_report.get("validator_performance", {})
        
        # Log top performing validators
        top_validators = sorted(
            [(node_id, metrics.get("reliability_score", 0)) 
             for node_id, metrics in validator_perf.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        logger.info("Top performing validators:")
        for node_id, score in top_validators:
            logger.info(f"  {node_id}: {score:.2f}")
    
    async def _simulate_consensus_phases(self):
        """Simulate consensus phases with adaptive timeouts"""
        phases = ["PROPOSING", "PREPARING", "COMMITTING", "FINALIZING"]
        
        for phase in phases:
            logger.info(f"Executing consensus phase: {phase}")
            
            # Start phase timer
            self.advanced_consensus.start_consensus_phase_timer(phase)
            
            # Simulate phase work with realistic delays
            phase_delay = self.advanced_consensus.get_adaptive_timeout(phase)
            actual_delay = phase_delay * random.uniform(0.8, 1.2)  # ±20% variation
            
            # Simulate network latency and processing time
            await asyncio.sleep(actual_delay / 1000)  # Convert to seconds
            
            # Simulate occasional phase failure
            phase_success = random.random() > 0.05  # 95% success rate
            
            # Stop phase timer
            self.advanced_consensus.stop_consensus_phase_timer(phase, phase_success)
            
            # Record message latency
            self.advanced_consensus.record_message_latency(
                message_type=phase,
                latency=actual_delay
            )
            
            if not phase_success:
                logger.warning(f"Consensus phase {phase} failed")
                await self._handle_error("consensus", 
                                       Exception(f"Phase {phase} failed"), 
                                       ErrorSeverity.MEDIUM)
    
    async def _performance_monitoring(self):
        """Monitor and visualize performance metrics"""
        while self.is_running:
            try:
                # Create consensus metrics
                metrics = ConsensusMetrics(
                    timestamp=time.time(),
                    view_number=self.consensus_round,
                    sequence_number=self.consensus_round * 10,
                    active_validators=25,
                    total_nodes=50,
                    consensus_state="PREPARING",
                    proposal_count=random.randint(5, 15),
                    prepare_count=random.randint(10, 25),
                    commit_count=random.randint(8, 20),
                    avg_response_time=random.uniform(50, 150),
                    success_rate=random.uniform(0.9, 0.99),
                    throughput=random.uniform(200, 800),
                    latency=random.uniform(30, 100)
                )
                
                # Update visualizer
                self.visualizer.update_metrics(metrics)
                
                # Log performance summary
                if self.consensus_round % 10 == 0:
                    report = self.advanced_consensus.get_performance_report()
                    timeout_stats = report.get("timeout_statistics", {})
                    
                    logger.info(f"Performance Report - Round {self.consensus_round}:")
                    logger.info(f"  Throughput: {metrics.throughput:.1f} TPS")
                    logger.info(f"  Latency: {metrics.latency:.1f} ms")
                    logger.info(f"  Success Rate: {metrics.success_rate:.2%}")
                    
                    if timeout_stats:
                        for phase, stats in timeout_stats.items():
                            logger.info(f"  {phase} timeout: {stats['current_timeout']:.2f}s")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                await self._handle_error("monitoring", e, ErrorSeverity.LOW)
    
    async def _error_simulation(self):
        """Simulate and handle various errors"""
        error_count = 0
        while self.is_running and error_count < 20:  # Limit errors for demo
            try:
                # Wait random time between errors
                await asyncio.sleep(random.uniform(3, 15))
                
                # Simulate different types of errors
                error_type = random.choice(["network", "storage", "validation"])
                
                if error_type == "network":
                    error = ConnectionError("Network timeout")
                    severity = ErrorSeverity.MEDIUM
                elif error_type == "storage":
                    error = IOError("Disk full")
                    severity = ErrorSeverity.HIGH
                else:  # validation
                    error = ValueError("Invalid transaction")
                    severity = ErrorSeverity.LOW
                
                # Handle error
                await self._handle_error(error_type, error, severity)
                error_count += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in error simulation: {e}")
    
    async def _handle_error(self, component: str, error: Exception, severity: ErrorSeverity):
        """Handle errors with recovery"""
        try:
            # Handle error with recovery manager
            recovery_success = await self.error_manager.handle_error(
                component=component,
                error=error,
                severity=severity,
                context={"round": self.consensus_round, "node": self.node_id}
            )
            
            self.performance_stats["errors_handled"] += 1
            
            if recovery_success:
                logger.info(f"Successfully recovered from {component} error: {error}")
            else:
                logger.warning(f"Failed to recover from {component} error: {error}")
                
        except Exception as e:
            logger.error(f"Error handling failed: {e}")


async def main():
    """Main demo function"""
    logger.info("Starting AEGIS Advanced Features Demonstration")
    
    # Create demo node
    node = AEGISDemoNode("demo_node_001")
    
    # Setup signal handler for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        node.is_running = False
    
    # Run demo for 60 seconds
    try:
        demo_task = asyncio.create_task(node.start())
        await asyncio.wait_for(demo_task, timeout=60.0)
    except asyncio.TimeoutError:
        logger.info("Demo completed after 60 seconds")
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        node.is_running = False
        logger.info("Demo shutdown complete")
        
        # Print final statistics
        logger.info("Final Performance Statistics:")
        for key, value in node.performance_stats.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
