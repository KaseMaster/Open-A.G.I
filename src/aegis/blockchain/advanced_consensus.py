"""
AEGIS Advanced Consensus Features
Dynamic validator selection and adaptive timeouts for improved consensus performance
"""

import time
import asyncio
import random
import statistics
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidatorMetrics:
    """Metrics for a consensus validator"""
    node_id: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 1.0
    last_participation: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_validations: int = 0
    successful_validations: int = 0
    avg_response_time: float = 0.0
    reliability_score: float = 1.0


@dataclass
class AdaptiveTimeoutConfig:
    """Configuration for adaptive timeouts"""
    base_timeout: float = 5.0  # Base timeout in seconds
    min_timeout: float = 1.0   # Minimum timeout
    max_timeout: float = 30.0  # Maximum timeout
    adjustment_factor: float = 0.1  # How quickly to adjust
    smoothing_factor: float = 0.3   # Exponential smoothing


class DynamicValidatorSelector:
    """Dynamic validator selection based on performance and reputation"""
    
    def __init__(self, consensus_instance):
        self.consensus = consensus_instance
        self.validator_metrics: Dict[str, ValidatorMetrics] = {}
        self.recent_selections: deque = deque(maxlen=50)
        self.selection_history: List[Dict[str, Any]] = []
        
        # Initialize metrics for all known nodes
        if hasattr(self.consensus, 'network_topology') and hasattr(self.consensus.network_topology, 'nodes'):
            for node_id in self.consensus.network_topology.nodes:
                self.validator_metrics[node_id] = ValidatorMetrics(node_id=node_id)
    
    def select_validators(
        self,
        num_validators: int,
        exclude_nodes: Optional[Set[str]] = None
    ) -> List[str]:
        """Select validators based on performance metrics and reputation"""
        exclude_nodes = exclude_nodes or set()
        
        # Get all eligible nodes
        eligible_nodes = []
        if hasattr(self.consensus, 'network_topology') and hasattr(self.consensus.network_topology, 'nodes'):
            eligible_nodes = [
                node_id for node_id in self.consensus.network_topology.nodes
                if node_id not in exclude_nodes and node_id != self.consensus.node_id
            ]
        
        if len(eligible_nodes) <= num_validators:
            return eligible_nodes
        
        # Calculate selection scores for each node
        scores = {}
        for node_id in eligible_nodes:
            score = self._calculate_selection_score(node_id)
            scores[node_id] = score
        
        # Sort by score (higher is better)
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top validators
        selected = [node_id for node_id, score in sorted_nodes[:num_validators]]
        
        # Record selection
        selection_record = {
            "timestamp": time.time(),
            "selected_validators": selected,
            "scores": scores,
            "total_eligible": len(eligible_nodes)
        }
        self.selection_history.append(selection_record)
        self.recent_selections.append(selected)
        
        logger.info(f"Selected {len(selected)} validators: {selected}")
        return selected
    
    def _calculate_selection_score(self, node_id: str) -> float:
        """Calculate selection score for a node"""
        if node_id not in self.validator_metrics:
            self.validator_metrics[node_id] = ValidatorMetrics(node_id=node_id)
        
        metrics = self.validator_metrics[node_id]
        
        # Base score components
        response_time_score = self._calculate_response_time_score(metrics)
        success_rate_score = metrics.success_rate
        reliability_score = metrics.reliability_score
        recency_score = self._calculate_recency_score(metrics.last_participation)
        
        # Weighted combination
        score = (
            response_time_score * 0.3 +
            success_rate_score * 0.3 +
            reliability_score * 0.2 +
            recency_score * 0.2
        )
        
        # Boost for nodes that haven't been selected recently
        if self._is_underrepresented(node_id):
            score *= 1.2
        
        return score
    
    def _calculate_response_time_score(self, metrics: ValidatorMetrics) -> float:
        """Calculate score based on response times"""
        if not metrics.response_times:
            return 1.0
        
        avg_response = statistics.mean(metrics.response_times)
        # Normalize to 0-1 range (faster is better)
        max_expected = 10.0  # Expected max response time
        return max(0.0, min(1.0, max_expected / (avg_response + 1.0)))
    
    def _calculate_recency_score(self, last_participation: float) -> float:
        """Calculate score based on recent participation"""
        time_since_participation = time.time() - last_participation
        # Normalize to 0-1 range (more recent is better)
        max_time = 300.0  # 5 minutes
        return max(0.0, min(1.0, 1.0 - (time_since_participation / max_time)))
    
    def _is_underrepresented(self, node_id: str) -> bool:
        """Check if node has been selected recently"""
        if not self.recent_selections:
            return False
        
        # Count how many times node was selected in recent rounds
        recent_count = sum(
            1 for selection in self.recent_selections
            if node_id in selection
        )
        
        # If selected less than 20% of recent rounds, consider underrepresented
        total_recent_rounds = len(self.recent_selections)
        return (recent_count / total_recent_rounds) < 0.2 if total_recent_rounds > 0 else True
    
    def record_validation_result(
        self,
        node_id: str,
        success: bool,
        response_time: float,
        message_type: str
    ):
        """Record the result of a validation"""
        if node_id not in self.validator_metrics:
            self.validator_metrics[node_id] = ValidatorMetrics(node_id=node_id)
        
        metrics = self.validator_metrics[node_id]
        
        # Update metrics
        metrics.response_times.append(response_time)
        metrics.total_validations += 1
        metrics.last_participation = time.time()
        
        if success:
            metrics.successful_validations += 1
            metrics.consecutive_failures = 0
        else:
            metrics.consecutive_failures += 1
        
        # Update success rate with exponential smoothing
        if metrics.total_validations > 1:
            alpha = 0.1  # Smoothing factor
            metrics.success_rate = (
                alpha * (1.0 if success else 0.0) +
                (1 - alpha) * metrics.success_rate
            )
        else:
            metrics.success_rate = 1.0 if success else 0.0
        
        # Update average response time
        if metrics.response_times:
            metrics.avg_response_time = statistics.mean(metrics.response_times)
        
        # Update reliability score
        self._update_reliability_score(metrics, success)
    
    def _update_reliability_score(self, metrics: ValidatorMetrics, success: bool):
        """Update reliability score based on recent performance"""
        # Consider last 20 validations
        recent_window = min(20, metrics.total_validations)
        if recent_window > 0:
            recent_success_rate = metrics.successful_validations / metrics.total_validations
            # Blend with long-term success rate
            blended_rate = (
                0.7 * recent_success_rate +
                0.3 * (metrics.success_rate if metrics.total_validations > 1 else (1.0 if success else 0.0))
            )
            metrics.reliability_score = blended_rate
    
    def get_validator_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all validators"""
        report = {}
        
        for node_id, metrics in self.validator_metrics.items():
            report[node_id] = {
                "success_rate": metrics.success_rate,
                "avg_response_time": metrics.avg_response_time,
                "total_validations": metrics.total_validations,
                "reliability_score": metrics.reliability_score,
                "consecutive_failures": metrics.consecutive_failures,
                "last_participation": metrics.last_participation
            }
        
        return report


class AdaptiveTimeoutManager:
    """Adaptive timeout management for consensus phases"""
    
    def __init__(self, consensus_instance):
        self.consensus = consensus_instance
        self.config = AdaptiveTimeoutConfig()
        self.phase_timeouts: Dict[str, float] = {
            "PROPOSING": self.config.base_timeout,
            "PREPARING": self.config.base_timeout,
            "COMMITTING": self.config.base_timeout,
            "FINALIZING": self.config.base_timeout * 0.5  # Shorter for finalizing
        }
        
        self.timeout_history: Dict[str, List[float]] = defaultdict(list)
        self.current_phase_start: Optional[float] = None
        self.current_phase: Optional[str] = None
    
    def start_phase_timer(self, phase: str):
        """Start timer for a consensus phase"""
        self.current_phase = phase
        self.current_phase_start = time.time()
    
    def stop_phase_timer(self, phase: str, success: bool = True):
        """Stop timer and adjust timeout based on performance"""
        if self.current_phase != phase or not self.current_phase_start:
            return
        
        elapsed_time = time.time() - self.current_phase_start
        self.current_phase_start = None
        self.current_phase = None
        
        # Record timeout
        self.timeout_history[phase].append(elapsed_time)
        
        # Adjust timeout if successful
        if success:
            self._adjust_timeout(phase, elapsed_time)
    
    def _adjust_timeout(self, phase: str, actual_time: float):
        """Adjust timeout based on actual performance"""
        current_timeout = self.phase_timeouts[phase]
        
        # Exponential smoothing
        smoothed_time = (
            self.config.smoothing_factor * actual_time +
            (1 - self.config.smoothing_factor) * current_timeout
        )
        
        # Apply adjustment
        adjustment = self.config.adjustment_factor * (smoothed_time - current_timeout)
        new_timeout = current_timeout + adjustment
        
        # Clamp to bounds
        new_timeout = max(
            self.config.min_timeout,
            min(self.config.max_timeout, new_timeout)
        )
        
        self.phase_timeouts[phase] = new_timeout
        logger.debug(f"Adjusted {phase} timeout: {current_timeout:.2f}s -> {new_timeout:.2f}s")
    
    def get_timeout(self, phase: str) -> float:
        """Get timeout for a specific phase"""
        return self.phase_timeouts.get(phase, self.config.base_timeout)
    
    def get_adaptive_timeout_task(
        self,
        phase: str,
        callback: Any
    ) -> asyncio.Task:
        """Get adaptive timeout task for a phase"""
        timeout = self.get_timeout(phase)
        
        async def timeout_task():
            await asyncio.sleep(timeout)
            await callback()
        
        return asyncio.create_task(timeout_task())
    
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get timeout adjustment statistics"""
        stats = {}
        
        for phase, timeouts in self.timeout_history.items():
            if timeouts:
                stats[phase] = {
                    "current_timeout": self.phase_timeouts[phase],
                    "avg_actual_time": statistics.mean(timeouts),
                    "min_actual_time": min(timeouts),
                    "max_actual_time": max(timeouts),
                    "total_phases": len(timeouts)
                }
        
        return stats


class AdvancedConsensusFeatures:
    """Advanced consensus features integration"""
    
    def __init__(self, consensus_instance):
        self.consensus = consensus_instance
        self.validator_selector = DynamicValidatorSelector(consensus_instance)
        self.timeout_manager = AdaptiveTimeoutManager(consensus_instance)
        
        # Track consensus performance
        self.consensus_rounds: List[Dict[str, Any]] = []
        self.message_latencies: Dict[str, List[float]] = defaultdict(list)
    
    async def select_optimal_validators(
        self,
        num_validators: int,
        exclude_nodes: Optional[Set[str]] = None
    ) -> List[str]:
        """Select optimal validators for current consensus round"""
        return self.validator_selector.select_validators(num_validators, exclude_nodes)
    
    def start_consensus_phase_timer(self, phase: str):
        """Start timer for consensus phase"""
        self.timeout_manager.start_phase_timer(phase)
    
    def stop_consensus_phase_timer(self, phase: str, success: bool = True):
        """Stop timer for consensus phase"""
        self.timeout_manager.stop_phase_timer(phase, success)
    
    def get_adaptive_timeout(self, phase: str) -> float:
        """Get adaptive timeout for phase"""
        return self.timeout_manager.get_timeout(phase)
    
    def record_validation_result(
        self,
        node_id: str,
        success: bool,
        response_time: float,
        message_type: str
    ):
        """Record validation result for performance tracking"""
        self.validator_selector.record_validation_result(
            node_id, success, response_time, message_type
        )
    
    def record_message_latency(self, message_type: str, latency: float):
        """Record message latency for performance analysis"""
        self.message_latencies[message_type].append(latency)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "validator_performance": self.validator_selector.get_validator_performance_report(),
            "timeout_statistics": self.timeout_manager.get_timeout_statistics(),
            "message_latencies": {
                msg_type: {
                    "avg": statistics.mean(latencies) if latencies else 0,
                    "min": min(latencies) if latencies else 0,
                    "max": max(latencies) if latencies else 0,
                    "count": len(latencies)
                }
                for msg_type, latencies in self.message_latencies.items()
            },
            "total_consensus_rounds": len(self.consensus_rounds)
        }
    
    async def handle_timeout(self, phase: str):
        """Handle consensus phase timeout"""
        logger.warning(f"Consensus phase {phase} timed out")
        
        # Record timeout
        self.stop_consensus_phase_timer(phase, success=False)
        
        # Trigger view change if needed
        if phase in ["PROPOSING", "PREPARING"]:
            await self._trigger_view_change()
    
    async def _trigger_view_change(self):
        """Trigger view change for leader failure recovery"""
        logger.info("Triggering view change due to timeout")
        
        # This would normally send a view change message
        # For now, we'll just log it
        pass
