"""
AEGIS Comprehensive Error Handling and Recovery
Robust error handling with automatic recovery for all system components
"""

import asyncio
import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    DEGRADED = "degraded"
    FAILSAFE = "failsafe"


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_id: str
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    timestamp: float
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_name: str
    status: str  # "healthy", "degraded", "failed", "recovering"
    error_count: int = 0
    last_error: Optional[float] = None
    uptime: float = field(default_factory=time.time)
    recovery_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ErrorRecoveryManager:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history: deque = deque(maxlen=max_error_history)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_implementations: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Error patterns that require immediate attention
        self.critical_error_patterns = [
            "out of memory",
            "segmentation fault",
            "permission denied",
            "connection refused",
            "database locked"
        ]
        
        # Initialize background tasks
        self.background_tasks = set()
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_task)
        health_task.add_done_callback(self.background_tasks.discard)
        
        # Error cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self.background_tasks.discard)
    
    async def _health_check_loop(self):
        """Periodic health checks"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Cleanup old error records"""
        while True:
            try:
                self._cleanup_old_errors()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on all components"""
        current_time = time.time()
        
        for component_name, health in self.component_health.items():
            # Check for degraded components that need recovery
            if health.status == "degraded" and health.last_error:
                if current_time - health.last_error > 300:  # 5 minutes
                    await self._attempt_recovery(component_name)
            
            # Check for components that have been healthy for a long time
            if health.status == "healthy":
                health.uptime = current_time - health.uptime
    
    def _cleanup_old_errors(self):
        """Remove resolved errors older than 1 hour"""
        current_time = time.time()
        cutoff_time = current_time - 3600
        
        # Remove old resolved errors
        while (self.error_history and 
               self.error_history[0].resolved and 
               self.error_history[0].resolution_time and
               self.error_history[0].resolution_time < cutoff_time):
            self.error_history.popleft()
    
    def register_component(self, component_name: str):
        """Register a component for health monitoring"""
        if component_name not in self.component_health:
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status="healthy"
            )
    
    def register_recovery_strategy(self, error_pattern: str, strategy: Callable):
        """Register a recovery strategy for specific error patterns"""
        self.recovery_strategies[error_pattern] = strategy
    
    def register_fallback(self, component: str, fallback_impl: Callable):
        """Register a fallback implementation for a component"""
        self.fallback_implementations[component] = fallback_impl
    
    def register_circuit_breaker(self, component: str, breaker: 'CircuitBreaker'):
        """Register a circuit breaker for a component"""
        self.circuit_breakers[component] = breaker
    
    async def handle_error(
        self,
        component: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle an error with appropriate recovery strategy"""
        error_id = hashlib.sha256(
            f"{component}_{str(error)}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            component=component,
            error_type=type(error).__name__,
            severity=severity,
            message=str(error),
            timestamp=time.time(),
            stack_trace=traceback.format_exc() if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        )
        
        # Add to history
        self.error_history.append(error_record)
        
        # Update component health
        self._update_component_health(component, error_record)
        
        # Log error
        self._log_error(error_record)
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error_record, context)
        error_record.recovery_strategy = strategy
        
        # Execute recovery
        recovery_success = await self._execute_recovery(component, error_record, strategy, context)
        
        # Update resolution status
        if recovery_success:
            error_record.resolved = True
            error_record.resolution_time = time.time()
            logger.info(f"Error {error_id} resolved successfully")
        else:
            logger.warning(f"Failed to resolve error {error_id}")
        
        return recovery_success
    
    def _update_component_health(self, component: str, error_record: ErrorRecord):
        """Update health status of a component"""
        if component not in self.component_health:
            self.component_health[component] = ComponentHealth(
                component_name=component,
                status="healthy"
            )
        
        health = self.component_health[component]
        health.error_count += 1
        health.last_error = error_record.timestamp
        
        # Update status based on severity and error count
        if error_record.severity == ErrorSeverity.CRITICAL:
            health.status = "failed"
        elif error_record.severity == ErrorSeverity.HIGH and health.error_count > 5:
            health.status = "degraded"
        elif error_record.severity == ErrorSeverity.MEDIUM and health.error_count > 10:
            health.status = "degraded"
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_message = (
            f"Error [{error_record.error_id}] in {error_record.component}: "
            f"{error_record.message} (Severity: {error_record.severity.value})"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _determine_recovery_strategy(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryStrategy:
        """Determine appropriate recovery strategy for an error"""
        error_message = error_record.message.lower()
        
        # Check for critical patterns
        for pattern in self.critical_error_patterns:
            if pattern in error_message:
                return RecoveryStrategy.FAILSAFE
        
        # Check for specific recovery strategies
        for pattern, strategy in self.recovery_strategies.items():
            if pattern in error_message:
                if callable(strategy):
                    # If strategy is a function, execute it to get the actual strategy
                    try:
                        return strategy(error_record, context)
                    except Exception:
                        pass
                elif isinstance(strategy, RecoveryStrategy):
                    return strategy
        
        # Default strategies based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAILSAFE
        elif error_record.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.RESTART
        elif error_record.severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.RETRY
    
    async def _execute_recovery(
        self,
        component: str,
        error_record: ErrorRecord,
        strategy: RecoveryStrategy,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Execute recovery strategy"""
        max_attempts = 3
        attempt_delay = 1.0
        
        for attempt in range(max_attempts):
            error_record.recovery_attempts += 1
            
            try:
                if strategy == RecoveryStrategy.RETRY:
                    success = await self._retry_operation(component, error_record, context)
                elif strategy == RecoveryStrategy.FALLBACK:
                    success = await self._use_fallback(component, error_record, context)
                elif strategy == RecoveryStrategy.RESTART:
                    success = await self._restart_component(component, error_record, context)
                elif strategy == RecoveryStrategy.DEGRADED:
                    success = await self._degraded_mode(component, error_record, context)
                elif strategy == RecoveryStrategy.FAILSAFE:
                    success = await self._failsafe_mode(component, error_record, context)
                else:
                    success = False
                
                if success:
                    # Update component health on successful recovery
                    if component in self.component_health:
                        health = self.component_health[component]
                        health.recovery_count += 1
                        if health.status != "failed":
                            health.status = "healthy"
                            health.error_count = max(0, health.error_count - 1)
                    
                    return True
                
            except Exception as e:
                logger.error(f"Recovery attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(attempt_delay * (2 ** attempt))  # Exponential backoff
        
        return False
    
    async def _retry_operation(
        self,
        component: str,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Retry the failed operation"""
        logger.info(f"Retrying operation for component {component}")
        
        # Check circuit breaker
        if component in self.circuit_breakers:
            breaker = self.circuit_breakers[component]
            if not breaker.allow_request():
                logger.warning(f"Circuit breaker open for {component}")
                return False
        
        # In a real implementation, this would retry the specific operation
        # For now, we'll simulate a successful retry
        return True
    
    async def _use_fallback(
        self,
        component: str,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Use fallback implementation"""
        if component not in self.fallback_implementations:
            logger.warning(f"No fallback available for component {component}")
            return False
        
        logger.info(f"Using fallback implementation for {component}")
        
        try:
            fallback_func = self.fallback_implementations[component]
            # Execute fallback (would be component-specific in real implementation)
            result = fallback_func(context)
            return result is not None
        except Exception as e:
            logger.error(f"Fallback implementation failed: {e}")
            return False
    
    async def _restart_component(
        self,
        component: str,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Restart a component"""
        logger.info(f"Restarting component {component}")
        
        # Update health status
        if component in self.component_health:
            self.component_health[component].status = "recovering"
        
        # In a real implementation, this would restart the actual component
        # For now, we'll simulate a successful restart
        await asyncio.sleep(1)  # Simulate restart time
        
        return True
    
    async def _degraded_mode(
        self,
        component: str,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Operate in degraded mode"""
        logger.info(f"Switching {component} to degraded mode")
        
        # Update health status
        if component in self.component_health:
            self.component_health[component].status = "degraded"
        
        # In a real implementation, this would switch to a degraded functionality
        return True
    
    async def _failsafe_mode(
        self,
        component: str,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Enter failsafe mode"""
        logger.critical(f"Entering failsafe mode for {component}")
        
        # Update health status
        if component in self.component_health:
            self.component_health[component].status = "failed"
        
        # In a real implementation, this would shut down non-critical functions
        # and operate in a minimal safe state
        return True
    
    async def _attempt_recovery(self, component_name: str):
        """Attempt to recover a degraded component"""
        logger.info(f"Attempting recovery for {component_name}")
        
        if component_name in self.component_health:
            health = self.component_health[component_name]
            health.status = "recovering"
            
            # Simulate recovery process
            await asyncio.sleep(2)
            
            # Assume recovery successful
            health.status = "healthy"
            health.error_count = max(0, health.error_count - 5)  # Reduce error count
            logger.info(f"Recovery successful for {component_name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and metrics"""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Group errors by component and severity
        component_errors = defaultdict(lambda: defaultdict(int))
        severity_counts = defaultdict(int)
        recent_errors = []
        
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        for error_record in self.error_history:
            component_errors[error_record.component][error_record.severity.value] += 1
            severity_counts[error_record.severity.value] += 1
            
            # Collect recent errors (last hour)
            if error_record.timestamp > one_hour_ago:
                recent_errors.append({
                    "id": error_record.error_id,
                    "component": error_record.component,
                    "type": error_record.error_type,
                    "severity": error_record.severity.value,
                    "message": error_record.message,
                    "timestamp": error_record.timestamp,
                    "resolved": error_record.resolved
                })
        
        return {
            "total_errors": len(self.error_history),
            "component_errors": dict(component_errors),
            "severity_distribution": dict(severity_counts),
            "recent_errors": recent_errors,
            "component_health": {
                name: {
                    "status": health.status,
                    "error_count": health.error_count,
                    "recovery_count": health.recovery_count,
                    "last_error": health.last_error
                }
                for name, health in self.component_health.items()
            }
        }
    
    def clear_error_history(self, component: Optional[str] = None):
        """Clear error history, optionally for a specific component"""
        if component:
            # Remove errors for specific component
            self.error_history = deque([
                error for error in self.error_history
                if error.component != component
            ], maxlen=self.max_error_history)
            
            # Reset component health
            if component in self.component_health:
                self.component_health[component] = ComponentHealth(
                    component_name=component,
                    status="healthy"
                )
        else:
            # Clear all errors
            self.error_history.clear()
            for health in self.component_health.values():
                health.error_count = 0
                health.last_error = None
                health.status = "healthy"


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # "closed", "open", "half_open"
    
    def allow_request(self) -> bool:
        """Check if a request is allowed through the circuit breaker"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if timeout has passed
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.timeout):
                self.state = "half_open"
                return True
            return False
        elif self.state == "half_open":
            # Allow one request to test
            return True
        return False
    
    def record_success(self):
        """Record a successful request"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
    
    def record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def record_test_result(self, success: bool):
        """Record the result of a test request in half-open state"""
        if success:
            self.record_success()
        else:
            self.record_failure()


# Example usage and integration
if __name__ == "__main__":
    # Create error recovery manager
    error_manager = ErrorRecoveryManager()
    
    # Register components
    error_manager.register_component("blockchain")
    error_manager.register_component("ml_engine")
    error_manager.register_component("network")
    
    # Register circuit breakers
    blockchain_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)
    error_manager.register_circuit_breaker("blockchain", blockchain_breaker)
    
    # Example error handling
    async def simulate_error():
        try:
            # Simulate an error
            raise ValueError("Simulated blockchain error")
        except Exception as e:
            await error_manager.handle_error(
                component="blockchain",
                error=e,
                severity=ErrorSeverity.MEDIUM
            )
    
    # Run simulation
    asyncio.run(simulate_error())
