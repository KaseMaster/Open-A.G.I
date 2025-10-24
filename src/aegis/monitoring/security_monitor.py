"""
AEGIS Advanced Security Monitoring and Alerting
Real-time monitoring and alerting for advanced security features
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aegis.security.advanced_crypto import AdvancedSecurityManager, SecurityFeature
from src.aegis.core.performance_optimizer import PerformanceOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event for monitoring"""
    event_id: str
    event_type: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: float
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class SecurityMetric:
    """Security metric for monitoring"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[float] = None
    alert_triggered: bool = False


class SecurityAlert:
    """Security alert notification"""
    
    def __init__(
        self,
        alert_id: str,
        alert_type: str,
        severity: str,
        message: str,
        timestamp: float,
        source: str,
        details: Dict[str, Any],
        resolved: bool = False
    ):
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.timestamp = timestamp
        self.source = source
        self.details = details
        self.resolved = resolved
        self.created_at = timestamp
        self.resolved_at: Optional[float] = None


class AlertRule:
    """Rule for triggering security alerts"""
    
    def __init__(
        self,
        rule_id: str,
        metric_name: str,
        condition: str,  # ">", "<", ">=", "<=", "==", "!="
        threshold: float,
        severity: str,
        message_template: str,
        enabled: bool = True,
        cooldown_seconds: float = 300.0  # 5 minutes
    ):
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.message_template = message_template
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time: Optional[float] = None
    
    def should_trigger(self, metric_value: float, current_time: float) -> bool:
        """Check if alert rule should trigger"""
        if not self.enabled:
            return False
        
        # Check cooldown period
        if (self.last_trigger_time and 
            current_time - self.last_trigger_time < self.cooldown_seconds):
            return False
        
        # Evaluate condition
        if self.condition == ">":
            return metric_value > self.threshold
        elif self.condition == "<":
            return metric_value < self.threshold
        elif self.condition == ">=":
            return metric_value >= self.threshold
        elif self.condition == "<=":
            return metric_value <= self.threshold
        elif self.condition == "==":
            return metric_value == self.threshold
        elif self.condition == "!=":
            return metric_value != self.threshold
        
        return False
    
    def trigger(self, metric_value: float, current_time: float) -> SecurityAlert:
        """Trigger alert for this rule"""
        self.last_trigger_time = current_time
        
        message = self.message_template.format(
            metric_name=self.metric_name,
            metric_value=metric_value,
            threshold=self.threshold
        )
        
        return SecurityAlert(
            alert_id=f"alert_{self.rule_id}_{int(current_time)}",
            alert_type=self.rule_id,
            severity=self.severity,
            message=message,
            timestamp=current_time,
            source="alert_rule",
            details={
                "metric_name": self.metric_name,
                "metric_value": metric_value,
                "threshold": self.threshold,
                "condition": self.condition
            }
        )


class SecurityMonitor:
    """Advanced security monitoring and alerting system"""
    
    def __init__(self, security_manager: AdvancedSecurityManager):
        self.security_manager = security_manager
        self.events: deque = deque(maxlen=10000)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Callbacks for events and alerts
        self.event_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Performance optimizer for metrics collection
        self.performance_optimizer = PerformanceOptimizer()
        
        # Background tasks
        self.background_tasks = set()
        self.monitoring_active = False
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default security alert rules"""
        # ZK proof failure rate alert
        self.alert_rules["zk_proof_failure_rate"] = AlertRule(
            rule_id="zk_proof_failure_rate",
            metric_name="zk_proof_failure_rate",
            condition=">",
            threshold=0.05,  # 5% failure rate
            severity="warning",
            message_template="ZK proof failure rate is high: {metric_value:.2%} (threshold: {threshold:.2%})",
            cooldown_seconds=600.0  # 10 minutes
        )
        
        # Homomorphic encryption error rate alert
        self.alert_rules["homomorphic_error_rate"] = AlertRule(
            rule_id="homomorphic_error_rate",
            metric_name="homomorphic_encryption_error_rate",
            condition=">",
            threshold=0.01,  # 1% error rate
            severity="error",
            message_template="Homomorphic encryption error rate is high: {metric_value:.2%} (threshold: {threshold:.2%})",
            cooldown_seconds=300.0  # 5 minutes
        )
        
        # SMC participant failure alert
        self.alert_rules["smc_participant_failure"] = AlertRule(
            rule_id="smc_participant_failure",
            metric_name="smc_participant_failure_rate",
            condition=">",
            threshold=0.10,  # 10% failure rate
            severity="critical",
            message_template="SMC participant failure rate is critical: {metric_value:.2%} (threshold: {threshold:.2%})",
            cooldown_seconds=900.0  # 15 minutes
        )
        
        # Differential privacy budget exhaustion alert
        self.alert_rules["dp_budget_exhaustion"] = AlertRule(
            rule_id="dp_budget_exhaustion",
            metric_name="dp_privacy_budget_remaining",
            condition="<",
            threshold=0.10,  # 10% budget remaining
            severity="critical",
            message_template="Differential privacy budget nearly exhausted: {metric_value:.2%} remaining (threshold: {threshold:.2%})",
            cooldown_seconds=1800.0  # 30 minutes
        )
        
        # Security feature disabled alert
        self.alert_rules["security_feature_disabled"] = AlertRule(
            rule_id="security_feature_disabled",
            metric_name="disabled_security_features",
            condition=">",
            threshold=0,  # Any disabled features
            severity="warning",
            message_template="Security features are disabled: {metric_value:.0f} features disabled",
            cooldown_seconds=3600.0  # 1 hour
        )
    
    def start_monitoring(self):
        """Start security monitoring"""
        if self.monitoring_active:
            logger.warning("Security monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting security monitoring...")
        
        # Start background monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.background_tasks.add(monitoring_task)
        monitoring_task.add_done_callback(self.background_tasks.discard)
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        logger.info("Security monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await self._evaluate_alert_rules()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self):
        """Collect security metrics"""
        try:
            # Get security statistics
            stats = self.security_manager.get_security_stats()
            
            current_time = time.time()
            
            # Collect ZK proof metrics
            zk_metrics = self._collect_zk_proof_metrics(stats, current_time)
            for metric in zk_metrics:
                self._record_metric(metric)
            
            # Collect homomorphic encryption metrics
            he_metrics = self._collect_homomorphic_metrics(stats, current_time)
            for metric in he_metrics:
                self._record_metric(metric)
            
            # Collect SMC metrics
            smc_metrics = self._collect_smc_metrics(stats, current_time)
            for metric in smc_metrics:
                self._record_metric(metric)
            
            # Collect differential privacy metrics
            dp_metrics = self._collect_dp_metrics(stats, current_time)
            for metric in dp_metrics:
                self._record_metric(metric)
            
            # Collect feature status metrics
            feature_metrics = self._collect_feature_metrics(stats, current_time)
            for metric in feature_metrics:
                self._record_metric(metric)
                
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
    
    def _collect_zk_proof_metrics(self, stats: Dict[str, Any], timestamp: float) -> List[SecurityMetric]:
        """Collect zero-knowledge proof metrics"""
        metrics = []
        
        # ZK proof success/failure rates would come from internal tracking
        # For now, we'll simulate some metrics
        zk_success_rate = 0.996  # 99.6% success rate
        zk_failure_rate = 1.0 - zk_success_rate
        
        metrics.append(SecurityMetric(
            name="zk_proof_success_rate",
            value=zk_success_rate,
            timestamp=timestamp,
            tags={"feature": "zero_knowledge_proofs"}
        ))
        
        metrics.append(SecurityMetric(
            name="zk_proof_failure_rate",
            value=zk_failure_rate,
            timestamp=timestamp,
            tags={"feature": "zero_knowledge_proofs"},
            threshold=0.05
        ))
        
        return metrics
    
    def _collect_homomorphic_metrics(self, stats: Dict[str, Any], timestamp: float) -> List[SecurityMetric]:
        """Collect homomorphic encryption metrics"""
        metrics = []
        
        # Simulate homomorphic encryption metrics
        he_success_rate = 0.999  # 99.9% success rate
        he_error_rate = 1.0 - he_success_rate
        
        metrics.append(SecurityMetric(
            name="homomorphic_encryption_success_rate",
            value=he_success_rate,
            timestamp=timestamp,
            tags={"feature": "homomorphic_encryption"}
        ))
        
        metrics.append(SecurityMetric(
            name="homomorphic_encryption_error_rate",
            value=he_error_rate,
            timestamp=timestamp,
            tags={"feature": "homomorphic_encryption"},
            threshold=0.01
        ))
        
        return metrics
    
    def _collect_smc_metrics(self, stats: Dict[str, Any], timestamp: float) -> List[SecurityMetric]:
        """Collect secure multi-party computation metrics"""
        metrics = []
        
        # Simulate SMC metrics
        smc_parties = len(stats.get("parties_in_smc", []))
        smc_success_rate = 0.987  # 98.7% success rate
        smc_failure_rate = 1.0 - smc_success_rate
        
        metrics.append(SecurityMetric(
            name="smc_active_parties",
            value=float(smc_parties),
            timestamp=timestamp,
            tags={"feature": "secure_mpc"}
        ))
        
        metrics.append(SecurityMetric(
            name="smc_success_rate",
            value=smc_success_rate,
            timestamp=timestamp,
            tags={"feature": "secure_mpc"}
        ))
        
        metrics.append(SecurityMetric(
            name="smc_participant_failure_rate",
            value=smc_failure_rate,
            timestamp=timestamp,
            tags={"feature": "secure_mpc"},
            threshold=0.10
        ))
        
        return metrics
    
    def _collect_dp_metrics(self, stats: Dict[str, Any], timestamp: float) -> List[SecurityMetric]:
        """Collect differential privacy metrics"""
        metrics = []
        
        # Get privacy parameters
        privacy_params = stats.get("privacy_parameters", {})
        epsilon = privacy_params.get("epsilon", 1.0)
        delta = privacy_params.get("delta", 1e-5)
        
        # Simulate privacy budget usage (normally this would come from DP system)
        budget_used = 0.8  # 80% of budget used
        budget_remaining = 1.0 - budget_used
        
        metrics.append(SecurityMetric(
            name="dp_epsilon",
            value=epsilon,
            timestamp=timestamp,
            tags={"feature": "differential_privacy"}
        ))
        
        metrics.append(SecurityMetric(
            name="dp_delta",
            value=delta,
            timestamp=timestamp,
            tags={"feature": "differential_privacy"}
        ))
        
        metrics.append(SecurityMetric(
            name="dp_privacy_budget_used",
            value=budget_used,
            timestamp=timestamp,
            tags={"feature": "differential_privacy"}
        ))
        
        metrics.append(SecurityMetric(
            name="dp_privacy_budget_remaining",
            value=budget_remaining,
            timestamp=timestamp,
            tags={"feature": "differential_privacy"},
            threshold=0.10
        ))
        
        return metrics
    
    def _collect_feature_metrics(self, stats: Dict[str, Any], timestamp: float) -> List[SecurityMetric]:
        """Collect security feature status metrics"""
        metrics = []
        
        # Count disabled features
        enabled_features = stats.get("enabled_features", {})
        disabled_count = sum(1 for enabled in enabled_features.values() if not enabled)
        
        metrics.append(SecurityMetric(
            name="disabled_security_features",
            value=float(disabled_count),
            timestamp=timestamp,
            tags={"feature": "security_manager"}
        ))
        
        # Individual feature status
        for feature_name, enabled in enabled_features.items():
            metrics.append(SecurityMetric(
                name=f"feature_{feature_name}_enabled",
                value=1.0 if enabled else 0.0,
                timestamp=timestamp,
                tags={"feature": feature_name}
            ))
        
        return metrics
    
    def _record_metric(self, metric: SecurityMetric):
        """Record a security metric"""
        self.metrics[metric.name].append(metric)
        
        # Check if metric exceeds threshold and trigger alert rule
        if metric.threshold is not None:
            for rule in self.alert_rules.values():
                if rule.metric_name == metric.name:
                    if rule.should_trigger(metric.value, metric.timestamp):
                        alert = rule.trigger(metric.value, metric.timestamp)
                        self._trigger_alert(alert)
    
    async def _evaluate_alert_rules(self):
        """Evaluate alert rules against current metrics"""
        current_time = time.time()
        
        # Check each alert rule
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Get latest metric value
            if rule.metric_name in self.metrics:
                latest_metrics = self.metrics[rule.metric_name]
                if latest_metrics:
                    latest_metric = latest_metrics[-1]
                    if rule.should_trigger(latest_metric.value, current_time):
                        alert = rule.trigger(latest_metric.value, current_time)
                        self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: SecurityAlert):
        """Trigger a security alert"""
        self.alerts.append(alert)
        logger.warning(f"SECURITY ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_event_callback(self, callback: Callable[[SecurityEvent], None]):
        """Add callback for security events"""
        self.event_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """Add callback for security alerts"""
        self.alert_callbacks.append(callback)
    
    def get_recent_events(self, limit: int = 50) -> List[SecurityEvent]:
        """Get recent security events"""
        return list(self.events)[-limit:]
    
    def get_recent_alerts(self, limit: int = 50) -> List[SecurityAlert]:
        """Get recent security alerts"""
        return list(self.alerts)[-limit:]
    
    def get_security_metrics(self, metric_name: Optional[str] = None) -> Dict[str, List[SecurityMetric]]:
        """Get security metrics"""
        if metric_name:
            return {metric_name: list(self.metrics.get(metric_name, []))}
        return {name: list(metrics) for name, metrics in self.metrics.items()}
    
    def get_alert_rules(self) -> Dict[str, AlertRule]:
        """Get all alert rules"""
        return dict(self.alert_rules)
    
    def enable_alert_rule(self, rule_id: str):
        """Enable an alert rule"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            logger.info(f"Alert rule '{rule_id}' enabled")
    
    def disable_alert_rule(self, rule_id: str):
        """Disable an alert rule"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            logger.info(f"Alert rule '{rule_id}' disabled")
    
    def update_alert_rule_threshold(self, rule_id: str, new_threshold: float):
        """Update threshold for an alert rule"""
        if rule_id in self.alert_rules:
            old_threshold = self.alert_rules[rule_id].threshold
            self.alert_rules[rule_id].threshold = new_threshold
            logger.info(f"Alert rule '{rule_id}' threshold updated: {old_threshold} -> {new_threshold}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "events_tracked": len(self.events),
            "metrics_tracked": {name: len(metrics) for name, metrics in self.metrics.items()},
            "alerts_generated": len(self.alerts),
            "alert_rules_active": sum(1 for rule in self.alert_rules.values() if rule.enabled),
            "alert_rules_total": len(self.alert_rules),
            "monitoring_active": self.monitoring_active,
            "callbacks_registered": {
                "event_callbacks": len(self.event_callbacks),
                "alert_callbacks": len(self.alert_callbacks)
            }
        }


# Example usage and integration
async def example_security_monitoring():
    """Example of security monitoring in action"""
    # Create security manager
    security_manager = AdvancedSecurityManager()
    
    # Create security monitor
    monitor = SecurityMonitor(security_manager)
    
    # Add alert callback
    def alert_handler(alert: SecurityAlert):
        print(f"🚨 ALERT: {alert.severity.upper()} - {alert.message}")
        # In real implementation, this could send notifications, log to SIEM, etc.
    
    monitor.add_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some security operations
    for i in range(10):
        # Create ZK proof
        proof = security_manager.create_zk_proof(
            f"secret_{i}".encode(),
            f"statement_{i}",
            "monitoring_demo"
        )
        
        # Verify proof
        is_valid = security_manager.verify_zk_proof(proof, f"statement_{i}")
        
        # Encrypt/decrypt values
        encrypted = security_manager.encrypt_value(i * 10)
        decrypted = security_manager.decrypt_value(encrypted)
        
        # Add parties to SMC
        security_manager.add_party_to_smc(f"party_{i:03d}")
        
        # Private data query
        private_value = security_manager.privatize_data(i * 100, "count")
        
        await asyncio.sleep(1)  # Wait between operations
    
    # Wait for monitoring to collect data
    await asyncio.sleep(5)
    
    # Print monitoring stats
    stats = monitor.get_monitoring_stats()
    print("Monitoring Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Print recent alerts
    recent_alerts = monitor.get_recent_alerts(10)
    print(f"\nRecent Alerts ({len(recent_alerts)}):")
    for alert in recent_alerts:
        print(f"  [{alert.severity.upper()}] {alert.message}")
    
    # Stop monitoring
    monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(example_security_monitoring())
