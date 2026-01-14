#!/usr/bin/env python3
"""
Validator Management Console for Quantum Currency
Implements node monitoring, metrics, and logs management
"""

import sys
import os
import json
import time
import threading
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing modules
from validator_staking import ValidatorStakingSystem
from openagi.onchain_governance import OnChainGovernanceSystem

@dataclass
class NodeMetrics:
    """Represents metrics for a validator node"""
    node_id: str
    timestamp: float
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    disk_usage: float  # Percentage
    network_in: float  # Bytes per second
    network_out: float  # Bytes per second
    block_height: int
    pending_transactions: int
    uptime: float  # Percentage
    last_block_time: float
    peer_count: int
    harmonic_coherence: float  # Coherence score
    chr_score: float  # Reputation score
    staked_amount: float
    delegated_amount: float

@dataclass
class NodeLog:
    """Represents a log entry from a validator node"""
    log_id: str
    node_id: str
    timestamp: float
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: str
    module: str
    details: Optional[Dict] = None

@dataclass
class NodeAlert:
    """Represents an alert for a validator node"""
    alert_id: str
    node_id: str
    timestamp: float
    alert_type: str  # "performance", "security", "network", "consensus"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None

class ValidatorManagementConsole:
    """
    Implements validator management console with node monitoring, metrics, and logs
    """
    
    def __init__(self, console_name: str = "default-console"):
        self.console_name = console_name
        self.console_id = f"console-{int(time.time())}-{hashlib.md5(console_name.encode()).hexdigest()[:8]}"
        self.nodes: Dict[str, NodeMetrics] = {}
        self.logs: List[NodeLog] = []
        self.alerts: List[NodeAlert] = []
        self.staking_system = ValidatorStakingSystem()
        self.governance_system = OnChainGovernanceSystem()
        self.monitoring_config = {
            "metrics_update_interval": 5.0,  # seconds
            "log_retention_days": 7,
            "alert_thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "uptime": 95.0,
                "network_latency": 1000.0,  # ms
                "coherence_score": 0.7
            }
        }
        self._setup_logging()
        self._start_monitoring_thread()
    
    def _setup_logging(self):
        """Set up logging for the console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"validator_console_{self.console_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"ValidatorConsole-{self.console_id}")
    
    def _start_monitoring_thread(self):
        """Start background thread for monitoring"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started monitoring thread")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                self._update_node_metrics()
                self._check_alerts()
                time.sleep(self.monitoring_config["metrics_update_interval"])
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _update_node_metrics(self):
        """Update metrics for all nodes"""
        # In a real implementation, this would collect actual metrics from nodes
        # For this demo, we'll simulate metrics
        
        # Get validators from staking system
        for validator_id, validator in self.staking_system.validators.items():
            # Simulate metrics
            metrics = NodeMetrics(
                node_id=validator_id,
                timestamp=time.time(),
                cpu_usage=np.random.uniform(10, 70),
                memory_usage=np.random.uniform(20, 60),
                disk_usage=np.random.uniform(30, 80),
                network_in=np.random.uniform(1000, 10000),
                network_out=np.random.uniform(1000, 10000),
                block_height=np.random.randint(10000, 99999),
                pending_transactions=np.random.randint(0, 50),
                uptime=validator.uptime * 100,
                last_block_time=time.time() - np.random.uniform(0, 30),
                peer_count=np.random.randint(10, 50),
                harmonic_coherence=np.random.uniform(0.6, 0.95),
                chr_score=validator.chr_score,
                staked_amount=validator.total_staked,
                delegated_amount=validator.total_delegated
            )
            
            self.nodes[validator_id] = metrics
    
    def _check_alerts(self):
        """Check for alerts based on metrics"""
        thresholds = self.monitoring_config["alert_thresholds"]
        
        for node_id, metrics in self.nodes.items():
            # Check CPU usage
            if metrics.cpu_usage > thresholds["cpu_usage"]:
                self._create_alert(
                    node_id, "performance", "high",
                    f"High CPU usage: {metrics.cpu_usage:.1f}%"
                )
            
            # Check memory usage
            if metrics.memory_usage > thresholds["memory_usage"]:
                self._create_alert(
                    node_id, "performance", "high",
                    f"High memory usage: {metrics.memory_usage:.1f}%"
                )
            
            # Check disk usage
            if metrics.disk_usage > thresholds["disk_usage"]:
                self._create_alert(
                    node_id, "performance", "high",
                    f"High disk usage: {metrics.disk_usage:.1f}%"
                )
            
            # Check uptime
            if metrics.uptime < thresholds["uptime"]:
                self._create_alert(
                    node_id, "performance", "medium",
                    f"Low uptime: {metrics.uptime:.1f}%"
                )
            
            # Check coherence score
            if metrics.harmonic_coherence < thresholds["coherence_score"]:
                self._create_alert(
                    node_id, "consensus", "medium",
                    f"Low coherence score: {metrics.harmonic_coherence:.3f}"
                )
    
    def _create_alert(self, node_id: str, alert_type: str, severity: str, message: str):
        """Create a new alert"""
        # Check if similar alert already exists and is unresolved
        for alert in self.alerts:
            if (alert.node_id == node_id and 
                alert.alert_type == alert_type and 
                not alert.resolved and
                alert.message == message):
                return  # Don't create duplicate alerts
        
        alert_id = f"alert-{int(time.time())}-{hashlib.md5(f'{node_id}{message}'.encode()).hexdigest()[:8]}"
        
        alert = NodeAlert(
            alert_id=alert_id,
            node_id=node_id,
            timestamp=time.time(),
            alert_type=alert_type,
            severity=severity,
            message=message
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"New alert for {node_id}: {message}")
    
    def add_node_log(self, node_id: str, level: str, message: str, 
                    module: str = "unknown", details: Optional[Dict] = None):
        """
        Add a log entry from a node
        
        Args:
            node_id: ID of the node
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            module: Module that generated the log
            details: Additional details
        """
        log_id = f"log-{int(time.time())}-{hashlib.md5(f'{node_id}{message}'.encode()).hexdigest()[:8]}"
        
        log_entry = NodeLog(
            log_id=log_id,
            node_id=node_id,
            timestamp=time.time(),
            level=level,
            message=message,
            module=module,
            details=details
        )
        
        self.logs.append(log_entry)
        
        # Log to file as well
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"[{node_id}] {message}")
    
    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """
        Get metrics for a specific node
        
        Args:
            node_id: ID of the node
            
        Returns:
            NodeMetrics if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def get_all_node_metrics(self) -> Dict[str, NodeMetrics]:
        """
        Get metrics for all nodes
        
        Returns:
            Dictionary of node metrics
        """
        return self.nodes.copy()
    
    def get_node_logs(self, node_id: str, limit: int = 50) -> List[NodeLog]:
        """
        Get logs for a specific node
        
        Args:
            node_id: ID of the node
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        node_logs = [log for log in self.logs if log.node_id == node_id]
        node_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return node_logs[:limit]
    
    def get_all_logs(self, limit: int = 100) -> List[NodeLog]:
        """
        Get all logs
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        sorted_logs = sorted(self.logs, key=lambda x: x.timestamp, reverse=True)
        return sorted_logs[:limit]
    
    def get_active_alerts(self) -> List[NodeAlert]:
        """
        Get all active (unresolved) alerts
        
        Returns:
            List of active alerts
        """
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_node(self, node_id: str) -> List[NodeAlert]:
        """
        Get alerts for a specific node
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of alerts for the node
        """
        return [alert for alert in self.alerts if alert.node_id == node_id]
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "admin"):
        """
        Resolve an alert
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = time.time()
                alert.resolved_by = resolved_by
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        return False
    
    def get_system_overview(self) -> Dict:
        """
        Get system overview statistics
        
        Returns:
            Dictionary with system overview
        """
        total_nodes = len(self.nodes)
        active_nodes = len([m for m in self.nodes.values() if m.uptime > 90])
        total_alerts = len(self.alerts)
        active_alerts = len([a for a in self.alerts if not a.resolved])
        
        # Calculate average metrics
        if self.nodes:
            avg_cpu = np.mean([m.cpu_usage for m in self.nodes.values()])
            avg_memory = np.mean([m.memory_usage for m in self.nodes.values()])
            avg_coherence = np.mean([m.harmonic_coherence for m in self.nodes.values()])
        else:
            avg_cpu = 0.0
            avg_memory = 0.0
            avg_coherence = 0.0
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "average_cpu_usage": avg_cpu,
            "average_memory_usage": avg_memory,
            "average_coherence": avg_coherence,
            "staking_stats": self.staking_system.get_system_metrics(),
            "governance_stats": self.governance_system.get_governance_stats()
        }
    
    def generate_health_report(self) -> Dict:
        """
        Generate a health report for the validator network
        
        Returns:
            Dictionary with health report
        """
        overview = self.get_system_overview()
        active_alerts = self.get_active_alerts()
        
        # Categorize alerts by severity
        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        high_alerts = [a for a in active_alerts if a.severity == "high"]
        medium_alerts = [a for a in active_alerts if a.severity == "medium"]
        low_alerts = [a for a in active_alerts if a.severity == "low"]
        
        health_score = 100.0
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
        if high_alerts:
            health_score -= len(high_alerts) * 10
        if medium_alerts:
            health_score -= len(medium_alerts) * 5
        if low_alerts:
            health_score -= len(low_alerts) * 1
        
        health_score = max(0.0, health_score)
        
        return {
            "report_timestamp": time.time(),
            "health_score": health_score,
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "critical",
            "overview": overview,
            "alerts": {
                "critical": len(critical_alerts),
                "high": len(high_alerts),
                "medium": len(medium_alerts),
                "low": len(low_alerts),
                "total": len(active_alerts)
            },
            "recent_alerts": [
                {
                    "node_id": alert.node_id,
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts[:10]
            ]
        }

def demo_validator_console():
    """Demonstrate validator management console capabilities"""
    print("🖥️  Validator Management Console Demo")
    print("=" * 40)
    
    # Create console instance
    console = ValidatorManagementConsole("Quantum Validator Console")
    
    # Show initial system overview
    print("\n📊 System Overview:")
    overview = console.get_system_overview()
    print(f"   Total Nodes: {overview['total_nodes']}")
    print(f"   Active Nodes: {overview['active_nodes']}")
    print(f"   Total Alerts: {overview['total_alerts']}")
    print(f"   Active Alerts: {overview['active_alerts']}")
    print(f"   Avg CPU Usage: {overview['average_cpu_usage']:.1f}%")
    print(f"   Avg Memory Usage: {overview['average_memory_usage']:.1f}%")
    print(f"   Avg Coherence: {overview['average_coherence']:.3f}")
    
    # Simulate adding some nodes and metrics
    print("\n📡 Simulating Node Metrics:")
    
    # Add some sample nodes
    sample_nodes = ["validator-001", "validator-002", "validator-003", "validator-004", "validator-005"]
    
    # Simulate metrics updates
    for _ in range(3):
        console._update_node_metrics()
        time.sleep(1)
    
    # Show node metrics
    print("\n📈 Node Metrics:")
    for node_id in sample_nodes[:3]:
        metrics = console.get_node_metrics(node_id)
        if metrics:
            print(f"   {node_id}:")
            print(f"      CPU: {metrics.cpu_usage:.1f}%")
            print(f"      Memory: {metrics.memory_usage:.1f}%")
            print(f"      Coherence: {metrics.harmonic_coherence:.3f}")
            print(f"      Uptime: {metrics.uptime:.1f}%")
            print(f"      Block Height: {metrics.block_height}")
    
    # Add some logs
    print("\n📝 Adding Node Logs:")
    log_messages = [
        ("INFO", "Block proposal accepted", "consensus"),
        ("WARNING", "High memory usage detected", "monitoring"),
        ("ERROR", "Network connection timeout", "network"),
        ("INFO", "Transaction pool updated", "mempool"),
        ("DEBUG", "Harmonic validation completed", "validation")
    ]
    
    for i, (level, message, module) in enumerate(log_messages):
        node_id = sample_nodes[i % len(sample_nodes)]
        console.add_node_log(node_id, level, message, module)
        print(f"   Added {level} log to {node_id}: {message}")
    
    # Show recent logs
    print("\n📋 Recent Logs:")
    recent_logs = console.get_all_logs(limit=5)
    for log in recent_logs:
        timestamp = datetime.fromtimestamp(log.timestamp).strftime('%H:%M:%S')
        print(f"   [{timestamp}] {log.node_id}: {log.level} - {log.message}")
    
    # Simulate some alerts
    print("\n⚠️  Simulating Alerts:")
    
    # Manually create some alerts for demo
    console._create_alert("validator-001", "performance", "high", "CPU usage exceeded 80%")
    console._create_alert("validator-002", "network", "medium", "Network latency high")
    console._create_alert("validator-003", "consensus", "critical", "Low coherence score detected")
    
    # Show active alerts
    print("\n🚨 Active Alerts:")
    active_alerts = console.get_active_alerts()
    for alert in active_alerts:
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
        print(f"   [{timestamp}] {alert.node_id}: {alert.severity.upper()} - {alert.message}")
    
    # Resolve an alert
    print("\n✅ Resolving an Alert:")
    if active_alerts:
        alert_to_resolve = active_alerts[0]
        success = console.resolve_alert(alert_to_resolve.alert_id, "demo_operator")
        if success:
            print(f"   Resolved alert {alert_to_resolve.alert_id} for {alert_to_resolve.node_id}")
        else:
            print("   Failed to resolve alert")
    
    # Show updated alerts
    print("\n🔔 Updated Alerts:")
    active_alerts = console.get_active_alerts()
    print(f"   Active alerts: {len(active_alerts)}")
    
    # Generate health report
    print("\n🏥 Health Report:")
    health_report = console.generate_health_report()
    print(f"   Health Score: {health_report['health_score']:.1f}/100")
    print(f"   Status: {health_report['status']}")
    print(f"   Critical Alerts: {health_report['alerts']['critical']}")
    print(f"   High Alerts: {health_report['alerts']['high']}")
    
    # Show recent alerts from health report
    if health_report['recent_alerts']:
        print("   Recent Alerts:")
        for alert in health_report['recent_alerts']:
            timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
            print(f"      [{timestamp}] {alert['node_id']}: {alert['severity'].upper()} - {alert['message']}")
    
    print("\n✅ Validator management console demo completed!")

if __name__ == "__main__":
    demo_validator_console()