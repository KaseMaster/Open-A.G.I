#!/usr/bin/env python3
"""
Metrics Exporter for Lambda Attunement Layer
Exports Prometheus metrics for the λ-Attunement Layer
"""

import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Mock Prometheus client - in a real implementation, you would use the actual prometheus_client library
class MockPrometheusClient:
    """Mock Prometheus client for demonstration purposes"""
    
    def __init__(self):
        self.metrics = {}
    
    def Gauge(self, name: str, documentation: str, labelnames: Optional[list] = None):
        """Create a gauge metric"""
        return MockGauge(name, documentation, labelnames)
    
    def Counter(self, name: str, documentation: str, labelnames: Optional[list] = None):
        """Create a counter metric"""
        return MockCounter(name, documentation, labelnames)

class MockGauge:
    """Mock Gauge metric"""
    
    def __init__(self, name: str, documentation: str, labelnames: Optional[list] = None):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
        self.values = {}
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        label_key = tuple(sorted((labels or {}).items())) if labels else None
        self.values[label_key] = value

class MockCounter:
    """Mock Counter metric"""
    
    def __init__(self, name: str, documentation: str, labelnames: Optional[list] = None):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
        self.values = {}
    
    def inc(self, amount: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment counter"""
        label_key = tuple(sorted((labels or {}).items())) if labels else None
        self.values[label_key] = self.values.get(label_key, 0) + amount

# Initialize mock Prometheus client
prometheus_client = MockPrometheusClient()

# Define Prometheus metrics
uhes_alpha_value = prometheus_client.Gauge(
    'uhes_alpha_value',
    'Current α(t) value in Lambda Attunement Controller',
    ['controller_id']
)

uhes_lambda_value = prometheus_client.Gauge(
    'uhes_lambda_value',
    'Computed λ(t,L) for representative L',
    ['controller_id', 'scale_level']
)

uhes_C_hat = prometheus_client.Gauge(
    'uhes_C_hat',
    'Measured Coherence Density proxy',
    ['controller_id']
)

uhes_C_gradient = prometheus_client.Gauge(
    'uhes_C_gradient',
    'Last gradient estimate',
    ['controller_id']
)

uhes_alpha_delta = prometheus_client.Gauge(
    'uhes_alpha_delta',
    'Last applied alpha change',
    ['controller_id']
)

uhes_alpha_accept = prometheus_client.Counter(
    'uhes_alpha_accept',
    'Counter of accepted steps',
    ['controller_id']
)

uhes_alpha_revert = prometheus_client.Counter(
    'uhes_alpha_revert',
    'Counter of reverted steps',
    ['controller_id']
)

uhes_attunement_mode = prometheus_client.Gauge(
    'uhes_attunement_mode',
    'Current operational mode (0=idle,1=gradient,2=pid,3=emergency)',
    ['controller_id']
)

# Add new metrics for the 5-token system
qc_token_T1_staked_total = prometheus_client.Gauge(
    'qc_token_T1_staked_total',
    'Total amount of T1 tokens staked by validators',
    ['network_id']
)

qc_token_T2_rewards_epoch = prometheus_client.Counter(
    'qc_token_T2_rewards_epoch',
    'Total T2 rewards distributed per epoch',
    ['network_id']
)

qc_token_T4_boosts_active = prometheus_client.Gauge(
    'qc_token_T4_boosts_active',
    'Number of active T4 boosts',
    ['network_id']
)

qc_token_T5_memory_contributions = prometheus_client.Counter(
    'qc_token_T5_memory_contributions',
    'Total T5 memory contributions',
    ['network_id']
)

qc_token_T3_governance_votes = prometheus_client.Counter(
    'qc_token_T3_governance_votes',
    'Total T3 governance votes cast',
    ['network_id']
)

@dataclass
class AttunementMetrics:
    """Data class to hold attunement metrics"""
    alpha_value: float
    lambda_value: float
    c_hat: float
    c_gradient: float
    alpha_delta: float
    accept_count: int
    revert_count: int
    mode: int

class MetricsExporter:
    """
    Exports metrics from Lambda Attunement Controller to Prometheus
    """
    
    def __init__(self, controller_id: str = "default"):
        self.controller_id = controller_id
        self.metrics_history: list = []
        self.is_exporting = False
        self.export_interval = 10.0  # seconds
        
    def export_metrics(self, metrics: AttunementMetrics):
        """
        Export attunement metrics to Prometheus
        
        Args:
            metrics: AttunementMetrics object containing current metrics
        """
        try:
            # Export alpha value
            uhes_alpha_value.set(metrics.alpha_value, {'controller_id': self.controller_id})
            
            # Export lambda value (using LΦ as representative scale level)
            uhes_lambda_value.set(metrics.lambda_value, {
                'controller_id': self.controller_id,
                'scale_level': 'LΦ'
            })
            
            # Export coherence density
            uhes_C_hat.set(metrics.c_hat, {'controller_id': self.controller_id})
            
            # Export gradient
            uhes_C_gradient.set(metrics.c_gradient, {'controller_id': self.controller_id})
            
            # Export alpha delta
            uhes_alpha_delta.set(metrics.alpha_delta, {'controller_id': self.controller_id})
            
            # Export mode
            uhes_attunement_mode.set(metrics.mode, {'controller_id': self.controller_id})
            
            # Store metrics history
            self.metrics_history.append({
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            # Keep only recent history (last 100 metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
                
        except Exception as e:
            print(f"Error exporting metrics: {e}")
    
    def increment_accept_counter(self):
        """Increment the accept counter"""
        uhes_alpha_accept.inc(1, {'controller_id': self.controller_id})
    
    def increment_revert_counter(self):
        """Increment the revert counter"""
        uhes_alpha_revert.inc(1, {'controller_id': self.controller_id})
    
    def start_exporting(self, metrics_callback, interval: float = 10.0):
        """
        Start continuous metrics exporting
        
        Args:
            metrics_callback: Function that returns current AttunementMetrics
            interval: Export interval in seconds
        """
        if self.is_exporting:
            print("Metrics exporter is already running")
            return
            
        self.is_exporting = True
        self.export_interval = interval
        
        def export_loop():
            while self.is_exporting:
                try:
                    metrics = metrics_callback()
                    if metrics:
                        self.export_metrics(metrics)
                    time.sleep(self.export_interval)
                except Exception as e:
                    print(f"Error in metrics export loop: {e}")
                    time.sleep(self.export_interval)
        
        export_thread = threading.Thread(target=export_loop, daemon=True)
        export_thread.start()
        print(f"Metrics exporter started for controller {self.controller_id}")
    
    def stop_exporting(self):
        """Stop continuous metrics exporting"""
        self.is_exporting = False
        print(f"Metrics exporter stopped for controller {self.controller_id}")
    
    def get_metrics_history(self) -> list:
        """Get metrics history"""
        return self.metrics_history
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        return {
            'alpha_value': uhes_alpha_value.values,
            'lambda_value': uhes_lambda_value.values,
            'c_hat': uhes_C_hat.values,
            'c_gradient': uhes_C_gradient.values,
            'alpha_delta': uhes_alpha_delta.values,
            'accept_count': uhes_alpha_accept.values,
            'revert_count': uhes_alpha_revert.values,
            'mode': uhes_attunement_mode.values
        }

# Example usage and testing
def demo_metrics_exporter():
    """Demonstrate metrics exporter functionality"""
    print("🔬 Metrics Exporter Demo")
    print("=" * 30)
    
    # Create metrics exporter
    exporter = MetricsExporter("test-controller")
    
    # Create sample metrics
    sample_metrics = AttunementMetrics(
        alpha_value=1.05,
        lambda_value=0.65,
        c_hat=0.85,
        c_gradient=0.02,
        alpha_delta=0.01,
        accept_count=15,
        revert_count=3,
        mode=1
    )
    
    # Export metrics
    exporter.export_metrics(sample_metrics)
    print("✅ Sample metrics exported")
    
    # Increment counters
    exporter.increment_accept_counter()
    exporter.increment_revert_counter()
    print("✅ Counters incremented")
    
    # Show current metrics
    current_metrics = exporter.get_current_metrics()
    print(f"📊 Current metrics: {len(current_metrics)} metric types exported")
    
    print("\n✅ Metrics exporter demo completed!")

if __name__ == "__main__":
    demo_metrics_exporter()