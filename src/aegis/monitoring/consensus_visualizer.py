"""
AEGIS Consensus Visualization Tools
Real-time monitoring and visualization for consensus performance and health
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    animation = None
    Figure = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None
    make_subplots = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConsensusMetrics:
    """Real-time consensus metrics"""
    timestamp: float
    view_number: int
    sequence_number: int
    active_validators: int
    total_nodes: int
    consensus_state: str
    proposal_count: int
    prepare_count: int
    commit_count: int
    avg_response_time: float
    success_rate: float
    throughput: float  # transactions per second
    latency: float  # milliseconds


@dataclass
class ValidatorMetrics:
    """Metrics for individual validators"""
    node_id: str
    response_time: float
    success_rate: float
    participation_rate: float
    reputation_score: float
    last_active: float
    consecutive_failures: int


class ConsensusVisualizer:
    """Real-time consensus visualization and monitoring"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.validator_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: deque = deque(maxlen=100)
        
        # Current state
        self.current_metrics: Optional[ConsensusMetrics] = None
        self.current_validators: Dict[str, ValidatorMetrics] = {}
        
        # Visualization components
        self.fig = None
        self.axes = {}
        self.animation = None
        self.is_running = False
        
        # Data collection
        self.data_lock = threading.Lock()
        self.collected_data = {
            'timestamps': [],
            'throughput': [],
            'latency': [],
            'success_rate': [],
            'active_validators': []
        }
    
    def update_metrics(self, metrics: ConsensusMetrics):
        """Update consensus metrics"""
        with self.data_lock:
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Update collected data for plotting
            self.collected_data['timestamps'].append(metrics.timestamp)
            self.collected_data['throughput'].append(metrics.throughput)
            self.collected_data['latency'].append(metrics.latency)
            self.collected_data['success_rate'].append(metrics.success_rate)
            self.collected_data['active_validators'].append(metrics.active_validators)
            
            # Keep only last 100 data points
            for key in self.collected_data:
                if len(self.collected_data[key]) > 100:
                    self.collected_data[key] = self.collected_data[key][-100:]
    
    def update_validator_metrics(self, validator_metrics: List[ValidatorMetrics]):
        """Update validator metrics"""
        with self.data_lock:
            self.current_validators = {vm.node_id: vm for vm in validator_metrics}
            
            # Update history for each validator
            current_time = time.time()
            for vm in validator_metrics:
                self.validator_history[vm.node_id].append({
                    'timestamp': current_time,
                    'response_time': vm.response_time,
                    'success_rate': vm.success_rate,
                    'reputation_score': vm.reputation_score
                })
    
    def add_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Add an alert"""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        self.alerts.append(alert)
        logger.info(f"Alert [{severity}]: {message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last minute
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'avg_latency': sum(m.latency for m in recent_metrics) / len(recent_metrics),
            'avg_success_rate': sum(m.success_rate for m in recent_metrics) / len(recent_metrics),
            'current_validators': recent_metrics[-1].active_validators if recent_metrics else 0,
            'total_nodes': recent_metrics[-1].total_nodes if recent_metrics else 0,
            'consensus_state': recent_metrics[-1].consensus_state if recent_metrics else "unknown"
        }
    
    def get_validator_performance(self) -> List[Dict[str, Any]]:
        """Get validator performance rankings"""
        if not self.current_validators:
            return []
        
        validators = []
        for node_id, metrics in self.current_validators.items():
            validators.append({
                'node_id': node_id,
                'response_time': metrics.response_time,
                'success_rate': metrics.success_rate,
                'reputation_score': metrics.reputation_score,
                'participation_rate': metrics.participation_rate,
                'status': 'active' if time.time() - metrics.last_active < 30 else 'inactive'
            })
        
        # Sort by reputation score
        validators.sort(key=lambda x: x['reputation_score'], reverse=True)
        return validators
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        anomalies = []
        
        if len(self.metrics_history) < 10:
            return anomalies
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Check for throughput drops
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        current_throughput = recent_metrics[-1].throughput
        
        if current_throughput < avg_throughput * 0.7:  # 30% drop
            anomalies.append({
                'type': 'throughput_drop',
                'severity': 'warning',
                'message': f'Throughput dropped by {((avg_throughput - current_throughput) / avg_throughput * 100):.1f}%'
            })
        
        # Check for latency spikes
        avg_latency = sum(m.latency for m in recent_metrics) / len(recent_metrics)
        current_latency = recent_metrics[-1].latency
        
        if current_latency > avg_latency * 2:  # 2x increase
            anomalies.append({
                'type': 'latency_spike',
                'severity': 'warning',
                'message': f'Latency increased by {((current_latency - avg_latency) / avg_latency * 100):.1f}%'
            })
        
        # Check for success rate drops
        avg_success = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        current_success = recent_metrics[-1].success_rate
        
        if current_success < avg_success * 0.9:  # 10% drop
            anomalies.append({
                'type': 'success_rate_drop',
                'severity': 'warning',
                'message': f'Success rate dropped by {((avg_success - current_success) / avg_success * 100):.1f}%'
            })
        
        return anomalies
    
    def start_visualization(self):
        """Start real-time visualization"""
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available, visualization disabled")
            return
        
        if self.is_running:
            logger.warning("Visualization already running")
            return
        
        self.is_running = True
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('AEGIS Consensus Real-Time Monitoring', fontsize=16)
        
        # Create subplots
        self.axes['throughput'] = plt.subplot(2, 3, 1)
        self.axes['latency'] = plt.subplot(2, 3, 2)
        self.axes['success_rate'] = plt.subplot(2, 3, 3)
        self.axes['validators'] = plt.subplot(2, 3, 4)
        self.axes['state'] = plt.subplot(2, 3, 5)
        self.axes['alerts'] = plt.subplot(2, 3, 6)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.update_interval * 1000,
            blit=False
        )
        
        plt.show()
    
    def _update_plot(self, frame):
        """Update plot with new data"""
        with self.data_lock:
            # Update throughput plot
            self.axes['throughput'].clear()
            self.axes['throughput'].plot(
                self.collected_data['timestamps'],
                self.collected_data['throughput'],
                'b-',
                label='Throughput (TPS)'
            )
            self.axes['throughput'].set_title('Transaction Throughput')
            self.axes['throughput'].set_ylabel('Transactions/Second')
            self.axes['throughput'].legend()
            self.axes['throughput'].grid(True)
            
            # Update latency plot
            self.axes['latency'].clear()
            self.axes['latency'].plot(
                self.collected_data['timestamps'],
                self.collected_data['latency'],
                'r-',
                label='Latency (ms)'
            )
            self.axes['latency'].set_title('Network Latency')
            self.axes['latency'].set_ylabel('Milliseconds')
            self.axes['latency'].legend()
            self.axes['latency'].grid(True)
            
            # Update success rate plot
            self.axes['success_rate'].clear()
            self.axes['success_rate'].plot(
                self.collected_data['timestamps'],
                self.collected_data['success_rate'],
                'g-',
                label='Success Rate'
            )
            self.axes['success_rate'].set_title('Consensus Success Rate')
            self.axes['success_rate'].set_ylabel('Rate (0-1)')
            self.axes['success_rate'].set_ylim(0, 1)
            self.axes['success_rate'].legend()
            self.axes['success_rate'].grid(True)
            
            # Update validator count plot
            self.axes['validators'].clear()
            self.axes['validators'].plot(
                self.collected_data['timestamps'],
                self.collected_data['active_validators'],
                'm-',
                label='Active Validators'
            )
            self.axes['validators'].set_title('Validator Participation')
            self.axes['validators'].set_ylabel('Count')
            self.axes['validators'].legend()
            self.axes['validators'].grid(True)
            
            # Update state visualization
            self.axes['state'].clear()
            if self.current_metrics:
                states = ['IDLE', 'PROPOSING', 'PREPARING', 'COMMITTING', 'FINALIZING']
                state_values = [0] * len(states)
                current_state_index = states.index(self.current_metrics.consensus_state) if self.current_metrics.consensus_state in states else 0
                state_values[current_state_index] = 1
                
                colors = ['lightgray'] * len(states)
                colors[current_state_index] = 'orange'
                
                self.axes['state'].bar(states, state_values, color=colors)
                self.axes['state'].set_title('Consensus State')
                self.axes['state'].set_ylabel('Active')
            
            # Update alerts
            self.axes['alerts'].clear()
            if self.alerts:
                recent_alerts = list(self.alerts)[-5:]  # Last 5 alerts
                alert_texts = [f"{alert['severity'].upper()}: {alert['message']}" for alert in recent_alerts]
                self.axes['alerts'].text(0.1, 0.9, '\n'.join(alert_texts), 
                                       verticalalignment='top',
                                       transform=self.axes['alerts'].transAxes,
                                       fontsize=8)
            self.axes['alerts'].set_title('Recent Alerts')
            self.axes['alerts'].set_xlim(0, 1)
            self.axes['alerts'].set_ylim(0, 1)
            self.axes['alerts'].axis('off')
    
    def stop_visualization(self):
        """Stop real-time visualization"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary()
        validators = self.get_validator_performance()
        anomalies = self.detect_anomalies()
        
        return {
            'timestamp': time.time(),
            'performance_summary': summary,
            'top_validators': validators[:10],  # Top 10 validators
            'anomalies': anomalies,
            'total_alerts': len(self.alerts),
            'active_validators': len(self.current_validators)
        }
    
    def export_data(self, filename: str = None) -> str:
        """Export collected data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consensus_metrics_{timestamp}.json"
        
        export_data = {
            'metrics_history': [
                {
                    'timestamp': m.timestamp,
                    'view_number': m.view_number,
                    'sequence_number': m.sequence_number,
                    'active_validators': m.active_validators,
                    'total_nodes': m.total_nodes,
                    'consensus_state': m.consensus_state,
                    'throughput': m.throughput,
                    'latency': m.latency,
                    'success_rate': m.success_rate
                }
                for m in self.metrics_history
            ],
            'validator_history': {
                node_id: [
                    {
                        'timestamp': entry['timestamp'],
                        'response_time': entry['response_time'],
                        'success_rate': entry['success_rate'],
                        'reputation_score': entry['reputation_score']
                    }
                    for entry in history
                ]
                for node_id, history in self.validator_history.items()
            },
            'alerts': list(self.alerts)
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename


class WebConsensusDashboard:
    """Web-based consensus dashboard using Plotly"""
    
    def __init__(self):
        if not HAS_PLOTLY:
            logger.warning("Plotly not available, web dashboard disabled")
            return
    
    def create_dashboard(self, visualizer: ConsensusVisualizer) -> go.Figure:
        """Create interactive web dashboard"""
        if not HAS_PLOTLY:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Transaction Throughput', 'Network Latency',
                'Success Rate', 'Validator Participation',
                'Validator Performance', 'System Alerts'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Add traces (this would be updated dynamically in a real implementation)
        with visualizer.data_lock:
            timestamps = visualizer.collected_data['timestamps']
            
            # Throughput
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=visualizer.collected_data['throughput'],
                    mode='lines',
                    name='Throughput (TPS)',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Latency
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=visualizer.collected_data['latency'],
                    mode='lines',
                    name='Latency (ms)',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            # Success Rate
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=visualizer.collected_data['success_rate'],
                    mode='lines',
                    name='Success Rate',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Validator Count
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=visualizer.collected_data['active_validators'],
                    mode='lines',
                    name='Active Validators',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='AEGIS Consensus Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_validator_heatmap(self, visualizer: ConsensusVisualizer) -> go.Figure:
        """Create validator performance heatmap"""
        if not HAS_PLOTLY:
            return None
        
        validators = visualizer.get_validator_performance()
        
        if not validators:
            return go.Figure()
        
        # Prepare data for heatmap
        node_ids = [v['node_id'] for v in validators]
        metrics = ['Response Time', 'Success Rate', 'Reputation']
        
        # Create heatmap data
        heatmap_data = []
        for validator in validators:
            heatmap_data.append([
                validator['response_time'],
                validator['success_rate'],
                validator['reputation_score']
            ])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=node_ids,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Validator Performance Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Validators'
        )
        
        return fig


# Example usage and integration
async def simulate_consensus_metrics(visualizer: ConsensusVisualizer):
    """Simulate consensus metrics for testing"""
    import random
    
    for i in range(1000):
        # Simulate metrics
        metrics = ConsensusMetrics(
            timestamp=time.time(),
            view_number=random.randint(1, 10),
            sequence_number=i,
            active_validators=random.randint(20, 50),
            total_nodes=50,
            consensus_state=random.choice(['IDLE', 'PROPOSING', 'PREPARING', 'COMMITTING', 'FINALIZING']),
            proposal_count=random.randint(0, 10),
            prepare_count=random.randint(0, 20),
            commit_count=random.randint(0, 15),
            avg_response_time=random.uniform(10, 100),
            success_rate=random.uniform(0.8, 1.0),
            throughput=random.uniform(100, 1000),
            latency=random.uniform(20, 200)
        )
        
        visualizer.update_metrics(metrics)
        
        # Simulate validator metrics
        validator_metrics = []
        for j in range(metrics.active_validators):
            vm = ValidatorMetrics(
                node_id=f"node_{j:03d}",
                response_time=random.uniform(5, 50),
                success_rate=random.uniform(0.9, 1.0),
                participation_rate=random.uniform(0.8, 1.0),
                reputation_score=random.uniform(80, 100),
                last_active=time.time(),
                consecutive_failures=random.randint(0, 3)
            )
            validator_metrics.append(vm)
        
        visualizer.update_validator_metrics(validator_metrics)
        
        # Randomly generate alerts
        if random.random() < 0.05:  # 5% chance per iteration
            alert_types = ['high_latency', 'low_success_rate', 'validator_down']
            severities = ['info', 'warning', 'critical']
            visualizer.add_alert(
                alert_type=random.choice(alert_types),
                message=f"Simulated alert {i}",
                severity=random.choice(severities)
            )
        
        await asyncio.sleep(0.1)  # Update every 100ms


if __name__ == "__main__":
    # Create visualizer
    visualizer = ConsensusVisualizer(update_interval=0.5)
    
    # Start simulation in background
    simulation_task = asyncio.create_task(simulate_consensus_metrics(visualizer))
    
    # Start visualization (if matplotlib is available)
    if HAS_MATPLOTLIB:
        try:
            visualizer.start_visualization()
        except Exception as e:
            logger.error(f"Error starting visualization: {e}")
    
    # Run simulation for a while
    asyncio.run(asyncio.sleep(30))
    
    # Stop visualization
    visualizer.stop_visualization()
    
    # Generate report
    report = visualizer.generate_report()
    print("Performance Report:")
    print(json.dumps(report, indent=2))
